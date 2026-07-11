from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waferinspectsr.baselines import heuristic_defect_probability, uncertainty_from_probability
from waferinspectsr.degradation import DegradationConfig, degrade
from waferinspectsr.device import device_report, resolve_device
from waferinspectsr.metrics import aggregate_metric_dicts, apply_temperature, summarize_prediction
from waferinspectsr.models import CompactInspectionSafeSR, CompactSRLite, CompactUNetDetector
from waferinspectsr.protocol import apply_prior_fusion
from waferinspectsr.synthetic import generate_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run trainable-model robustness sweeps over controlled degradations.")
    parser.add_argument("--srlite-checkpoint", default="experiments/runs/scaled_seed20260602/neural_full/srlite.pt")
    parser.add_argument("--srlite-json", default="experiments/runs/scaled_seed20260602/neural_full/srlite.json")
    parser.add_argument("--unet-checkpoint", default="experiments/runs/scaled_seed20260602/neural_full/unet_detector.pt")
    parser.add_argument("--unet-json", default="experiments/runs/scaled_seed20260602/neural_full/unet_detector.json")
    parser.add_argument("--dpu-checkpoint", default="experiments/runs/scaled_seed20260602/neural_full/dpu_wafersr.pt")
    parser.add_argument("--dpu-json", default="experiments/runs/scaled_seed20260602/neural_full/dpu_wafersr.json")
    parser.add_argument("--output", default="experiments/runs/paper_robustness/trainable_robustness_scaled.json")
    parser.add_argument("--num-samples", type=int, default=36)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def defect_size_bin(mask: np.ndarray) -> str:
    area = int(np.asarray(mask).astype(bool).sum())
    if area == 0:
        return "clean"
    if area < 80:
        return "small"
    if area < 350:
        return "medium"
    return "large"


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def load_srlite(path: str | Path, device: torch.device) -> CompactSRLite:
    checkpoint = load_checkpoint(path, device)
    model = CompactSRLite(scale=2, channels=24, blocks=2).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def load_unet(path: str | Path, run_doc: dict[str, Any], device: torch.device) -> CompactUNetDetector:
    checkpoint = load_checkpoint(path, device)
    channels = int(run_doc.get("protocol", {}).get("channels", 16))
    model = CompactUNetDetector(scale=2, channels=channels).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def load_dpu(path: str | Path, device: torch.device) -> CompactInspectionSafeSR:
    checkpoint = load_checkpoint(path, device)
    model = CompactInspectionSafeSR(scale=2, channels=24, blocks=2, dropout=0.05).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def apply_protocol(prob: np.ndarray, protocol: dict[str, Any], sr: np.ndarray | None = None) -> np.ndarray:
    out = prob
    if sr is not None:
        out = apply_prior_fusion(
            out,
            sr,
            str(protocol.get("prior_fusion", "none")),
            float(protocol.get("prior_weight", 0.5)),
        )
    temperature = float(protocol.get("temperature", 1.0))
    if temperature > 0.0:
        out = np.stack([apply_temperature(p, temperature) for p in out]).astype(np.float32)
    return out


def predict_srlite(
    model: CompactSRLite,
    lr_batch: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    sr_chunks = []
    synchronize(device)
    start = time.perf_counter()
    with torch.no_grad():
        for offset in range(0, len(lr_batch), batch_size):
            batch = lr_batch[offset : offset + batch_size].to(device, non_blocking=True)
            sr_chunks.append(model(batch).squeeze(1).cpu().numpy())
    synchronize(device)
    runtime_ms = 1000.0 * (time.perf_counter() - start) / max(len(lr_batch), 1)
    sr = np.concatenate(sr_chunks, axis=0)
    prob = np.stack([heuristic_defect_probability(image, sigma=2.0) for image in sr]).astype(np.float32)
    risk = np.stack([uncertainty_from_probability(p) for p in prob]).astype(np.float32)
    return sr, prob, risk, runtime_ms


def predict_unet(
    model: CompactUNetDetector,
    lr_batch: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    prob_chunks = []
    synchronize(device)
    start = time.perf_counter()
    with torch.no_grad():
        for offset in range(0, len(lr_batch), batch_size):
            batch = lr_batch[offset : offset + batch_size].to(device, non_blocking=True)
            prob_chunks.append(model(batch).squeeze(1).cpu().numpy())
    synchronize(device)
    runtime_ms = 1000.0 * (time.perf_counter() - start) / max(len(lr_batch), 1)
    prob = np.concatenate(prob_chunks, axis=0)
    risk = np.stack([uncertainty_from_probability(p) for p in prob]).astype(np.float32)
    return prob, risk, runtime_ms


def predict_dpu(
    model: CompactInspectionSafeSR,
    lr_batch: torch.Tensor,
    protocol: dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    mc_samples = int(protocol.get("mc_samples", 1))
    sr_chunks = []
    prob_chunks = []
    risk_chunks = []
    synchronize(device)
    start = time.perf_counter()
    with torch.no_grad():
        for offset in range(0, len(lr_batch), batch_size):
            batch = lr_batch[offset : offset + batch_size].to(device, non_blocking=True)
            if mc_samples > 1:
                model.train()
                probs, risks, srs = [], [], []
                for _ in range(mc_samples):
                    outputs = model(batch)
                    probs.append(outputs["defect_prob"].squeeze(1))
                    risks.append(outputs["risk"].squeeze(1))
                    srs.append(outputs["sr"].squeeze(1))
                prob_tensor = torch.stack(probs)
                risk_tensor = torch.stack(risks)
                sr_tensor = torch.stack(srs)
                mc_var = prob_tensor.var(dim=0, unbiased=False)
                prob_chunks.append(prob_tensor.mean(dim=0).cpu().numpy())
                risk_chunks.append(torch.clamp(risk_tensor.mean(dim=0) + 4.0 * mc_var, 0.0, 1.0).cpu().numpy())
                sr_chunks.append(sr_tensor.mean(dim=0).cpu().numpy())
                model.eval()
            else:
                model.eval()
                outputs = model(batch)
                prob_chunks.append(outputs["defect_prob"].squeeze(1).cpu().numpy())
                risk_chunks.append(outputs["risk"].squeeze(1).cpu().numpy())
                sr_chunks.append(outputs["sr"].squeeze(1).cpu().numpy())
    synchronize(device)
    runtime_ms = 1000.0 * (time.perf_counter() - start) / max(len(lr_batch), 1)
    return (
        np.concatenate(sr_chunks, axis=0),
        np.concatenate(prob_chunks, axis=0),
        np.concatenate(risk_chunks, axis=0),
        runtime_ms,
    )


def summarize_method(
    method: str,
    prob: np.ndarray,
    risk: np.ndarray,
    samples,
    runtime_ms: float,
    threshold: float,
    sr: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    rows_by_size: dict[str, list[dict[str, float]]] = {}
    for index, sample in enumerate(samples):
        size_bin = defect_size_bin(sample.defect_mask)
        metrics = summarize_prediction(
            prob[index],
            sample.defect_mask,
            sample.clean_mask,
            sample.edge_mask,
            risk=risk[index],
            threshold=threshold,
            sr_image=sr[index] if sr is not None else None,
            hr_image=sample.image if sr is not None else None,
        )
        scalar = {key: value for key, value in metrics.items() if isinstance(value, (float, int))}
        scalar["runtime_ms"] = runtime_ms
        rows_by_size.setdefault("all", []).append(scalar)
        rows_by_size.setdefault(size_bin, []).append(scalar)
        if size_bin == "clean":
            rows_by_size.setdefault("clean_false_calls", []).append(scalar)
    out = []
    for size_bin, rows in rows_by_size.items():
        out.append({"defect_size_bin": size_bin, "baseline": method, **aggregate_metric_dicts(rows)})
    return out


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    report = device_report(device)
    srlite_doc = load_json(args.srlite_json)
    unet_doc = load_json(args.unet_json)
    dpu_doc = load_json(args.dpu_json)
    srlite = load_srlite(args.srlite_checkpoint, device)
    unet = load_unet(args.unet_checkpoint, unet_doc, device)
    dpu = load_dpu(args.dpu_checkpoint, device)

    samples = generate_dataset(
        num_samples=args.num_samples,
        height=args.height,
        width=args.width,
        pattern_types=["line_space", "contact_hole"],
        defect_types=["bridge", "gap", "particle", "scratch", "missing_pattern", "none"],
        seed=args.seed,
    )

    sweeps = []
    for blur_sigma in [0.6, 1.2, 1.8]:
        for noise in [0.01, 0.03, 0.06]:
            for contrast in [0.8, 1.0, 1.2]:
                cfg = DegradationConfig(
                    scale=2,
                    blur_sigma=blur_sigma,
                    gaussian_noise=noise,
                    shot_noise=noise,
                    contrast_low=contrast,
                    contrast_high=contrast,
                )
                lrs = [degrade(sample.image, cfg, seed=args.seed + index) for index, sample in enumerate(samples)]
                lr_batch = torch.from_numpy(np.asarray(lrs, dtype=np.float32)[:, None])

                sr, prob, risk, runtime_ms = predict_srlite(srlite, lr_batch, device, args.batch_size)
                prob = apply_protocol(prob, srlite_doc["protocol"])
                for row in summarize_method("srlite_detector", prob, risk, samples, runtime_ms, float(srlite_doc["protocol"]["threshold"]), sr=sr):
                    sweeps.append({"blur_sigma": blur_sigma, "noise": noise, "contrast": contrast, **row})

                prob, risk, runtime_ms = predict_unet(unet, lr_batch, device, args.batch_size)
                prob = apply_protocol(prob, unet_doc["protocol"])
                for row in summarize_method("unet_detector", prob, risk, samples, runtime_ms, float(unet_doc["protocol"]["threshold"])):
                    sweeps.append({"blur_sigma": blur_sigma, "noise": noise, "contrast": contrast, **row})

                sr, prob, risk, runtime_ms = predict_dpu(dpu, lr_batch, dpu_doc["protocol"], device, args.batch_size)
                prob = apply_protocol(prob, dpu_doc["protocol"], sr=sr)
                for row in summarize_method("dpu_wafersr", prob, risk, samples, runtime_ms, float(dpu_doc["protocol"]["threshold"]), sr=sr):
                    sweeps.append({"blur_sigma": blur_sigma, "noise": noise, "contrast": contrast, **row})

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "protocol": {
                    "device": report,
                    "num_samples": args.num_samples,
                    "height": args.height,
                    "width": args.width,
                    "seed": args.seed,
                    "batch_size": args.batch_size,
                    "srlite_checkpoint": args.srlite_checkpoint,
                    "unet_checkpoint": args.unet_checkpoint,
                    "dpu_checkpoint": args.dpu_checkpoint,
                    "srlite_json": args.srlite_json,
                    "unet_json": args.unet_json,
                    "dpu_json": args.dpu_json,
                },
                "sweeps": sweeps,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out}")
    print(f"sweeps={len(sweeps)}")


if __name__ == "__main__":
    main()
