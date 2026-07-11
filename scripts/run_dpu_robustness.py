from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waferinspectsr.degradation import DegradationConfig, degrade
from waferinspectsr.device import device_report, resolve_device
from waferinspectsr.metrics import aggregate_metric_dicts, apply_temperature, summarize_prediction
from waferinspectsr.models import CompactInspectionSafeSR
from waferinspectsr.protocol import apply_prior_fusion
from waferinspectsr.synthetic import generate_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DPU-WaferSR robustness sweeps over controlled degradations.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-json", required=True)
    parser.add_argument("--output", default="experiments/runs/paper_robustness/dpu_robustness.json")
    parser.add_argument("--num-samples", type=int, default=36)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mc-samples", type=int, default=None)
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


def run_model(
    model: CompactInspectionSafeSR,
    lr_batch: torch.Tensor,
    device: torch.device,
    mc_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.no_grad():
        eval_lr = lr_batch.to(device)
        if mc_samples > 1:
            model.train()
            probs, risks, srs = [], [], []
            for _ in range(mc_samples):
                outputs = model(eval_lr)
                probs.append(outputs["defect_prob"].squeeze(1))
                risks.append(outputs["risk"].squeeze(1))
                srs.append(outputs["sr"].squeeze(1))
            prob_tensor = torch.stack(probs)
            risk_tensor = torch.stack(risks)
            sr_tensor = torch.stack(srs)
            prob = prob_tensor.mean(dim=0).cpu().numpy()
            risk = torch.clamp(risk_tensor.mean(dim=0) + 4.0 * prob_tensor.var(dim=0, unbiased=False), 0.0, 1.0)
            risk_np = risk.cpu().numpy()
            sr = sr_tensor.mean(dim=0).cpu().numpy()
            model.eval()
        else:
            model.eval()
            outputs = model(eval_lr)
            prob = outputs["defect_prob"].squeeze(1).cpu().numpy()
            risk_np = outputs["risk"].squeeze(1).cpu().numpy()
            sr = outputs["sr"].squeeze(1).cpu().numpy()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    return sr, prob, risk_np, 1000.0 * elapsed / max(len(lr_batch), 1)


def apply_protocol(prob: np.ndarray, sr: np.ndarray, protocol: dict) -> np.ndarray:
    fused = apply_prior_fusion(prob, sr, protocol.get("prior_fusion", "none"), float(protocol.get("prior_weight", 0.5)))
    temperature = float(protocol.get("temperature", 1.0))
    if temperature > 0.0:
        fused = np.stack([apply_temperature(p, temperature) for p in fused]).astype(np.float32)
    return fused


def main() -> None:
    args = parse_args()
    run_doc = json.loads(Path(args.run_json).read_text(encoding="utf-8"))
    protocol = run_doc["protocol"]
    device = resolve_device(args.device)
    report = device_report(device)
    mc_samples = args.mc_samples if args.mc_samples is not None else int(protocol.get("mc_samples", 1))
    threshold = float(protocol.get("threshold", 0.5))

    model = CompactInspectionSafeSR(scale=2, channels=24, blocks=2, dropout=0.05).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model_state"])

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
                sr, prob, risk, runtime_ms = run_model(model, lr_batch, device, mc_samples)
                prob = apply_protocol(prob, sr, protocol)

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
                        sr_image=sr[index],
                        hr_image=sample.image,
                    )
                    scalar = {key: value for key, value in metrics.items() if isinstance(value, (float, int))}
                    scalar["runtime_ms"] = runtime_ms
                    rows_by_size.setdefault("all", []).append(scalar)
                    rows_by_size.setdefault(size_bin, []).append(scalar)
                    if size_bin == "clean":
                        rows_by_size.setdefault("clean_false_calls", []).append(scalar)
                for size_bin, rows in rows_by_size.items():
                    sweeps.append(
                        {
                            "blur_sigma": blur_sigma,
                            "noise": noise,
                            "contrast": contrast,
                            "defect_size_bin": size_bin,
                            "baseline": "dpu_wafersr",
                            **aggregate_metric_dicts(rows),
                        }
                    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "protocol": {
                    "checkpoint": args.checkpoint,
                    "run_json": args.run_json,
                    "device": report,
                    "mc_samples": mc_samples,
                    "threshold": threshold,
                    "source_protocol": protocol,
                },
                "sweeps": sweeps,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
