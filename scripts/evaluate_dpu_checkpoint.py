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

from waferinspectsr.device import device_report, resolve_device
from waferinspectsr.io import load_npz
from waferinspectsr.metrics import aggregate_metric_dicts, aggregate_recall_fpr_curves, aggregate_risk_coverage
from waferinspectsr.metrics import apply_temperature, fit_temperature, summarize_prediction, threshold_for_target_fpr
from waferinspectsr.models import CompactInspectionSafeSR
from waferinspectsr.protocol import apply_prior_fusion, indices_for_splits, parse_split_names, require_indices


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained DPU-WaferSR checkpoint under a chosen protocol.")
    parser.add_argument("--input", required=True, help="Materialized benchmark NPZ.")
    parser.add_argument("--checkpoint", required=True, help="DPU checkpoint produced by train_dpu_wafersr.py.")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--matched-fpr", type=float, default=0.0003)
    parser.add_argument("--validation-split", default="val_calib")
    parser.add_argument("--calibration-splits", default=None)
    parser.add_argument("--mc-samples", type=int, default=4)
    parser.add_argument("--fit-temperature", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument(
        "--prior-fusion",
        choices=["none", "geometric", "average"],
        default="geometric",
        help="Fuse learned defect probabilities with an SR local-contrast inspection prior.",
    )
    parser.add_argument("--prior-weight", type=float, default=0.5)
    parser.add_argument("--min-component-area", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    return parser.parse_args()


def checkpoint_state(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError(f"Checkpoint {path} does not contain a model_state entry")
    return checkpoint


def evaluate_probabilities(
    model: CompactInspectionSafeSR,
    lr: torch.Tensor,
    device: torch.device,
    mc_samples: int,
    eval_batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    prob_chunks = []
    risk_chunks = []
    sr_chunks = []
    synchronize_if_needed(device)
    start_time = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(lr), eval_batch_size):
            eval_lr = lr[start : start + eval_batch_size].to(device, non_blocking=True)
            if mc_samples > 1:
                model.train()
                probs = []
                risks = []
                srs = []
                for _ in range(mc_samples):
                    outputs = model(eval_lr)
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
                outputs = model(eval_lr)
                prob_chunks.append(outputs["defect_prob"].squeeze(1).cpu().numpy())
                risk_chunks.append(outputs["risk"].squeeze(1).cpu().numpy())
                sr_chunks.append(outputs["sr"].squeeze(1).cpu().numpy())
    synchronize_if_needed(device)
    elapsed = time.perf_counter() - start_time
    return (
        np.concatenate(prob_chunks, axis=0),
        np.concatenate(risk_chunks, axis=0),
        np.concatenate(sr_chunks, axis=0),
        elapsed,
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    report = device_report(device)
    data = load_npz(args.input)
    splits = [str(value) for value in data.get("split", ["train"] * len(data["hr"]))]
    lr = torch.from_numpy(data["lr"][:, None].astype("float32"))
    hr = data["hr"].astype("float32")
    scale = int(hr.shape[-1] // lr.shape[-1])

    checkpoint = checkpoint_state(Path(args.checkpoint))
    model = CompactInspectionSafeSR(
        scale=scale,
        channels=24,
        blocks=2,
        dropout=0.05,
        use_sr=bool(checkpoint.get("use_sr", True)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    eval_batch_size = args.eval_batch_size if args.eval_batch_size > 0 else len(lr)
    prob, risk, sr, eval_seconds = evaluate_probabilities(
        model=model,
        lr=lr,
        device=device,
        mc_samples=args.mc_samples,
        eval_batch_size=eval_batch_size,
    )
    prob = apply_prior_fusion(prob, sr, args.prior_fusion, args.prior_weight)
    runtime_ms_per_sample = 1000.0 * eval_seconds / max(len(prob), 1)

    calibration_splits = parse_split_names(args.calibration_splits, args.validation_split)
    val_indices = indices_for_splits(splits, calibration_splits)
    if args.matched_fpr is not None:
        require_indices(val_indices, calibration_splits, "calibration")
    if val_indices and args.fit_temperature:
        temperature = fit_temperature(
            [prob[idx] for idx in val_indices],
            [data["defect_mask"][idx] for idx in val_indices],
        )
        prob = np.stack([apply_temperature(item, temperature) for item in prob])
    else:
        temperature = 1.0
    if val_indices and args.matched_fpr is not None:
        threshold = threshold_for_target_fpr(
            [prob[idx] for idx in val_indices],
            [data["clean_mask"][idx] for idx in val_indices],
            args.matched_fpr,
            min_component_area=args.min_component_area,
        )
    else:
        threshold = 0.5

    metric_rows = []
    risk_curves = []
    operating_curves = []
    for index in range(len(prob)):
        metrics = summarize_prediction(
            prob[index],
            data["defect_mask"][index],
            data["clean_mask"][index],
            data["edge_mask"][index],
            risk=risk[index],
            threshold=threshold,
            sr_image=sr[index],
            hr_image=hr[index],
            min_component_area=args.min_component_area,
        )
        scalar = {key: value for key, value in metrics.items() if isinstance(value, (float, int))}
        scalar["threshold"] = threshold
        scalar["runtime_ms"] = runtime_ms_per_sample
        metric_rows.append(scalar)
        risk_curves.append(metrics["risk_coverage"])
        operating_curves.append(metrics["recall_fpr_curve"])

    summary = aggregate_metric_dicts(metric_rows)
    by_split = {}
    for split in sorted(set(splits)):
        rows = [row for row, row_split in zip(metric_rows, splits) if row_split == split]
        if rows:
            by_split[split] = aggregate_metric_dicts(rows)

    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "method": "dpu_wafersr_checkpoint",
                "source_checkpoint": str(Path(args.checkpoint)),
                "checkpoint_ablation": checkpoint.get("ablation"),
                "checkpoint_history": checkpoint.get("history"),
                "protocol": {
                    "matched_fpr": args.matched_fpr,
                    "validation_split": args.validation_split,
                    "calibration_splits": sorted(calibration_splits),
                    "seed": args.seed,
                    "mc_samples": args.mc_samples,
                    "temperature": temperature,
                    "temperature_fit": bool(args.fit_temperature and val_indices),
                    "threshold": threshold,
                    "device": report,
                    "prior_fusion": args.prior_fusion,
                    "prior_weight": args.prior_weight,
                    "min_component_area": args.min_component_area,
                    "eval_batch_size": args.eval_batch_size,
                    "timing": {
                        "eval_seconds": eval_seconds,
                        "runtime_ms_per_sample": runtime_ms_per_sample,
                    },
                },
                "summary": summary,
                "risk_coverage": aggregate_risk_coverage(risk_curves),
                "operating_curve": aggregate_recall_fpr_curves(operating_curves),
                "by_split": by_split,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {output}")
    print(f"threshold={threshold:.6g} runtime_ms={runtime_ms_per_sample:.3f}")


if __name__ == "__main__":
    main()
