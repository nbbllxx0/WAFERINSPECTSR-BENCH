from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waferinspectsr.baselines import heuristic_defect_probability, uncertainty_from_probability
from waferinspectsr.device import device_report, resolve_device
from waferinspectsr.io import load_npz
from waferinspectsr.metrics import aggregate_metric_dicts, aggregate_recall_fpr_curves, aggregate_risk_coverage
from waferinspectsr.metrics import apply_temperature, fit_temperature
from waferinspectsr.metrics import summarize_prediction
from waferinspectsr.metrics import threshold_for_target_fpr
from waferinspectsr.models import CompactNAFSR, CompactSRLite
from waferinspectsr.protocol import indices_for_splits, parse_split_names, require_indices


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SR-lite reconstruction baseline and evaluate via detector.")
    parser.add_argument("--input", default="data/generated/smoke/samples.npz")
    parser.add_argument("--output", default="experiments/runs/smoke/srlite.pt")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-splits", default="train")
    parser.add_argument("--validation-split", default="val_calib")
    parser.add_argument("--calibration-splits", default=None)
    parser.add_argument("--matched-fpr", type=float, default=0.01)
    parser.add_argument("--fit-temperature", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--min-component-area", type=int, default=1)
    parser.add_argument("--architecture", choices=["residual", "naf"], default="residual")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    report = device_report(device)
    print(f"device={device} torch={report['torch_version']} cuda_available={report['cuda_available']}")
    data = load_npz(args.input)
    splits = [str(value) for value in data.get("split", ["train"] * len(data["hr"]))]
    train_splits = {split.strip() for split in args.train_splits.split(",") if split.strip()}
    train_indices = [idx for idx, split in enumerate(splits) if split in train_splits]
    if not train_indices:
        raise ValueError(f"No samples found for train_splits={sorted(train_splits)}")

    lr = torch.from_numpy(data["lr"][:, None].astype("float32"))
    hr = torch.from_numpy(data["hr"][:, None].astype("float32"))
    loader = DataLoader(
        TensorDataset(lr[train_indices], hr[train_indices]),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
    )

    scale = int(hr.shape[-1] // lr.shape[-1])
    if args.architecture == "naf":
        model = CompactNAFSR(scale=scale, channels=24, blocks=2).to(device)
        method_name = "nafnet_lite_detector"
    else:
        model = CompactSRLite(scale=scale, channels=24, blocks=2).to(device)
        method_name = "srlite_detector"
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    history = []
    synchronize_if_needed(device)
    train_start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch_lr, batch_hr in loader:
            batch_lr = batch_lr.to(device, non_blocking=True)
            batch_hr = batch_hr.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            sr = model(batch_lr)
            loss = F.l1_loss(sr, batch_hr)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch, "loss": float(np.mean(losses))})
        print(f"epoch={epoch} loss={history[-1]['loss']:.4f}")
    synchronize_if_needed(device)
    train_seconds = time.perf_counter() - train_start

    model.eval()
    synchronize_if_needed(device)
    eval_start = time.perf_counter()
    with torch.no_grad():
        sr = model(lr.to(device)).squeeze(1).cpu().numpy()
    probs = [heuristic_defect_probability(sr_image, sigma=2.0) for sr_image in sr]
    risks = [uncertainty_from_probability(prob) for prob in probs]
    synchronize_if_needed(device)
    eval_seconds = time.perf_counter() - eval_start
    runtime_ms_per_sample = 1000.0 * eval_seconds / max(len(sr), 1)

    calibration_splits = parse_split_names(args.calibration_splits, args.validation_split)
    val_indices = indices_for_splits(splits, calibration_splits)
    if args.matched_fpr is not None:
        require_indices(val_indices, calibration_splits, "calibration")
    if val_indices and args.fit_temperature:
        temperature = fit_temperature(
            [probs[idx] for idx in val_indices],
            [data["defect_mask"][idx] for idx in val_indices],
        )
        probs = [apply_temperature(prob, temperature) for prob in probs]
    else:
        temperature = 1.0
    if val_indices and args.matched_fpr is not None:
        threshold = threshold_for_target_fpr(
            [probs[idx] for idx in val_indices],
            [data["clean_mask"][idx] for idx in val_indices],
            args.matched_fpr,
            min_component_area=args.min_component_area,
        )
    else:
        threshold = 0.5

    metric_rows = []
    risk_curves = []
    operating_curves = []
    per_sample = []
    for idx, prob in enumerate(probs):
        metrics = summarize_prediction(
            prob,
            data["defect_mask"][idx],
            data["clean_mask"][idx],
            data["edge_mask"][idx],
            risk=risks[idx],
            threshold=threshold,
            sr_image=sr[idx],
            hr_image=data["hr"][idx],
            min_component_area=args.min_component_area,
        )
        scalar = {key: value for key, value in metrics.items() if isinstance(value, (float, int))}
        scalar["threshold"] = threshold
        scalar["runtime_ms"] = runtime_ms_per_sample
        metric_rows.append(scalar)
        risk_curves.append(metrics["risk_coverage"])
        operating_curves.append(metrics["recall_fpr_curve"])
        per_sample.append({"index": idx, "split": splits[idx], "baseline": method_name, **scalar})

    summary = aggregate_metric_dicts(metric_rows)
    by_split = {}
    for split in sorted(set(splits)):
        rows = [row for row, row_split in zip(metric_rows, splits) if row_split == split]
        if rows:
            by_split[split] = aggregate_metric_dicts(rows)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "history": history}, out)
    out.with_suffix(".json").write_text(
        json.dumps(
            {
                "method": method_name,
                "protocol": {
                    "architecture": args.architecture,
                    "matched_fpr": args.matched_fpr,
                    "validation_split": args.validation_split,
                    "calibration_splits": sorted(calibration_splits),
                    "train_splits": sorted(train_splits),
                    "train_count": len(train_indices),
                    "temperature": temperature,
                    "temperature_fit": bool(args.fit_temperature and val_indices),
                    "threshold": threshold,
                    "device": report,
                    "min_component_area": args.min_component_area,
                    "timing": {
                        "train_seconds": train_seconds,
                        "eval_seconds": eval_seconds,
                        "runtime_ms_per_sample": runtime_ms_per_sample,
                    },
                },
                "history": history,
                "summary": summary,
                "risk_coverage": aggregate_risk_coverage(risk_curves),
                "operating_curve": aggregate_recall_fpr_curves(operating_curves),
                "by_split": by_split,
                "per_sample": per_sample,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
