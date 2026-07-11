from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waferinspectsr.device import device_report, resolve_device
from waferinspectsr.io import load_npz
from waferinspectsr.metrics import aggregate_metric_dicts, aggregate_recall_fpr_curves, aggregate_risk_coverage
from waferinspectsr.metrics import summarize_prediction
from waferinspectsr.metrics import apply_temperature, fit_temperature, threshold_for_target_fpr
from waferinspectsr.models import CompactInspectionSafeSR, inspection_safe_loss
from waferinspectsr.protocol import apply_prior_fusion, indices_for_splits, parse_split_names, require_indices


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate DPU-WaferSR.")
    parser.add_argument("--input", default="data/generated/smoke/samples.npz")
    parser.add_argument("--output", default="experiments/runs/smoke/dpu_wafersr.pt")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--ablation",
        choices=["sr_only", "task", "task_hallucination", "full"],
        default="full",
    )
    parser.add_argument(
        "--use-sr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When disabled, train the same stem/trunk/defect pathway without the SR reconstruction branch.",
    )
    parser.add_argument("--matched-fpr", type=float, default=0.01)
    parser.add_argument("--validation-split", default="val_calib")
    parser.add_argument("--calibration-splits", default=None)
    parser.add_argument("--train-splits", default="train")
    parser.add_argument("--mc-samples", type=int, default=4)
    parser.add_argument("--fit-temperature", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--positive-weight", type=float, default=64.0)
    parser.add_argument("--hallucination-weight", type=float, default=None)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument(
        "--prior-fusion",
        choices=["none", "geometric", "average"],
        default="geometric",
        help="Fuse learned defect probabilities with an SR local-contrast inspection prior.",
    )
    parser.add_argument("--prior-weight", type=float, default=0.5)
    parser.add_argument("--min-component-area", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=64, help="Batch size for full-dataset evaluation.")
    return parser.parse_args()


def weights_for_ablation(name: str, positive_weight: float) -> dict[str, float]:
    presets = {
        "sr_only": {
            "recon": 1.0,
            "defect": 0.0,
            "hallucination": 0.0,
            "edge": 0.0,
            "calibration": 0.0,
            "positive": 1.0,
        },
        "task": {
            "recon": 1.0,
            "defect": 1.0,
            "hallucination": 0.0,
            "edge": 0.0,
            "calibration": 0.0,
            "positive": positive_weight,
        },
        "task_hallucination": {
            "recon": 1.0,
            "defect": 1.0,
            "hallucination": 1.0,
            "edge": 0.0,
            "calibration": 0.0,
            "positive": positive_weight,
        },
        "full": {
            "recon": 1.0,
            "defect": 1.0,
            "hallucination": 1.0,
            "edge": 0.2,
            "calibration": 0.05,
            "positive": positive_weight,
        },
    }
    return presets[name]


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
    defect = torch.from_numpy(data["defect_mask"][:, None].astype("float32"))
    clean = torch.from_numpy(data["clean_mask"][:, None].astype("float32"))
    train_tensors = TensorDataset(lr[train_indices], hr[train_indices], defect[train_indices], clean[train_indices])
    loader = DataLoader(
        train_tensors,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
    )

    scale = int(hr.shape[-1] // lr.shape[-1])
    model = CompactInspectionSafeSR(
        scale=scale, channels=24, blocks=2, dropout=0.05, use_sr=args.use_sr
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    weights = weights_for_ablation(args.ablation, args.positive_weight)
    if args.hallucination_weight is not None:
        weights["hallucination"] = args.hallucination_weight
    if not args.use_sr:
        # No reconstruction target when the SR branch is removed.
        weights["recon"] = 0.0
    history = []
    synchronize_if_needed(device)
    train_start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch_lr, batch_hr, batch_defect, batch_clean in loader:
            batch_lr = batch_lr.to(device, non_blocking=True)
            batch_hr = batch_hr.to(device, non_blocking=True)
            batch_defect = batch_defect.to(device, non_blocking=True)
            batch_clean = batch_clean.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch_lr)
            loss_dict = inspection_safe_loss(outputs, batch_hr, batch_defect, batch_clean, weights=weights)
            loss_dict["total"].backward()
            optimizer.step()
            losses.append(float(loss_dict["total"].detach().cpu()))
        history.append({"epoch": epoch, "loss": float(np.mean(losses))})
        print(f"epoch={epoch} loss={history[-1]['loss']:.4f}")
    synchronize_if_needed(device)
    train_seconds = time.perf_counter() - train_start

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    metric_rows = []
    risk_curves = []
    operating_curves = []
    synchronize_if_needed(device)
    eval_start = time.perf_counter()
    eval_batch_size = args.eval_batch_size if args.eval_batch_size > 0 else len(lr)
    prob_chunks = []
    risk_chunks = []
    sr_chunks = []
    with torch.no_grad():
        for start in range(0, len(lr), eval_batch_size):
            eval_lr = lr[start : start + eval_batch_size].to(device, non_blocking=True)
            if args.mc_samples > 1:
                model.train()
                probs = []
                risks = []
                srs = []
                for _ in range(args.mc_samples):
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
    prob = np.concatenate(prob_chunks, axis=0)
    risk = np.concatenate(risk_chunks, axis=0)
    sr = np.concatenate(sr_chunks, axis=0)
    prob = apply_prior_fusion(prob, sr, args.prior_fusion, args.prior_weight)
    synchronize_if_needed(device)
    eval_seconds = time.perf_counter() - eval_start
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
        prob = np.stack([apply_temperature(p, temperature) for p in prob])
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
    for index in range(len(prob)):
        metrics = summarize_prediction(
            prob[index],
            data["defect_mask"][index],
            data["clean_mask"][index],
            data["edge_mask"][index],
            risk=risk[index],
            threshold=threshold,
            sr_image=sr[index],
            hr_image=data["hr"][index],
            min_component_area=args.min_component_area,
        )
        scalar = {key: value for key, value in metrics.items() if isinstance(value, (float, int))}
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

    torch.save(
        {
            "model_state": model.state_dict(),
            "history": history,
            "ablation": args.ablation,
            "use_sr": bool(args.use_sr),
        },
        out,
    )
    out.with_suffix(".json").write_text(
        json.dumps(
            {
                "ablation": args.ablation,
                "use_sr": bool(args.use_sr),
                "weights": weights,
                "protocol": {
                    "matched_fpr": args.matched_fpr,
                    "validation_split": args.validation_split,
                    "calibration_splits": sorted(calibration_splits),
                    "train_splits": sorted(train_splits),
                    "train_count": len(train_indices),
                    "mc_samples": args.mc_samples,
                    "temperature": temperature,
                    "temperature_fit": bool(args.fit_temperature and val_indices),
                    "threshold": threshold,
                    "device": report,
                    "prior_fusion": args.prior_fusion,
                    "prior_weight": args.prior_weight,
                    "use_sr": bool(args.use_sr),
                    "min_component_area": args.min_component_area,
                    "eval_batch_size": args.eval_batch_size,
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
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
