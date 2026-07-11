from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waferinspectsr.baselines import uncertainty_from_probability
from waferinspectsr.device import device_report, resolve_device
from waferinspectsr.io import load_npz
from waferinspectsr.metrics import aggregate_metric_dicts, aggregate_recall_fpr_curves, aggregate_risk_coverage
from waferinspectsr.metrics import apply_temperature, fit_temperature, summarize_prediction
from waferinspectsr.metrics import threshold_for_target_fpr
from waferinspectsr.models import CompactASPPDetector, CompactUNetDetector, edge_loss
from waferinspectsr.protocol import indices_for_splits, parse_split_names, require_indices


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a detector-only U-Net-style baseline on LR crops.")
    parser.add_argument("--input", default="data/generated/smoke/samples.npz")
    parser.add_argument("--output", default="experiments/runs/smoke/unet_detector.pt")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-splits", default="train")
    parser.add_argument("--validation-split", default="val_calib")
    parser.add_argument("--calibration-splits", default=None)
    parser.add_argument("--matched-fpr", type=float, default=0.01)
    parser.add_argument("--fit-temperature", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--architecture", choices=("unet", "aspp", "deeplabv3"), default="unet")
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--positive-weight", type=float, default=64.0)
    parser.add_argument("--clean-weight", type=float, default=1.0)
    parser.add_argument("--edge-weight", type=float, default=0.1)
    parser.add_argument("--min-component-area", type=int, default=1)
    return parser.parse_args()


class TorchvisionDeepLabDetector(nn.Module):
    """Torchvision DeepLabV3 detector adapted for grayscale LR-to-HR masks."""

    def __init__(self, scale: int = 2) -> None:
        super().__init__()
        try:
            from torchvision.models.segmentation import deeplabv3_resnet50
        except Exception as exc:  # pragma: no cover - depends on optional torchvision install
            raise RuntimeError("torchvision with segmentation models is required for --architecture deeplabv3") from exc

        self.scale = scale
        self.input_adapter = nn.Conv2d(1, 3, kernel_size=1)
        self.model = deeplabv3_resnet50(weights=None, weights_backbone=None, num_classes=1, aux_loss=False)

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        logits = self.model(self.input_adapter(lr))["out"]
        if self.scale != 1:
            logits = F.interpolate(logits, scale_factor=self.scale, mode="bilinear", align_corners=False)
        return torch.sigmoid(logits)


def detector_loss(
    prob: torch.Tensor,
    defect_mask: torch.Tensor,
    clean_mask: torch.Tensor,
    positive_weight: float,
    clean_weight: float,
    edge_weight: float,
) -> dict[str, torch.Tensor]:
    prob = prob.clamp(1e-5, 1.0 - 1e-5)
    weights = 1.0 + (float(positive_weight) - 1.0) * defect_mask
    defect = F.binary_cross_entropy(prob, defect_mask, weight=weights)
    clean_values = prob[clean_mask > 0.5]
    if clean_values.numel() == 0:
        clean = prob.new_tensor(0.0)
    else:
        clean = F.binary_cross_entropy(clean_values, torch.zeros_like(clean_values))
    edges = edge_loss(prob, defect_mask)
    total = defect + float(clean_weight) * clean + float(edge_weight) * edges
    return {"total": total, "defect": defect, "clean": clean, "edge": edges}


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
    train_tensors = TensorDataset(lr[train_indices], defect[train_indices], clean[train_indices])
    loader = DataLoader(
        train_tensors,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
    )

    scale = int(hr.shape[-1] // lr.shape[-1])
    if args.architecture == "aspp":
        model = CompactASPPDetector(scale=scale, channels=args.channels).to(device)
        method = "aspp_detector"
        architecture = "compact_aspp_detector"
    elif args.architecture == "deeplabv3":
        model = TorchvisionDeepLabDetector(scale=scale).to(device)
        method = "deeplabv3_resnet50_detector"
        architecture = "torchvision_deeplabv3_resnet50"
    else:
        model = CompactUNetDetector(scale=scale, channels=args.channels).to(device)
        method = "unet_detector"
        architecture = "compact_unet_detector"
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    history = []
    synchronize_if_needed(device)
    train_start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        epoch_losses: list[dict[str, float]] = []
        for batch_lr, batch_defect, batch_clean in loader:
            batch_lr = batch_lr.to(device, non_blocking=True)
            batch_defect = batch_defect.to(device, non_blocking=True)
            batch_clean = batch_clean.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prob = model(batch_lr)
            losses = detector_loss(
                prob,
                batch_defect,
                batch_clean,
                args.positive_weight,
                args.clean_weight,
                args.edge_weight,
            )
            losses["total"].backward()
            optimizer.step()
            epoch_losses.append({key: float(value.detach().cpu()) for key, value in losses.items()})
        history.append(
            {
                "epoch": epoch,
                "loss": float(np.mean([item["total"] for item in epoch_losses])),
                "defect": float(np.mean([item["defect"] for item in epoch_losses])),
                "clean": float(np.mean([item["clean"] for item in epoch_losses])),
                "edge": float(np.mean([item["edge"] for item in epoch_losses])),
            }
        )
        print(f"epoch={epoch} loss={history[-1]['loss']:.4f}")
    synchronize_if_needed(device)
    train_seconds = time.perf_counter() - train_start

    model.eval()
    synchronize_if_needed(device)
    eval_start = time.perf_counter()
    with torch.no_grad():
        prob = model(lr.to(device)).squeeze(1).cpu().numpy()
        sr = F.interpolate(lr.to(device), size=hr.shape[-2:], mode="bilinear", align_corners=False).squeeze(1).cpu().numpy()
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
        prob = np.stack([apply_temperature(item, temperature) for item in prob])
    else:
        temperature = 1.0
    risks = [uncertainty_from_probability(item) for item in prob]
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
    per_sample = []
    for idx, score in enumerate(prob):
        metrics = summarize_prediction(
            score,
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
        per_sample.append({"index": idx, "split": splits[idx], "baseline": method, **scalar})

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
                "method": method,
                "protocol": {
                    "architecture": architecture,
                    "matched_fpr": args.matched_fpr,
                    "validation_split": args.validation_split,
                    "calibration_splits": sorted(calibration_splits),
                    "train_splits": sorted(train_splits),
                    "train_count": len(train_indices),
                    "temperature": temperature,
                    "temperature_fit": bool(args.fit_temperature and val_indices),
                    "threshold": threshold,
                    "device": report,
                    "channels": args.channels,
                    "positive_weight": args.positive_weight,
                    "clean_weight": args.clean_weight,
                    "edge_weight": args.edge_weight,
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
