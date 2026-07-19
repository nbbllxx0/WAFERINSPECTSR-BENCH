"""Re-evaluate primary methods with one detector and one calibration protocol.

Track A changes only the image transformation. Every transformed image is scored
by the same local-residual detector with the same smoothing parameter. Track B
reuses trained DeepLabV3 checkpoints but applies the same three-stage operating-
point rule: fit calibration on validation data, select on an independent clean
calibration split, and evaluate held-out splits once.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from waferinspectsr.baselines import (  # noqa: E402
    _wiener_sharpen,
    heuristic_defect_probability,
    uncertainty_from_probability,
)
from waferinspectsr.degradation import upsample_lr  # noqa: E402
from waferinspectsr.image_ops import (  # noqa: E402
    gaussian_filter,
    gaussian_gradient_magnitude,
    resize_image,
)
from waferinspectsr.io import load_npz  # noqa: E402
from waferinspectsr.metrics import (  # noqa: E402
    aggregate_metric_dicts,
    apply_temperature,
    fit_temperature,
    summarize_prediction,
    threshold_for_target_fpr,
)
from waferinspectsr.models import CompactNAFSR, CompactSRLite  # noqa: E402
from waferinspectsr.protocol import indices_for_splits, parse_split_names  # noqa: E402


CANONICAL_SEEDS = (7, 11, 13, 17, 19, 23, 29, 31, 37, 41)
TARGET_FPR = 3e-4
FEASIBILITY_TOLERANCE = 1.5
CALIBRATION_TARGETS = (3e-4, 2e-4, 1.5e-4, 1e-4, 7.5e-5, 5e-5, 3e-5)
COMMON_DETECTOR_SIGMA = 2.0


class TorchvisionDeepLabDetector(nn.Module):
    """The exact grayscale LR-to-HR DeepLabV3 architecture used in training."""

    def __init__(self, scale: int = 2) -> None:
        super().__init__()
        from torchvision.models.segmentation import deeplabv3_resnet50

        self.scale = scale
        self.input_adapter = nn.Conv2d(1, 3, kernel_size=1)
        self.model = deeplabv3_resnet50(
            weights=None,
            weights_backbone=None,
            num_classes=1,
            aux_loss=False,
        )

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        logits = self.model(self.input_adapter(lr))["out"]
        if self.scale != 1:
            logits = F.interpolate(
                logits,
                scale_factor=self.scale,
                mode="bilinear",
                align_corners=False,
            )
        return torch.sigmoid(logits)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "runs" / "tsm_seed_sweep",
    )
    parser.add_argument(
        "--naf-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "runs" / "tsm_seed_sweep",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "tsm_unified_protocol",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in CANONICAL_SEEDS),
    )
    parser.add_argument("--skip-deeplab", action="store_true")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def infer_sr_model(
    model: nn.Module,
    lr: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    tensor = torch.from_numpy(lr[:, None].astype("float32"))
    with torch.no_grad():
        for start in range(0, len(tensor), batch_size):
            batch = tensor[start : start + batch_size].to(device, non_blocking=True)
            outputs.append(model(batch).squeeze(1).cpu().numpy())
    return np.concatenate(outputs)


def infer_deeplab(
    checkpoint: Path,
    lr: np.ndarray,
    scale: int,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    model = TorchvisionDeepLabDetector(scale=scale).to(device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state"])
    model.eval()
    outputs: list[np.ndarray] = []
    tensor = torch.from_numpy(lr[:, None].astype("float32"))
    with torch.no_grad():
        for start in range(0, len(tensor), batch_size):
            batch = tensor[start : start + batch_size].to(device, non_blocking=True)
            outputs.append(model(batch).squeeze(1).cpu().numpy())
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(outputs)


def make_track_a_images(
    data: dict[str, np.ndarray],
    seed_dir: Path,
    naf_checkpoint: Path,
    device: torch.device,
) -> dict[str, np.ndarray]:
    lr = data["lr"]
    target_shape = tuple(int(v) for v in data["hr"].shape[-2:])
    images: dict[str, list[np.ndarray] | np.ndarray] = {
        "bicubic_reference": [],
        "nearest": [],
        "bilinear": [],
        "lanczos": [],
        "denoise_lanczos": [],
        "wiener": [],
        "sharpened": [],
        "naf_heuristic": [],
    }
    for sample in lr:
        bicubic = upsample_lr(sample, target_shape)
        images["bicubic_reference"].append(bicubic)
        images["nearest"].append(resize_image(sample, target_shape, mode="nearest"))
        images["bilinear"].append(resize_image(sample, target_shape, mode="bilinear"))
        images["lanczos"].append(resize_image(sample, target_shape, mode="lanczos"))
        denoised = gaussian_filter(sample, sigma=0.55)
        images["denoise_lanczos"].append(
            resize_image(denoised, target_shape, mode="lanczos")
        )
        images["wiener"].append(
            _wiener_sharpen(bicubic, sigma=1.0, balance=0.03)
        )
        blur = gaussian_filter(bicubic, sigma=1.0)
        images["sharpened"].append(
            np.clip(bicubic + 1.2 * (bicubic - blur), 0.0, 1.0).astype(np.float32)
        )
        low = gaussian_filter(bicubic, sigma=1.4)
        detail = bicubic - low
        gate = gaussian_gradient_magnitude(bicubic, sigma=1.0)
        if gate.max() > 0:
            gate = gate / gate.max()
        images["naf_heuristic"].append(
            np.clip(bicubic + 0.9 * detail * (0.5 + 0.5 * gate), 0.0, 1.0).astype(
                np.float32
            )
        )

    stacked = {
        name: np.stack(values).astype(np.float32)
        for name, values in images.items()
        if isinstance(values, list)
    }
    scale = int(data["hr"].shape[-1] // data["lr"].shape[-1])

    srlite_state = torch.load(seed_dir / "srlite.pt", map_location="cpu", weights_only=False)
    srlite = CompactSRLite(scale=scale, channels=24, blocks=2).to(device)
    srlite.load_state_dict(srlite_state["model_state"])
    stacked["sr_lite"] = infer_sr_model(srlite, lr, device)
    del srlite

    naf_state = torch.load(naf_checkpoint, map_location="cpu", weights_only=False)
    naf = CompactNAFSR(scale=scale, channels=24, blocks=2).to(device)
    naf.load_state_dict(naf_state["model_state"])
    stacked["naf_trained"] = infer_sr_model(naf, lr, device)
    del naf
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return stacked


def clean_fpr(
    probabilities: np.ndarray,
    clean_masks: np.ndarray,
    indices: list[int],
    threshold: float,
) -> float:
    values = []
    for index in indices:
        mask = clean_masks[index] > 0.5
        denominator = int(mask.sum())
        values.append(
            float(np.logical_and(probabilities[index] >= threshold, mask).sum())
            / max(denominator, 1)
        )
    return float(np.mean(values))


def summarize_split(
    probabilities: np.ndarray,
    risks: np.ndarray,
    sr_images: np.ndarray,
    data: dict[str, np.ndarray],
    indices: list[int],
    threshold: float,
) -> dict[str, float]:
    rows = []
    for index in indices:
        metrics = summarize_prediction(
            probabilities[index],
            data["defect_mask"][index],
            data["clean_mask"][index],
            data["edge_mask"][index],
            risk=risks[index],
            threshold=threshold,
            sr_image=sr_images[index],
            hr_image=data["hr"][index],
        )
        rows.append(
            {key: value for key, value in metrics.items() if isinstance(value, (float, int))}
        )
    return aggregate_metric_dicts(rows)


def select_and_evaluate(
    method: str,
    raw_probabilities: np.ndarray,
    sr_images: np.ndarray,
    data: dict[str, np.ndarray],
    splits: list[str],
) -> dict[str, object]:
    val_indices = indices_for_splits(splits, parse_split_names("val_calib", "val_calib"))
    clean_indices = indices_for_splits(
        splits, parse_split_names("clean_calib", "clean_calib")
    )
    temperature = fit_temperature(
        [raw_probabilities[index] for index in val_indices],
        [data["defect_mask"][index] for index in val_indices],
    )
    probabilities = np.stack(
        [apply_temperature(probability, temperature) for probability in raw_probabilities]
    )
    risks = np.stack([uncertainty_from_probability(probability) for probability in probabilities])

    candidates = []
    selected = None
    for target in CALIBRATION_TARGETS:
        threshold = threshold_for_target_fpr(
            [probabilities[index] for index in val_indices],
            [data["clean_mask"][index] for index in val_indices],
            target,
        )
        row = {
            "calibration_target": target,
            "threshold": float(threshold),
            "clean_calibration_fpr": clean_fpr(
                probabilities,
                data["clean_mask"],
                clean_indices,
                threshold,
            ),
        }
        candidates.append(row)
        if (
            selected is None
            and row["clean_calibration_fpr"]
            <= TARGET_FPR * FEASIBILITY_TOLERANCE
        ):
            selected = dict(row)
            selected["selection_fallback"] = False
    if selected is None:
        selected = dict(candidates[-1])
        selected["selection_fallback"] = True

    by_split = {}
    for split in ("test", "weak_test", "clean_test", "ood_test"):
        indices = [index for index, value in enumerate(splits) if value == split]
        if indices:
            by_split[split] = summarize_split(
                probabilities,
                risks,
                sr_images,
                data,
                indices,
                float(selected["threshold"]),
            )
    test_fpr = float(by_split["test"]["false_positive_rate_mean"])
    return {
        "method": method,
        "temperature": float(temperature),
        "selected": selected,
        "candidates": candidates,
        "test_feasible": test_fpr <= TARGET_FPR * FEASIBILITY_TOLERANCE,
        "by_split": by_split,
    }


def flatten_result(seed: int, result: dict[str, object]) -> dict[str, object]:
    by_split = result["by_split"]
    selected = result["selected"]
    row: dict[str, object] = {
        "seed": seed,
        "method": result["method"],
        "temperature": result["temperature"],
        "calibration_target": selected["calibration_target"],
        "threshold": selected["threshold"],
        "clean_calibration_fpr": selected["clean_calibration_fpr"],
        "selection_fallback": selected["selection_fallback"],
        "test_feasible": result["test_feasible"],
    }
    mapping = {
        "defect_recall_mean": "recall",
        "false_positive_rate_mean": "fpr",
        "no_defect_hallucination_rate_mean": "nhr",
        "precision_mean": "precision",
        "component_recall_mean": "component_recall",
        "component_iou10_recall_mean": "component_iou10_recall",
        "edge_f1_mean": "edge_f1",
        "psnr_mean": "psnr",
        "ssim_mean": "ssim",
    }
    for split, metrics in by_split.items():
        for source, destination in mapping.items():
            if source in metrics:
                row[f"{split}_{destination}"] = metrics[source]
    return row


def aggregate_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)
    aggregate = {}
    for method, method_rows in grouped.items():
        numeric_keys = sorted(
            {
                key
                for row in method_rows
                for key, value in row.items()
                if isinstance(value, (float, int)) and not isinstance(value, bool)
                and key != "seed"
            }
        )
        summary: dict[str, object] = {
            "n_seeds": len(method_rows),
            "selection_feasible_seeds": sum(
                not bool(row["selection_fallback"]) for row in method_rows
            ),
            "test_feasible_seeds": sum(bool(row["test_feasible"]) for row in method_rows),
        }
        for key in numeric_keys:
            values = np.asarray([float(row[key]) for row in method_rows if key in row])
            if len(values):
                summary[f"{key}_mean"] = float(np.mean(values))
                summary[f"{key}_std"] = float(np.std(values))
        aggregate[method] = summary
    return aggregate


def main() -> None:
    args = parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    all_rows: list[dict[str, object]] = []
    detailed = {}

    for seed in seeds:
        seed_dir = args.run_dir / f"seed_{seed}"
        data = load_npz(seed_dir / "data" / "samples.npz")
        splits = [str(value) for value in data["split"]]
        naf_checkpoint = args.naf_dir / f"seed_{seed}" / "naf_sr.pt"
        track_a_images = make_track_a_images(data, seed_dir, naf_checkpoint, device)
        seed_results = {}

        for method, images in track_a_images.items():
            raw_probabilities = np.stack(
                [
                    heuristic_defect_probability(image, sigma=COMMON_DETECTOR_SIGMA)
                    for image in images
                ]
            )
            result = select_and_evaluate(method, raw_probabilities, images, data, splits)
            seed_results[method] = result
            row = flatten_result(seed, result)
            all_rows.append(row)
            print(
                f"seed={seed} method={method} "
                f"recall={row['test_recall']:.4f} fpr={row['test_fpr']:.6f} "
                f"feasible={int(bool(row['test_feasible']))}"
            )

        if not args.skip_deeplab:
            scale = int(data["hr"].shape[-1] // data["lr"].shape[-1])
            deeplab_probability = infer_deeplab(
                seed_dir / "deeplabv3_detector_e20.pt",
                data["lr"],
                scale,
                device,
            )
            deeplab_sr = np.stack(
                [upsample_lr(sample, tuple(data["hr"].shape[-2:])) for sample in data["lr"]]
            )
            result = select_and_evaluate(
                "deeplabv3",
                deeplab_probability,
                deeplab_sr,
                data,
                splits,
            )
            seed_results["deeplabv3"] = result
            row = flatten_result(seed, result)
            all_rows.append(row)
            print(
                f"seed={seed} method=deeplabv3 "
                f"recall={row['test_recall']:.4f} fpr={row['test_fpr']:.6f} "
                f"feasible={int(bool(row['test_feasible']))}"
            )

        detailed[str(seed)] = seed_results
        (args.output_dir / f"seed_{seed}.json").write_text(
            json.dumps(seed_results, indent=2), encoding="utf-8"
        )

    aggregate = aggregate_rows(all_rows)
    output = {
        "protocol": {
            "seeds": seeds,
            "target_fpr": TARGET_FPR,
            "feasibility_tolerance": FEASIBILITY_TOLERANCE,
            "calibration_targets": CALIBRATION_TARGETS,
            "selection_rule": (
                "fit temperature and candidate thresholds on val_calib; select the most "
                "permissive clean_calib-feasible candidate; evaluate held-out splits once"
            ),
            "track_a_detector": "local residual contrast",
            "track_a_detector_sigma": COMMON_DETECTOR_SIGMA,
            "track_a_invariant": (
                "identical detector, detector hyperparameters, calibration rule, and metrics "
                "for every image transformation"
            ),
        },
        "aggregate": aggregate,
        "per_seed": all_rows,
    }
    (args.output_dir / "unified_protocol_aggregate.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    fieldnames = sorted({key for row in all_rows for key in row})
    with (args.output_dir / "unified_protocol_per_seed.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
