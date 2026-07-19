"""Evaluate protocol-compatible transfer on the public Carinthia-S masks.

The external data are never used to train a model, fit a temperature, or choose
a threshold.  Each of the ten synthetic-run policies is applied unchanged to
the same public SEM images.  The role-based comparison uses the bicubic
fixed-detector reference, the highest-SSIM learned reconstruction
(NAF-trained), and the task-trained direct detector (DeepLabV3).  It is a
compact transfer stress test rather than an exhaustive external ranking.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from waferinspectsr.baselines import heuristic_defect_probability  # noqa: E402
from waferinspectsr.degradation import upsample_lr  # noqa: E402
from waferinspectsr.image_ops import binary_dilation  # noqa: E402
from waferinspectsr.models import CompactNAFSR  # noqa: E402


SEEDS = (7, 11, 13, 17, 19, 23, 29, 31, 37, 41)
METHODS = ("bicubic_reference", "naf_trained", "deeplabv3")
METHOD_LABELS = {
    "bicubic_reference": "Bicubic reference",
    "naf_trained": "Trained NAF-style SR",
    "deeplabv3": "DeepLabV3 detector",
}


class TorchvisionDeepLabDetector(nn.Module):
    """The detector architecture used by the synthetic benchmark runs."""

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
        probability = self.model(self.input_adapter(lr))["out"]
        probability = torch.sigmoid(probability)
        if self.scale != 1:
            probability = F.interpolate(
                probability,
                scale_factor=self.scale,
                mode="bilinear",
                align_corners=False,
            )
        return probability


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=PROJECT_ROOT / "data/raw/carinthia_s/data/data/carinthia-s.csv",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT / "data/raw/carinthia_s/data/data",
    )
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
        "--policy-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "tsm_unified_protocol",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "tsm_external_transfer",
    )
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in SEEDS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-per-class", type=int, default=0)
    parser.add_argument("--hr-size", type=int, default=256)
    return parser.parse_args()


def load_manifest(
    csv_path: Path,
    root: Path,
    max_per_class: int,
) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    counts: dict[str, int] = defaultdict(int)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter=";"):
            label = str(row["label"])
            if max_per_class > 0 and counts[label] >= max_per_class:
                continue
            image_path = root / row["image_path"]
            mask_path = root / row["mask_path"]
            if image_path.exists() and mask_path.exists():
                selected.append(
                    {
                        "label": label,
                        "image": str(image_path),
                        "mask": str(mask_path),
                    }
                )
                counts[label] += 1
    if not selected:
        raise FileNotFoundError(f"No image/mask pairs found from {csv_path}")
    return selected


def load_arrays(
    rows: list[dict[str, str]],
    hr_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lr_images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    clean_masks: list[np.ndarray] = []
    labels: list[int] = []
    for row in rows:
        image = Image.open(row["image"]).convert("L").resize(
            (hr_size, hr_size), Image.Resampling.BICUBIC
        )
        mask = Image.open(row["mask"]).convert("L").resize(
            (hr_size, hr_size), Image.Resampling.NEAREST
        )
        lr = image.resize((hr_size // 2, hr_size // 2), Image.Resampling.BICUBIC)
        defect = np.asarray(mask, dtype=np.uint8) > 0
        clean = ~binary_dilation(defect, iterations=3)
        lr_images.append(np.asarray(lr, dtype=np.float32) / 255.0)
        masks.append(defect)
        clean_masks.append(clean)
        labels.append(int(row["label"]))
    return (
        np.stack(lr_images).astype(np.float32),
        np.stack(masks).astype(bool),
        np.stack(clean_masks).astype(bool),
        np.asarray(labels, dtype=np.int16),
    )


def raw_threshold(temperature: float, calibrated_threshold: float) -> float:
    """Map a threshold after temperature scaling back to raw probability."""

    probability = float(np.clip(calibrated_threshold, 1e-5, 1.0 - 1e-5))
    logit = math.log(probability / (1.0 - probability))
    return float(1.0 / (1.0 + math.exp(-max(float(temperature), 1e-3) * logit)))


def load_policies(policy_path: Path) -> dict[str, dict[str, float]]:
    doc = json.loads(policy_path.read_text(encoding="utf-8"))
    policies = {}
    for method in METHODS:
        result = doc[method]
        policies[method] = {
            "temperature": float(result["temperature"]),
            "threshold": float(result["selected"]["threshold"]),
            "raw_threshold": raw_threshold(
                float(result["temperature"]),
                float(result["selected"]["threshold"]),
            ),
        }
    return policies


def add_batch_metrics(
    accumulator: dict[str, list[float]],
    prediction: np.ndarray,
    defect_mask: np.ndarray,
    clean_mask: np.ndarray,
    labels: np.ndarray,
) -> None:
    axes = tuple(range(1, prediction.ndim))
    defect_count = defect_mask.sum(axis=axes)
    clean_count = clean_mask.sum(axis=axes)
    true_positive = np.logical_and(prediction, defect_mask).sum(axis=axes)
    false_positive = np.logical_and(prediction, clean_mask).sum(axis=axes)
    recall = true_positive / np.maximum(defect_count, 1)
    fpr = false_positive / np.maximum(clean_count, 1)
    for index, label in enumerate(labels):
        accumulator["fpr"].append(float(fpr[index]))
        if defect_count[index] > 0:
            accumulator["recall"].append(float(recall[index]))
            accumulator[f"recall_class_{int(label)}"].append(float(recall[index]))
        if int(label) == 6:
            accumulator["nhr"].append(float(fpr[index]))
            accumulator["clean_crop_called"].append(float(prediction[index].any()))


def summarize_accumulator(
    seed: int,
    method: str,
    policy: dict[str, float],
    accumulator: dict[str, list[float]],
) -> dict[str, object]:
    class_means = [
        float(np.mean(accumulator[f"recall_class_{label}"]))
        for label in range(1, 6)
        if accumulator[f"recall_class_{label}"]
    ]
    return {
        "seed": seed,
        "method": method,
        "method_label": METHOD_LABELS[method],
        "temperature": policy["temperature"],
        "calibrated_threshold": policy["threshold"],
        "raw_threshold": policy["raw_threshold"],
        "defect_image_macro_recall": float(np.mean(accumulator["recall"])),
        "defect_class_macro_recall": float(np.mean(class_means)),
        "clean_region_fpr": float(np.mean(accumulator["fpr"])),
        "no_defect_hallucination_rate": float(np.mean(accumulator["nhr"])),
        "no_defect_crop_call_rate": float(np.mean(accumulator["clean_crop_called"])),
    }


def aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)
    output = {}
    metrics = (
        "defect_image_macro_recall",
        "defect_class_macro_recall",
        "clean_region_fpr",
        "no_defect_hallucination_rate",
        "no_defect_crop_call_rate",
    )
    for method, subset in grouped.items():
        summary: dict[str, object] = {"n_seeds": len(subset)}
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in subset])
            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_std"] = float(np.std(values))
        output[method] = summary
    return output


def main() -> None:
    args = parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    manifest = load_manifest(args.csv, args.root, args.max_per_class)
    print(f"loading {len(manifest)} external image/mask pairs")
    lr, defect_mask, clean_mask, labels = load_arrays(manifest, args.hr_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, object]] = []
    batch_size = max(int(args.batch_size), 1)
    for seed in seeds:
        policies = load_policies(args.policy_dir / f"seed_{seed}.json")
        run_dir = args.run_dir / f"seed_{seed}"

        naf_state = torch.load(
            args.naf_dir / f"seed_{seed}" / "naf_sr.pt",
            map_location="cpu",
            weights_only=False,
        )
        naf = CompactNAFSR(scale=2, channels=24, blocks=2).to(device)
        naf.load_state_dict(naf_state["model_state"])
        naf.eval()

        deeplab_state = torch.load(
            run_dir / "deeplabv3_detector_e20.pt",
            map_location="cpu",
            weights_only=False,
        )
        deeplab = TorchvisionDeepLabDetector(scale=2).to(device)
        deeplab.load_state_dict(deeplab_state["model_state"])
        deeplab.eval()

        accumulators = {method: defaultdict(list) for method in METHODS}
        with torch.no_grad():
            for start in range(0, len(lr), batch_size):
                stop = min(start + batch_size, len(lr))
                lr_batch = lr[start:stop]
                tensor = torch.from_numpy(lr_batch[:, None]).to(device, non_blocking=True)

                bicubic = np.stack(
                    [
                        upsample_lr(image, (args.hr_size, args.hr_size))
                        for image in lr_batch
                    ]
                )
                bicubic_probability = np.stack(
                    [
                        heuristic_defect_probability(image, sigma=2.0)
                        for image in bicubic
                    ]
                )
                naf_image = naf(tensor).squeeze(1).cpu().numpy()
                naf_probability = np.stack(
                    [
                        heuristic_defect_probability(image, sigma=2.0)
                        for image in naf_image
                    ]
                )
                deeplab_probability = deeplab(tensor).squeeze(1).cpu().numpy()

                probabilities = {
                    "bicubic_reference": bicubic_probability,
                    "naf_trained": naf_probability,
                    "deeplabv3": deeplab_probability,
                }
                for method, probability in probabilities.items():
                    prediction = probability >= policies[method]["raw_threshold"]
                    add_batch_metrics(
                        accumulators[method],
                        prediction,
                        defect_mask[start:stop],
                        clean_mask[start:stop],
                        labels[start:stop],
                    )

        for method in METHODS:
            row = summarize_accumulator(
                seed,
                method,
                policies[method],
                accumulators[method],
            )
            all_rows.append(row)
            print(
                f"seed={seed} method={method} "
                f"recall={row['defect_image_macro_recall']:.4f} "
                f"fpr={row['clean_region_fpr']:.6f} "
                f"clean_crop_calls={row['no_defect_crop_call_rate']:.4f}"
            )

        del naf, deeplab
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = aggregate(all_rows)
    payload = {
        "protocol": {
            "dataset": "Carinthia-S",
            "sample_count": len(manifest),
            "seeds": seeds,
            "external_data_use": (
                "inference only; no training, temperature fitting, or threshold selection"
            ),
            "pseudo_lr": "bicubic downsampling from 256 x 256 SEM crop to 128 x 128",
            "track_a_detector": "local residual contrast, sigma 2.0",
            "reported_unit": "image-macro metrics followed by mean and SD across seed policies",
            "class_counts": {
                str(label): int(np.sum(labels == label)) for label in sorted(set(labels))
            },
        },
        "aggregate": summary,
        "per_seed": all_rows,
    }
    (args.output_dir / "external_transfer_aggregate.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "external_transfer_per_seed.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
