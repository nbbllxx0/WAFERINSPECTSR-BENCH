from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waferinspectsr.baselines import uncertainty_from_probability
from waferinspectsr.degradation import upsample_lr
from waferinspectsr.image_ops import gaussian_filter, gaussian_gradient_magnitude, uniform_filter
from waferinspectsr.io import load_npz
from waferinspectsr.metrics import aggregate_metric_dicts, aggregate_recall_fpr_curves, aggregate_risk_coverage
from waferinspectsr.metrics import summarize_prediction, threshold_for_target_fpr
from waferinspectsr.protocol import indices_for_splits, parse_split_names, require_indices


METHOD = "padim_local_gaussian_anomaly"
TABLE_METRICS = [
    "defect_recall",
    "false_positive_rate",
    "no_defect_hallucination_rate",
    "component_hallucination_count",
    "weak_defect_recall",
    "pixel_ap",
    "mask_iou",
    "edge_f1",
    "runtime_ms",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a dependency-free PaDiM-style local Gaussian anomaly-localization baseline."
    )
    parser.add_argument("--input", action="append", required=True, help="Input samples.npz. Repeat for seed aggregates.")
    parser.add_argument("--output-json", default="paper/tables/padim_anomaly_results.json")
    parser.add_argument("--output-csv", default="paper/tables/padim_anomaly_results.csv")
    parser.add_argument("--output-md", default="paper/tables/padim_anomaly_results.md")
    parser.add_argument("--normal-splits", default="train", help="Splits used to fit normal feature statistics.")
    parser.add_argument("--calibration-splits", default="val_calib,clean_calib")
    parser.add_argument("--matched-fpr", type=float, default=0.0003)
    parser.add_argument("--score-scale-percentile", type=float, default=99.9)
    parser.add_argument("--min-std", type=float, default=1e-3)
    parser.add_argument("--min-component-area", type=int, default=1)
    return parser.parse_args()


def feature_stack(lr: np.ndarray, target_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    sr = upsample_lr(lr, target_shape)
    low_1 = gaussian_filter(sr, sigma=1.0)
    low_3 = gaussian_filter(sr, sigma=3.0)
    residual_1 = np.abs(sr - low_1)
    residual_3 = np.abs(sr - low_3)
    grad = gaussian_gradient_magnitude(sr, sigma=1.0)
    local_mean = uniform_filter(sr, size=9)
    local_var = np.maximum(uniform_filter((sr - local_mean) ** 2, size=9), 0.0)
    local_std = np.sqrt(local_var).astype(np.float32)
    features = np.stack([sr, low_1, low_3, residual_1, residual_3, grad, local_std]).astype(np.float32)
    return features, sr


def fit_feature_stats(
    data: dict[str, Any],
    splits: list[str],
    normal_splits: set[str],
    min_std: float,
) -> dict[str, np.ndarray]:
    normal_indices = [idx for idx, split in enumerate(splits) if split in normal_splits]
    require_indices(normal_indices, normal_splits, "normal-statistics")
    target_shape = tuple(int(value) for value in data["hr"].shape[-2:])
    first_features, _ = feature_stack(data["lr"][normal_indices[0]], target_shape)
    sums = np.zeros_like(first_features, dtype=np.float64)
    squares = np.zeros_like(first_features, dtype=np.float64)
    counts = np.zeros(first_features.shape[1:], dtype=np.float64)

    for idx in normal_indices:
        features, _ = feature_stack(data["lr"][idx], target_shape)
        normal_mask = np.asarray(data["clean_mask"][idx], dtype=bool)
        weights = normal_mask.astype(np.float64)
        sums += features * weights[None, :, :]
        squares += np.square(features, dtype=np.float64) * weights[None, :, :]
        counts += weights

    valid = counts > 0
    global_count = max(float(counts.sum()), 1.0)
    global_mean = sums.sum(axis=(1, 2)) / global_count
    global_var = np.maximum(squares.sum(axis=(1, 2)) / global_count - global_mean**2, min_std**2)

    mean = np.empty_like(sums, dtype=np.float32)
    std = np.empty_like(sums, dtype=np.float32)
    for channel in range(sums.shape[0]):
        channel_mean = np.full(sums.shape[1:], global_mean[channel], dtype=np.float64)
        channel_var = np.full(sums.shape[1:], global_var[channel], dtype=np.float64)
        channel_mean[valid] = sums[channel][valid] / counts[valid]
        channel_var[valid] = squares[channel][valid] / counts[valid] - channel_mean[valid] ** 2
        mean[channel] = channel_mean.astype(np.float32)
        std[channel] = np.sqrt(np.maximum(channel_var, min_std**2)).astype(np.float32)
    return {"mean": mean, "std": std}


def raw_anomaly_score(features: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    z = (features - stats["mean"]) / stats["std"]
    score = np.sqrt(np.mean(z * z, axis=0))
    return score.astype(np.float32)


def normalize_score(score: np.ndarray, scale: float) -> np.ndarray:
    arr = np.maximum(np.asarray(score, dtype=np.float32), 0.0)
    denom = arr + max(float(scale), 1e-6)
    return (arr / denom).astype(np.float32)


def fit_score_scale(
    data: dict[str, Any],
    splits: list[str],
    calibration_splits: set[str],
    stats: dict[str, np.ndarray],
    percentile: float,
) -> float:
    calibration_indices = indices_for_splits(splits, calibration_splits)
    require_indices(calibration_indices, calibration_splits, "score-scale calibration")
    target_shape = tuple(int(value) for value in data["hr"].shape[-2:])
    values = []
    for idx in calibration_indices:
        features, _ = feature_stack(data["lr"][idx], target_shape)
        score = raw_anomaly_score(features, stats)
        clean_values = score[np.asarray(data["clean_mask"][idx], dtype=bool)]
        if clean_values.size:
            values.append(clean_values)
    if not values:
        return 1.0
    scale = float(np.percentile(np.concatenate(values), np.clip(percentile, 50.0, 100.0)))
    return max(scale, 1e-6)


def evaluate_one(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    data = load_npz(path)
    splits = [str(value) for value in data.get("split", ["train"] * len(data["hr"]))]
    normal_splits = parse_split_names(args.normal_splits, "")
    calibration_splits = parse_split_names(args.calibration_splits, "")
    stats = fit_feature_stats(data, splits, normal_splits, args.min_std)
    score_scale = fit_score_scale(data, splits, calibration_splits, stats, args.score_scale_percentile)
    target_shape = tuple(int(value) for value in data["hr"].shape[-2:])

    calibration_indices = indices_for_splits(splits, calibration_splits)
    calibration_scores = []
    calibration_clean = []
    raw_outputs: list[tuple[np.ndarray, np.ndarray, float]] = []
    for idx in range(len(data["hr"])):
        start = time.perf_counter()
        features, sr = feature_stack(data["lr"][idx], target_shape)
        score = normalize_score(raw_anomaly_score(features, stats), score_scale)
        runtime_ms = (time.perf_counter() - start) * 1000.0
        raw_outputs.append((score, sr, runtime_ms))
        if idx in calibration_indices:
            calibration_scores.append(score)
            calibration_clean.append(data["clean_mask"][idx])
    threshold = threshold_for_target_fpr(
        calibration_scores,
        calibration_clean,
        args.matched_fpr,
        min_component_area=args.min_component_area,
    )

    metric_rows = []
    per_sample = []
    risk_curves = []
    operating_curves = []
    for idx, (score, sr, runtime_ms) in enumerate(raw_outputs):
        risk = uncertainty_from_probability(score)
        metrics = summarize_prediction(
            score,
            data["defect_mask"][idx],
            data["clean_mask"][idx],
            data["edge_mask"][idx],
            risk=risk,
            threshold=threshold,
            sr_image=sr,
            hr_image=data["hr"][idx],
            min_component_area=args.min_component_area,
        )
        scalar = {key: value for key, value in metrics.items() if isinstance(value, (float, int))}
        scalar["threshold"] = threshold
        scalar["runtime_ms"] = runtime_ms
        scalar["weak_defect_recall"] = scalar["defect_recall"] if splits[idx] == "weak_test" else 0.0
        metric_rows.append(scalar)
        risk_curves.append(metrics["risk_coverage"])
        operating_curves.append(metrics["recall_fpr_curve"])
        per_sample.append(
            {
                "index": idx,
                "split": splits[idx],
                "severity": str(data.get("severity", ["unknown"] * len(data["hr"]))[idx]),
                "baseline": METHOD,
                **scalar,
            }
        )

    by_split = {}
    for split in sorted(set(splits)):
        rows = [row for row, row_split in zip(metric_rows, splits) if row_split == split]
        if rows:
            by_split[split] = aggregate_metric_dicts(rows)

    return {
        "input": str(path),
        "summary": aggregate_metric_dicts(metric_rows),
        "by_split": by_split,
        "risk_coverage": aggregate_risk_coverage(risk_curves),
        "operating_curve": aggregate_recall_fpr_curves(operating_curves),
        "per_sample": per_sample,
        "protocol": {
            "method": METHOD,
            "feature_channels": ["intensity", "gauss1", "gauss3", "residual1", "residual3", "gradient", "local_std"],
            "normal_splits": sorted(normal_splits),
            "calibration_splits": sorted(calibration_splits),
            "matched_fpr": args.matched_fpr,
            "score_scale_percentile": args.score_scale_percentile,
            "score_scale": score_scale,
            "threshold": threshold,
            "min_std": args.min_std,
            "min_component_area": args.min_component_area,
        },
    }


def aggregate_seed_summaries(seed_docs: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    by_split_rows: dict[str, list[dict[str, float]]] = defaultdict(list)
    by_split_rows["all"] = [mean_metrics(doc["summary"]) for doc in seed_docs]
    for doc in seed_docs:
        for split, metrics in doc["by_split"].items():
            by_split_rows[split].append(mean_metrics(metrics))
    return {split: aggregate_metric_dicts(rows) for split, rows in sorted(by_split_rows.items()) if rows}


def mean_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {key.removesuffix("_mean"): value for key, value in metrics.items() if key.endswith("_mean")}


def flatten_repeated_metric(metrics: dict[str, float], key: str) -> float | str:
    return metrics.get(f"{key}_mean", "")


def write_tables(doc: dict[str, Any], csv_path: Path, md_path: Path) -> None:
    rows = []
    repeated = doc["repeated_by_split"]
    for split in ["all", "test", "weak_test", "clean_test", "ood_test"]:
        if split not in repeated:
            continue
        metrics = repeated[split]
        row = {"method": METHOD, "split": split}
        row.update({metric: flatten_repeated_metric(metrics, metric) for metric in TABLE_METRICS})
        rows.append(row)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["method", "split", *TABLE_METRICS]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "| method | split | recall | FPR | clean NHR | clean comps | weak recall | AP | mask IoU | edge F1 | runtime ms |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["method"]),
                    str(row["split"]),
                    f"{float(row['defect_recall']):.4f}" if row["defect_recall"] != "" else "",
                    f"{float(row['false_positive_rate']):.6f}" if row["false_positive_rate"] != "" else "",
                    f"{float(row['no_defect_hallucination_rate']):.6f}"
                    if row["no_defect_hallucination_rate"] != ""
                    else "",
                    f"{float(row['component_hallucination_count']):.3f}"
                    if row["component_hallucination_count"] != ""
                    else "",
                    f"{float(row['weak_defect_recall']):.4f}" if row["weak_defect_recall"] != "" else "",
                    f"{float(row['pixel_ap']):.4f}" if row["pixel_ap"] != "" else "",
                    f"{float(row['mask_iou']):.4f}" if row["mask_iou"] != "" else "",
                    f"{float(row['edge_f1']):.4f}" if row["edge_f1"] != "" else "",
                    f"{float(row['runtime_ms']):.2f}" if row["runtime_ms"] != "" else "",
                ]
            )
            + " |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    seed_docs = [evaluate_one(Path(input_path), args) for input_path in args.input]
    all_rows = []
    for seed_doc in seed_docs:
        all_rows.extend(
            {
                key: value
                for key, value in row.items()
                if isinstance(value, (float, int))
            }
            for row in seed_doc["per_sample"]
        )
    doc = {
        "method": METHOD,
        "protocol": {
            "normal_splits": sorted(parse_split_names(args.normal_splits, "")),
            "calibration_splits": sorted(parse_split_names(args.calibration_splits, "")),
            "matched_fpr": args.matched_fpr,
            "score_scale_percentile": args.score_scale_percentile,
            "min_std": args.min_std,
            "min_component_area": args.min_component_area,
            "inputs": args.input,
        },
        "summary": aggregate_metric_dicts(all_rows),
        "repeated_by_split": aggregate_seed_summaries(seed_docs),
        "seeds": seed_docs,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    write_tables(doc, Path(args.output_csv), Path(args.output_md))
    print(f"wrote {output_json}")
    print(f"wrote {args.output_csv}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
