from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waferinspectsr.baselines import run_v1_baselines
from waferinspectsr.io import load_npz
from waferinspectsr.metrics import aggregate_metric_dicts, aggregate_recall_fpr_curves, aggregate_risk_coverage
from waferinspectsr.metrics import summarize_prediction, threshold_for_target_fpr
from waferinspectsr.protocol import parse_split_names, require_indices


METHOD_LABELS = {
    "no_sr_task_detector": "No-SR Task",
    "lr_detector": "LR",
    "nearest_detector": "Nearest",
    "bilinear_detector": "Bilinear",
    "bicubic_detector": "Bicubic",
    "lanczos_detector": "Lanczos",
    "denoise_upsample_detector": "Denoise+Upsample",
    "wiener_deconv_detector": "Wiener Deconv.",
    "prior_only_detector": "Prior-only",
    "sharpened_sr_detector": "SharpSR",
    "naf_style_sr_detector": "NAF-style SR",
    "hr_detector_reference": "HR reference",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate local baselines over a sharded scaled synthetic manifest.")
    parser.add_argument("--manifest", default="data/generated/scaled_seed20260602/manifest.json")
    parser.add_argument("--output-json", default="experiments/runs/scaled_seed20260602/baselines.json")
    parser.add_argument("--output-md", default="paper/tables/scaled_baseline_results.md")
    parser.add_argument("--output-csv", default="paper/tables/scaled_baseline_results.csv")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--matched-fpr", type=float, default=0.0003)
    parser.add_argument("--calibration-splits", default="val_calib,clean_calib")
    parser.add_argument("--include-train", action="store_true", help="Include train split in per-method summary aggregation.")
    return parser.parse_args()


def load_manifest(path: str | Path) -> tuple[Path, dict]:
    manifest_path = Path(path)
    doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    return manifest_path.parent, doc


def iter_shards(base_dir: Path, manifest: dict):
    for row in manifest["shards"]:
        shard_path = base_dir / row["path"]
        yield row, load_npz(shard_path)


def calibration_indices(data: dict, split_names: set[str]) -> list[int]:
    splits = [str(split) for split in data.get("split", ["unknown"] * len(data["hr"]))]
    return [idx for idx, split in enumerate(splits) if split in split_names]


def scalar_metrics(metrics: dict, runtime_ms: float, threshold: float) -> dict[str, float]:
    out = {key: value for key, value in metrics.items() if isinstance(value, (float, int))}
    out["runtime_ms"] = runtime_ms
    out["threshold"] = threshold
    return out


def evaluate_one_shard(data: dict, thresholds: dict[str, float], fallback_threshold: float):
    for idx, hr in enumerate(data["hr"]):
        split = str(data.get("split", ["unknown"] * len(data["hr"]))[idx])
        severity = str(data.get("severity", ["unknown"] * len(data["hr"]))[idx])
        for pred in run_v1_baselines(data["lr"][idx], hr.shape, hr=hr, include_oracle=True):
            threshold = thresholds.get(pred.name, fallback_threshold)
            metrics = summarize_prediction(
                pred.defect_prob,
                data["defect_mask"][idx],
                data["clean_mask"][idx],
                data["edge_mask"][idx],
                risk=pred.risk,
                threshold=threshold,
                sr_image=pred.sr_image,
                hr_image=hr,
            )
            yield split, severity, pred.name, scalar_metrics(metrics, pred.runtime_ms, threshold), metrics


def write_summary_tables(doc: dict, md_path: Path, csv_path: Path) -> None:
    rows = []
    by_split = doc["by_split"]
    ordered_methods = [method for method in METHOD_LABELS if method in doc["summary"]]
    ordered_methods.extend(sorted(method for method in doc["summary"] if method not in METHOD_LABELS))
    for method in ordered_methods:
        test = by_split.get("test", {}).get(method, {})
        clean = by_split.get("clean_test", {}).get(method, {})
        weak = by_split.get("weak_test", {}).get(method, {})
        ood = by_split.get("ood_test", {}).get(method, {})
        summary = doc["summary"][method]
        rows.append(
            {
                "Method": METHOD_LABELS.get(method, method),
                "Test recall": fmt(test.get("defect_recall_mean", "")),
                "Test FPR": fmt(test.get("false_positive_rate_mean", "")),
                "Clean NHR": fmt(clean.get("no_defect_hallucination_rate_mean", "")),
                "Weak recall": fmt(weak.get("defect_recall_mean", "")),
                "OOD recall": fmt(ood.get("defect_recall_mean", "")),
                "pAUC": fmt(test.get("partial_auc_fpr_1e_3_mean", "")),
                "ECE": fmt(summary.get("ece_mean", "")),
                "Runtime ms": fmt(summary.get("runtime_ms_mean", "")),
            }
        )
    headers = list(rows[0].keys()) if rows else ["Method"]
    md_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row[header]) for header in headers) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object) -> str:
    if value == "":
        return ""
    if isinstance(value, (float, int)):
        return f"{float(value):.6f}"
    return str(value)


def main() -> None:
    args = parse_args()
    base_dir, manifest = load_manifest(args.manifest)
    calibration_splits = parse_split_names(args.calibration_splits, "")
    threshold_scores: dict[str, list] = defaultdict(list)
    threshold_clean: dict[str, list] = defaultdict(list)
    total_calibration = 0
    for _row, data in iter_shards(base_dir, manifest):
        indices = calibration_indices(data, calibration_splits)
        total_calibration += len(indices)
        for idx in indices:
            for pred in run_v1_baselines(
                data["lr"][idx],
                data["hr"][idx].shape,
                hr=data["hr"][idx],
                include_oracle=True,
            ):
                threshold_scores[pred.name].append(pred.defect_prob)
                threshold_clean[pred.name].append(data["clean_mask"][idx])
    require_indices(list(range(total_calibration)), calibration_splits, "scaled calibration")
    thresholds = {
        name: threshold_for_target_fpr(threshold_scores[name], threshold_clean[name], args.matched_fpr)
        for name in threshold_scores
    }

    rows_by_name: dict[str, list[dict[str, float]]] = defaultdict(list)
    rows_by_split: dict[str, dict[str, list[dict[str, float]]]] = defaultdict(lambda: defaultdict(list))
    risk_by_name: dict[str, list[list[dict[str, float]]]] = defaultdict(list)
    operating_by_name: dict[str, list[list[dict[str, float]]]] = defaultdict(list)
    per_sample = []
    evaluated = 0
    for shard_row, data in iter_shards(base_dir, manifest):
        for split, severity, name, scalar, metrics in evaluate_one_shard(data, thresholds, args.threshold):
            if args.include_train or split != "train":
                rows_by_name[name].append(scalar)
            rows_by_split[split][name].append(scalar)
            risk_by_name[name].append(metrics["risk_coverage"])
            operating_by_name[name].append(metrics["recall_fpr_curve"])
            per_sample.append(
                {
                    "shard": shard_row["path"],
                    "split": split,
                    "severity": severity,
                    "baseline": name,
                    **scalar,
                }
            )
            evaluated += 1
    summary = {name: aggregate_metric_dicts(rows) for name, rows in rows_by_name.items()}
    by_split = {
        split: {name: aggregate_metric_dicts(rows) for name, rows in methods.items()}
        for split, methods in rows_by_split.items()
    }
    doc = {
        "method": "scaled_local_baseline_suite",
        "protocol": {
            "manifest": args.manifest,
            "matched_fpr": args.matched_fpr,
            "calibration_splits": sorted(calibration_splits),
            "calibration_samples": total_calibration,
            "include_train_in_summary": args.include_train,
            "thresholds": thresholds,
            "total_samples": manifest["total_samples"],
            "split_counts": manifest["split_counts"],
        },
        "summary": summary,
        "by_split": by_split,
        "risk_coverage": {name: aggregate_risk_coverage(curves) for name, curves in risk_by_name.items()},
        "operating_curves": {name: aggregate_recall_fpr_curves(curves) for name, curves in operating_by_name.items()},
        "per_sample": per_sample,
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    write_summary_tables(doc, Path(args.output_md), Path(args.output_csv))
    print(f"wrote {out}")
    print(f"wrote {args.output_md}")
    print(f"evaluated_predictions={evaluated} calibration_samples={total_calibration}")


if __name__ == "__main__":
    main()
