from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waferinspectsr.baselines import run_v1_baselines
from waferinspectsr.io import load_npz
from waferinspectsr.metrics import aggregate_metric_dicts, aggregate_risk_coverage
from waferinspectsr.metrics import aggregate_recall_fpr_curves
from waferinspectsr.metrics import summarize_prediction, threshold_for_target_fpr
from waferinspectsr.protocol import indices_for_splits, parse_split_names, require_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v1 baseline benchmark.")
    parser.add_argument("--input", default="data/generated/smoke/samples.npz")
    parser.add_argument("--output", default="experiments/runs/smoke/baselines.json")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--matched-fpr", type=float, default=None)
    parser.add_argument("--validation-split", default="val_calib")
    parser.add_argument("--calibration-splits", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_npz(args.input)
    rows_by_name: dict[str, list[dict[str, float]]] = defaultdict(list)
    risk_by_name: dict[str, list[list[dict[str, float]]]] = defaultdict(list)
    operating_by_name: dict[str, list[list[dict[str, float]]]] = defaultdict(list)
    per_sample = []
    thresholds: dict[str, float] = {}
    if args.matched_fpr is not None:
        calibration_splits = parse_split_names(args.calibration_splits, args.validation_split)
        split_values = [str(split) for split in data.get("split", ["val_calib"] * len(data["hr"]))]
        val_indices = indices_for_splits(split_values, calibration_splits)
        require_indices(val_indices, calibration_splits, "calibration")
        val_scores: dict[str, list] = defaultdict(list)
        val_clean: dict[str, list] = defaultdict(list)
        for idx in val_indices:
            for pred in run_v1_baselines(
                data["lr"][idx],
                data["hr"][idx].shape,
                hr=data["hr"][idx],
                include_oracle=True,
            ):
                val_scores[pred.name].append(pred.defect_prob)
                val_clean[pred.name].append(data["clean_mask"][idx])
        thresholds = {
            name: threshold_for_target_fpr(val_scores[name], val_clean[name], args.matched_fpr)
            for name in val_scores
        }

    for idx, hr in enumerate(data["hr"]):
        predictions = run_v1_baselines(data["lr"][idx], hr.shape, hr=hr, include_oracle=True)
        for pred in predictions:
            threshold = thresholds.get(pred.name, args.threshold)
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
            scalar_metrics = {key: value for key, value in metrics.items() if isinstance(value, (float, int))}
            scalar_metrics["runtime_ms"] = pred.runtime_ms
            scalar_metrics["threshold"] = threshold
            rows_by_name[pred.name].append(scalar_metrics)
            risk_by_name[pred.name].append(metrics["risk_coverage"])
            operating_by_name[pred.name].append(metrics["recall_fpr_curve"])
            per_sample.append(
                {
                    "index": idx,
                    "split": str(data.get("split", ["unknown"] * len(data["hr"]))[idx]),
                    "severity": str(data.get("severity", ["unknown"] * len(data["hr"]))[idx]),
                    "baseline": pred.name,
                    **scalar_metrics,
                }
            )

    summary = {name: aggregate_metric_dicts(rows) for name, rows in rows_by_name.items()}
    risk_coverage = {name: aggregate_risk_coverage(curves) for name, curves in risk_by_name.items()}
    operating_curves = {name: aggregate_recall_fpr_curves(curves) for name, curves in operating_by_name.items()}
    by_split: dict[str, dict[str, dict[str, float]]] = {}
    for split in sorted({row["split"] for row in per_sample}):
        by_split[split] = {}
        for name in rows_by_name:
            rows = [
                {key: value for key, value in row.items() if isinstance(value, (float, int))}
                for row in per_sample
                if row["split"] == split and row["baseline"] == name
            ]
            if rows:
                by_split[split][name] = aggregate_metric_dicts(rows)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "protocol": {
                    "threshold": args.threshold,
                    "matched_fpr": args.matched_fpr,
                    "validation_split": args.validation_split,
                    "calibration_splits": sorted(calibration_splits) if args.matched_fpr is not None else None,
                    "thresholds": thresholds,
                },
                "summary": summary,
                "risk_coverage": risk_coverage,
                "operating_curves": operating_curves,
                "by_split": by_split,
                "per_sample": per_sample,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out}")
    for name, metrics in summary.items():
        recall = metrics["defect_recall_mean"]
        fpr = metrics["false_positive_rate_mean"]
        hallucination = metrics["no_defect_hallucination_rate_mean"]
        print(f"{name}: recall={recall:.3f} fpr={fpr:.3f} hallucination={hallucination:.3f}")


if __name__ == "__main__":
    main()
