from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waferinspectsr.baselines import heuristic_defect_probability, uncertainty_from_probability
from waferinspectsr.io import load_npz
from waferinspectsr.metrics import aggregate_metric_dicts, aggregate_recall_fpr_curves, aggregate_risk_coverage
from waferinspectsr.metrics import summarize_prediction, threshold_for_target_fpr
from waferinspectsr.pretrained_sr import load_manifest, load_runner
from waferinspectsr.protocol import indices_for_splits, parse_split_names, require_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate checkpoint-backed pretrained SR baselines.")
    parser.add_argument("--input", default="data/generated/default/samples.npz")
    parser.add_argument("--manifest", default="configs/pretrained_sr_manifest.example.json")
    parser.add_argument("--output", default="experiments/runs/pretrained_sr/pretrained_sr.json")
    parser.add_argument("--status-md", default="paper/tables/pretrained_sr_status.md")
    parser.add_argument("--status-csv", default="paper/tables/pretrained_sr_status.csv")
    parser.add_argument("--results-md", default="paper/tables/pretrained_sr_results.md")
    parser.add_argument("--results-csv", default="paper/tables/pretrained_sr_results.csv")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--matched-fpr", type=float, default=None)
    parser.add_argument("--validation-split", default="val_calib")
    parser.add_argument("--calibration-splits", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--require-all", action="store_true", help="Fail if any declared checkpoint is missing or unloadable.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for smoke/debug evaluation.")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
        return torch.device("cuda")
    if name == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def predict_probability(runner, lr, target_shape):
    pred = runner.predict(lr, target_shape)
    prob = heuristic_defect_probability(pred.sr_image, sigma=2.2)
    risk = uncertainty_from_probability(prob)
    return pred, prob, risk


def write_status_table(status: list[dict[str, object]], md_path: Path, csv_path: Path) -> None:
    headers = ["Method", "Family", "Architecture", "Available", "Checkpoint", "Reason"]
    rows = [
        {
            "Method": row["name"],
            "Family": row["family"],
            "Architecture": row["architecture"],
            "Available": row["available"],
            "Checkpoint": row["checkpoint"],
            "Reason": row.get("reason", "loaded"),
        }
        for row in status
    ]
    md_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(one_line(row[header]) for header in headers) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows([{header: one_line(row[header]) for header in headers} for row in rows])


def fmt(value: object, digits: int = 6) -> str:
    if isinstance(value, (float, int)):
        return f"{float(value):.{digits}f}"
    return ""


def one_line(value: object) -> str:
    return str(value).replace("\r", " ").replace("\n", " ")


def split_metric(
    by_split: dict[str, dict[str, dict[str, float]]],
    split: str,
    name: str,
    key: str,
    digits: int = 6,
) -> str:
    return fmt(by_split.get(split, {}).get(name, {}).get(key), digits)


def write_results_table(
    status: list[dict[str, object]],
    by_split: dict[str, dict[str, dict[str, float]]],
    md_path: Path,
    csv_path: Path,
) -> None:
    headers = [
        "Method",
        "Available",
        "Status",
        "Test recall",
        "Test FPR",
        "Clean NHR",
        "Weak recall",
        "OOD recall",
        "PSNR",
        "Runtime ms",
        "Threshold",
    ]
    rows = []
    for row in status:
        name = str(row["name"])
        available = bool(row["available"])
        rows.append(
            {
                "Method": name,
                "Available": available,
                "Status": "loaded" if available else str(row.get("reason", "")),
                "Test recall": split_metric(by_split, "test", name, "defect_recall_mean"),
                "Test FPR": split_metric(by_split, "test", name, "false_positive_rate_mean"),
                "Clean NHR": split_metric(by_split, "clean_test", name, "no_defect_hallucination_rate_mean"),
                "Weak recall": split_metric(by_split, "weak_test", name, "defect_recall_mean"),
                "OOD recall": split_metric(by_split, "ood_test", name, "defect_recall_mean"),
                "PSNR": split_metric(by_split, "test", name, "psnr_mean", digits=3),
                "Runtime ms": split_metric(by_split, "test", name, "runtime_ms_mean", digits=3),
                "Threshold": split_metric(by_split, "test", name, "threshold_mean", digits=6),
            }
        )
    md_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(one_line(row[header]) for header in headers) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows([{header: one_line(row[header]) for header in headers} for row in rows])


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    data = load_npz(args.input)
    total = len(data["hr"]) if args.max_samples is None else min(int(args.max_samples), len(data["hr"]))
    sample_indices = list(range(total))
    splits = [str(value) for value in data.get("split", ["unknown"] * len(data["hr"]))]
    specs = load_manifest(args.manifest)
    runners = []
    status = []
    for spec in specs:
        runner, runner_status = load_runner(spec, args.manifest, device)
        status.append(runner_status)
        if runner is not None:
            runners.append(runner)
    if runners and sample_indices:
        valid_runners = []
        probe_idx = sample_indices[0]
        for runner in runners:
            try:
                runner.predict(data["lr"][probe_idx], data["hr"][probe_idx].shape)
            except Exception as exc:  # pragma: no cover - integration guard for external checkpoints.
                for row in status:
                    if row["name"] == runner.spec.name:
                        row["available"] = False
                        row["reason"] = f"runtime_failed: {exc}"
                        break
            else:
                valid_runners.append(runner)
        runners = valid_runners
    if args.require_all and any(not row["available"] for row in status):
        missing = [row["name"] for row in status if not row["available"]]
        raise RuntimeError(f"Unavailable pretrained SR baselines: {missing}")

    thresholds: dict[str, float] = {}
    calibration_splits = None
    if runners and args.matched_fpr is not None:
        calibration_splits = parse_split_names(args.calibration_splits, args.validation_split)
        val_indices = [idx for idx in indices_for_splits(splits, calibration_splits) if idx in sample_indices]
        require_indices(val_indices, calibration_splits, "calibration")
        val_scores: dict[str, list] = defaultdict(list)
        val_clean: dict[str, list] = defaultdict(list)
        for idx in val_indices:
            for runner in runners:
                _pred, prob, _risk = predict_probability(runner, data["lr"][idx], data["hr"][idx].shape)
                val_scores[runner.spec.name].append(prob)
                val_clean[runner.spec.name].append(data["clean_mask"][idx])
        thresholds = {
            name: threshold_for_target_fpr(val_scores[name], val_clean[name], args.matched_fpr)
            for name in val_scores
        }

    rows_by_name: dict[str, list[dict[str, float]]] = defaultdict(list)
    risk_by_name: dict[str, list[list[dict[str, float]]]] = defaultdict(list)
    operating_by_name: dict[str, list[list[dict[str, float]]]] = defaultdict(list)
    per_sample = []
    for idx in sample_indices:
        for runner in runners:
            pred, prob, risk = predict_probability(runner, data["lr"][idx], data["hr"][idx].shape)
            threshold = thresholds.get(runner.spec.name, args.threshold)
            metrics = summarize_prediction(
                prob,
                data["defect_mask"][idx],
                data["clean_mask"][idx],
                data["edge_mask"][idx],
                risk=risk,
                threshold=threshold,
                sr_image=pred.sr_image,
                hr_image=data["hr"][idx],
            )
            scalar = {key: value for key, value in metrics.items() if isinstance(value, (float, int))}
            scalar["runtime_ms"] = pred.runtime_ms
            scalar["threshold"] = threshold
            rows_by_name[runner.spec.name].append(scalar)
            risk_by_name[runner.spec.name].append(metrics["risk_coverage"])
            operating_by_name[runner.spec.name].append(metrics["recall_fpr_curve"])
            per_sample.append(
                {
                    "index": idx,
                    "split": splits[idx],
                    "severity": str(data.get("severity", ["unknown"] * len(data["hr"]))[idx]),
                    "baseline": runner.spec.name,
                    **scalar,
                }
            )

    summary = {name: aggregate_metric_dicts(rows) for name, rows in rows_by_name.items()}
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
    risk_coverage = {name: aggregate_risk_coverage(curves) for name, curves in risk_by_name.items()}
    operating_curves = {name: aggregate_recall_fpr_curves(curves) for name, curves in operating_by_name.items()}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "method": "pretrained_sr_suite",
                "protocol": {
                    "input": args.input,
                    "manifest": args.manifest,
                    "sample_count": total,
                    "threshold": args.threshold,
                    "matched_fpr": args.matched_fpr,
                    "validation_split": args.validation_split,
                    "calibration_splits": sorted(calibration_splits) if calibration_splits else None,
                    "thresholds": thresholds,
                    "device": str(device),
                },
                "status": status,
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
    write_status_table(status, Path(args.status_md), Path(args.status_csv))
    write_results_table(status, by_split, Path(args.results_md), Path(args.results_csv))
    print(f"wrote {out}")
    print(f"wrote {args.status_md}")
    print(f"wrote {args.results_md}")
    available = [row["name"] for row in status if row["available"]]
    missing = [row["name"] for row in status if not row["available"]]
    print(f"available={available}")
    print(f"unavailable={missing}")


if __name__ == "__main__":
    main()
