from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


MAIN_METRICS = [
    "defect_recall_mean",
    "false_positive_rate_mean",
    "no_defect_hallucination_rate_mean",
    "component_hallucination_count_mean",
    "pixel_ap_mean",
    "mask_iou_mean",
    "edge_f1_mean",
    "brier_score_mean",
    "ece_mean",
    "runtime_ms_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper tables from saved JSON artifacts.")
    parser.add_argument("--baseline-json", default="experiments/runs/smoke/baselines.json")
    parser.add_argument("--proposed-json", default="experiments/runs/smoke/dpu_wafersr.json")
    parser.add_argument(
        "--extra-json",
        action="append",
        default=[],
        help="Additional method JSON files with a top-level summary and optional method name.",
    )
    parser.add_argument("--output-csv", default="experiments/runs/smoke/main_table.csv")
    parser.add_argument("--output-md", default="experiments/runs/smoke/main_table.md")
    return parser.parse_args()


def fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    args = parse_args()
    baseline = json.loads(Path(args.baseline_json).read_text(encoding="utf-8"))["summary"]
    proposed_doc = json.loads(Path(args.proposed_json).read_text(encoding="utf-8"))
    proposed = proposed_doc["summary"]
    rows = []
    for name, metrics in baseline.items():
        row = {"method": name, **{metric: metrics.get(metric, "") for metric in MAIN_METRICS}}
        rows.append(row)
    for extra_path in args.extra_json:
        extra_doc = json.loads(Path(extra_path).read_text(encoding="utf-8"))
        name = extra_doc.get("method", Path(extra_path).stem)
        metrics = extra_doc["summary"]
        rows.append({"method": name, **{metric: metrics.get(metric, "") for metric in MAIN_METRICS}})
    rows.append({"method": "dpu_wafersr", **{metric: proposed.get(metric, "") for metric in MAIN_METRICS}})

    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", *MAIN_METRICS])
        writer.writeheader()
        writer.writerows(rows)

    md_path = Path(args.output_md)
    header = ["method", *MAIN_METRICS]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(col, "")) for col in header) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
