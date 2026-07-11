from __future__ import annotations

import argparse
import json
from pathlib import Path


def method_split(doc: dict, split: str, method: str | None = None) -> dict:
    by_split = doc.get("by_split", {})
    if split not in by_split:
        return {}
    if method is None:
        return by_split[split]
    return by_split[split].get(method, {})


def first_method_split(doc: dict, splits: tuple[str, ...], method: str | None = None) -> dict:
    for split in splits:
        out = method_split(doc, split, method)
        if out:
            return out
    return {}


def value(metrics: dict, key: str) -> float:
    return float(metrics.get(key, 0.0))


def risk_coverage_useful(doc: dict) -> bool:
    curve = doc.get("risk_coverage", [])
    if len(curve) < 2:
        return False
    full = next((row for row in curve if abs(float(row["coverage"]) - 1.0) < 1e-8), curve[0])
    half = min(curve, key=lambda row: abs(float(row["coverage"]) - 0.5))
    return float(half["error_mean"]) < float(full["error_mean"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check v1 paper-readiness gates from baseline/proposed summaries.")
    parser.add_argument("--baseline-json", default="experiments/runs/smoke/baselines.json")
    parser.add_argument("--proposed-json", default="experiments/runs/smoke/dpu_wafersr.json")
    parser.add_argument("--modern-sr-json", default=None)
    parser.add_argument("--output", default="experiments/runs/smoke/gates.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_doc = json.loads(Path(args.baseline_json).read_text(encoding="utf-8"))
    proposed_doc = json.loads(Path(args.proposed_json).read_text(encoding="utf-8"))
    baseline = baseline_doc["summary"]
    proposed = proposed_doc["summary"]
    if args.modern_sr_json:
        modern_doc = json.loads(Path(args.modern_sr_json).read_text(encoding="utf-8"))
        generic = modern_doc["summary"]
        modern_name = modern_doc.get("method", "modern_sr")
    else:
        generic = baseline["sharpened_sr_detector"]
        modern_name = "sharpened_sr_detector"
    lr = baseline["lr_detector"]
    lr_test = method_split(baseline_doc, "test", "lr_detector")
    modern_test = method_split(modern_doc, "test") if args.modern_sr_json else method_split(baseline_doc, "test", modern_name)
    proposed_test = method_split(proposed_doc, "test")
    modern_weak = first_method_split(modern_doc, ("weak_test", "weak")) if args.modern_sr_json else first_method_split(baseline_doc, ("weak_test", "weak"), modern_name)
    proposed_weak = first_method_split(proposed_doc, ("weak_test", "weak"))
    modern_clean = first_method_split(modern_doc, ("clean_test", "clean")) if args.modern_sr_json else first_method_split(baseline_doc, ("clean_test", "clean"), modern_name)
    proposed_clean = first_method_split(proposed_doc, ("clean_test", "clean"))
    hallucination_reduction = 1.0 - (
        proposed["no_defect_hallucination_rate_mean"] / max(generic["no_defect_hallucination_rate_mean"], 1e-8)
    )
    clean_hallucination_reduction = 1.0 - (
        value(proposed_clean, "no_defect_hallucination_rate_mean")
        / max(value(modern_clean, "no_defect_hallucination_rate_mean"), 1e-8)
    )
    clean_component_reduction = 1.0 - (
        value(proposed_clean, "component_hallucination_count_mean")
        / max(value(modern_clean, "component_hallucination_count_mean"), 1e-8)
    )
    test_recall_gain_vs_modern = (
        value(proposed_test, "defect_recall_mean") - value(modern_test, "defect_recall_mean")
    )
    weak_recall_gain_vs_modern = value(proposed_weak, "defect_recall_mean") - value(
        modern_weak, "defect_recall_mean"
    )
    gates = {
        "aggregate_hallucination_reduction_20pct": hallucination_reduction >= 0.20,
        "clean_hallucination_reduction_20pct": clean_hallucination_reduction >= 0.20,
        "clean_component_hallucination_reduction_20pct": clean_component_reduction >= 0.20,
        "test_recall_beats_modern_sr": test_recall_gain_vs_modern > 0.0,
        "weak_recall_beats_modern_sr": weak_recall_gain_vs_modern > 0.0,
        "test_recall_beats_lr": value(proposed_test, "defect_recall_mean") > value(lr_test, "defect_recall_mean"),
        "test_fpr_no_worse_than_lr": value(proposed_test, "false_positive_rate_mean")
        <= value(lr_test, "false_positive_rate_mean"),
        "aggregate_fpr_no_worse_than_lr": proposed["false_positive_rate_mean"] <= lr["false_positive_rate_mean"],
        "aggregate_hallucination_reduction_value": hallucination_reduction,
        "clean_hallucination_reduction_value": clean_hallucination_reduction,
        "clean_component_hallucination_reduction_value": clean_component_reduction,
        "test_recall_gain_vs_modern": test_recall_gain_vs_modern,
        "weak_recall_gain_vs_modern": weak_recall_gain_vs_modern,
        "risk_coverage_useful": risk_coverage_useful(proposed_doc),
        "modern_sr_reference": modern_name,
        "matched_fpr_protocol_present": "protocol" in proposed_doc,
        "split_metrics_present": bool(proposed_test and proposed_weak and proposed_clean),
        "note": "Single-seed smoke runs are not paper-ready evidence; aggregate multiple seeds before claims.",
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(gates, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    print(json.dumps(gates, indent=2))


if __name__ == "__main__":
    main()
