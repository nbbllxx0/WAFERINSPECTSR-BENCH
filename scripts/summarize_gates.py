from __future__ import annotations

import argparse
import json
from pathlib import Path


def mean_std(values: list[float]) -> dict[str, float]:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {"mean": mean, "std": variance**0.5}


def get_metric(doc: dict, split: str | None, key: str, method: str | None = None) -> float:
    if split is None:
        source = doc["summary"] if method is None else doc["summary"][method]
    else:
        source = doc["by_split"][split] if method is None else doc["by_split"][split][method]
    return float(source[key])


def split_metric(doc: dict, candidates: tuple[str, ...], key: str, method: str | None = None) -> float:
    by_split = doc.get("by_split", {})
    for split in candidates:
        if split in by_split:
            source = by_split[split] if method is None else by_split[split][method]
            return float(source[key])
    raise KeyError(f"None of the candidate splits exist: {candidates}")


def summarize_publication_gates(run_dir: Path) -> dict[str, object]:
    rows = []
    for seed_dir in sorted(run_dir.glob("seed_*")):
        baseline_path = seed_dir / "baselines.json"
        srlite_path = seed_dir / "srlite.json"
        proposed_path = seed_dir / "dpu_wafersr.json"
        if not (baseline_path.exists() and srlite_path.exists() and proposed_path.exists()):
            continue
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        srlite = json.loads(srlite_path.read_text(encoding="utf-8"))
        proposed = json.loads(proposed_path.read_text(encoding="utf-8"))
        risk_by_coverage = {float(row["coverage"]): float(row["error_mean"]) for row in proposed["risk_coverage"]}
        risk_improvement = risk_by_coverage.get(1.0, 0.0) - risk_by_coverage.get(0.5, 0.0)
        rows.append(
            {
                "seed": seed_dir.name,
                "lr_test_recall": get_metric(baseline, "test", "defect_recall_mean", "lr_detector"),
                "lr_test_fpr": get_metric(baseline, "test", "false_positive_rate_mean", "lr_detector"),
                "srlite_test_recall": get_metric(srlite, "test", "defect_recall_mean"),
                "srlite_weak_recall": split_metric(srlite, ("weak_test", "weak"), "defect_recall_mean"),
                "srlite_hallucination": get_metric(srlite, None, "no_defect_hallucination_rate_mean"),
                "dpu_test_recall": get_metric(proposed, "test", "defect_recall_mean"),
                "dpu_test_fpr": get_metric(proposed, "test", "false_positive_rate_mean"),
                "dpu_weak_recall": split_metric(proposed, ("weak_test", "weak"), "defect_recall_mean"),
                "dpu_hallucination": get_metric(proposed, None, "no_defect_hallucination_rate_mean"),
                "dpu_runtime_ms": get_metric(proposed, None, "runtime_ms_mean"),
                "risk_coverage_error_reduction_at_50pct": risk_improvement,
            }
        )
    if not rows:
        return {"available": False, "reason": "No complete seed directories found."}

    stats = {key: mean_std([row[key] for row in rows]) for key in rows[0] if key != "seed"}
    hallucination_reduction = 1.0 - stats["dpu_hallucination"]["mean"] / max(
        stats["srlite_hallucination"]["mean"], 1e-12
    )
    hard_gates = {
        "minimum_three_seed_evidence": len(rows) >= 3,
        "minimum_five_seed_evidence": len(rows) >= 5,
        "minimum_ten_seed_evidence": len(rows) >= 10,
        "hallucination_reduction_20pct": hallucination_reduction >= 0.20,
        "test_recall_beats_lr": stats["dpu_test_recall"]["mean"] > stats["lr_test_recall"]["mean"],
        "test_recall_beats_srlite": stats["dpu_test_recall"]["mean"] > stats["srlite_test_recall"]["mean"],
        "weak_recall_beats_srlite": stats["dpu_weak_recall"]["mean"] > stats["srlite_weak_recall"]["mean"],
        "test_fpr_no_worse_than_lr": stats["dpu_test_fpr"]["mean"] <= stats["lr_test_fpr"]["mean"],
        "runtime_reported": stats["dpu_runtime_ms"]["mean"] > 0.0,
        "risk_coverage_improves_under_abstention": stats["risk_coverage_error_reduction_at_50pct"]["mean"] > 0.0,
    }
    return {
        "available": True,
        "seed_count": len(rows),
        "rows": rows,
        "stats": stats,
        "hallucination_reduction_vs_srlite": hallucination_reduction,
        "hard_gates": hard_gates,
        "all_hard_gates_pass": all(hard_gates.values()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize per-seed gate JSON files.")
    parser.add_argument("--run-dir", default="experiments/runs/default_seed_sweep_gpu")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    paths = sorted(run_dir.glob("seed_*/gates.json"))
    if not paths:
        raise FileNotFoundError(f"No seed_*/gates.json files under {run_dir}")
    docs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    bool_keys = sorted(key for key, value in docs[0].items() if isinstance(value, bool))
    numeric_keys = sorted(key for key, value in docs[0].items() if isinstance(value, (int, float)) and not isinstance(value, bool))
    summary: dict[str, object] = {
        "run_dir": str(run_dir),
        "seeds": [path.parent.name for path in paths],
        "gate_pass_counts": {key: sum(1 for doc in docs if bool(doc.get(key))) for key in bool_keys},
        "all_pass": {key: all(bool(doc.get(key)) for doc in docs) for key in bool_keys},
        "numeric_mean": {},
        "publication_aggregate": summarize_publication_gates(run_dir),
    }
    for key in numeric_keys:
        values = [float(doc[key]) for doc in docs]
        summary["numeric_mean"][key] = mean_std(values)
    out = Path(args.output) if args.output else run_dir / "gate_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    print(json.dumps(summary["gate_pass_counts"], indent=2))


if __name__ == "__main__":
    main()
