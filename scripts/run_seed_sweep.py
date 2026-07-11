from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reproducible multi-seed benchmark sweeps.")
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--seeds", default="7,11,13,17,19")
    parser.add_argument("--output-dir", default="experiments/runs/seed_sweep")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--matched-fpr", type=float, default=0.01)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--positive-weight", type=float, default=64.0)
    parser.add_argument("--hallucination-weight", type=float, default=2.0)
    parser.add_argument("--prior-fusion", choices=["none", "geometric", "average"], default="geometric")
    parser.add_argument("--prior-weight", type=float, default=0.5)
    parser.add_argument("--mc-samples", type=int, default=4)
    parser.add_argument("--min-component-area", type=int, default=1)
    parser.add_argument("--calibration-splits", default="val_calib,clean_calib")
    parser.add_argument(
        "--train-python",
        default=None,
        help="Optional Python executable for PyTorch training scripts, e.g. a CUDA-enabled conda env.",
    )
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate_rows(rows: list[dict]) -> dict[str, dict[str, float]]:
    methods = sorted({row["method"] for row in rows})
    out: dict[str, dict[str, float]] = {}
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        keys = [
            key
            for key, value in method_rows[0].items()
            if key not in {"seed", "method"} and isinstance(value, (int, float))
        ]
        out[method] = {}
        for key in keys:
            values = [float(row[key]) for row in method_rows]
            mean = sum(values) / len(values)
            var = sum((value - mean) ** 2 for value in values) / len(values)
            out[method][f"{key}_across_seed_mean"] = mean
            out[method][f"{key}_across_seed_std"] = var**0.5
    return out


def collect_complete_seed_rows(output_dir: Path) -> tuple[list[int], list[dict]]:
    seeds: list[int] = []
    rows: list[dict] = []
    candidates: list[tuple[int, Path]] = []
    for seed_dir in output_dir.glob("seed_*"):
        try:
            seed = int(seed_dir.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        candidates.append((seed, seed_dir))
    for seed, seed_dir in sorted(candidates):
        baseline_path = seed_dir / "baselines.json"
        detector_path = seed_dir / "unet_detector.json"
        srlite_path = seed_dir / "srlite.json"
        proposed_path = seed_dir / "dpu_wafersr.json"
        if not all(path.exists() for path in (baseline_path, detector_path, srlite_path, proposed_path)):
            continue
        baseline = load_json(baseline_path)["summary"]
        detector_doc = load_json(detector_path)
        srlite_doc = load_json(srlite_path)
        proposed = load_json(proposed_path)["summary"]
        for method, summary in baseline.items():
            rows.append({"seed": seed, "method": method, **summary})
        rows.append({"seed": seed, "method": detector_doc.get("method", "unet_detector"), **detector_doc["summary"]})
        rows.append({"seed": seed, "method": srlite_doc.get("method", "srlite_detector"), **srlite_doc["summary"]})
        rows.append({"seed": seed, "method": "dpu_wafersr", **proposed})
        seeds.append(seed)
    return seeds, rows


def main() -> None:
    args = parse_args()
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    base_cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_python = args.train_python or sys.executable

    for seed in seeds:
        run_dir = output_dir / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg = dict(base_cfg)
        cfg["seed"] = seed
        cfg["output_dir"] = str(run_dir / "data")
        cfg_path = run_dir / "config.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")
        data_path = run_dir / "data" / "samples.npz"
        baseline_json = run_dir / "baselines.json"
        proposed_pt = run_dir / "dpu_wafersr.pt"
        proposed_json = run_dir / "dpu_wafersr.json"
        detector_pt = run_dir / "unet_detector.pt"
        detector_json = run_dir / "unet_detector.json"
        srlite_pt = run_dir / "srlite.pt"
        srlite_json = run_dir / "srlite.json"
        table_csv = run_dir / "main_table.csv"
        table_md = run_dir / "main_table.md"
        gates_json = run_dir / "gates.json"

        run([sys.executable, "scripts/prepare_sample_data.py", "--config", str(cfg_path)])
        run(
            [
                sys.executable,
                "scripts/run_baselines.py",
                "--input",
                str(data_path),
                "--output",
                str(baseline_json),
                "--matched-fpr",
                str(args.matched_fpr),
                *(
                    ["--calibration-splits", args.calibration_splits]
                    if args.calibration_splits
                    else []
                ),
            ]
        )
        run(
            [
                train_python,
                "scripts/train_srlite_baseline.py",
                "--input",
                str(data_path),
                "--output",
                str(srlite_pt),
                "--epochs",
                str(args.epochs),
                "--matched-fpr",
                str(args.matched_fpr),
                "--device",
                args.device,
                *(
                    ["--calibration-splits", args.calibration_splits]
                    if args.calibration_splits
                    else []
                ),
                "--min-component-area",
                str(args.min_component_area),
            ]
        )
        run(
            [
                train_python,
                "scripts/train_detector_baseline.py",
                "--input",
                str(data_path),
                "--output",
                str(detector_pt),
                "--epochs",
                str(args.epochs),
                "--matched-fpr",
                str(args.matched_fpr),
                "--device",
                args.device,
                *(
                    ["--calibration-splits", args.calibration_splits]
                    if args.calibration_splits
                    else []
                ),
                "--positive-weight",
                str(args.positive_weight),
                "--min-component-area",
                str(args.min_component_area),
            ]
        )
        run(
            [
                train_python,
                "scripts/train_dpu_wafersr.py",
                "--input",
                str(data_path),
                "--output",
                str(proposed_pt),
                "--epochs",
                str(args.epochs),
                "--matched-fpr",
                str(args.matched_fpr),
                "--device",
                args.device,
                *(
                    ["--calibration-splits", args.calibration_splits]
                    if args.calibration_splits
                    else []
                ),
                "--mc-samples",
                str(args.mc_samples),
                "--positive-weight",
                str(args.positive_weight),
                "--hallucination-weight",
                str(args.hallucination_weight),
                "--prior-fusion",
                args.prior_fusion,
                "--prior-weight",
                str(args.prior_weight),
                "--min-component-area",
                str(args.min_component_area),
            ]
        )
        run(
            [
                sys.executable,
                "scripts/make_tables.py",
                "--baseline-json",
                str(baseline_json),
                "--extra-json",
                str(srlite_json),
                "--extra-json",
                str(detector_json),
                "--proposed-json",
                str(proposed_json),
                "--output-csv",
                str(table_csv),
                "--output-md",
                str(table_md),
            ]
        )
        run(
            [
                sys.executable,
                "scripts/check_publication_gates.py",
                "--baseline-json",
                str(baseline_json),
                "--modern-sr-json",
                str(srlite_json),
                "--proposed-json",
                str(proposed_json),
                "--output",
                str(gates_json),
            ]
        )

    complete_seeds, combined = collect_complete_seed_rows(output_dir)
    out = output_dir / "combined_summary.json"
    out.write_text(
        json.dumps(
            {"seeds": complete_seeds, "summary_across_seeds": aggregate_rows(combined), "rows": combined},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
