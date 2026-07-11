from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run/resume scaled neural evidence over multiple scaled seeds.")
    parser.add_argument("--config", default="configs/scaled.yaml")
    parser.add_argument("--seeds", default="20260602,20260603,20260604,20260605,20260606,20260607,20260608,20260609")
    parser.add_argument("--data-root", default="data/generated")
    parser.add_argument("--run-root", default="experiments/runs")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--matched-fpr", type=float, default=0.0003)
    parser.add_argument("--calibration-splits", default="val_calib,clean_calib")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-python", default=None)
    parser.add_argument("--mc-samples", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--positive-weight", type=float, default=64.0)
    parser.add_argument("--hallucination-weight", type=float, default=2.0)
    parser.add_argument("--prior-fusion", choices=["none", "geometric", "average"], default="geometric")
    parser.add_argument("--prior-weight", type=float, default=0.5)
    parser.add_argument("--force", action="store_true", help="Rerun steps even when their expected outputs exist.")
    return parser.parse_args()


def parse_seeds(raw: str) -> list[int]:
    seeds = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def run(cmd: list[str], expected: Path | None = None, force: bool = False) -> None:
    if expected is not None and expected.exists() and not force:
        print(f"skip existing {expected}")
        return
    print("running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def seed_paths(seed: int, data_root: Path, run_root: Path) -> dict[str, Path]:
    run_dir = run_root / f"scaled_seed{seed}"
    neural_dir = run_dir / "neural_full"
    return {
        "run_dir": run_dir,
        "data_dir": data_root / f"scaled_seed{seed}",
        "config": run_dir / "scaled_config.yaml",
        "manifest": data_root / f"scaled_seed{seed}" / "manifest.json",
        "samples": neural_dir / "data" / "samples.npz",
        "baseline_json": neural_dir / "baselines.json",
        "srlite_pt": neural_dir / "srlite.pt",
        "srlite_json": neural_dir / "srlite.json",
        "unet_pt": neural_dir / "unet_detector.pt",
        "unet_json": neural_dir / "unet_detector.json",
        "dpu_pt": neural_dir / "dpu_wafersr.pt",
        "dpu_json": neural_dir / "dpu_wafersr.json",
        "table_csv": neural_dir / "main_table.csv",
        "table_md": neural_dir / "main_table.md",
        "gates": neural_dir / "gates.json",
    }


def write_seed_config(base_cfg: dict, seed: int, paths: dict[str, Path]) -> None:
    paths["run_dir"].mkdir(parents=True, exist_ok=True)
    cfg = copy.deepcopy(base_cfg)
    cfg["seed"] = seed
    cfg["output_dir"] = paths["data_dir"].as_posix()
    paths["config"].write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    base_cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if not isinstance(base_cfg, dict):
        raise ValueError(f"Config must be a mapping: {args.config}")
    data_root = Path(args.data_root)
    run_root = Path(args.run_root)
    train_python = args.train_python or sys.executable

    for seed in seeds:
        paths = seed_paths(seed, data_root, run_root)
        write_seed_config(base_cfg, seed, paths)
        neural_dir = paths["samples"].parent.parent
        neural_dir.mkdir(parents=True, exist_ok=True)
        run(
            [
                sys.executable,
                "scripts/prepare_scaled_data.py",
                "--config",
                str(paths["config"]),
                "--summary-md",
                str(paths["run_dir"] / "scaled_synthetic_manifest.md"),
                "--summary-csv",
                str(paths["run_dir"] / "scaled_synthetic_manifest.csv"),
            ],
            expected=paths["manifest"],
            force=args.force,
        )
        run(
            [
                sys.executable,
                "scripts/materialize_scaled_npz.py",
                "--manifest",
                str(paths["manifest"]),
                "--output",
                str(paths["samples"]),
                "--max-per-split",
                "0",
                "--summary-md",
                str(neural_dir / "scaled_neural_materialization.md"),
                "--summary-csv",
                str(neural_dir / "scaled_neural_materialization.csv"),
            ],
            expected=paths["samples"],
            force=args.force,
        )
        run(
            [
                sys.executable,
                "scripts/run_baselines.py",
                "--input",
                str(paths["samples"]),
                "--output",
                str(paths["baseline_json"]),
                "--matched-fpr",
                str(args.matched_fpr),
                "--calibration-splits",
                args.calibration_splits,
            ],
            expected=paths["baseline_json"],
            force=args.force,
        )
        run(
            [
                train_python,
                "scripts/train_srlite_baseline.py",
                "--input",
                str(paths["samples"]),
                "--output",
                str(paths["srlite_pt"]),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--matched-fpr",
                str(args.matched_fpr),
                "--calibration-splits",
                args.calibration_splits,
                "--device",
                args.device,
            ],
            expected=paths["srlite_json"],
            force=args.force,
        )
        run(
            [
                train_python,
                "scripts/train_detector_baseline.py",
                "--input",
                str(paths["samples"]),
                "--output",
                str(paths["unet_pt"]),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--matched-fpr",
                str(args.matched_fpr),
                "--calibration-splits",
                args.calibration_splits,
                "--device",
                args.device,
                "--positive-weight",
                str(args.positive_weight),
            ],
            expected=paths["unet_json"],
            force=args.force,
        )
        run(
            [
                train_python,
                "scripts/train_dpu_wafersr.py",
                "--input",
                str(paths["samples"]),
                "--output",
                str(paths["dpu_pt"]),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--matched-fpr",
                str(args.matched_fpr),
                "--calibration-splits",
                args.calibration_splits,
                "--positive-weight",
                str(args.positive_weight),
                "--hallucination-weight",
                str(args.hallucination_weight),
                "--prior-fusion",
                args.prior_fusion,
                "--prior-weight",
                str(args.prior_weight),
                "--mc-samples",
                str(args.mc_samples),
                "--eval-batch-size",
                str(args.eval_batch_size),
                "--device",
                args.device,
            ],
            expected=paths["dpu_json"],
            force=args.force,
        )
        run(
            [
                sys.executable,
                "scripts/make_tables.py",
                "--baseline-json",
                str(paths["baseline_json"]),
                "--extra-json",
                str(paths["srlite_json"]),
                "--extra-json",
                str(paths["unet_json"]),
                "--proposed-json",
                str(paths["dpu_json"]),
                "--output-csv",
                str(paths["table_csv"]),
                "--output-md",
                str(paths["table_md"]),
            ],
            expected=paths["table_md"],
            force=args.force,
        )
        run(
            [
                sys.executable,
                "scripts/check_publication_gates.py",
                "--baseline-json",
                str(paths["baseline_json"]),
                "--modern-sr-json",
                str(paths["srlite_json"]),
                "--proposed-json",
                str(paths["dpu_json"]),
                "--output",
                str(paths["gates"]),
            ],
            expected=paths["gates"],
            force=args.force,
        )
    print(f"scaled neural sweep complete seeds={seeds}")


if __name__ == "__main__":
    main()
