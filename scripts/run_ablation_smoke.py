from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the four essential v1 ablations as short smoke trainings.")
    parser.add_argument("--input", default="data/generated/smoke/samples.npz")
    parser.add_argument("--output-dir", default="experiments/runs/smoke/ablations")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--train-python", default=sys.executable)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--matched-fpr", type=float, default=0.01)
    parser.add_argument("--calibration-splits", default=None)
    parser.add_argument("--mc-samples", type=int, default=4)
    parser.add_argument("--positive-weight", type=float, default=64.0)
    parser.add_argument("--hallucination-weight", type=float, default=None)
    parser.add_argument("--prior-fusion", choices=["none", "geometric", "average"], default="geometric")
    parser.add_argument("--prior-weight", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ablation in ["sr_only", "task", "task_hallucination", "full"]:
        out = out_dir / f"{ablation}.pt"
        cmd = [
            args.train_python,
            "scripts/train_dpu_wafersr.py",
            "--input",
            args.input,
            "--output",
            str(out),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--ablation",
            ablation,
            "--matched-fpr",
            str(args.matched_fpr),
            "--device",
            args.device,
            "--mc-samples",
            str(args.mc_samples),
            "--positive-weight",
            str(args.positive_weight),
            "--prior-fusion",
            args.prior_fusion,
            "--prior-weight",
            str(args.prior_weight),
        ]
        if args.calibration_splits:
            cmd.extend(["--calibration-splits", args.calibration_splits])
        if args.hallucination_weight is not None:
            cmd.extend(["--hallucination-weight", str(args.hallucination_weight)])
        print("running:", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
