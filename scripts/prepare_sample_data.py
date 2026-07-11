from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waferinspectsr.config import ensure_dir, load_config
from waferinspectsr.degradation import DegradationConfig
from waferinspectsr.io import save_sample_npz
from waferinspectsr.synthetic import generate_benchmark_splits, generate_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic WaferInspectSR-Bench data.")
    parser.add_argument("--config", default="configs/smoke.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    output_dir = ensure_dir(cfg["output_dir"])
    degradation = DegradationConfig(scale=int(cfg["scale"]), **cfg["degradation"])
    if "split_counts" in cfg:
        samples = generate_benchmark_splits(
            split_counts={str(key): int(value) for key, value in cfg["split_counts"].items()},
            height=int(cfg["height"]),
            width=int(cfg["width"]),
            pattern_types=cfg["pattern_types"],
            defect_types=cfg["defect_types"],
            seed=int(cfg["seed"]),
        )
    else:
        samples = generate_dataset(
            num_samples=int(cfg["num_samples"]),
            height=int(cfg["height"]),
            width=int(cfg["width"]),
            pattern_types=cfg["pattern_types"],
            defect_types=cfg["defect_types"],
            seed=int(cfg["seed"]),
        )
    out = save_sample_npz(samples, Path(output_dir) / "samples.npz", degradation, seed=int(cfg["seed"]))
    print(f"wrote {len(samples)} samples to {out}")


if __name__ == "__main__":
    main()
