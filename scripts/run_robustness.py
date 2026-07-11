from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waferinspectsr.baselines import run_v1_baselines
from waferinspectsr.degradation import DegradationConfig, degrade
from waferinspectsr.metrics import aggregate_metric_dicts, summarize_prediction
from waferinspectsr.synthetic import generate_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run robustness sweeps for v1 baselines.")
    parser.add_argument("--output", default="experiments/runs/smoke/robustness.json")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    return parser.parse_args()


def defect_size_bin(mask) -> str:
    area = int(np.asarray(mask).astype(bool).sum())
    if area == 0:
        return "clean"
    if area < 80:
        return "small"
    if area < 350:
        return "medium"
    return "large"


def main() -> None:
    args = parse_args()
    samples = generate_dataset(
        num_samples=args.num_samples,
        height=args.height,
        width=args.width,
        pattern_types=["line_space", "contact_hole"],
        defect_types=["bridge", "gap", "particle", "scratch", "missing_pattern", "none"],
        seed=args.seed,
    )
    sweeps = []
    for blur_sigma in [0.6, 1.2, 1.8]:
        for noise in [0.01, 0.03, 0.06]:
            for contrast in [0.8, 1.0, 1.2]:
                cfg = DegradationConfig(
                    scale=2,
                    blur_sigma=blur_sigma,
                    gaussian_noise=noise,
                    shot_noise=noise,
                    contrast_low=contrast,
                    contrast_high=contrast,
                )
                rows_by_name: dict[str, list[dict[str, float]]] = {}
                rows_by_size: dict[tuple[str, str], list[dict[str, float]]] = {}
                clean_rows_by_name: dict[str, list[dict[str, float]]] = {}
                for index, sample in enumerate(samples):
                    lr = degrade(sample.image, cfg, seed=args.seed + index)
                    size_bin = defect_size_bin(sample.defect_mask)
                    for pred in run_v1_baselines(lr, sample.image.shape, hr=sample.image, include_oracle=True):
                        metrics = summarize_prediction(
                            pred.defect_prob,
                            sample.defect_mask,
                            sample.clean_mask,
                            sample.edge_mask,
                            risk=pred.risk,
                        )
                        scalar = {key: value for key, value in metrics.items() if isinstance(value, (float, int))}
                        scalar["runtime_ms"] = pred.runtime_ms
                        rows_by_name.setdefault(pred.name, []).append(scalar)
                        rows_by_size.setdefault((pred.name, size_bin), []).append(scalar)
                        if size_bin == "clean":
                            clean_rows_by_name.setdefault(pred.name, []).append(scalar)
                for name, rows in rows_by_name.items():
                    sweeps.append(
                        {
                            "blur_sigma": blur_sigma,
                            "noise": noise,
                            "contrast": contrast,
                            "defect_size_bin": "all",
                            "baseline": name,
                            **aggregate_metric_dicts(rows),
                        }
                    )
                for (name, size_bin), rows in rows_by_size.items():
                    sweeps.append(
                        {
                            "blur_sigma": blur_sigma,
                            "noise": noise,
                            "contrast": contrast,
                            "defect_size_bin": size_bin,
                            "baseline": name,
                            **aggregate_metric_dicts(rows),
                        }
                    )
                for name, rows in clean_rows_by_name.items():
                    sweeps.append(
                        {
                            "blur_sigma": blur_sigma,
                            "noise": noise,
                            "contrast": contrast,
                            "defect_size_bin": "clean_false_calls",
                            "baseline": name,
                            **aggregate_metric_dicts(rows),
                        }
                    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"sweeps": sweeps}, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
