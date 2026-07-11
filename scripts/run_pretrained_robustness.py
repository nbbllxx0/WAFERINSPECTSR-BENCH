from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waferinspectsr.baselines import heuristic_defect_probability, uncertainty_from_probability
from waferinspectsr.degradation import DegradationConfig, degrade
from waferinspectsr.device import device_report, resolve_device
from waferinspectsr.metrics import aggregate_metric_dicts, summarize_prediction
from waferinspectsr.pretrained_sr import load_manifest, load_runner
from waferinspectsr.synthetic import generate_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run robustness sweeps for metric-ready pretrained SR baselines.")
    parser.add_argument("--manifest", default="configs/pretrained_sr_manifest.example.json")
    parser.add_argument("--run-json", default="experiments/runs/pretrained_sr/pretrained_sr.json")
    parser.add_argument("--output", default="experiments/runs/paper_robustness/pretrained_robustness.json")
    parser.add_argument("--num-samples", type=int, default=36)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        return {}
    return json.loads(target.read_text(encoding="utf-8"))


def defect_size_bin(mask: np.ndarray) -> str:
    area = int(np.asarray(mask).astype(bool).sum())
    if area == 0:
        return "clean"
    if area < 80:
        return "small"
    if area < 350:
        return "medium"
    return "large"


def predict_probability(runner, lr: np.ndarray, target_shape: tuple[int, int]):
    pred = runner.predict(lr, target_shape)
    prob = heuristic_defect_probability(pred.sr_image, sigma=2.2)
    risk = uncertainty_from_probability(prob)
    return pred, prob, risk


def summarize_method(method: str, rows: list[dict[str, float]], by_size: dict[str, list[dict[str, float]]]) -> list[dict[str, Any]]:
    out = [{"defect_size_bin": "all", "baseline": method, **aggregate_metric_dicts(rows)}]
    for size_bin, items in sorted(by_size.items()):
        out.append({"defect_size_bin": size_bin, "baseline": method, **aggregate_metric_dicts(items)})
    clean_rows = by_size.get("clean")
    if clean_rows:
        out.append({"defect_size_bin": "clean_false_calls", "baseline": method, **aggregate_metric_dicts(clean_rows)})
    return out


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    report = device_report(device)
    run_doc = load_json(args.run_json)
    thresholds = {
        str(name): float(value)
        for name, value in run_doc.get("protocol", {}).get("thresholds", {}).items()
    }

    status = []
    runners = []
    for spec in load_manifest(args.manifest):
        runner, row = load_runner(spec, args.manifest, device)
        status.append(row)
        if runner is not None:
            runners.append(runner)

    samples = generate_dataset(
        num_samples=args.num_samples,
        height=args.height,
        width=args.width,
        pattern_types=["line_space", "contact_hole"],
        defect_types=["bridge", "gap", "particle", "scratch", "missing_pattern", "none"],
        seed=args.seed,
    )

    if runners and samples:
        valid_runners = []
        probe_cfg = DegradationConfig(scale=2, blur_sigma=0.6, gaussian_noise=0.01, shot_noise=0.01)
        probe_lr = degrade(samples[0].image, probe_cfg, seed=args.seed)
        for runner in runners:
            try:
                runner.predict(probe_lr, samples[0].image.shape)
            except Exception as exc:  # pragma: no cover - external-checkpoint guard.
                for row in status:
                    if row["name"] == runner.spec.name:
                        row["available"] = False
                        row["reason"] = f"runtime_failed: {exc}"
                        break
            else:
                valid_runners.append(runner)
        runners = valid_runners

    if not runners:
        raise RuntimeError("No metric-ready pretrained SR runners are available for robustness evaluation.")

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
                rows_by_name: dict[str, list[dict[str, float]]] = {runner.spec.name: [] for runner in runners}
                rows_by_size: dict[str, dict[str, list[dict[str, float]]]] = {
                    runner.spec.name: {} for runner in runners
                }
                for index, sample in enumerate(samples):
                    lr = degrade(sample.image, cfg, seed=args.seed + index)
                    size_bin = defect_size_bin(sample.defect_mask)
                    for runner in runners:
                        pred, prob, risk = predict_probability(runner, lr, sample.image.shape)
                        threshold = thresholds.get(runner.spec.name, args.threshold)
                        metrics = summarize_prediction(
                            prob,
                            sample.defect_mask,
                            sample.clean_mask,
                            sample.edge_mask,
                            risk=risk,
                            threshold=threshold,
                            sr_image=pred.sr_image,
                            hr_image=sample.image,
                        )
                        scalar = {key: value for key, value in metrics.items() if isinstance(value, (float, int))}
                        scalar["runtime_ms"] = pred.runtime_ms
                        scalar["threshold"] = threshold
                        rows_by_name[runner.spec.name].append(scalar)
                        rows_by_size[runner.spec.name].setdefault(size_bin, []).append(scalar)
                for runner in runners:
                    for row in summarize_method(runner.spec.name, rows_by_name[runner.spec.name], rows_by_size[runner.spec.name]):
                        sweeps.append({"blur_sigma": blur_sigma, "noise": noise, "contrast": contrast, **row})

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "protocol": {
                    "device": report,
                    "manifest": args.manifest,
                    "run_json": args.run_json,
                    "num_samples": args.num_samples,
                    "height": args.height,
                    "width": args.width,
                    "seed": args.seed,
                    "thresholds": thresholds,
                },
                "status": status,
                "sweeps": sweeps,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out}")
    print(f"available={[runner.spec.name for runner in runners]}")
    print(f"sweeps={len(sweeps)}")


if __name__ == "__main__":
    main()
