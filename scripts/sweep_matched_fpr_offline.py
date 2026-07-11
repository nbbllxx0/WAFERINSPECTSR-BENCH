"""Select DPU operating points without test-set feedback, then evaluate once."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from waferinspectsr.device import device_report, resolve_device
from waferinspectsr.io import load_npz
from waferinspectsr.metrics import (
    aggregate_metric_dicts,
    apply_temperature,
    fit_temperature,
    summarize_prediction,
    threshold_for_target_fpr,
)
from waferinspectsr.models import CompactInspectionSafeSR
from waferinspectsr.protocol import apply_prior_fusion, indices_for_splits, parse_split_names

CANONICAL_SEEDS = [7, 11, 13, 17, 19, 23, 29, 31, 37, 41]
TARGET = 3e-4
TOL = 1.5
CALIB_TARGETS = [3e-4, 2e-4, 1.5e-4, 1e-4, 7.5e-5, 5e-5, 3e-5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "experiments" / "runs" / "default_seed_sweep_gpu_runtime_10seed",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "outputs"
        / "matched_clean_calib",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prior-fusion", default="geometric", choices=["none", "geometric", "average"])
    parser.add_argument("--seeds", default=",".join(str(s) for s in CANONICAL_SEEDS))
    return parser.parse_args()


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def infer(model, lr, device, mc_samples: int = 4, batch: int = 64):
    probs, risks, srs = [], [], []
    sync(device)
    with torch.no_grad():
        for start in range(0, len(lr), batch):
            x = lr[start : start + batch].to(device, non_blocking=True)
            if mc_samples > 1:
                model.train()
                pstack, rstack, sstack = [], [], []
                for _ in range(mc_samples):
                    out = model(x)
                    pstack.append(out["defect_prob"].squeeze(1))
                    rstack.append(out["risk"].squeeze(1))
                    sstack.append(out["sr"].squeeze(1))
                pt = torch.stack(pstack)
                rt = torch.stack(rstack)
                st = torch.stack(sstack)
                probs.append(pt.mean(0).cpu().numpy())
                risks.append(torch.clamp(rt.mean(0) + 4.0 * pt.var(0, unbiased=False), 0, 1).cpu().numpy())
                srs.append(st.mean(0).cpu().numpy())
                model.eval()
            else:
                model.eval()
                out = model(x)
                probs.append(out["defect_prob"].squeeze(1).cpu().numpy())
                risks.append(out["risk"].squeeze(1).cpu().numpy())
                srs.append(out["sr"].squeeze(1).cpu().numpy())
    sync(device)
    return np.concatenate(probs), np.concatenate(risks), np.concatenate(srs)


def eval_at_threshold(prob, risk, sr, data, splits, threshold):
    rows = []
    for i in range(len(prob)):
        m = summarize_prediction(
            prob[i],
            data["defect_mask"][i],
            data["clean_mask"][i],
            data["edge_mask"][i],
            risk=risk[i],
            threshold=threshold,
            sr_image=sr[i],
            hr_image=data["hr"][i],
        )
        rows.append({k: v for k, v in m.items() if isinstance(v, (float, int))})
    by_split = {}
    for split in sorted(set(splits)):
        subset = [row for row, s in zip(rows, splits) if s == split]
        if subset:
            by_split[split] = aggregate_metric_dicts(subset)
    return by_split

def eval_indices_at_threshold(prob, risk, sr, data, indices, threshold):
    rows = []
    for i in indices:
        m = summarize_prediction(
            prob[i],
            data["defect_mask"][i],
            data["clean_mask"][i],
            data["edge_mask"][i],
            risk=risk[i],
            threshold=threshold,
            sr_image=sr[i],
            hr_image=data["hr"][i],
        )
        rows.append({k: v for k, v in m.items() if isinstance(v, (float, int))})
    return aggregate_metric_dicts(rows)


def main() -> None:
    args = parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    report = device_report(device)
    chosen_rows = []
    diagnostics = []

    for seed in seeds:
        seed_dir = args.run_dir / f"seed_{seed}"
        data = load_npz(seed_dir / "data" / "samples.npz")
        splits = [str(v) for v in data.get("split", ["train"] * len(data["hr"]))]
        lr = torch.from_numpy(data["lr"][:, None].astype("float32"))
        scale = int(data["hr"].shape[-1] // data["lr"].shape[-1])
        ckpt = torch.load(seed_dir / "dpu_wafersr.pt", map_location="cpu")
        model = CompactInspectionSafeSR(
            scale=scale, channels=24, blocks=2, dropout=0.05, use_sr=bool(ckpt.get("use_sr", True))
        ).to(device)
        model.load_state_dict(ckpt["model_state"])
        prob, risk, sr = infer(model, lr, device)
        prob = apply_prior_fusion(prob, sr, args.prior_fusion, 0.5)

        val_idx = indices_for_splits(splits, parse_split_names("val_calib", "val_calib"))
        clean_calib_idx = indices_for_splits(splits, parse_split_names("clean_calib", "clean_calib"))
        temperature = fit_temperature(
            [prob[i] for i in val_idx],
            [data["defect_mask"][i] for i in val_idx],
        )
        prob_t = np.stack([apply_temperature(p, temperature) for p in prob])

        seed_out = {
            "seed": seed,
            "temperature": temperature,
            "device": report,
            "selection_rule": "fit on val_calib; select on clean_calib; evaluate test once after freezing",
            "targets": {},
        }
        chosen = None
        candidates = []
        for target in CALIB_TARGETS:
            thr = threshold_for_target_fpr(
                [prob_t[i] for i in val_idx],
                [data["clean_mask"][i] for i in val_idx],
                target,
            )
            clean_calib = eval_indices_at_threshold(
                prob_t, risk, sr, data, clean_calib_idx, thr
            )
            clean_calib_fpr = float(clean_calib["false_positive_rate_mean"])
            row = {
                "seed": seed,
                "calib_target": target,
                "threshold": thr,
                "clean_calib_fpr": clean_calib_fpr,
            }
            seed_out["targets"][f"{target:g}"] = row
            diagnostics.append(row)
            candidates.append(row)
            if chosen is None and clean_calib_fpr <= TARGET * TOL:
                chosen = dict(row)
                chosen["matched_fallback"] = 0.0
        if chosen is None:
            chosen = dict(candidates[-1])
            chosen["matched_fallback"] = 1.0

        by_split = eval_at_threshold(prob_t, risk, sr, data, splits, chosen["threshold"])
        chosen.update(
            {
                "recall": float(by_split["test"]["defect_recall_mean"]),
                "fpr": float(by_split["test"]["false_positive_rate_mean"]),
                "weak_recall": float(by_split.get("weak_test", {}).get("defect_recall_mean", float("nan"))),
                "clean_nhr": float(by_split.get("clean_test", {}).get("no_defect_hallucination_rate_mean", float("nan"))),
                "edge_f1": float(by_split["test"]["edge_f1_mean"]),
                "ssim": float(by_split["test"].get("ssim_mean", float("nan"))),
            }
        )
        seed_out["selected"] = chosen
        chosen_rows.append(chosen)
        out_path = args.output_dir / f"seed_{seed}_sweep.json"
        out_path.write_text(json.dumps(seed_out, indent=2), encoding="utf-8")
        print(
            f"seed={seed} chosen_target={chosen['calib_target']:g} "
            f"recall={chosen['recall']:.4f} fpr={chosen['fpr']:.6f} fallback={chosen['matched_fallback']}"
        )

    recalls = [r["recall"] for r in chosen_rows]
    fprs = [r["fpr"] for r in chosen_rows]
    weaks = [r["weak_recall"] for r in chosen_rows]
    agg = {
        "prior_fusion": args.prior_fusion,
        "target_fpr": TARGET,
        "tolerance": TOL,
        "n_seeds": len(chosen_rows),
        "feasible_matched_seeds": int(sum(1 for r in chosen_rows if r.get("matched_fallback", 1) < 0.5)),
        "test_feasible_seeds": int(sum(1 for r in chosen_rows if r["fpr"] <= TARGET * TOL)),
        "test_recall_mean": float(np.mean(recalls)),
        "test_recall_std": float(np.std(recalls)),
        "test_fpr_mean": float(np.mean(fprs)),
        "test_fpr_std": float(np.std(fprs)),
        "weak_recall_mean": float(np.mean(weaks)),
        "weak_recall_std": float(np.std(weaks)),
        "per_seed": chosen_rows,
        "diagnostics": diagnostics,
    }
    (args.output_dir / "matched_feasible_aggregate.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")
    print(json.dumps({k: agg[k] for k in agg if k not in {"per_seed", "diagnostics"}}, indent=2))


if __name__ == "__main__":
    main()
