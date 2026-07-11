from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waferinspectsr.baselines import heuristic_defect_probability, lr_detector, uncertainty_from_probability
from waferinspectsr.device import resolve_device
from waferinspectsr.io import load_npz
from waferinspectsr.metrics import aggregate_metric_dicts, apply_temperature, fit_temperature, summarize_prediction, threshold_for_target_fpr
from waferinspectsr.models import CompactInspectionSafeSR, CompactSRLite
from waferinspectsr.protocol import apply_prior_fusion, indices_for_splits, parse_split_names, require_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate shared local-contrast prior-fusion fairness sweeps.")
    parser.add_argument("--run-dir", default="experiments/runs/default_seed_sweep_gpu_runtime_10seed")
    parser.add_argument("--output-md", default="paper/tables/prior_fusion_sweep.md")
    parser.add_argument("--output-json", default="paper/tables/prior_fusion_sweep.json")
    parser.add_argument("--weights", default="0.0,0.25,0.5,0.75")
    parser.add_argument("--calibration-splits", default="val_calib,clean_calib")
    parser.add_argument("--matched-fpr", type=float, default=0.0003)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def run_srlite(seed_dir: Path, data: dict, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    lr = torch.from_numpy(data["lr"][:, None].astype("float32")).to(device)
    scale = int(data["hr"].shape[-1] // data["lr"].shape[-1])
    model = CompactSRLite(scale=scale, channels=24, blocks=2).to(device)
    state = torch.load(seed_dir / "srlite.pt", map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()
    with torch.no_grad():
        sr = model(lr).squeeze(1).cpu().numpy()
    probs = np.stack([heuristic_defect_probability(item, sigma=2.0) for item in sr]).astype(np.float32)
    return sr, probs


def run_dpu(seed_dir: Path, data: dict, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    protocol = json.loads((seed_dir / "dpu_wafersr.json").read_text(encoding="utf-8"))["protocol"]
    mc_samples = int(protocol.get("mc_samples", 1))
    scale = int(data["hr"].shape[-1] // data["lr"].shape[-1])
    model = CompactInspectionSafeSR(scale=scale, channels=24, blocks=2, dropout=0.05).to(device)
    state = torch.load(seed_dir / "dpu_wafersr.pt", map_location=device)
    model.load_state_dict(state["model_state"])
    lr = torch.from_numpy(data["lr"][:, None].astype("float32")).to(device)
    with torch.no_grad():
        if mc_samples > 1:
            model.train()
            probs, risks, srs = [], [], []
            for _ in range(mc_samples):
                out = model(lr)
                probs.append(out["defect_prob"].squeeze(1))
                risks.append(out["risk"].squeeze(1))
                srs.append(out["sr"].squeeze(1))
            prob_tensor = torch.stack(probs)
            risk_tensor = torch.stack(risks)
            sr_tensor = torch.stack(srs)
            prob = prob_tensor.mean(dim=0).cpu().numpy()
            risk = torch.clamp(risk_tensor.mean(dim=0) + 4.0 * prob_tensor.var(dim=0, unbiased=False), 0.0, 1.0).cpu().numpy()
            sr = sr_tensor.mean(dim=0).cpu().numpy()
            model.eval()
        else:
            model.eval()
            out = model(lr)
            sr = out["sr"].squeeze(1).cpu().numpy()
            prob = out["defect_prob"].squeeze(1).cpu().numpy()
            risk = out["risk"].squeeze(1).cpu().numpy()
    temperature = float(protocol.get("temperature", 1.0))
    return sr, prob, risk, temperature


def summarize_method(data: dict, prob: np.ndarray, sr: np.ndarray, risk: np.ndarray, splits: list[str], calib_indices: list[int], target_fpr: float) -> dict:
    temperature = fit_temperature([prob[idx] for idx in calib_indices], [data["defect_mask"][idx] for idx in calib_indices])
    prob = np.stack([apply_temperature(item, temperature) for item in prob]).astype(np.float32)
    threshold = threshold_for_target_fpr([prob[idx] for idx in calib_indices], [data["clean_mask"][idx] for idx in calib_indices], target_fpr)
    rows = []
    for idx in range(len(prob)):
        metrics = summarize_prediction(
            prob[idx],
            data["defect_mask"][idx],
            data["clean_mask"][idx],
            data["edge_mask"][idx],
            risk=risk[idx],
            threshold=threshold,
            sr_image=sr[idx],
            hr_image=data["hr"][idx],
        )
        rows.append({key: value for key, value in metrics.items() if isinstance(value, (float, int))})
    by_split = {}
    for split in sorted(set(splits)):
        split_rows = [row for row, row_split in zip(rows, splits) if row_split == split]
        if split_rows:
            by_split[split] = aggregate_metric_dicts(split_rows)
    return {"threshold": threshold, "temperature": temperature, "summary": aggregate_metric_dicts(rows), "by_split": by_split}


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    device = resolve_device(args.device)
    weights = [float(item.strip()) for item in args.weights.split(",") if item.strip()]
    all_rows = []
    for seed_dir in sorted(run_dir.glob("seed_*")):
        data = load_npz(seed_dir / "data" / "samples.npz")
        splits = [str(value) for value in data["split"]]
        split_names = parse_split_names(args.calibration_splits, "val_calib,clean_calib")
        calib_indices = indices_for_splits(splits, split_names)
        require_indices(calib_indices, split_names, "calibration")
        lr_sr = np.stack([lr_detector(item, data["hr"][0].shape).sr_image for item in data["lr"]]).astype(np.float32)
        lr_prob = np.stack([lr_detector(item, data["hr"][0].shape).defect_prob for item in data["lr"]]).astype(np.float32)
        srlite_sr, srlite_prob = run_srlite(seed_dir, data, device)
        dpu_sr, dpu_prob, dpu_risk, _ = run_dpu(seed_dir, data, device)
        sources = {
            "LR+prior": (lr_sr, lr_prob, np.stack([uncertainty_from_probability(item) for item in lr_prob])),
            "SR-lite+prior": (srlite_sr, srlite_prob, np.stack([uncertainty_from_probability(item) for item in srlite_prob])),
            "DPU+prior": (dpu_sr, dpu_prob, dpu_risk),
        }
        for weight in weights:
            for method, (sr, prob, risk) in sources.items():
                fused = apply_prior_fusion(prob, sr, "geometric", weight)
                summary = summarize_method(data, fused, sr, risk, splits, calib_indices, args.matched_fpr)
                test = summary["by_split"]["test"]
                clean = summary["by_split"].get("clean_test", summary["by_split"].get("clean", {}))
                all_rows.append(
                    {
                        "seed": seed_dir.name,
                        "method": method,
                        "prior_weight": weight,
                        "test_recall": float(test["defect_recall_mean"]),
                        "test_fpr": float(test["false_positive_rate_mean"]),
                        "clean_nhr": float(clean.get("no_defect_hallucination_rate_mean", 0.0)),
                        "component_recall": float(test["component_recall_mean"]),
                        "precision": float(test["precision_mean"]),
                    }
                )
    grouped = {}
    for row in all_rows:
        key = (row["method"], row["prior_weight"])
        grouped.setdefault(key, []).append(row)
    table_rows = []
    for (method, weight), rows in sorted(grouped.items()):
        out = {"Method": method, "Prior weight": weight}
        for key in ["test_recall", "test_fpr", "clean_nhr", "component_recall", "precision"]:
            vals = np.asarray([row[key] for row in rows], dtype=np.float64)
            out[key] = f"{vals.mean():.6f} +/- {vals.std(ddof=0):.6f}"
        table_rows.append(out)
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps({"rows": all_rows, "summary": table_rows}, indent=2), encoding="utf-8")
    headers = list(table_rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in table_rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.output_md}")
    print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
