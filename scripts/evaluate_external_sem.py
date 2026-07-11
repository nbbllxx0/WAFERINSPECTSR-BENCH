from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waferinspectsr.device import resolve_device
from waferinspectsr.degradation import upsample_lr
from waferinspectsr.metrics import apply_temperature
from waferinspectsr.models import CompactInspectionSafeSR
from waferinspectsr.protocol import apply_prior_fusion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference-only external SEM stress evaluation on Carinthia.")
    parser.add_argument("--csv", default="data/raw/carinthia/data/data/carinthia.csv")
    parser.add_argument("--root", default="data/raw/carinthia/data")
    parser.add_argument("--checkpoint", default="experiments/runs/default_seed_sweep_gpu_runtime_10seed/seed_7/dpu_wafersr.pt")
    parser.add_argument("--run-json", default="experiments/runs/default_seed_sweep_gpu_runtime_10seed/seed_7/dpu_wafersr.json")
    parser.add_argument("--output-json", default="paper/tables/external_sem_stress.json")
    parser.add_argument("--output-md", default="paper/tables/external_sem_stress.md")
    parser.add_argument("--output-csv", default="paper/tables/external_sem_stress.csv")
    parser.add_argument("--max-per-class", type=int, default=64)
    parser.add_argument("--hr-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_rows(csv_path: Path, root: Path, max_per_class: int) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    counts: dict[str, int] = defaultdict(int)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        for row in reader:
            label = str(row["label"])
            if counts[label] >= max_per_class:
                continue
            image_path = root / row["image_path"]
            if image_path.exists():
                selected.append({"label": label, "image": str(image_path), "file_name": row["file_name"]})
                counts[label] += 1
    return selected


def load_image(path: Path, hr_size: int) -> tuple[np.ndarray, np.ndarray]:
    image = Image.open(path).convert("L").resize((hr_size, hr_size), Image.Resampling.BICUBIC)
    hr = np.asarray(image, dtype=np.float32) / 255.0
    lr_size = hr_size // 2
    lr = np.asarray(image.resize((lr_size, lr_size), Image.Resampling.BICUBIC), dtype=np.float32) / 255.0
    return lr.astype(np.float32), hr.astype(np.float32)


def run_model(
    model: CompactInspectionSafeSR,
    lr_batch: torch.Tensor,
    device: torch.device,
    mc_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eval_lr = lr_batch.to(device)
    with torch.no_grad():
        if mc_samples > 1:
            model.train()
            probs, risks, srs = [], [], []
            for _ in range(mc_samples):
                outputs = model(eval_lr)
                probs.append(outputs["defect_prob"].squeeze(1))
                risks.append(outputs["risk"].squeeze(1))
                srs.append(outputs["sr"].squeeze(1))
            prob_tensor = torch.stack(probs)
            risk_tensor = torch.stack(risks)
            sr_tensor = torch.stack(srs)
            prob = prob_tensor.mean(dim=0)
            sr = sr_tensor.mean(dim=0)
            risk = torch.clamp(risk_tensor.mean(dim=0) + 4.0 * prob_tensor.var(dim=0, unbiased=False), 0.0, 1.0)
        else:
            model.eval()
            outputs = model(eval_lr)
            sr = outputs["sr"].squeeze(1)
            prob = outputs["defect_prob"].squeeze(1)
            risk = outputs["risk"].squeeze(1)
    return sr.cpu().numpy(), prob.cpu().numpy(), risk.cpu().numpy()


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = float(np.mean(values))
    return mean, float(np.std(values))


def write_tables(rows: list[dict[str, object]], output_md: Path, output_csv: Path) -> None:
    by_label: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_label[str(row["label"])].append(row)
    table_rows = []
    for label, group in sorted(by_label.items(), key=lambda item: int(item[0])):
        density = [float(row["predicted_defect_density"]) for row in group]
        risk = [float(row["mean_risk"]) for row in group]
        response = [float(row["mean_probability"]) for row in group]
        density_mean, density_std = mean_std(density)
        risk_mean, risk_std = mean_std(risk)
        response_mean, response_std = mean_std(response)
        table_rows.append(
            {
                "Label": label,
                "Samples": len(group),
                "Predicted density": f"{density_mean:.6f} +/- {density_std:.6f}",
                "Mean probability": f"{response_mean:.6f} +/- {response_std:.6f}",
                "Mean risk": f"{risk_mean:.6f} +/- {risk_std:.6f}",
            }
        )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    headers = ["Label", "Samples", "Predicted density", "Mean probability", "Mean risk"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in table_rows:
        lines.append("| " + " | ".join(str(row[header]) for header in headers) + " |")
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(table_rows)


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    root = Path(args.root)
    run_doc = json.loads(Path(args.run_json).read_text(encoding="utf-8"))
    protocol = run_doc["protocol"]
    rows = load_rows(csv_path, root, args.max_per_class)
    if not rows:
        raise ValueError(f"No external SEM images found from {csv_path}")

    device = resolve_device(args.device)
    model = CompactInspectionSafeSR(scale=2, channels=24, blocks=2, dropout=0.05).to(device)
    try:
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model_state"])

    lr_images, hr_images = [], []
    for row in rows:
        lr, hr = load_image(Path(str(row["image"])), args.hr_size)
        lr_images.append(lr)
        hr_images.append(hr)
    lr_batch = torch.from_numpy(np.asarray(lr_images, dtype=np.float32)[:, None])

    start = time.perf_counter()
    sr, prob, risk = run_model(model, lr_batch, device, int(protocol.get("mc_samples", 1)))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    runtime_ms = 1000.0 * (time.perf_counter() - start) / max(len(rows), 1)

    prob = apply_prior_fusion(prob, sr, protocol.get("prior_fusion", "none"), float(protocol.get("prior_weight", 0.5)))
    temperature = float(protocol.get("temperature", 1.0))
    if temperature > 0:
        prob = np.stack([apply_temperature(item, temperature) for item in prob]).astype(np.float32)

    threshold = float(protocol["threshold"])
    out_rows = []
    for index, row in enumerate(rows):
        predicted = prob[index] >= threshold
        bicubic = upsample_lr(lr_images[index], (args.hr_size, args.hr_size))
        out_rows.append(
            {
                **row,
                "runtime_ms": runtime_ms,
                "threshold": threshold,
                "predicted_defect_density": float(predicted.mean()),
                "mean_probability": float(prob[index].mean()),
                "mean_risk": float(risk[index].mean()),
                "sr_l1_to_bicubic": float(np.mean(np.abs(sr[index] - bicubic))),
                "image_mean": float(np.mean(hr_images[index])),
                "image_std": float(np.std(hr_images[index])),
            }
        )

    payload = {
        "source": {
            "dataset": "Carinthia",
            "csv": str(csv_path),
            "root": str(root),
            "sample_count": len(out_rows),
            "max_per_class": args.max_per_class,
            "mask_available": False,
            "paired_hr_lr_available": False,
            "interpretation": "inference-only external SEM stress; no recall/FPR claims without masks",
        },
        "protocol": protocol,
        "rows": out_rows,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_tables(out_rows, Path(args.output_md), Path(args.output_csv))
    print(f"wrote {len(out_rows)} rows to {args.output_json}")
    print(f"runtime_ms_per_sample={runtime_ms:.3f}")


if __name__ == "__main__":
    main()
