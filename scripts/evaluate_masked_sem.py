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

from waferinspectsr.baselines import run_v1_baselines, uncertainty_from_probability
from waferinspectsr.device import resolve_device
from waferinspectsr.image_ops import binary_dilation
from waferinspectsr.metrics import apply_temperature, summarize_prediction
from waferinspectsr.models import CompactInspectionSafeSR, CompactUNetDetector
from waferinspectsr.protocol import apply_prior_fusion
from waferinspectsr.synthetic import mask_edges


METHOD_ORDER = [
    "no_sr_task_detector",
    "lr_detector",
    "nearest_detector",
    "bilinear_detector",
    "bicubic_detector",
    "lanczos_detector",
    "denoise_upsample_detector",
    "wiener_deconv_detector",
    "prior_only_detector",
    "sharpened_sr_detector",
    "naf_style_sr_detector",
    "unet_detector",
    "dpu_wafersr",
]

METHOD_LABELS = {
    "no_sr_task_detector": "No-SR Task",
    "lr_detector": "LR",
    "nearest_detector": "Nearest",
    "bilinear_detector": "Bilinear",
    "bicubic_detector": "Bicubic",
    "lanczos_detector": "Lanczos",
    "denoise_upsample_detector": "Denoise+Upsample",
    "wiener_deconv_detector": "Wiener Deconv.",
    "prior_only_detector": "Prior-only",
    "sharpened_sr_detector": "SharpSR",
    "naf_style_sr_detector": "NAF-style SR",
    "unet_detector": "U-Net detector",
    "dpu_wafersr": "DPU-WaferSR",
}

LABEL_NAMES = {
    "1": "defect class 1",
    "2": "defect class 2",
    "3": "defect class 3",
    "4": "defect class 4",
    "5": "defect class 5",
    "6": "no-defect class 6",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate frozen protocols on masked external SEM data.")
    parser.add_argument("--csv", default="data/raw/carinthia_s/data/data/carinthia-s.csv")
    parser.add_argument("--root", default="data/raw/carinthia_s/data/data")
    parser.add_argument("--checkpoint", default="experiments/runs/default_seed_sweep_gpu_runtime_10seed/seed_7/dpu_wafersr.pt")
    parser.add_argument("--run-json", default="experiments/runs/default_seed_sweep_gpu_runtime_10seed/seed_7/dpu_wafersr.json")
    parser.add_argument("--detector-checkpoint", default="experiments/runs/default_seed_sweep_gpu_runtime_10seed/seed_7/unet_detector.pt")
    parser.add_argument("--detector-json", default="experiments/runs/default_seed_sweep_gpu_runtime_10seed/seed_7/unet_detector.json")
    parser.add_argument("--baseline-json", default="experiments/runs/default_seed_sweep_gpu_runtime_10seed/seed_7/baselines.json")
    parser.add_argument("--output-json", default="paper/tables/carinthia_s_masked_validation.json")
    parser.add_argument("--output-md", default="paper/tables/carinthia_s_masked_validation.md")
    parser.add_argument("--output-csv", default="paper/tables/carinthia_s_masked_validation.csv")
    parser.add_argument("--max-per-class", type=int, default=128, help="Maximum samples per label. Use <=0 for all rows.")
    parser.add_argument("--batch-size", type=int, default=128)
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
            if max_per_class > 0 and counts[label] >= max_per_class:
                continue
            image_path = root / row["image_path"]
            mask_path = root / row["mask_path"]
            if image_path.exists() and mask_path.exists():
                selected.append(
                    {
                        "label": label,
                        "image": str(image_path),
                        "mask": str(mask_path),
                        "file_name": row.get("filename", image_path.stem),
                    }
                )
                counts[label] += 1
    return selected


def load_image_mask(row: dict[str, str], hr_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = Image.open(row["image"]).convert("L").resize((hr_size, hr_size), Image.Resampling.BICUBIC)
    mask = Image.open(row["mask"]).convert("L").resize((hr_size, hr_size), Image.Resampling.NEAREST)
    hr = np.asarray(image, dtype=np.float32) / 255.0
    defect_mask = np.asarray(mask, dtype=np.uint8) > 0
    lr = np.asarray(image.resize((hr_size // 2, hr_size // 2), Image.Resampling.BICUBIC), dtype=np.float32) / 255.0
    return lr.astype(np.float32), hr.astype(np.float32), defect_mask


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


def run_unet_detector(
    model: CompactUNetDetector,
    lr_batch: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    eval_lr = lr_batch.to(device)
    with torch.no_grad():
        model.eval()
        prob = model(eval_lr).squeeze(1)
        sr = torch.nn.functional.interpolate(eval_lr, size=prob.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
    return sr.cpu().numpy(), prob.cpu().numpy()


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    return float(np.mean(values)), float(np.std(values))


def predict_dpu_batched(
    model: CompactInspectionSafeSR,
    lr_images: list[np.ndarray],
    device: torch.device,
    batch_size: int,
    mc_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sr_chunks: list[np.ndarray] = []
    prob_chunks: list[np.ndarray] = []
    risk_chunks: list[np.ndarray] = []
    batch_size = max(int(batch_size), 1)
    for start in range(0, len(lr_images), batch_size):
        lr_batch = torch.from_numpy(np.asarray(lr_images[start : start + batch_size], dtype=np.float32)[:, None])
        sr, prob, risk = run_model(model, lr_batch, device, mc_samples)
        sr_chunks.append(sr)
        prob_chunks.append(prob)
        risk_chunks.append(risk)
    return np.concatenate(sr_chunks), np.concatenate(prob_chunks), np.concatenate(risk_chunks)


def predict_unet_batched(
    model: CompactUNetDetector,
    lr_images: list[np.ndarray],
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    sr_chunks: list[np.ndarray] = []
    prob_chunks: list[np.ndarray] = []
    batch_size = max(int(batch_size), 1)
    for start in range(0, len(lr_images), batch_size):
        lr_batch = torch.from_numpy(np.asarray(lr_images[start : start + batch_size], dtype=np.float32)[:, None])
        sr, prob = run_unet_detector(model, lr_batch, device)
        sr_chunks.append(sr)
        prob_chunks.append(prob)
    return np.concatenate(sr_chunks), np.concatenate(prob_chunks)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["method"]), str(row["label"]))].append(row)
    for (method, label), group in sorted(groups.items(), key=lambda item: (METHOD_ORDER.index(item[0][0]), int(item[0][1]))):
        result: dict[str, object] = {"Method": METHOD_LABELS[method], "Class": LABEL_NAMES.get(label, f"class {label}"), "Samples": len(group)}
        for source_key, out_key, digits in [
            ("defect_recall", "Recall", 4),
            ("false_positive_rate", "FPR", 6),
            ("no_defect_hallucination_rate", "NHR", 6),
            ("component_recall", "Component recall", 4),
            ("component_iou10_recall", "Comp IoU>=0.10", 4),
            ("component_centroid8_recall", "Comp centroid<=8px", 4),
            ("precision", "Precision", 4),
            ("mask_iou", "Mask IoU", 4),
            ("edge_f1", "Edge F1", 4),
            ("false_components_per_mpx", "False comp/Mpx", 2),
        ]:
            m, s = mean_std([float(row[source_key]) for row in group])
            result[out_key] = f"{m:.{digits}f} +/- {s:.{digits}f}"
        out.append(result)
    aggregate: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        aggregate[str(row["method"])].append(row)
    for method, group in sorted(aggregate.items(), key=lambda item: METHOD_ORDER.index(item[0])):
        result = {"Method": METHOD_LABELS[method], "Class": "all classes", "Samples": len(group)}
        for source_key, out_key, digits in [
            ("defect_recall", "Recall", 4),
            ("false_positive_rate", "FPR", 6),
            ("no_defect_hallucination_rate", "NHR", 6),
            ("component_recall", "Component recall", 4),
            ("component_iou10_recall", "Comp IoU>=0.10", 4),
            ("component_centroid8_recall", "Comp centroid<=8px", 4),
            ("precision", "Precision", 4),
            ("mask_iou", "Mask IoU", 4),
            ("edge_f1", "Edge F1", 4),
            ("false_components_per_mpx", "False comp/Mpx", 2),
        ]:
            m, s = mean_std([float(row[source_key]) for row in group])
            result[out_key] = f"{m:.{digits}f} +/- {s:.{digits}f}"
        out.insert(0, result)
    return out


def write_tables(summary_rows: list[dict[str, object]], output_md: Path, output_csv: Path) -> None:
    output_md.parent.mkdir(parents=True, exist_ok=True)
    headers = list(summary_rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in summary_rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(summary_rows)


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    root = Path(args.root)
    rows = load_rows(csv_path, root, args.max_per_class)
    if not rows:
        raise ValueError(f"No paired image/mask rows found from {csv_path}")

    run_doc = json.loads(Path(args.run_json).read_text(encoding="utf-8"))
    detector_doc = json.loads(Path(args.detector_json).read_text(encoding="utf-8"))
    baseline_doc = json.loads(Path(args.baseline_json).read_text(encoding="utf-8"))
    protocol = run_doc["protocol"]
    detector_protocol = detector_doc["protocol"]
    thresholds = baseline_doc["protocol"]["thresholds"]

    lr_images, hr_images, defect_masks = [], [], []
    for row in rows:
        lr, hr, defect_mask = load_image_mask(row, args.hr_size)
        lr_images.append(lr)
        hr_images.append(hr)
        defect_masks.append(defect_mask)

    device = resolve_device(args.device)
    model = CompactInspectionSafeSR(scale=2, channels=24, blocks=2, dropout=0.05).to(device)
    try:
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model_state"])
    start = time.perf_counter()
    sr, dpu_prob, dpu_risk = predict_dpu_batched(
        model, lr_images, device, int(args.batch_size), int(protocol.get("mc_samples", 1))
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    dpu_runtime_ms = 1000.0 * (time.perf_counter() - start) / max(len(rows), 1)
    dpu_prob = apply_prior_fusion(dpu_prob, sr, protocol.get("prior_fusion", "none"), float(protocol.get("prior_weight", 0.5)))
    dpu_prob = np.stack([apply_temperature(item, float(protocol.get("temperature", 1.0))) for item in dpu_prob]).astype(np.float32)

    unet = CompactUNetDetector(scale=2, channels=int(detector_protocol.get("channels", 16))).to(device)
    try:
        unet_state = torch.load(args.detector_checkpoint, map_location=device, weights_only=True)
    except TypeError:
        unet_state = torch.load(args.detector_checkpoint, map_location=device)
    unet.load_state_dict(unet_state["model_state"])
    start = time.perf_counter()
    unet_sr, unet_prob = predict_unet_batched(unet, lr_images, device, int(args.batch_size))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    unet_runtime_ms = 1000.0 * (time.perf_counter() - start) / max(len(rows), 1)
    unet_prob = np.stack([apply_temperature(item, float(detector_protocol.get("temperature", 1.0))) for item in unet_prob]).astype(np.float32)
    unet_risk = np.stack([uncertainty_from_probability(item) for item in unet_prob]).astype(np.float32)

    out_rows: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        defect_mask = defect_masks[index]
        clean_mask = ~binary_dilation(defect_mask, iterations=3)
        edge_mask = mask_edges(defect_mask)
        for pred in run_v1_baselines(lr_images[index], (args.hr_size, args.hr_size), hr=hr_images[index], include_oracle=False):
            threshold = float(thresholds[pred.name])
            metrics = summarize_prediction(
                pred.defect_prob,
                defect_mask,
                clean_mask,
                edge_mask,
                risk=pred.risk,
                threshold=threshold,
                sr_image=pred.sr_image,
                hr_image=hr_images[index],
            )
            out_rows.append(
                {
                    "dataset": "Carinthia-S",
                    "label": row["label"],
                    "class_name": LABEL_NAMES.get(row["label"], f"class {row['label']}"),
                    "image": row["image"],
                    "mask": row["mask"],
                    "method": pred.name,
                    "threshold": threshold,
                    "runtime_ms": pred.runtime_ms,
                    **{k: v for k, v in metrics.items() if isinstance(v, (int, float))},
                }
            )
        metrics = summarize_prediction(
            unet_prob[index],
            defect_mask,
            clean_mask,
            edge_mask,
            risk=unet_risk[index],
            threshold=float(detector_protocol["threshold"]),
            sr_image=unet_sr[index],
            hr_image=hr_images[index],
        )
        out_rows.append(
            {
                "dataset": "Carinthia-S",
                "label": row["label"],
                "class_name": LABEL_NAMES.get(row["label"], f"class {row['label']}"),
                "image": row["image"],
                "mask": row["mask"],
                "method": "unet_detector",
                "threshold": float(detector_protocol["threshold"]),
                "runtime_ms": unet_runtime_ms,
                **{k: v for k, v in metrics.items() if isinstance(v, (int, float))},
            }
        )
        metrics = summarize_prediction(
            dpu_prob[index],
            defect_mask,
            clean_mask,
            edge_mask,
            risk=dpu_risk[index],
            threshold=float(protocol["threshold"]),
            sr_image=sr[index],
            hr_image=hr_images[index],
        )
        out_rows.append(
            {
                "dataset": "Carinthia-S",
                "label": row["label"],
                "class_name": LABEL_NAMES.get(row["label"], f"class {row['label']}"),
                "image": row["image"],
                "mask": row["mask"],
                "method": "dpu_wafersr",
                "threshold": float(protocol["threshold"]),
                "runtime_ms": dpu_runtime_ms,
                **{k: v for k, v in metrics.items() if isinstance(v, (int, float))},
            }
        )

    summary_rows = summarize(out_rows)
    payload = {
        "source": {
            "dataset": "Carinthia-S",
            "csv": str(csv_path),
            "root": str(root),
            "sample_count": len(rows),
            "max_per_class": args.max_per_class,
            "label_names": LABEL_NAMES,
            "mask_available": True,
            "paired_hr_lr_available": "pseudo-LR generated by downsampling real SEM images",
            "interpretation": "external mask-level stress using frozen synthetic thresholds; not retrained on Carinthia-S",
        },
        "protocol": {
            "dpu_protocol": protocol,
            "detector_protocol": detector_protocol,
            "baseline_thresholds": thresholds,
            "hr_size": args.hr_size,
            "device": str(device),
        },
        "summary": summary_rows,
        "rows": out_rows,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_tables(summary_rows, Path(args.output_md), Path(args.output_csv))
    print(f"wrote {len(rows)} masked external samples and {len(out_rows)} method rows to {args.output_json}")


if __name__ == "__main__":
    main()
