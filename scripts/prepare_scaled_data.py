from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waferinspectsr.config import ensure_dir, load_config
from waferinspectsr.degradation import DegradationConfig
from waferinspectsr.io import save_sample_npz
from waferinspectsr.synthetic import generate_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate sharded scaled synthetic benchmark data.")
    parser.add_argument("--config", default="configs/scaled.yaml")
    parser.add_argument("--shard-size", type=int, default=None)
    parser.add_argument("--summary-md", default="paper/tables/scaled_synthetic_manifest.md")
    parser.add_argument("--summary-csv", default="paper/tables/scaled_synthetic_manifest.csv")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_defects(split: str, defect_types: list[str]) -> tuple[list[str], str]:
    nominal = [name for name in defect_types if name not in {"none", "clean", "residue"}]
    if split in {"clean", "clean_calib", "clean_test"}:
        return ["none"], "clean"
    if split in {"weak", "weak_test"}:
        return nominal, "weak"
    if split in {"ood", "ood_calib_optional", "ood_test"}:
        return ["residue"], "ood"
    return nominal, "nominal"


def write_summary(rows: list[dict[str, object]], md_path: Path, csv_path: Path) -> None:
    md_path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["Split", "Samples", "Shards", "First shard"]
    totals: dict[str, dict[str, object]] = {}
    for row in rows:
        split = str(row["split"])
        entry = totals.setdefault(split, {"Split": split, "Samples": 0, "Shards": 0, "First shard": row["path"]})
        entry["Samples"] = int(entry["Samples"]) + int(row["samples"])
        entry["Shards"] = int(entry["Shards"]) + 1
    out_rows = list(totals.values())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in out_rows:
        lines.append("| " + " | ".join(str(row[header]) for header in headers) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(out_rows)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    output_dir = ensure_dir(cfg["output_dir"])
    shard_dir = ensure_dir(Path(output_dir) / "shards")
    shard_size = int(args.shard_size or cfg.get("shard_size", 128))
    degradation = DegradationConfig(scale=int(cfg["scale"]), **cfg["degradation"])
    pattern_types = list(cfg["pattern_types"])
    defect_types = list(cfg["defect_types"])
    split_counts = {str(key): int(value) for key, value in cfg["split_counts"].items()}
    rows: list[dict[str, object]] = []
    seed = int(cfg["seed"])
    split_seed_base = seed * 1000003
    for split_index, (split, count) in enumerate(split_counts.items()):
        defects, severity = split_defects(split, defect_types)
        remaining = count
        shard_index = 0
        while remaining > 0:
            this_count = min(shard_size, remaining)
            shard_seed = split_seed_base + split_index * 10007 + shard_index
            samples = generate_dataset(
                num_samples=this_count,
                height=int(cfg["height"]),
                width=int(cfg["width"]),
                pattern_types=pattern_types,
                defect_types=defects,
                seed=shard_seed,
                split=split,
                severity=severity,
            )
            shard_path = shard_dir / f"{split}_{shard_index:04d}.npz"
            save_sample_npz(samples, shard_path, degradation, seed=shard_seed)
            rel_path = shard_path.relative_to(Path(output_dir)).as_posix()
            rows.append(
                {
                    "path": rel_path,
                    "split": split,
                    "samples": this_count,
                    "shard_index": shard_index,
                    "seed": shard_seed,
                    "bytes": shard_path.stat().st_size,
                    "sha256": sha256(shard_path),
                }
            )
            remaining -= this_count
            shard_index += 1
    manifest = {
        "schema_version": 1,
        "config": args.config,
        "seed": seed,
        "height": int(cfg["height"]),
        "width": int(cfg["width"]),
        "scale": int(cfg["scale"]),
        "total_samples": sum(split_counts.values()),
        "split_counts": split_counts,
        "shard_size": shard_size,
        "shards": rows,
    }
    manifest_path = Path(output_dir) / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_summary(rows, Path(args.summary_md), Path(args.summary_csv))
    print(f"wrote {manifest_path}")
    print(f"wrote {args.summary_md}")
    print(f"total_samples={manifest['total_samples']} shards={len(rows)}")


if __name__ == "__main__":
    main()
