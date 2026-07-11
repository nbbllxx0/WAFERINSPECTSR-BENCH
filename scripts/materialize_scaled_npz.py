from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waferinspectsr.io import load_npz


ARRAY_KEYS = [
    "hr",
    "lr",
    "defect_mask",
    "clean_mask",
    "edge_mask",
    "pattern_type",
    "defect_type",
    "split",
    "severity",
    "sample_seed",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize selected scaled benchmark shards into one trainable NPZ.")
    parser.add_argument("--manifest", default="data/generated/scaled_seed20260602/manifest.json")
    parser.add_argument("--output", default="experiments/runs/scaled_seed20260602/neural_smoke/data/samples.npz")
    parser.add_argument(
        "--splits",
        default="train,val_calib,test,clean_calib,clean_test,weak_test,ood_test",
        help="Comma-separated splits to include. Use 'all' for every manifest split.",
    )
    parser.add_argument(
        "--max-per-split",
        type=int,
        default=32,
        help="Maximum samples per split. Use 0 or a negative value for all available samples.",
    )
    parser.add_argument("--summary-md", default="paper/tables/scaled_neural_materialization.md")
    parser.add_argument("--summary-csv", default="paper/tables/scaled_neural_materialization.csv")
    return parser.parse_args()


def selected_splits(raw: str, manifest: dict[str, Any]) -> set[str]:
    if raw.strip().lower() == "all":
        return set(str(key) for key in manifest["split_counts"])
    return {item.strip() for item in raw.split(",") if item.strip()}


def take_from_shard(data: dict[str, np.ndarray], start: int, count: int) -> dict[str, np.ndarray]:
    return {key: data[key][start : start + count] for key in ARRAY_KEYS}


def append_chunk(target: dict[str, list[np.ndarray]], chunk: dict[str, np.ndarray]) -> None:
    for key in ARRAY_KEYS:
        target[key].append(chunk[key])


def write_summary(rows: list[dict[str, object]], md_path: Path, csv_path: Path) -> None:
    headers = ["Split", "Samples", "Shards used"]
    md_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row[header]) for header in headers) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent
    wanted = selected_splits(args.splits, manifest)
    max_per_split = int(args.max_per_split)
    unlimited = max_per_split <= 0
    chunks: dict[str, list[np.ndarray]] = defaultdict(list)
    selected_counts: Counter[str] = Counter()
    shards_used: Counter[str] = Counter()

    for shard in manifest["shards"]:
        split = str(shard["split"])
        if split not in wanted:
            continue
        if not unlimited and selected_counts[split] >= max_per_split:
            continue
        shard_path = root / str(shard["path"])
        data = load_npz(shard_path)
        available = int(len(data["hr"]))
        remaining = available if unlimited else max_per_split - selected_counts[split]
        take = min(available, remaining)
        if take <= 0:
            continue
        append_chunk(chunks, take_from_shard(data, 0, take))
        selected_counts[split] += take
        shards_used[split] += 1

    missing = sorted(split for split in wanted if selected_counts[split] == 0)
    if missing:
        raise ValueError(f"No samples selected for splits={missing}")
    if not chunks:
        raise ValueError("No scaled shard samples selected.")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: np.concatenate(value, axis=0) for key, value in chunks.items()}
    arrays["source_manifest"] = np.asarray([manifest_path.as_posix()])
    arrays["materialized_max_per_split"] = np.asarray([max_per_split], dtype=np.int64)
    np.savez_compressed(output, **arrays)

    rows = [
        {"Split": split, "Samples": selected_counts[split], "Shards used": shards_used[split]}
        for split in sorted(selected_counts)
    ]
    write_summary(rows, Path(args.summary_md), Path(args.summary_csv))
    print(f"wrote {output}")
    print(f"total_samples={sum(selected_counts.values())} splits={dict(selected_counts)}")
    print(f"wrote {args.summary_md}")


if __name__ == "__main__":
    main()
