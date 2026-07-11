from __future__ import annotations

import argparse
import json
from pathlib import Path


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a local manifest for Carinthia/Carinthia-S style SEM data.")
    parser.add_argument("--root", required=True, help="Local dataset root. This script does not download data.")
    parser.add_argument("--output", default="data/raw/carinthia_manifest.json")
    parser.add_argument("--mask-dir-name", default="masks")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(root)
    mask_dir = root / args.mask_dir_name
    rows = []
    for image_path in sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTS):
        if args.mask_dir_name in image_path.parts:
            continue
        rel = image_path.relative_to(root)
        candidates = [
            mask_dir / rel,
            image_path.with_name(f"{image_path.stem}_mask{image_path.suffix}"),
            image_path.with_name(f"{image_path.stem}.png"),
        ]
        mask = next((candidate for candidate in candidates if candidate.exists()), None)
        rows.append(
            {
                "image": str(image_path),
                "mask": str(mask) if mask else None,
                "class_hint": image_path.parent.name,
                "has_mask": mask is not None,
            }
        )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"root": str(root), "items": rows}, indent=2), encoding="utf-8")
    print(f"wrote {len(rows)} items to {out}")
    print("No files were copied or downloaded.")


if __name__ == "__main__":
    main()
