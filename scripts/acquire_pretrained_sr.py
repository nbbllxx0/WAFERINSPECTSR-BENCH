from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse


DIRECT_EXTENSIONS = {
    ".ckpt",
    ".onnx",
    ".pth",
    ".pt",
    ".safetensors",
    ".tar",
    ".torchscript",
    ".zip",
}

MANUAL_MODES = {
    "archive_extract",
    "manual_export",
    "scale_mismatch_export",
    "source_only",
    "torchscript_export",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Acquire or audit heavyweight pretrained SR source artifacts.")
    parser.add_argument("--manifest", default="configs/pretrained_sr_manifest.example.json")
    parser.add_argument("--root", default=".")
    parser.add_argument("--sources-md", default="paper/tables/pretrained_sr_sources.md")
    parser.add_argument("--sources-csv", default="paper/tables/pretrained_sr_sources.csv")
    parser.add_argument("--download", action="store_true", help="Download direct source URLs declared by the manifest.")
    parser.add_argument("--only", action="append", default=None, help="Limit to a method name. May be passed repeatedly.")
    parser.add_argument("--force", action="store_true", help="Replace existing downloaded source artifacts.")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-bytes", type=int, default=0, help="Abort any one download above this size; 0 means unlimited.")
    return parser.parse_args()


def load_entries(path: Path) -> list[dict[str, object]]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    entries = doc.get("baselines", doc) if isinstance(doc, dict) else doc
    if not isinstance(entries, list):
        raise ValueError("Pretrained SR manifest must be a list or contain a 'baselines' list.")
    out = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Each pretrained SR manifest entry must be an object.")
        out.append(entry)
    return out


def resolve(root: Path, raw: object | None) -> Path | None:
    if raw is None:
        return None
    path = Path(str(raw))
    return path if path.is_absolute() else root / path


def is_direct_url(url: str | None) -> bool:
    if not url:
        return False
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    suffix = Path(parsed.path).suffix.lower()
    if suffix in DIRECT_EXTENSIONS:
        return True
    return "releases/download" in parsed.path


def default_download_path(root: Path, entry: dict[str, object], checkpoint_path: Path) -> Path | None:
    source_url = str(entry.get("source_url", "") or "")
    mode = str(entry.get("acquisition_mode", "manual_export"))
    declared = entry.get("download_path")
    if declared:
        return resolve(root, declared)
    if mode == "direct_checkpoint":
        return checkpoint_path
    if not is_direct_url(source_url):
        return None
    filename = Path(urlparse(source_url).path).name
    if not filename:
        filename = f"{entry['name']}.source"
    return root / "models" / "pretrained_sr" / "sources" / filename


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path, timeout: float, max_bytes: int, force: bool) -> tuple[str, int | None]:
    if destination.exists() and not force:
        return "already_present", destination.stat().st_size
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.name + ".download")
    if tmp.exists():
        tmp.unlink()
    request = urllib.request.Request(url, headers={"User-Agent": "WaferInspectSR-Bench/1.0"})
    total = 0
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            length = response.headers.get("Content-Length")
            if length and max_bytes and int(length) > max_bytes:
                raise RuntimeError(f"remote size {length} exceeds --max-bytes {max_bytes}")
            with tmp.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if max_bytes and total > max_bytes:
                        raise RuntimeError(f"download exceeded --max-bytes {max_bytes}")
                    handle.write(chunk)
        shutil.move(str(tmp), str(destination))
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    return "downloaded", total


def status_for_entry(
    root: Path,
    entry: dict[str, object],
    do_download: bool,
    timeout: float,
    max_bytes: int,
    force: bool,
) -> dict[str, object]:
    name = str(entry["name"])
    mode = str(entry.get("acquisition_mode", "manual_export"))
    checkpoint = resolve(root, entry["checkpoint"])
    if checkpoint is None:
        raise ValueError(f"{name} has no checkpoint path.")
    source_url = str(entry.get("source_url", "") or "")
    source_note = str(entry.get("source_note", "") or "")
    download_path = default_download_path(root, entry, checkpoint)
    direct = is_direct_url(source_url)
    download_result = ""
    error = ""

    if do_download and direct and download_path is not None:
        try:
            download_result, _ = download(source_url, download_path, timeout, max_bytes, force)
        except (RuntimeError, urllib.error.URLError, OSError) as exc:
            error = str(exc)

    checkpoint_exists = checkpoint.exists()
    source_exists = bool(download_path and download_path.exists())
    if checkpoint_exists:
        status = "checkpoint_present"
    elif error:
        status = "download_failed"
    elif source_exists and mode != "direct_checkpoint":
        status = "source_present_conversion_required"
    elif do_download and direct and download_path is not None and mode == "direct_checkpoint":
        status = "checkpoint_missing_after_download"
    elif direct:
        status = "downloadable_not_present"
    elif mode in MANUAL_MODES:
        status = "manual_source_required"
    else:
        status = "source_missing"

    return {
        "Method": name,
        "Family": str(entry.get("family", name)),
        "Architecture": str(entry.get("architecture", "")),
        "Acquisition mode": mode,
        "Checkpoint": checkpoint.relative_to(root).as_posix() if checkpoint.is_relative_to(root) else str(checkpoint),
        "Checkpoint present": checkpoint_exists,
        "Checkpoint bytes": checkpoint.stat().st_size if checkpoint_exists else "",
        "Checkpoint SHA256": sha256(checkpoint) if checkpoint_exists else "",
        "Source artifact": (
            download_path.relative_to(root).as_posix()
            if download_path is not None and download_path.is_relative_to(root)
            else (str(download_path) if download_path is not None else "")
        ),
        "Source present": source_exists,
        "Source bytes": download_path.stat().st_size if source_exists and download_path is not None else "",
        "Source SHA256": sha256(download_path) if source_exists and download_path is not None else "",
        "Source URL": source_url,
        "Source note": source_note,
        "Download result": download_result,
        "Status": status,
        "Error": error,
    }


def escape_md(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_tables(rows: list[dict[str, object]], md_path: Path, csv_path: Path) -> None:
    md_headers = [
        "Method",
        "Family",
        "Acquisition mode",
        "Checkpoint present",
        "Source present",
        "Status",
        "Source URL",
        "Source note",
    ]
    md_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| " + " | ".join(md_headers) + " |",
        "| " + " | ".join(["---"] * len(md_headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(escape_md(row[header]) for header in md_headers) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    manifest = resolve(root, args.manifest)
    if manifest is None:
        raise ValueError("manifest path is required")
    entries = load_entries(manifest)
    only = set(args.only or [])
    if only:
        entries = [entry for entry in entries if str(entry["name"]) in only]
        missing = only.difference(str(entry["name"]) for entry in entries)
        if missing:
            raise ValueError(f"Unknown --only method(s): {sorted(missing)}")
    rows = [
        status_for_entry(root, entry, args.download, args.timeout, args.max_bytes, args.force)
        for entry in entries
    ]
    if not rows:
        raise ValueError("No pretrained SR entries selected.")
    write_tables(rows, resolve(root, args.sources_md) or Path(args.sources_md), resolve(root, args.sources_csv) or Path(args.sources_csv))
    print(f"wrote {args.sources_md}")
    print(f"wrote {args.sources_csv}")
    for row in rows:
        print(f"{row['Method']}: {row['Status']}")
    if any(row["Status"] == "download_failed" for row in rows):
        sys.exit(2)


if __name__ == "__main__":
    main()
