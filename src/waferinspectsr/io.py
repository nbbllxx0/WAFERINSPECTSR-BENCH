"""Dataset serialization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from waferinspectsr.degradation import DegradationConfig, degrade
from waferinspectsr.synthetic import SyntheticSample


def save_sample_npz(samples: list[SyntheticSample], path: str | Path, degradation: DegradationConfig, seed: int) -> Path:
    """Save synthetic samples and degraded observations into one NPZ file."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    hr = []
    lr = []
    defect_masks = []
    clean_masks = []
    edge_masks = []
    pattern_types = []
    defect_types = []
    sample_seeds = []
    for index, sample in enumerate(samples):
        hr.append(sample.image)
        lr.append(degrade(sample.image, degradation, seed + index))
        defect_masks.append(sample.defect_mask.astype(np.uint8))
        clean_masks.append(sample.clean_mask.astype(np.uint8))
        edge_masks.append(sample.edge_mask.astype(np.uint8))
        pattern_types.append(sample.pattern_type)
        defect_types.append(sample.defect_type)
        sample_seeds.append(sample.seed)
    splits = [sample.split for sample in samples]
    severities = [sample.severity for sample in samples]
    np.savez_compressed(
        out,
        hr=np.stack(hr).astype(np.float32),
        lr=np.stack(lr).astype(np.float32),
        defect_mask=np.stack(defect_masks).astype(np.uint8),
        clean_mask=np.stack(clean_masks).astype(np.uint8),
        edge_mask=np.stack(edge_masks).astype(np.uint8),
        pattern_type=np.asarray(pattern_types),
        defect_type=np.asarray(defect_types),
        split=np.asarray(splits),
        severity=np.asarray(severities),
        sample_seed=np.asarray(sample_seeds, dtype=np.int64),
        degradation=np.asarray([degradation.__dict__], dtype=object),
    )
    return out


def load_npz(path: str | Path) -> dict[str, Any]:
    """Load an NPZ dataset into memory."""
    with np.load(Path(path), allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def write_json(data: Any, path: str | Path) -> Path:
    """Write JSON-serializable data with stable formatting."""
    import json

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return out


def save_image(image: np.ndarray, path: str | Path) -> Path:
    """Save a float image as an 8-bit PNG."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    Image.fromarray((arr * 255).astype(np.uint8)).save(out)
    return out
