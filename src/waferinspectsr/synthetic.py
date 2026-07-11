"""Synthetic wafer-pattern benchmark generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from waferinspectsr.image_ops import binary_dilation, binary_erosion, gaussian_filter


@dataclass(frozen=True)
class SyntheticSample:
    """One high-resolution synthetic inspection sample."""

    image: np.ndarray
    defect_mask: np.ndarray
    clean_mask: np.ndarray
    edge_mask: np.ndarray
    pattern_type: str
    defect_type: str
    split: str
    severity: str
    seed: int


_PITCHES = {
    "train": [16, 20, 24],
    "val": [18, 22, 26],
    "val_calib": [18, 22, 26],
    "test": [28, 32, 36],
    "clean": [16, 22, 28],
    "clean_calib": [16, 22, 28],
    "clean_test": [16, 22, 28],
    "weak": [18, 24, 32],
    "weak_test": [18, 24, 32],
    "ood": [40, 44],
    "ood_calib_optional": [40, 44],
    "ood_test": [40, 44],
}

_HOLE_PITCHES = {
    "train": [18, 22, 26],
    "val": [20, 24, 28],
    "val_calib": [20, 24, 28],
    "test": [30, 34, 38],
    "clean": [18, 26, 34],
    "clean_calib": [18, 26, 34],
    "clean_test": [18, 26, 34],
    "weak": [20, 28, 36],
    "weak_test": [20, 28, 36],
    "ood": [42, 46],
    "ood_calib_optional": [42, 46],
    "ood_test": [42, 46],
}


def _rng(seed: int | np.random.Generator) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def _line_space_pattern(height: int, width: int, rng: np.random.Generator, split: str) -> np.ndarray:
    yy, xx = np.mgrid[:height, :width]
    pitch = int(rng.choice(_PITCHES.get(split, _PITCHES["train"])))
    duty = rng.uniform(0.35, 0.55)
    phase = int(rng.integers(0, pitch))
    lines = ((xx + phase) % pitch) < int(max(2, pitch * duty))
    base = np.where(lines, 0.72, 0.26).astype(np.float32)
    base += 0.04 * np.sin(2 * np.pi * yy / max(height, 1)).astype(np.float32)
    return np.clip(base, 0.0, 1.0)


def _contact_hole_pattern(height: int, width: int, rng: np.random.Generator, split: str) -> np.ndarray:
    image = np.full((height, width), 0.68, dtype=np.float32)
    pitch = int(rng.choice(_HOLE_PITCHES.get(split, _HOLE_PITCHES["train"])))
    radius = int(max(3, pitch * rng.uniform(0.18, 0.26)))
    y_offset = int(rng.integers(0, pitch))
    x_offset = int(rng.integers(0, pitch))
    yy, xx = np.mgrid[:height, :width]
    for y in range(y_offset, height, pitch):
        for x in range(x_offset, width, pitch):
            dist = (yy - y) ** 2 + (xx - x) ** 2
            hole = dist <= radius**2
            image[hole] = 0.22
    image = gaussian_filter(image, sigma=0.6)
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def _draw_bridge(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> None:
    height, width = image.shape
    y = int(rng.integers(height // 5, 4 * height // 5))
    x = int(rng.integers(width // 5, 4 * width // 5))
    length = int(rng.integers(width // 12, width // 5))
    thickness = int(rng.integers(3, 7))
    x0, x1 = max(0, x - length // 2), min(width, x + length // 2)
    y0, y1 = max(0, y - thickness // 2), min(height, y + thickness // 2 + 1)
    image[y0:y1, x0:x1] = 0.78
    mask[y0:y1, x0:x1] = True


def _draw_gap(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> None:
    height, width = image.shape
    y = int(rng.integers(height // 5, 4 * height // 5))
    x = int(rng.integers(width // 5, 4 * width // 5))
    size = int(rng.integers(8, 18))
    y0, y1 = max(0, y - size // 2), min(height, y + size // 2)
    x0, x1 = max(0, x - size // 2), min(width, x + size // 2)
    image[y0:y1, x0:x1] = 0.18
    mask[y0:y1, x0:x1] = True


def _draw_particle(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> None:
    height, width = image.shape
    yy, xx = np.mgrid[:height, :width]
    y = int(rng.integers(height // 8, 7 * height // 8))
    x = int(rng.integers(width // 8, 7 * width // 8))
    radius = int(rng.integers(4, 11))
    particle = (yy - y) ** 2 + (xx - x) ** 2 <= radius**2
    image[particle] = rng.uniform(0.82, 0.95)
    mask[particle] = True


def _draw_scratch(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> None:
    height, width = image.shape
    y0 = int(rng.integers(height // 6, 5 * height // 6))
    x0 = int(rng.integers(width // 8, width // 3))
    length = int(rng.integers(width // 4, width // 2))
    slope = rng.uniform(-0.4, 0.4)
    thickness = int(rng.integers(2, 5))
    yy, xx = np.mgrid[:height, :width]
    x1 = min(width - 1, x0 + length)
    y_line = y0 + slope * (xx - x0)
    scratch = (xx >= x0) & (xx <= x1) & (np.abs(yy - y_line) <= thickness)
    image[scratch] = 0.08
    mask[scratch] = True


def _draw_missing_pattern(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> None:
    height, width = image.shape
    y = int(rng.integers(height // 4, 3 * height // 4))
    x = int(rng.integers(width // 4, 3 * width // 4))
    h = int(rng.integers(10, 22))
    w = int(rng.integers(10, 28))
    y0, y1 = max(0, y - h // 2), min(height, y + h // 2)
    x0, x1 = max(0, x - w // 2), min(width, x + w // 2)
    fill = float(np.median(image[max(0, y0 - 5) : min(height, y1 + 5), max(0, x0 - 5) : min(width, x1 + 5)]))
    image[y0:y1, x0:x1] = fill
    mask[y0:y1, x0:x1] = True


def _draw_residue(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> None:
    """OOD thin residue shape not used in nominal training."""
    height, width = image.shape
    yy, xx = np.mgrid[:height, :width]
    y = int(rng.integers(height // 5, 4 * height // 5))
    x = int(rng.integers(width // 5, 4 * width // 5))
    radius_y = int(rng.integers(3, 7))
    radius_x = int(rng.integers(18, 34))
    angle = rng.uniform(-0.8, 0.8)
    x_rot = (xx - x) * np.cos(angle) + (yy - y) * np.sin(angle)
    y_rot = -(xx - x) * np.sin(angle) + (yy - y) * np.cos(angle)
    residue = (x_rot / radius_x) ** 2 + (y_rot / radius_y) ** 2 <= 1.0
    image[residue] = 0.88
    mask[residue] = True


_DEFECT_DRAWERS = {
    "bridge": _draw_bridge,
    "gap": _draw_gap,
    "particle": _draw_particle,
    "scratch": _draw_scratch,
    "missing_pattern": _draw_missing_pattern,
    "residue": _draw_residue,
}


def mask_edges(mask: np.ndarray) -> np.ndarray:
    """Return a one-pixel binary boundary mask."""
    binary = mask.astype(bool)
    eroded = binary_erosion(binary, iterations=1)
    return binary & ~eroded


def generate_sample(
    height: int,
    width: int,
    pattern_type: str,
    defect_type: str,
    seed: int | np.random.Generator,
    split: str = "train",
    severity: str = "nominal",
) -> SyntheticSample:
    """Generate one synthetic HR sample with exact masks."""
    rng = _rng(seed)
    if pattern_type == "line_space":
        image = _line_space_pattern(height, width, rng, split)
    elif pattern_type == "contact_hole":
        image = _contact_hole_pattern(height, width, rng, split)
    else:
        raise ValueError(f"Unknown pattern_type: {pattern_type}")

    defect_mask = np.zeros((height, width), dtype=bool)
    if defect_type not in {"none", "clean"}:
        before_defect = image.copy()
        try:
            drawer = _DEFECT_DRAWERS[defect_type]
        except KeyError as exc:
            raise ValueError(f"Unknown defect_type: {defect_type}") from exc
        drawer(image, defect_mask, rng)
        if severity == "weak":
            # Low-contrast borderline defects support missed-defect evaluation.
            image[defect_mask] = 0.72 * before_defect[defect_mask] + 0.28 * image[defect_mask]
        elif severity == "ood":
            image[defect_mask] = np.clip(0.55 * before_defect[defect_mask] + 0.45 * image[defect_mask], 0.0, 1.0)

    image = gaussian_filter(image, sigma=0.25).astype(np.float32)
    image = np.clip(image + rng.normal(0.0, 0.01, image.shape).astype(np.float32), 0.0, 1.0)
    clean_mask = ~binary_dilation(defect_mask, iterations=3)
    edge_mask = mask_edges(defect_mask)
    sample_seed = int(seed) if not isinstance(seed, np.random.Generator) else -1
    return SyntheticSample(image, defect_mask, clean_mask, edge_mask, pattern_type, defect_type, split, severity, sample_seed)


def generate_dataset(
    num_samples: int,
    height: int,
    width: int,
    pattern_types: Iterable[str],
    defect_types: Iterable[str],
    seed: int,
    split: str = "train",
    severity: str = "nominal",
) -> list[SyntheticSample]:
    """Generate a deterministic synthetic dataset."""
    rng = np.random.default_rng(seed)
    patterns = list(pattern_types)
    defects = list(defect_types)
    if not patterns or not defects:
        raise ValueError("pattern_types and defect_types must be non-empty")
    samples: list[SyntheticSample] = []
    for index in range(num_samples):
        sample_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        pattern = patterns[index % len(patterns)]
        defect = defects[(index // len(patterns)) % len(defects)]
        samples.append(generate_sample(height, width, pattern, defect, sample_seed, split=split, severity=severity))
    return samples


def generate_benchmark_splits(
    split_counts: dict[str, int],
    height: int,
    width: int,
    pattern_types: Iterable[str],
    defect_types: Iterable[str],
    seed: int,
) -> list[SyntheticSample]:
    """Generate clean, weak, train/val/test, and OOD parameter-disjoint splits."""
    all_samples: list[SyntheticSample] = []
    rng = np.random.default_rng(seed)
    nominal_defects = [d for d in defect_types if d not in {"none", "clean", "residue"}]
    for split, count in split_counts.items():
        split_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        if split in {"clean", "clean_calib", "clean_test"}:
            defects = ["none"]
            severity = "clean"
        elif split in {"weak", "weak_test"}:
            defects = nominal_defects
            severity = "weak"
        elif split in {"ood", "ood_calib_optional", "ood_test"}:
            defects = ["residue"]
            severity = "ood"
        else:
            defects = nominal_defects
            severity = "nominal"
        all_samples.extend(
            generate_dataset(
                count,
                height,
                width,
                pattern_types,
                defects,
                split_seed,
                split=split,
                severity=severity,
            )
        )
    return all_samples
