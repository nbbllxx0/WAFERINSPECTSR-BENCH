"""Shared evaluation protocol helpers."""

from __future__ import annotations

import numpy as np

from waferinspectsr.baselines import heuristic_defect_probability


def parse_split_names(value: str | None, default: str) -> set[str]:
    source = value if value is not None else default
    return {item.strip() for item in source.split(",") if item.strip()}


def indices_for_splits(splits: list[str], split_names: set[str]) -> list[int]:
    return [idx for idx, split in enumerate(splits) if split in split_names]


def require_indices(indices: list[int], split_names: set[str], purpose: str) -> None:
    if not indices:
        raise ValueError(f"No samples found for {purpose} splits={sorted(split_names)}")


def local_contrast_prior(sr_images: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Compute the same HR-image local-contrast prior available to all methods."""
    return np.stack([heuristic_defect_probability(sr_image, sigma=sigma) for sr_image in sr_images]).astype(np.float32)


def apply_prior_fusion(
    prob: np.ndarray,
    sr_images: np.ndarray,
    fusion: str = "none",
    weight: float = 0.5,
) -> np.ndarray:
    """Fuse learned probabilities with a local-contrast prior.

    The prior is computed only from each method's reconstructed image. It does
    not use HR targets, masks, defect labels, split metadata, or degradation
    metadata.
    """
    if fusion == "none":
        return np.clip(prob, 0.0, 1.0).astype(np.float32)
    prior = local_contrast_prior(sr_images)
    alpha = float(np.clip(weight, 0.0, 1.0))
    p = np.clip(prob, 1e-6, 1.0)
    q = np.clip(prior, 1e-6, 1.0)
    if fusion == "geometric":
        fused = np.power(p, 1.0 - alpha) * np.power(q, alpha)
    elif fusion == "average":
        fused = (1.0 - alpha) * p + alpha * q
    else:
        raise ValueError(f"Unknown prior fusion mode: {fusion}")
    return np.clip(fused, 0.0, 1.0).astype(np.float32)
