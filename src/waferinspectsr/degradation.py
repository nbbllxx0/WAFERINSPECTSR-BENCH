"""Controlled wafer-inspection degradation pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from waferinspectsr.image_ops import as_float_image, gaussian_filter, resize_image, uniform_filter


@dataclass(frozen=True)
class DegradationConfig:
    """Parameters for deterministic LR generation."""

    scale: int = 2
    blur_sigma: float = 1.0
    defocus_sigma: float = 0.5
    shot_noise: float = 0.02
    gaussian_noise: float = 0.015
    poisson_peak: float = 60.0
    scanline_strength: float = 0.025
    contrast_low: float = 0.9
    contrast_high: float = 1.1
    aliasing: float = 0.015


def _as_float_image(image: np.ndarray) -> np.ndarray:
    return as_float_image(image)


def bicubic_resize(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize a 2D image with anti-aliased cubic interpolation."""
    return resize_image(_as_float_image(image), shape, bicubic=True)


def degrade(image: np.ndarray, config: DegradationConfig, seed: int) -> np.ndarray:
    """Generate a low-resolution observation from a high-resolution image."""
    rng = np.random.default_rng(seed)
    hr = _as_float_image(image)

    degraded = gaussian_filter(hr, sigma=max(config.blur_sigma, 0.0))
    if config.defocus_sigma > 0:
        degraded = uniform_filter(degraded, size=max(1, int(round(config.defocus_sigma * 3))))

    contrast = rng.uniform(config.contrast_low, config.contrast_high)
    degraded = np.clip((degraded - 0.5) * contrast + 0.5, 0.0, 1.0)

    if config.aliasing > 0:
        yy, xx = np.mgrid[: hr.shape[0], : hr.shape[1]]
        phase = rng.uniform(0, 2 * np.pi)
        degraded += config.aliasing * np.sin(2 * np.pi * xx / max(6, config.scale * 4) + phase)

    if config.shot_noise > 0:
        degraded += rng.normal(0.0, config.shot_noise * np.sqrt(np.maximum(degraded, 1e-4)), degraded.shape)

    if config.poisson_peak > 0:
        degraded = rng.poisson(np.clip(degraded, 0.0, 1.0) * config.poisson_peak) / config.poisson_peak

    if config.gaussian_noise > 0:
        degraded += rng.normal(0.0, config.gaussian_noise, degraded.shape)

    if config.scanline_strength > 0:
        line_offsets = rng.normal(0.0, config.scanline_strength, size=(hr.shape[0], 1))
        degraded += line_offsets

    degraded = np.clip(degraded, 0.0, 1.0)
    lr_shape = (hr.shape[0] // config.scale, hr.shape[1] // config.scale)
    if min(lr_shape) < 1:
        raise ValueError(f"Scale {config.scale} is too large for image shape {hr.shape}")
    return resize_image(degraded, lr_shape, bicubic=False)


def upsample_lr(lr: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Bicubic-upsample an LR observation to HR shape."""
    return bicubic_resize(lr, target_shape)
