"""Small NumPy/Pillow image operations.

This module avoids compiled SciPy/scikit-image dependencies so the benchmark
can run in mismatched local Python environments.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def as_float_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale image, got shape {arr.shape}")
    return np.clip(arr, 0.0, 1.0)


def gaussian_kernel1d(sigma: float) -> np.ndarray:
    if sigma <= 0:
        return np.asarray([1.0], dtype=np.float32)
    radius = max(1, int(round(3 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel.astype(np.float32)


def convolve_axis(image: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    radius = len(kernel) // 2
    pad_width = [(0, 0), (0, 0)]
    pad_width[axis] = (radius, radius)
    padded = np.pad(arr, pad_width, mode="reflect")
    out = np.zeros_like(arr, dtype=np.float32)
    for offset, weight in enumerate(kernel):
        start = offset
        end = start + arr.shape[axis]
        if axis == 0:
            out += weight * padded[start:end, :]
        else:
            out += weight * padded[:, start:end]
    return out


def gaussian_filter(image: np.ndarray, sigma: float) -> np.ndarray:
    arr = as_float_image(image)
    kernel = gaussian_kernel1d(float(sigma))
    return convolve_axis(convolve_axis(arr, kernel, axis=0), kernel, axis=1)


def uniform_filter(image: np.ndarray, size: int) -> np.ndarray:
    arr = as_float_image(image)
    size = max(1, int(size))
    kernel = np.full(size, 1.0 / size, dtype=np.float32)
    return convolve_axis(convolve_axis(arr, kernel, axis=0), kernel, axis=1)


def resize_image(image: np.ndarray, shape: tuple[int, int], bicubic: bool = True, mode: str | None = None) -> np.ndarray:
    arr = as_float_image(image)
    pil = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
    if mode is None:
        mode = "bicubic" if bicubic else "bilinear"
    resampling = {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
    }
    try:
        resample = resampling[mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported resize mode: {mode}") from exc
    resized = pil.resize((int(shape[1]), int(shape[0])), resample=resample)
    return (np.asarray(resized, dtype=np.float32) / 255.0).astype(np.float32)


def _shift_bool(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    padded = np.pad(mask.astype(bool), ((1, 1), (1, 1)), mode="constant", constant_values=False)
    y0 = 1 + dy
    x0 = 1 + dx
    return padded[y0 : y0 + mask.shape[0], x0 : x0 + mask.shape[1]]


def binary_dilation(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    out = np.asarray(mask, dtype=bool)
    for _ in range(max(0, int(iterations))):
        neighbors = [_shift_bool(out, dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
        out = np.logical_or.reduce(neighbors)
    return out


def binary_erosion(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    out = np.asarray(mask, dtype=bool)
    for _ in range(max(0, int(iterations))):
        neighbors = [_shift_bool(out, dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
        out = np.logical_and.reduce(neighbors)
    return out


def gaussian_gradient_magnitude(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    smoothed = gaussian_filter(image, sigma=sigma)
    gy, gx = np.gradient(smoothed)
    return np.sqrt(gx**2 + gy**2).astype(np.float32)
