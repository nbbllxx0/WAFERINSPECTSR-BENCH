"""Lightweight v1 baselines for inspection-safe SR evaluation."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from waferinspectsr.degradation import upsample_lr
from waferinspectsr.image_ops import gaussian_filter, gaussian_gradient_magnitude, resize_image


@dataclass(frozen=True)
class BaselinePrediction:
    name: str
    sr_image: np.ndarray
    defect_prob: np.ndarray
    risk: np.ndarray
    runtime_ms: float


def heuristic_defect_probability(image: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """Estimate defect response from local residual contrast."""
    arr = np.asarray(image, dtype=np.float32)
    smooth = gaussian_filter(arr, sigma=sigma)
    residual = np.abs(arr - smooth)
    scale = np.percentile(residual, 98) + 1e-6
    prob = np.clip(residual / scale, 0.0, 1.0)
    return gaussian_filter(prob, sigma=0.6).astype(np.float32)


def uncertainty_from_probability(prob: np.ndarray) -> np.ndarray:
    """High risk near uncertain probability values and strong local variation."""
    p = np.clip(np.asarray(prob, dtype=np.float32), 0.0, 1.0)
    entropy_like = 1.0 - 2.0 * np.abs(p - 0.5)
    texture = gaussian_gradient_magnitude(p, sigma=1.0)
    if texture.max() > 0:
        texture = texture / texture.max()
    return np.clip(0.7 * entropy_like + 0.3 * texture, 0.0, 1.0).astype(np.float32)


def lr_detector(lr: np.ndarray, target_shape: tuple[int, int]) -> BaselinePrediction:
    start = time.perf_counter()
    sr = upsample_lr(lr, target_shape)
    prob = heuristic_defect_probability(sr, sigma=3.5)
    risk = uncertainty_from_probability(prob)
    runtime_ms = (time.perf_counter() - start) * 1000.0
    return BaselinePrediction("lr_detector", sr, prob, risk, runtime_ms)


def no_sr_task_detector(lr: np.ndarray, target_shape: tuple[int, int]) -> BaselinePrediction:
    """Task-only baseline: detect at LR, then upsample the probability map."""
    start = time.perf_counter()
    lr_prob = heuristic_defect_probability(lr, sigma=1.6)
    prob = upsample_lr(lr_prob, target_shape)
    sr = upsample_lr(lr, target_shape)
    risk = uncertainty_from_probability(prob)
    runtime_ms = (time.perf_counter() - start) * 1000.0
    return BaselinePrediction("no_sr_task_detector", sr, prob, risk, runtime_ms)


def nearest_detector(lr: np.ndarray, target_shape: tuple[int, int]) -> BaselinePrediction:
    start = time.perf_counter()
    sr = resize_image(lr, target_shape, mode="nearest")
    prob = heuristic_defect_probability(sr, sigma=2.8)
    risk = uncertainty_from_probability(prob)
    runtime_ms = (time.perf_counter() - start) * 1000.0
    return BaselinePrediction("nearest_detector", sr, prob, risk, runtime_ms)


def bilinear_detector(lr: np.ndarray, target_shape: tuple[int, int]) -> BaselinePrediction:
    start = time.perf_counter()
    sr = resize_image(lr, target_shape, mode="bilinear")
    prob = heuristic_defect_probability(sr, sigma=2.7)
    risk = uncertainty_from_probability(prob)
    runtime_ms = (time.perf_counter() - start) * 1000.0
    return BaselinePrediction("bilinear_detector", sr, prob, risk, runtime_ms)


def bicubic_detector(lr: np.ndarray, target_shape: tuple[int, int]) -> BaselinePrediction:
    start = time.perf_counter()
    sr = upsample_lr(lr, target_shape)
    prob = heuristic_defect_probability(sr, sigma=2.5)
    risk = uncertainty_from_probability(prob)
    runtime_ms = (time.perf_counter() - start) * 1000.0
    return BaselinePrediction("bicubic_detector", sr, prob, risk, runtime_ms)


def lanczos_detector(lr: np.ndarray, target_shape: tuple[int, int]) -> BaselinePrediction:
    start = time.perf_counter()
    sr = resize_image(lr, target_shape, mode="lanczos")
    prob = heuristic_defect_probability(sr, sigma=2.4)
    risk = uncertainty_from_probability(prob)
    runtime_ms = (time.perf_counter() - start) * 1000.0
    return BaselinePrediction("lanczos_detector", sr, prob, risk, runtime_ms)


def denoise_upsample_detector(lr: np.ndarray, target_shape: tuple[int, int]) -> BaselinePrediction:
    """Classical denoise-then-upsample baseline."""
    start = time.perf_counter()
    denoised_lr = gaussian_filter(lr, sigma=0.55)
    sr = resize_image(denoised_lr, target_shape, mode="lanczos")
    prob = heuristic_defect_probability(sr, sigma=2.5)
    risk = uncertainty_from_probability(prob)
    runtime_ms = (time.perf_counter() - start) * 1000.0
    return BaselinePrediction("denoise_upsample_detector", sr, prob, risk, runtime_ms)


def _wiener_sharpen(image: np.ndarray, sigma: float = 1.0, balance: float = 0.025) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    h, w = arr.shape
    yy, xx = np.mgrid[:h, :w]
    yy = yy - h // 2
    xx = xx - w // 2
    psf = np.exp(-(xx**2 + yy**2) / (2.0 * sigma * sigma)).astype(np.float32)
    psf /= psf.sum()
    psf = np.fft.ifftshift(psf)
    spectrum = np.fft.fft2(arr)
    transfer = np.fft.fft2(psf)
    restored = np.fft.ifft2(spectrum * np.conj(transfer) / (np.abs(transfer) ** 2 + balance)).real
    low, high = np.percentile(restored, [0.5, 99.5])
    if high > low:
        restored = (restored - low) / (high - low)
    return np.clip(restored, 0.0, 1.0).astype(np.float32)


def wiener_deconv_detector(lr: np.ndarray, target_shape: tuple[int, int]) -> BaselinePrediction:
    """Classical Wiener-style deconvolution baseline after interpolation."""
    start = time.perf_counter()
    bicubic = upsample_lr(lr, target_shape)
    sr = _wiener_sharpen(bicubic, sigma=1.0, balance=0.03)
    prob = heuristic_defect_probability(sr, sigma=2.1)
    risk = uncertainty_from_probability(prob)
    runtime_ms = (time.perf_counter() - start) * 1000.0
    return BaselinePrediction("wiener_deconv_detector", sr, prob, risk, runtime_ms)


def prior_only_detector(lr: np.ndarray, target_shape: tuple[int, int]) -> BaselinePrediction:
    """Detector that exposes only the local-contrast prior as its probability map."""
    start = time.perf_counter()
    sr = upsample_lr(lr, target_shape)
    prob = heuristic_defect_probability(sr, sigma=2.0)
    risk = uncertainty_from_probability(prob)
    runtime_ms = (time.perf_counter() - start) * 1000.0
    return BaselinePrediction("prior_only_detector", sr, prob, risk, runtime_ms)


def sharpened_sr_detector(lr: np.ndarray, target_shape: tuple[int, int]) -> BaselinePrediction:
    """A generic sharpness-oriented SR proxy used as a v1 failure contrast."""
    start = time.perf_counter()
    bicubic = upsample_lr(lr, target_shape)
    blur = gaussian_filter(bicubic, sigma=1.0)
    sr = np.clip(bicubic + 1.2 * (bicubic - blur), 0.0, 1.0).astype(np.float32)
    prob = heuristic_defect_probability(sr, sigma=2.0)
    risk = uncertainty_from_probability(prob)
    runtime_ms = (time.perf_counter() - start) * 1000.0
    return BaselinePrediction("sharpened_sr_detector", sr, prob, risk, runtime_ms)


def naf_style_sr_detector(lr: np.ndarray, target_shape: tuple[int, int]) -> BaselinePrediction:
    """Activation-free SR-style proxy inspired by gated NAFNet restoration blocks."""
    start = time.perf_counter()
    bicubic = upsample_lr(lr, target_shape)
    low = gaussian_filter(bicubic, sigma=1.4)
    detail = bicubic - low
    gate = gaussian_gradient_magnitude(bicubic, sigma=1.0)
    if gate.max() > 0:
        gate = gate / gate.max()
    sr = np.clip(bicubic + 0.9 * detail * (0.5 + 0.5 * gate), 0.0, 1.0).astype(np.float32)
    prob = heuristic_defect_probability(sr, sigma=2.1)
    risk = uncertainty_from_probability(prob)
    runtime_ms = (time.perf_counter() - start) * 1000.0
    return BaselinePrediction("naf_style_sr_detector", sr, prob, risk, runtime_ms)


def hr_detector_reference(hr: np.ndarray) -> BaselinePrediction:
    """Diagnostic heuristic detector on the undegraded HR image."""
    start = time.perf_counter()
    sr = np.asarray(hr, dtype=np.float32)
    prob = heuristic_defect_probability(sr, sigma=2.0)
    risk = uncertainty_from_probability(prob)
    runtime_ms = (time.perf_counter() - start) * 1000.0
    return BaselinePrediction("hr_detector_reference", sr, prob, risk, runtime_ms)


def run_v1_baselines(
    lr: np.ndarray,
    target_shape: tuple[int, int],
    hr: np.ndarray | None = None,
    include_oracle: bool = False,
) -> list[BaselinePrediction]:
    predictions = [
        no_sr_task_detector(lr, target_shape),
        lr_detector(lr, target_shape),
        nearest_detector(lr, target_shape),
        bilinear_detector(lr, target_shape),
        bicubic_detector(lr, target_shape),
        lanczos_detector(lr, target_shape),
        denoise_upsample_detector(lr, target_shape),
        wiener_deconv_detector(lr, target_shape),
        prior_only_detector(lr, target_shape),
        sharpened_sr_detector(lr, target_shape),
        naf_style_sr_detector(lr, target_shape),
    ]
    if include_oracle:
        if hr is None:
            raise ValueError("hr must be provided when include_oracle=True")
        predictions.append(hr_detector_reference(hr))
    return predictions
