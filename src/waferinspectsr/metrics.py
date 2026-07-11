"""Inspection-first metrics for super-resolution benchmark outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from waferinspectsr.image_ops import binary_dilation, binary_erosion


EPS = 1e-8


@dataclass(frozen=True)
class MetricResult:
    """Flat metric result for JSON serialization."""

    name: str
    value: float


def _bool(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr).astype(bool)


def _score(arr: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(arr, dtype=np.float32), 0.0, 1.0)


def logit(prob: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(prob, dtype=np.float32), 1e-5, 1.0 - 1e-5)
    return np.log(p / (1.0 - p)).astype(np.float32)


def apply_temperature(prob: np.ndarray, temperature: float) -> np.ndarray:
    """Calibrate probabilities by dividing logits by temperature."""
    temp = max(float(temperature), 1e-3)
    z = logit(prob) / temp
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


def binary_cross_entropy_np(prob: np.ndarray, target: np.ndarray) -> float:
    p = np.clip(np.asarray(prob, dtype=np.float32), 1e-5, 1.0 - 1e-5)
    y = _bool(target).astype(np.float32)
    return float(np.mean(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))))


def fit_temperature(
    probs: list[np.ndarray],
    targets: list[np.ndarray],
    grid: Iterable[float] = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0),
) -> float:
    """Grid-search a probability temperature on validation BCE."""
    if not probs:
        return 1.0
    best_temp = 1.0
    best_loss = float("inf")
    for temp in grid:
        losses = [binary_cross_entropy_np(apply_temperature(prob, temp), target) for prob, target in zip(probs, targets)]
        loss = float(np.mean(losses))
        if loss < best_loss:
            best_loss = loss
            best_temp = float(temp)
    return best_temp


def threshold_mask(score: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return _score(score) >= threshold


def filter_components(mask: np.ndarray, min_area: int = 1, max_area: int | None = None) -> np.ndarray:
    """Filter binary connected components by area."""
    if min_area <= 1 and max_area is None:
        return _bool(mask)
    out = np.zeros_like(_bool(mask), dtype=bool)
    for component in connected_components(mask):
        area = int(component.sum())
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue
        out |= component
    return out


def threshold_filtered_mask(
    score: np.ndarray,
    threshold: float = 0.5,
    min_component_area: int = 1,
    max_component_area: int | None = None,
) -> np.ndarray:
    return filter_components(threshold_mask(score, threshold), min_component_area, max_component_area)


def defect_recall(
    score: np.ndarray,
    defect_mask: np.ndarray,
    threshold: float = 0.5,
    min_component_area: int = 1,
    max_component_area: int | None = None,
) -> float:
    pred = threshold_filtered_mask(score, threshold, min_component_area, max_component_area)
    gt = _bool(defect_mask)
    return float((pred & gt).sum() / max(int(gt.sum()), 1))


def false_positive_rate(
    score: np.ndarray,
    clean_mask: np.ndarray,
    threshold: float = 0.5,
    min_component_area: int = 1,
    max_component_area: int | None = None,
) -> float:
    pred = threshold_filtered_mask(score, threshold, min_component_area, max_component_area)
    clean = _bool(clean_mask)
    return float((pred & clean).sum() / max(int(clean.sum()), 1))


def no_defect_hallucination_rate(
    score: np.ndarray,
    clean_mask: np.ndarray,
    threshold: float = 0.5,
    min_component_area: int = 1,
    max_component_area: int | None = None,
) -> float:
    """Area-weighted no-defect hallucination rate.

    NHR(tau) = |{p in C : P_defect(p) >= tau}| / |C|, where C is the
    clean-region mask and tau is the operating threshold.
    """
    return false_positive_rate(score, clean_mask, threshold, min_component_area, max_component_area)


def connected_components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 8-connected binary components as boolean masks."""
    binary = _bool(mask)
    seen = np.zeros(binary.shape, dtype=bool)
    components: list[np.ndarray] = []
    height, width = binary.shape
    for y in range(height):
        for x in range(width):
            if not binary[y, x] or seen[y, x]:
                continue
            stack = [(y, x)]
            seen[y, x] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cy, cx = stack.pop()
                coords.append((cy, cx))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < height and 0 <= nx < width and binary[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
            comp = np.zeros(binary.shape, dtype=bool)
            ys, xs = zip(*coords)
            comp[np.asarray(ys), np.asarray(xs)] = True
            components.append(comp)
    return components


def component_hallucination_rate(
    score: np.ndarray,
    clean_mask: np.ndarray,
    threshold: float = 0.5,
    min_area: int = 4,
) -> float:
    """Component-weighted hallucination count normalized per clean image."""
    pred_clean = threshold_mask(score, threshold) & _bool(clean_mask)
    components = connected_components(pred_clean)
    return float(sum(int(component.sum()) >= min_area for component in components))


def component_recall(
    score: np.ndarray,
    defect_mask: np.ndarray,
    threshold: float = 0.5,
    min_area: int = 4,
) -> float:
    """Fraction of ground-truth defect components touched by a prediction."""
    gt_components = [component for component in connected_components(defect_mask) if int(component.sum()) >= min_area]
    if not gt_components:
        return 0.0
    pred = threshold_mask(score, threshold)
    hits = sum(bool((pred & component).any()) for component in gt_components)
    return float(hits / len(gt_components))


def component_iou_recall(
    score: np.ndarray,
    defect_mask: np.ndarray,
    threshold: float = 0.5,
    min_area: int = 4,
    iou_threshold: float = 0.1,
) -> float:
    """Fraction of GT components matched by a predicted component at IoU >= threshold."""
    gt_components = [component for component in connected_components(defect_mask) if int(component.sum()) >= min_area]
    if not gt_components:
        return 0.0
    pred_components = [component for component in connected_components(threshold_mask(score, threshold)) if int(component.sum()) >= min_area]
    hits = 0
    for gt in gt_components:
        matched = False
        for pred in pred_components:
            union = gt | pred
            if not union.any():
                continue
            if float((gt & pred).sum() / union.sum()) >= iou_threshold:
                matched = True
                break
        hits += int(matched)
    return float(hits / len(gt_components))


def component_centroid_recall(
    score: np.ndarray,
    defect_mask: np.ndarray,
    threshold: float = 0.5,
    min_area: int = 4,
    max_distance: float = 8.0,
) -> float:
    """Fraction of GT components with a predicted centroid within max_distance pixels."""
    gt_components = [component for component in connected_components(defect_mask) if int(component.sum()) >= min_area]
    if not gt_components:
        return 0.0
    pred_components = [component for component in connected_components(threshold_mask(score, threshold)) if int(component.sum()) >= min_area]
    if not pred_components:
        return 0.0

    def centroid(component: np.ndarray) -> np.ndarray:
        ys, xs = np.where(component)
        return np.asarray([float(ys.mean()), float(xs.mean())], dtype=np.float64)

    pred_centroids = [centroid(component) for component in pred_components]
    hits = 0
    for gt in gt_components:
        gt_centroid = centroid(gt)
        distances = [float(np.linalg.norm(gt_centroid - pred_centroid)) for pred_centroid in pred_centroids]
        hits += int(min(distances) <= max_distance)
    return float(hits / len(gt_components))


def false_components_per_mpx(
    score: np.ndarray,
    clean_mask: np.ndarray,
    threshold: float = 0.5,
    min_area: int = 4,
) -> float:
    """Clean-region false components normalized by megapixels of clean area."""
    clean = _bool(clean_mask)
    clean_pixels = int(clean.sum())
    if clean_pixels <= 0:
        return 0.0
    pred_clean = threshold_mask(score, threshold) & clean
    components = connected_components(pred_clean)
    count = sum(int(component.sum()) >= min_area for component in components)
    return float(count / (clean_pixels / 1_000_000.0))


def precision_at_threshold(
    score: np.ndarray,
    defect_mask: np.ndarray,
    threshold: float = 0.5,
    min_component_area: int = 1,
    max_component_area: int | None = None,
) -> float:
    """Pixel precision at the operating threshold."""
    pred = threshold_filtered_mask(score, threshold, min_component_area, max_component_area)
    gt = _bool(defect_mask)
    tp = float((pred & gt).sum())
    fp = float((pred & ~gt).sum())
    if tp + fp <= EPS:
        return 0.0
    return float(tp / (tp + fp))


def mask_iou(
    score: np.ndarray,
    defect_mask: np.ndarray,
    threshold: float = 0.5,
    min_component_area: int = 1,
    max_component_area: int | None = None,
) -> float:
    pred = threshold_filtered_mask(score, threshold, min_component_area, max_component_area)
    gt = _bool(defect_mask)
    union = pred | gt
    if not union.any():
        return 1.0
    return float((pred & gt).sum() / union.sum())


def average_precision_score(prob: np.ndarray, target: np.ndarray) -> float:
    """Pixel-level average precision for defect localization."""
    p = _score(prob).ravel()
    y = _bool(target).astype(np.uint8).ravel()
    positives = int(y.sum())
    if positives == 0:
        return 0.0
    order = np.argsort(-p)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / positives
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - recall_prev) * precision))


def boundary(mask: np.ndarray) -> np.ndarray:
    gt = _bool(mask)
    eroded = binary_erosion(gt, iterations=1)
    return gt & ~eroded


def edge_f1(
    score: np.ndarray,
    edge_mask: np.ndarray,
    threshold: float = 0.5,
    tolerance: int = 1,
    min_component_area: int = 1,
    max_component_area: int | None = None,
) -> float:
    pred_edge = boundary(threshold_filtered_mask(score, threshold, min_component_area, max_component_area))
    gt_edge = _bool(edge_mask)
    if tolerance > 0:
        pred_match = binary_dilation(pred_edge, iterations=tolerance)
        gt_match = binary_dilation(gt_edge, iterations=tolerance)
    else:
        pred_match = pred_edge
        gt_match = gt_edge
    tp = float((pred_edge & gt_match).sum())
    fp = float((pred_edge & ~gt_match).sum())
    fn = float((gt_edge & ~pred_match).sum())
    precision = tp / max(tp + fp, EPS)
    recall = tp / max(tp + fn, EPS)
    return float(2 * precision * recall / max(precision + recall, EPS))


def brier_score(prob: np.ndarray, target: np.ndarray) -> float:
    p = _score(prob)
    y = _bool(target).astype(np.float32)
    return float(np.mean((p - y) ** 2))


def psnr(pred: np.ndarray, target: np.ndarray, max_value: float = 1.0) -> float:
    """Secondary reconstruction sanity metric."""
    mse = float(np.mean((_score(pred) - _score(target)) ** 2))
    if mse <= EPS:
        return 99.0
    return float(20.0 * np.log10(max_value) - 10.0 * np.log10(mse))


def ssim_simple(pred: np.ndarray, target: np.ndarray) -> float:
    """Small global SSIM approximation for secondary sanity checks."""
    x = _score(pred)
    y = _score(target)
    c1 = 0.01**2
    c2 = 0.03**2
    mux = float(x.mean())
    muy = float(y.mean())
    varx = float(x.var())
    vary = float(y.var())
    cov = float(((x - mux) * (y - muy)).mean())
    numerator = (2 * mux * muy + c1) * (2 * cov + c2)
    denominator = (mux**2 + muy**2 + c1) * (varx + vary + c2)
    return float(numerator / max(denominator, EPS))


def expected_calibration_error(prob: np.ndarray, target: np.ndarray, bins: int = 10) -> float:
    p = _score(prob).ravel()
    y = _bool(target).astype(np.float32).ravel()
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        if high == 1.0:
            keep = (p >= low) & (p <= high)
        else:
            keep = (p >= low) & (p < high)
        if not keep.any():
            continue
        confidence = float(p[keep].mean())
        accuracy = float(y[keep].mean())
        ece += float(keep.mean()) * abs(confidence - accuracy)
    return float(ece)


def risk_coverage_curve(
    prob: np.ndarray,
    target: np.ndarray,
    risk: np.ndarray,
    coverages: Iterable[float] = (1.0, 0.9, 0.75, 0.5, 0.25),
    threshold: float = 0.5,
) -> list[dict[str, float]]:
    """Compute error after abstaining from the highest-risk pixels."""
    p = _score(prob).ravel()
    y = _bool(target).ravel()
    r = _score(risk).ravel()
    order = np.argsort(r)
    pred = p >= threshold
    rows: list[dict[str, float]] = []
    for coverage in coverages:
        count = max(1, int(round(float(coverage) * len(order))))
        keep = order[:count]
        error = float(np.mean(pred[keep] != y[keep]))
        rows.append({"coverage": float(coverage), "error": error})
    return rows


def area_under_risk_coverage(curve: list[dict[str, float]]) -> float:
    """Trapezoidal area under error-vs-coverage curve."""
    if len(curve) < 2:
        return 0.0
    points = sorted((float(row["coverage"]), float(row["error"])) for row in curve)
    return float(np.trapezoid([p[1] for p in points], [p[0] for p in points]))


def recall_fpr_curve(
    prob: np.ndarray,
    defect_mask: np.ndarray,
    clean_mask: np.ndarray,
    thresholds: Iterable[float] = (0.999, 0.995, 0.99, 0.975, 0.95, 0.9, 0.75, 0.5, 0.25, 0.1, 0.05),
) -> list[dict[str, float]]:
    """Inspection operating curve over score thresholds."""
    return [
        {
            "threshold": float(threshold),
            "recall": defect_recall(prob, defect_mask, threshold),
            "fpr": false_positive_rate(prob, clean_mask, threshold),
            "nhr": no_defect_hallucination_rate(prob, clean_mask, threshold),
        }
        for threshold in thresholds
    ]


def partial_auc_low_fpr(curve: list[dict[str, float]], max_fpr: float = 1e-3) -> float:
    """Area under recall-vs-FPR restricted to the low-FPR inspection region."""
    if not curve:
        return 0.0
    points = sorted((float(row["fpr"]), float(row["recall"])) for row in curve)
    clipped: list[tuple[float, float]] = []
    prev_fpr, prev_recall = points[0]
    if prev_fpr <= max_fpr:
        clipped.append((prev_fpr, prev_recall))
    for fpr, recall in points[1:]:
        if fpr <= max_fpr:
            clipped.append((fpr, recall))
        elif prev_fpr < max_fpr:
            ratio = (max_fpr - prev_fpr) / max(fpr - prev_fpr, EPS)
            interp = prev_recall + ratio * (recall - prev_recall)
            clipped.append((max_fpr, float(interp)))
            break
        prev_fpr, prev_recall = fpr, recall
    if len(clipped) < 2:
        return 0.0
    xs = [point[0] for point in clipped]
    ys = [point[1] for point in clipped]
    return float(np.trapezoid(ys, xs) / max(max_fpr, EPS))


def summarize_prediction(
    prob: np.ndarray,
    defect_mask: np.ndarray,
    clean_mask: np.ndarray,
    edge_mask: np.ndarray,
    risk: np.ndarray | None = None,
    threshold: float = 0.5,
    sr_image: np.ndarray | None = None,
    hr_image: np.ndarray | None = None,
    min_component_area: int = 1,
    max_component_area: int | None = None,
) -> dict[str, float | list[dict[str, float]]]:
    """Return the core inspection metric bundle for one prediction."""
    if risk is None:
        risk = np.abs(_score(prob) - 0.5) * -2.0 + 1.0
        risk = np.clip(risk, 0.0, 1.0)
    out: dict[str, float | list[dict[str, float]]] = {
        "defect_recall": defect_recall(prob, defect_mask, threshold, min_component_area, max_component_area),
        "false_positive_rate": false_positive_rate(prob, clean_mask, threshold, min_component_area, max_component_area),
        "precision": precision_at_threshold(prob, defect_mask, threshold, min_component_area, max_component_area),
        "component_recall": component_recall(prob, defect_mask, threshold),
        "component_iou10_recall": component_iou_recall(prob, defect_mask, threshold, iou_threshold=0.1),
        "component_iou25_recall": component_iou_recall(prob, defect_mask, threshold, iou_threshold=0.25),
        "component_centroid8_recall": component_centroid_recall(prob, defect_mask, threshold, max_distance=8.0),
        "mask_iou": mask_iou(prob, defect_mask, threshold, min_component_area, max_component_area),
        "pixel_ap": average_precision_score(prob, defect_mask),
        "edge_f1": edge_f1(prob, edge_mask, threshold, min_component_area=min_component_area, max_component_area=max_component_area),
        "no_defect_hallucination_rate": no_defect_hallucination_rate(
            prob, clean_mask, threshold, min_component_area, max_component_area
        ),
        "component_hallucination_count": component_hallucination_rate(prob, clean_mask, threshold),
        "false_components_per_mpx": false_components_per_mpx(prob, clean_mask, threshold),
        "brier_score": brier_score(prob, defect_mask),
        "ece": expected_calibration_error(prob, defect_mask),
    }
    risk_curve = risk_coverage_curve(prob, defect_mask, risk, threshold=threshold)
    operating_curve = recall_fpr_curve(prob, defect_mask, clean_mask)
    out["risk_coverage_auc"] = area_under_risk_coverage(risk_curve)
    out["partial_auc_fpr_1e_3"] = partial_auc_low_fpr(operating_curve, max_fpr=1e-3)
    out["risk_coverage"] = risk_curve
    out["recall_fpr_curve"] = operating_curve
    if sr_image is not None and hr_image is not None:
        out["psnr"] = psnr(sr_image, hr_image)
        out["ssim"] = ssim_simple(sr_image, hr_image)
    return out


def threshold_for_target_fpr(
    scores: list[np.ndarray],
    clean_masks: list[np.ndarray],
    target_fpr: float,
    min_component_area: int = 1,
    max_component_area: int | None = None,
) -> float:
    """Choose tau so validation clean-region FPR is at or below target_fpr."""
    if min_component_area <= 1 and max_component_area is None:
        clean_scores = []
        for score, clean in zip(scores, clean_masks):
            values = _score(score)[_bool(clean)]
            if values.size:
                clean_scores.append(values)
        if not clean_scores:
            return 0.5
        values = np.concatenate(clean_scores)
        target_fpr = float(np.clip(target_fpr, 0.0, 1.0))
        if target_fpr <= 0:
            return float(np.nextafter(values.max(), 1.0))
        quantile = max(0.0, min(1.0, 1.0 - target_fpr))
        return float(np.quantile(values, quantile))
    if not scores:
        return 0.5
    target_fpr = float(np.clip(target_fpr, 0.0, 1.0))
    candidates = np.unique(np.concatenate([_score(score).ravel() for score in scores]))
    if candidates.size > 512:
        candidates = np.quantile(candidates, np.linspace(0.0, 1.0, 512))
    best = float(np.nextafter(float(candidates.max()), 1.0))
    for threshold in sorted((float(x) for x in candidates), reverse=True):
        fprs = [
            false_positive_rate(score, clean, threshold, min_component_area, max_component_area)
            for score, clean in zip(scores, clean_masks)
        ]
        if float(np.mean(fprs)) <= target_fpr:
            best = threshold
        else:
            break
    return best


def aggregate_metric_dicts(rows: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate scalar metric dictionaries with mean and std keys."""
    if not rows:
        raise ValueError("No metric rows to aggregate")
    keys = [key for key, value in rows[0].items() if isinstance(value, (int, float))]
    out: dict[str, float] = {}
    for key in keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        out[f"{key}_mean"] = float(values.mean())
        out[f"{key}_std"] = float(values.std(ddof=0))
    return out


def aggregate_risk_coverage(curves: list[list[dict[str, float]]]) -> list[dict[str, float]]:
    """Aggregate risk-coverage curves with mean/std error at each coverage."""
    if not curves:
        return []
    coverages = [row["coverage"] for row in curves[0]]
    out = []
    for index, coverage in enumerate(coverages):
        errors = np.asarray([curve[index]["error"] for curve in curves], dtype=np.float64)
        out.append(
            {
                "coverage": float(coverage),
                "error_mean": float(errors.mean()),
                "error_std": float(errors.std(ddof=0)),
            }
        )
    return out


def aggregate_recall_fpr_curves(curves: list[list[dict[str, float]]]) -> list[dict[str, float]]:
    """Aggregate threshold operating curves."""
    if not curves:
        return []
    thresholds = [row["threshold"] for row in curves[0]]
    out = []
    for index, threshold in enumerate(thresholds):
        row: dict[str, float] = {"threshold": float(threshold)}
        for key in ("recall", "fpr", "nhr"):
            values = np.asarray([curve[index][key] for curve in curves], dtype=np.float64)
            row[f"{key}_mean"] = float(values.mean())
            row[f"{key}_std"] = float(values.std(ddof=0))
        out.append(row)
    return out
