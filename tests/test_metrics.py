import numpy as np

from waferinspectsr.metrics import (
    defect_recall,
    expected_calibration_error,
    fit_temperature,
    apply_temperature,
    aggregate_risk_coverage,
    area_under_risk_coverage,
    average_precision_score,
    component_centroid_recall,
    component_iou_recall,
    component_recall,
    false_components_per_mpx,
    false_positive_rate,
    mask_iou,
    no_defect_hallucination_rate,
    summarize_prediction,
    threshold_for_target_fpr,
)


def test_metrics_perfect_prediction():
    target = np.zeros((8, 8), dtype=bool)
    target[2:4, 2:4] = True
    clean = ~target
    score = target.astype(np.float32)
    assert defect_recall(score, target) == 1.0
    assert false_positive_rate(score, clean) == 0.0
    assert no_defect_hallucination_rate(score, clean) == 0.0
    assert mask_iou(score, target) == 1.0
    assert expected_calibration_error(score, target) == 0.0


def test_summarize_prediction_includes_core_keys():
    target = np.zeros((8, 8), dtype=bool)
    target[2:5, 2:5] = True
    clean = ~target
    score = target.astype(np.float32) * 0.9
    result = summarize_prediction(score, target, clean, target)
    for key in [
        "defect_recall",
        "false_positive_rate",
        "precision",
        "component_recall",
        "component_iou10_recall",
        "component_iou25_recall",
        "component_centroid8_recall",
        "mask_iou",
        "pixel_ap",
        "edge_f1",
        "no_defect_hallucination_rate",
        "component_hallucination_count",
        "false_components_per_mpx",
        "brier_score",
        "ece",
        "risk_coverage_auc",
        "partial_auc_fpr_1e_3",
        "risk_coverage",
        "recall_fpr_curve",
    ]:
        assert key in result

    result_with_recon = summarize_prediction(score, target, clean, target, sr_image=score, hr_image=score)
    assert "psnr" in result_with_recon
    assert "ssim" in result_with_recon


def test_threshold_for_target_fpr_uses_clean_regions():
    scores = [np.asarray([[0.1, 0.2], [0.8, 0.9]], dtype=np.float32)]
    clean = [np.asarray([[1, 1], [0, 0]], dtype=bool)]
    threshold = threshold_for_target_fpr(scores, clean, target_fpr=0.0)
    assert threshold > 0.2


def test_temperature_scaling_changes_probabilities():
    prob = np.asarray([[0.2, 0.8]], dtype=np.float32)
    colder = apply_temperature(prob, 0.5)
    warmer = apply_temperature(prob, 2.0)
    assert colder[0, 0] < prob[0, 0]
    assert warmer[0, 0] > prob[0, 0]
    temp = fit_temperature([prob], [np.asarray([[0, 1]], dtype=bool)])
    assert temp > 0


def test_aggregate_risk_coverage():
    curves = [
        [{"coverage": 1.0, "error": 0.2}, {"coverage": 0.5, "error": 0.1}],
        [{"coverage": 1.0, "error": 0.4}, {"coverage": 0.5, "error": 0.2}],
    ]
    agg = aggregate_risk_coverage(curves)
    assert agg[0]["coverage"] == 1.0
    assert abs(agg[0]["error_mean"] - 0.3) < 1e-9


def test_average_precision_score_orders_defects():
    target = np.asarray([1, 0, 1, 0], dtype=bool)
    good = np.asarray([0.9, 0.1, 0.8, 0.2], dtype=np.float32)
    bad = np.asarray([0.1, 0.9, 0.2, 0.8], dtype=np.float32)
    assert average_precision_score(good, target) > average_precision_score(bad, target)


def test_component_recall_and_risk_auc():
    target = np.zeros((8, 8), dtype=bool)
    target[1:3, 1:3] = True
    target[5:7, 5:7] = True
    score = np.zeros((8, 8), dtype=np.float32)
    score[1:3, 1:3] = 0.9
    curve = [{"coverage": 1.0, "error": 0.4}, {"coverage": 0.5, "error": 0.2}]
    assert component_recall(score, target, threshold=0.5) == 0.5
    assert area_under_risk_coverage(curve) > 0.0


def test_strict_component_variants_and_false_component_density():
    target = np.zeros((16, 16), dtype=bool)
    target[2:6, 2:6] = True
    score = np.zeros((16, 16), dtype=np.float32)
    score[2:6, 2:6] = 0.9
    score[10:12, 10:12] = 0.9
    clean = ~target
    assert component_iou_recall(score, target, threshold=0.5, iou_threshold=0.25) == 1.0
    assert component_centroid_recall(score, target, threshold=0.5, max_distance=2.0) == 1.0
    assert false_components_per_mpx(score, clean, threshold=0.5, min_area=4) > 0.0
