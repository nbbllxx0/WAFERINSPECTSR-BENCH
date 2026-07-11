from pathlib import Path

from waferinspectsr.baselines import run_v1_baselines
from waferinspectsr.degradation import DegradationConfig
from waferinspectsr.io import load_npz, save_sample_npz
from waferinspectsr.metrics import summarize_prediction
from waferinspectsr.synthetic import generate_dataset


def test_synthetic_baseline_smoke(tmp_path: Path):
    samples = generate_dataset(
        num_samples=2,
        height=64,
        width=64,
        pattern_types=["line_space", "contact_hole"],
        defect_types=["particle", "gap"],
        seed=9,
    )
    path = save_sample_npz(samples, tmp_path / "samples.npz", DegradationConfig(scale=2), seed=9)
    data = load_npz(path)
    predictions = run_v1_baselines(data["lr"][0], data["hr"][0].shape)
    assert {pred.name for pred in predictions} == {
        "no_sr_task_detector",
        "lr_detector",
        "nearest_detector",
        "bilinear_detector",
        "bicubic_detector",
        "lanczos_detector",
        "denoise_upsample_detector",
        "wiener_deconv_detector",
        "prior_only_detector",
        "sharpened_sr_detector",
        "naf_style_sr_detector",
    }
    predictions_with_oracle = run_v1_baselines(data["lr"][0], data["hr"][0].shape, hr=data["hr"][0], include_oracle=True)
    assert "hr_detector_reference" in {pred.name for pred in predictions_with_oracle}
    metrics = summarize_prediction(
        predictions[0].defect_prob,
        data["defect_mask"][0],
        data["clean_mask"][0],
        data["edge_mask"][0],
        risk=predictions[0].risk,
    )
    assert 0.0 <= metrics["defect_recall"] <= 1.0
    assert 0.0 <= metrics["no_defect_hallucination_rate"] <= 1.0
