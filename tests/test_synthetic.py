import numpy as np

from waferinspectsr.synthetic import generate_benchmark_splits, generate_dataset, generate_sample


def test_generate_sample_has_exact_masks():
    sample = generate_sample(96, 96, "line_space", "particle", 123)
    assert sample.image.shape == (96, 96)
    assert sample.defect_mask.shape == (96, 96)
    assert sample.defect_mask.any()
    assert sample.clean_mask.any()
    assert not np.any(sample.clean_mask & sample.defect_mask)
    assert sample.edge_mask.sum() > 0


def test_generate_dataset_is_deterministic():
    a = generate_dataset(3, 64, 64, ["line_space"], ["gap"], 5)
    b = generate_dataset(3, 64, 64, ["line_space"], ["gap"], 5)
    assert np.allclose(a[0].image, b[0].image)
    assert np.array_equal(a[1].defect_mask, b[1].defect_mask)


def test_clean_and_weak_splits_are_explicit():
    samples = generate_benchmark_splits(
        {"clean_calib": 2, "clean_test": 2, "weak_test": 2, "ood_calib_optional": 1, "ood_test": 2, "test": 2},
        64,
        64,
        ["line_space", "contact_hole"],
        ["gap", "particle"],
        seed=21,
    )
    by_split = {sample.split for sample in samples}
    assert {"clean_calib", "clean_test", "weak_test", "ood_calib_optional", "ood_test", "test"} <= by_split
    clean = [sample for sample in samples if sample.split in {"clean_calib", "clean_test"}]
    weak = [sample for sample in samples if sample.split == "weak_test"]
    ood = [sample for sample in samples if sample.split in {"ood_calib_optional", "ood_test"}]
    assert all(not sample.defect_mask.any() for sample in clean)
    assert all(sample.defect_mask.any() for sample in weak)
    assert all(sample.defect_type == "residue" for sample in ood)
