# WaferInspectSR-Bench

WaferInspectSR-Bench is the reference code release for evaluating super-resolution under semiconductor inspection constraints. It measures downstream defect recall together with clean-region false-positive behavior, weak-defect sensitivity, calibration, and abstention risk.

The accompanying manuscript's central protocol is deliberately conservative: candidate operating points are fit on `val_calib`, selected on the disjoint `clean_calib` split, frozen, and then evaluated once on held-out test data. The release includes the corrected no-test-feedback operating-point sweep used for that audit.

## Included

- `src/waferinspectsr/`: benchmark, degradation, model, metric, and protocol implementation.
- `configs/`: smoke, default, scaled, ablation, and pretrained-model configurations.
- `scripts/`: data preparation, training, evaluation, robustness, and protocol-audit entry points.
- `tests/`: deterministic unit and smoke tests.

Datasets, generated tensors, model checkpoints, experiment outputs, manuscript sources, and figure-generation code are intentionally not included. This boundary keeps the public repository focused on executable benchmark code and avoids redistributing third-party data or large artifacts.

## Installation

Python 3.10 or newer is required.

```bash
python -m pip install -e ".[dev]"
pytest
```

Optional pretrained-SR adapters require:

```bash
python -m pip install -e ".[dev,pretrained]"
```

## Smoke workflow

```bash
python scripts/prepare_sample_data.py --config configs/smoke.yaml
python scripts/run_baselines.py --input data/generated/smoke/samples.npz --output outputs/baselines.json --matched-fpr 0.01
python scripts/train_dpu_wafersr.py --input data/generated/smoke/samples.npz --output outputs/dpu_wafersr.pt --epochs 1 --device auto
pytest
```

Generated data and outputs are ignored by Git. Use `--help` on each script for its full interface.

## Leak-free operating-point audit

The manuscript audit used ten canonical seeds and a target clean-region FPR of `3e-4`. Candidate thresholds are fitted without test feedback, selected by clean calibration feasibility, and only then scored on the test split:

```bash
python scripts/sweep_matched_fpr_offline.py --run-dir PATH/TO/SEED_RUNS --output-dir outputs/matched_clean_calib --device cuda
```

This audit found that clean-calibration feasibility did not transfer to the held-out test split for DPU-WaferSR (10/10 feasible on `clean_calib`, 0/10 feasible on test). The paper therefore reports this as a failed constraint-transfer result rather than a matched-FPR performance gain.

## Data

The synthetic benchmark can be generated locally with the included scripts. Carinthia-S remains governed by its original dataset terms and must be obtained separately. No third-party data are redistributed here.

## License

Code in this repository is released under the MIT License. Dataset and external checkpoint licenses remain with their respective owners.

## Citation

Please cite the accompanying arXiv manuscript. Formal citation metadata will be added once the arXiv identifier is assigned.

