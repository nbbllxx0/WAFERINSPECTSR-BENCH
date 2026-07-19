# WaferInspectSR-Bench

WaferInspectSR-Bench is the reference code release for evaluating super-resolution under semiconductor inspection constraints. It measures downstream defect recall together with clean-region false-positive behavior, weak-defect sensitivity, calibration, and abstention risk.

The accompanying manuscript's central protocol is deliberately conservative: candidate operating points are fit on `val_calib`, selected on the disjoint `clean_calib` split, frozen, and then evaluated once on held-out test data. The release includes the corrected no-test-feedback operating-point sweep used for that audit.

## Included

- `src/waferinspectsr/`: benchmark, degradation, model, metric, and protocol implementation.
- `configs/`: smoke, default, scaled, ablation, pretrained-model, and stable TSM manuscript configurations.
- `scripts/`: data preparation, training, unified-protocol evaluation, external SEM stress, robustness, and protocol-audit entry points.
- `tests/`: deterministic unit and smoke tests.

Datasets, generated tensors, model checkpoints, experiment outputs, manuscript sources, and figure-generation code are intentionally not included. This boundary keeps the public repository focused on executable benchmark code and avoids redistributing third-party data or large artifacts. The exact code path, canonical seeds, configuration, and commands used for the IEEE TSM manuscript are documented in [TSM_REPRODUCIBILITY.md](TSM_REPRODUCIBILITY.md).

## Installation

Python 3.10 or newer is required. The manuscript DeepLabV3 path also requires torchvision; install the optional tsm dependency group for that workflow.

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

The manuscript uses ten canonical seeds and a target clean-region FPR of `3e-4`. Candidate thresholds are fitted without test feedback, selected by clean calibration feasibility, and only then scored on the test split. The complete code-only workflow is in [TSM_REPRODUCIBILITY.md](TSM_REPRODUCIBILITY.md):

```bash
python scripts/sweep_matched_fpr_offline.py --run-dir experiments/runs/tsm_seed_sweep --output-dir outputs/tsm_dpu_transfer --device cuda
```

This audit found that clean-calibration feasibility did not transfer to the held-out test split for DPU-WaferSR (10/10 feasible on `clean_calib`, 0/10 feasible on test). The paper therefore reports this as a failed constraint-transfer result rather than a matched-FPR performance gain.

## Data

The synthetic benchmark can be generated locally with the included scripts. Carinthia-S remains governed by its original dataset terms and must be obtained separately. No third-party data are redistributed here.

## License

Code in this repository is released under the MIT License. Dataset and external checkpoint licenses remain with their respective owners.

## Citation

Please cite the accompanying manuscript. Formal citation metadata will be added when its permanent identifier is assigned.

