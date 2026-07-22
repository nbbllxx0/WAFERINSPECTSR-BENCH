# IEEE TSM manuscript code path

This repository is the code implementation accompanying the manuscript
"Does Super-Resolution Preserve Defect Evidence? A Low-False-Call Benchmark
for Semiconductor Inspection."

Preprint: https://arxiv.org/abs/2607.17401

It intentionally contains code and configuration only. Datasets, generated
sample arrays, checkpoints, experiment outputs, manuscript sources, and
figure-rendering assets are not distributed. Running the commands below
creates those artifacts locally.

## Reference environment

- Python 3.11
- PyTorch 2.11.0 with CUDA 12.8
- torchvision compatible with the installed PyTorch build
- NVIDIA GeForce RTX 5090 for the reported neural training and external
  inference timings

Other compatible CPU or CUDA environments can execute the code, but runtime
measurements are hardware dependent.

## Install

~~~powershell
python -m pip install -e ".[dev,tsm]"
pytest
~~~

## Canonical synthetic runs

The stable manuscript configuration is "configs/tsm_submission.yaml". The ten
complete data-generation and training repetitions use seeds 7, 11, 13, 17, 19,
23, 29, 31, 37, and 41. The target clean-region false-positive rate is 3e-4,
with validation fitting followed by independent clean-calibration selection.

~~~powershell
$seeds = 7,11,13,17,19,23,29,31,37,41
$run = "experiments/runs/tsm_seed_sweep"

python scripts/run_seed_sweep.py --config configs/tsm_submission.yaml --seeds 7,11,13,17,19,23,29,31,37,41 --epochs 5 --matched-fpr 0.0003 --calibration-splits val_calib,clean_calib --output-dir $run --device cuda --positive-weight 64 --hallucination-weight 2 --prior-fusion geometric --prior-weight 0.5 --mc-samples 4

foreach ($seed in $seeds) {
  python scripts/train_srlite_baseline.py --input "$run/seed_$seed/data/samples.npz" --output "$run/seed_$seed/naf_sr.pt" --architecture naf --epochs 20 --batch-size 4 --seed $seed --matched-fpr 0.0003 --calibration-splits val_calib --device cuda
  python scripts/train_detector_baseline.py --input "$run/seed_$seed/data/samples.npz" --output "$run/seed_$seed/deeplabv3_detector_e20.pt" --architecture deeplabv3 --epochs 20 --batch-size 4 --seed $seed --matched-fpr 0.0003 --calibration-splits val_calib,clean_calib --device cuda --positive-weight 64 --clean-weight 1 --edge-weight 0.1
}
~~~

The first command also trains SR-lite and DPU-WaferSR for five epochs in every
repetition. The separate loops train the NAF-style reconstruction model and
DeepLabV3 direct detector used in the manuscript.

## Unified reconstruction and detector protocol

This command applies the common local-residual detector to every Track A image
transformation, fits candidate operating points on validation data, selects
among them on independent clean calibration, and evaluates held-out splits
once. DeepLabV3 is evaluated under the same three-stage policy rule as the
task-trained Track B reference.

~~~powershell
$run = "experiments/runs/tsm_seed_sweep"
python scripts/run_unified_protocol.py --run-dir $run --naf-dir $run --output-dir outputs/tsm_unified_protocol --device cuda
~~~

## Joint-model operating-point transfer

The joint-model audit uses validation calibration to construct candidates,
independent clean calibration to select one candidate, and nominal held-out
data only after the policy is frozen.

~~~powershell
$run = "experiments/runs/tsm_seed_sweep"
python scripts/sweep_matched_fpr_offline.py --run-dir $run --output-dir outputs/tsm_dpu_transfer --device cuda
~~~

## Carinthia-S stress test

Obtain Carinthia-S from its original source and extract it so that
"data/raw/carinthia_s/data/data" contains "carinthia-s.csv", "images", and
"masks". The command below applies the three role-based synthetic policies
unchanged: bicubic as the fixed-detector reference, NAF-trained as the
highest-SSIM learned reconstruction, and DeepLabV3 as the direct detector.
It does not train, recalibrate, or select thresholds on Carinthia-S.

~~~powershell
$run = "experiments/runs/tsm_seed_sweep"
python scripts/run_external_transfer.py --csv data/raw/carinthia_s/data/data/carinthia-s.csv --root data/raw/carinthia_s/data/data --run-dir $run --naf-dir $run --policy-dir outputs/tsm_unified_protocol --output-dir outputs/tsm_external_transfer --device cuda
~~~

A nonzero "--max-per-class" value is only for a smoke test and is not the
4,591-image manuscript evaluation.

## Release boundary

This is a code-side release. Users generate all data and outputs locally.
Third-party datasets remain subject to their original terms, and checkpoints
or manuscript result files are intentionally excluded.
