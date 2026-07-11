# WaferInspectSR-Bench code release

Version: `0.2.0-arxiv`

Repository: https://github.com/nbbllxx0/WAFERINSPECTSR-BENCH

This release contains the executable benchmark package, configurations, training and evaluation entry points, tests, and the corrected leak-free operating-point audit.

Not included: manuscript or arXiv sources, figure/plot/render generation code, datasets, generated samples, experiment outputs, model checkpoints, or third-party pretrained weights.

The corrected audit selects candidate operating points using validation and clean-calibration data only, freezes the selected rule, and evaluates held-out test data once. Its reported DPU-WaferSR result is a constraint-transfer failure, not a matched-FPR improvement claim.

