# WaferInspectSR-Bench code release

Version: 0.3.0-tsm-code

Repository: https://github.com/nbbllxx0/WAFERINSPECTSR-BENCH

This code-side release contains the executable benchmark package, the stable IEEE TSM manuscript configuration, exact canonical commands, training and evaluation entry points, unified fixed-detector and operating-policy evaluation, external SEM stress code, and deterministic tests.

The release boundary is intentional. It does not redistribute manuscript sources, figure-rendering assets, third-party datasets, generated samples, experiment outputs, model checkpoints, or external pretrained weights. Users generate those artifacts locally by following TSM_REPRODUCIBILITY.md.

The operating-point implementation fits candidates on validation data, selects a policy using independent clean calibration, freezes it, and evaluates held-out or external data only afterward. The reported DPU-WaferSR result is a constraint-transfer failure, not a matched-FPR improvement claim.
