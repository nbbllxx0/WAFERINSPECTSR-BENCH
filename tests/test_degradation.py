import numpy as np

from waferinspectsr.degradation import DegradationConfig, degrade, upsample_lr


def test_degradation_shape_and_determinism():
    image = np.linspace(0, 1, 64 * 64, dtype=np.float32).reshape(64, 64)
    cfg = DegradationConfig(scale=2)
    a = degrade(image, cfg, seed=11)
    b = degrade(image, cfg, seed=11)
    assert a.shape == (32, 32)
    assert np.allclose(a, b)
    assert 0.0 <= float(a.min()) <= float(a.max()) <= 1.0


def test_upsample_lr_restores_target_shape():
    lr = np.zeros((16, 20), dtype=np.float32)
    up = upsample_lr(lr, (32, 40))
    assert up.shape == (32, 40)
