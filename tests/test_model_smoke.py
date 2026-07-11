import numpy as np
import torch

from waferinspectsr.models import (
    CompactASPPDetector,
    CompactInspectionSafeSR,
    CompactNAFSR,
    CompactSRLite,
    CompactUNetDetector,
    inspection_safe_loss,
)
from waferinspectsr.device import resolve_device
from waferinspectsr.pretrained_sr import EDSR, RCAN, OnnxSRRunner, RRDBNet, SRBaselineSpec, SwinIR, build_model, load_manifest


def test_model_forward_and_loss_shapes():
    model = CompactInspectionSafeSR(scale=2, channels=12, blocks=1)
    lr = torch.rand(2, 1, 16, 16)
    hr = torch.rand(2, 1, 32, 32)
    defect = torch.zeros(2, 1, 32, 32)
    defect[:, :, 10:14, 10:14] = 1.0
    clean = 1.0 - defect
    outputs = model(lr)
    assert outputs["sr"].shape == hr.shape
    assert outputs["defect_prob"].shape == hr.shape
    assert outputs["risk"].shape == hr.shape
    loss = inspection_safe_loss(outputs, hr, defect, clean)
    assert loss["total"].ndim == 0


def test_model_dropout_changes_mc_predictions():
    torch.manual_seed(3)
    model = CompactInspectionSafeSR(scale=2, channels=12, blocks=1, dropout=0.5)
    model.train()
    lr = torch.rand(1, 1, 16, 16)
    first = model(lr)["defect_prob"]
    second = model(lr)["defect_prob"]
    assert not torch.allclose(first, second)


def test_srlite_forward_shape():
    model = CompactSRLite(scale=2, channels=12, blocks=1)
    lr = torch.rand(2, 1, 16, 16)
    sr = model(lr)
    assert sr.shape == (2, 1, 32, 32)


def test_naf_sr_forward_shape():
    model = CompactNAFSR(scale=2, channels=12, blocks=1)
    lr = torch.rand(2, 1, 16, 16)
    sr = model(lr)
    assert sr.shape == (2, 1, 32, 32)


def test_unet_detector_forward_shape():
    model = CompactUNetDetector(scale=2, channels=8)
    lr = torch.rand(2, 1, 16, 16)
    prob = model(lr)
    assert prob.shape == (2, 1, 32, 32)
    assert float(prob.min()) >= 0.0
    assert float(prob.max()) <= 1.0


def test_aspp_detector_forward_shape():
    model = CompactASPPDetector(scale=2, channels=8)
    lr = torch.rand(2, 1, 16, 16)
    prob = model(lr)
    assert prob.shape == (2, 1, 32, 32)
    assert float(prob.min()) >= 0.0
    assert float(prob.max()) <= 1.0


def test_resolve_device_auto_returns_torch_device():
    device = resolve_device("auto")
    assert device.type in {"cpu", "cuda"}


def test_pretrained_sr_architecture_smoke_shapes():
    lr = torch.rand(1, 1, 8, 8)
    specs = [
        SRBaselineSpec("edsr", "EDSR", "edsr", "missing.pt", kwargs={"channels": 8, "blocks": 1}),
        SRBaselineSpec(
            "rcan",
            "RCAN",
            "rcan",
            "missing.pt",
            kwargs={"channels": 8, "groups": 1, "blocks_per_group": 1, "reduction": 4},
        ),
        SRBaselineSpec("rrdb", "RRDB", "rrdb", "missing.pt", kwargs={"channels": 8, "blocks": 1, "growth_channels": 4}),
        SRBaselineSpec(
            "swinir",
            "SwinIR",
            "swinir",
            "missing.pt",
            kwargs={
                "img_size": 8,
                "embed_dim": 8,
                "depths": [1],
                "num_heads": [2],
                "window_size": 4,
                "upscale_channels": 8,
            },
        ),
    ]
    for spec in specs:
        model = build_model(spec)
        out = model(lr)
        assert out.shape == (1, 1, 16, 16)


def test_pretrained_sr_classes_are_torch_modules():
    assert isinstance(EDSR(scale=2, channels=8, blocks=1), torch.nn.Module)
    assert isinstance(RCAN(scale=2, channels=8, groups=1, blocks_per_group=1), torch.nn.Module)
    assert isinstance(RRDBNet(scale=2, channels=8, blocks=1, growth_channels=4), torch.nn.Module)
    assert isinstance(
        SwinIR(scale=2, img_size=8, embed_dim=8, depths=(1,), num_heads=(2,), window_size=4, upscale_channels=8),
        torch.nn.Module,
    )


def test_pretrained_sr_manifest_preserves_source_metadata(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        """
        {
          "baselines": [
            {
              "name": "Real-ESRGAN",
              "family": "Real-ESRGAN",
              "architecture": "rrdb",
              "checkpoint": "models/pretrained_sr/realesrgan_rrdb_x2.pt",
              "source_url": "https://example.test/realesrgan.pth",
              "source_note": "direct checkpoint",
              "min_loaded_fraction": 0.9
            }
          ]
        }
        """,
        encoding="utf-8",
    )
    spec = load_manifest(manifest)[0]
    assert spec.source_url == "https://example.test/realesrgan.pth"
    assert spec.source_note == "direct checkpoint"
    assert spec.min_loaded_fraction == 0.9


class _OnnxMeta:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape


class _NearestOnnxSession:
    def get_inputs(self):
        return [_OnnxMeta("input", [1, 4, 4, 1])]

    def get_outputs(self):
        return [_OnnxMeta("output", [1, 8, 8, 1])]

    def run(self, _outputs, feed):
        tensor = feed["input"]
        assert tensor.shape == (1, 4, 4, 1)
        return [np.repeat(np.repeat(tensor, 2, axis=1), 2, axis=2)]


def test_onnx_fixed_input_tiling_covers_larger_crop():
    spec = SRBaselineSpec(
        "onnx-tiled",
        "Fake",
        "onnx",
        "missing.onnx",
        input_channels=1,
        output_channels=1,
        kwargs={"input_layout": "nhwc", "output_layout": "nhwc", "tile_fixed_input": True, "tile_overlap": 1},
    )
    runner = OnnxSRRunner(spec, _NearestOnnxSession())
    lr = np.arange(36, dtype=np.float32).reshape(6, 6) / 35.0
    pred = runner.predict(lr, (12, 12))
    expected = np.repeat(np.repeat(lr, 2, axis=0), 2, axis=1)
    assert pred.sr_image.shape == (12, 12)
    assert np.allclose(pred.sr_image, expected)
