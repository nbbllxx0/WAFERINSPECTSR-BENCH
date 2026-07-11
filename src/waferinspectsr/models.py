"""Compact proposed model skeleton for DPU-WaferSR-style experiments."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    """Small residual block with dropout for MC uncertainty."""

    def __init__(self, channels: int, dropout: float = 0.05) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class CompactInspectionSafeSR(nn.Module):
    """Compact SR backbone with defect and hallucination-risk heads.

    When ``use_sr`` is False, the stem/trunk stay identical but the reconstruction
    head is bypassed: trunk features are PixelShuffled to HR and fed to the same
    defect/risk pathway. The reported ``sr`` image is bilinear upsampling of LR
    (for metric compatibility), isolating the causal role of the SR branch.
    """

    def __init__(
        self,
        scale: int = 2,
        channels: int = 48,
        blocks: int = 4,
        dropout: float = 0.05,
        use_sr: bool = True,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.use_sr = bool(use_sr)
        self.stem = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.trunk = nn.Sequential(*[ResidualBlock(channels, dropout=dropout) for _ in range(blocks)])
        upsample_head = nn.Sequential(
            nn.Conv2d(channels, channels * scale * scale, kernel_size=3, padding=1),
            nn.PixelShuffle(scale),
            nn.GELU(),
            nn.Conv2d(channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        # Matched capacity: either the SR image head or the feature-upsample head is active.
        self.sr_head = upsample_head if self.use_sr else None
        self.feature_upsample = None if self.use_sr else upsample_head
        self.shared_hr = nn.Sequential(
            nn.Conv2d(1, channels // 2, kernel_size=3, padding=1),
            nn.GELU(),
            ResidualBlock(channels // 2, dropout=dropout),
        )
        self.defect_head = nn.Sequential(nn.Conv2d(channels // 2, 1, kernel_size=1), nn.Sigmoid())
        self.risk_head = nn.Sequential(nn.Conv2d(channels // 2, 1, kernel_size=1), nn.Sigmoid())

    def forward(self, lr: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.trunk(self.stem(lr))
        if self.use_sr:
            assert self.sr_head is not None
            sr = self.sr_head(features)
            shared_input = sr
        else:
            assert self.feature_upsample is not None
            shared_input = self.feature_upsample(features)
            # Metric-facing reconstruction proxy; not used by the defect pathway.
            sr = F.interpolate(lr, scale_factor=self.scale, mode="bilinear", align_corners=False).clamp(0.0, 1.0)
        shared = self.shared_hr(shared_input)
        defect_prob = self.defect_head(shared)
        risk = self.risk_head(shared)
        return {"sr": sr, "defect_prob": defect_prob, "risk": risk}


class CompactSRLite(nn.Module):
    """Small reconstruction-only SR baseline."""

    def __init__(self, scale: int = 2, channels: int = 32, blocks: int = 3, dropout: float = 0.0) -> None:
        super().__init__()
        self.scale = scale
        self.net = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, padding=1),
            nn.GELU(),
            *[ResidualBlock(channels, dropout=dropout) for _ in range(blocks)],
            nn.Conv2d(channels, channels * scale * scale, kernel_size=3, padding=1),
            nn.PixelShuffle(scale),
            nn.GELU(),
            nn.Conv2d(channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        return self.net(lr)


class NAFBlock(nn.Module):
    """Small activation-free gated block inspired by NAFNet-style restoration."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(channels)
        self.expand = nn.Conv2d(channels, channels * 2, kernel_size=1)
        self.depthwise = nn.Conv2d(channels * 2, channels * 2, kernel_size=3, padding=1, groups=channels * 2)
        self.project = nn.Conv2d(channels, channels, kernel_size=1)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.depthwise(self.expand(self.norm(x)))
        left, right = y.chunk(2, dim=1)
        gated = left * right
        return x + self.beta * self.project(gated)


class CompactNAFSR(nn.Module):
    """Dependency-free NAFNet-style SR baseline for controlled comparisons."""

    def __init__(self, scale: int = 2, channels: int = 24, blocks: int = 4) -> None:
        super().__init__()
        self.scale = scale
        self.stem = nn.Conv2d(1, channels, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(*[NAFBlock(channels) for _ in range(blocks)])
        self.head = nn.Sequential(
            nn.Conv2d(channels, channels * scale * scale, kernel_size=3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        features = self.blocks(self.stem(lr))
        return self.head(features)


class CompactUNetDetector(nn.Module):
    """Small detector-only U-Net-style baseline that maps LR crops to HR masks."""

    def __init__(self, scale: int = 2, channels: int = 16, dropout: float = 0.05) -> None:
        super().__init__()
        self.scale = scale
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels * 2, channels * 2, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.bridge = ResidualBlock(channels * 2, dropout=dropout)
        self.dec1 = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Conv2d(channels, scale * scale, kernel_size=3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        enc1 = self.enc1(lr)
        enc2 = self.enc2(enc1)
        bridge = self.bridge(enc2)
        up = F.interpolate(bridge, size=enc1.shape[-2:], mode="bilinear", align_corners=False)
        dec = self.dec1(torch.cat([up, enc1], dim=1))
        return self.head(dec)


class CompactASPPDetector(nn.Module):
    """Dilated-context detector baseline inspired by DeepLab-style ASPP."""

    def __init__(self, scale: int = 2, channels: int = 24, dropout: float = 0.05) -> None:
        super().__init__()
        self.scale = scale
        self.stem = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.aspp = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
                    nn.GELU(),
                    nn.Dropout2d(dropout),
                )
                for dilation in (1, 2, 4, 8)
            ]
        )
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GELU(),
        )
        self.project = nn.Sequential(
            nn.Conv2d(channels * 5, channels, kernel_size=1),
            nn.GELU(),
            ResidualBlock(channels, dropout=dropout),
        )
        self.head = nn.Sequential(
            nn.Conv2d(channels, channels * scale * scale, kernel_size=3, padding=1),
            nn.PixelShuffle(scale),
            nn.GELU(),
            nn.Conv2d(channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        features = self.stem(lr)
        pooled = F.interpolate(self.image_pool(features), size=features.shape[-2:], mode="bilinear", align_corners=False)
        context = torch.cat([branch(features) for branch in self.aspp] + [pooled], dim=1)
        return self.head(self.project(context))


def edge_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Sobel edge L1 loss for metrology-sensitive boundaries."""
    kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=pred.dtype, device=pred.device)
    kernel_y = kernel_x.t()
    kernel_x = kernel_x.view(1, 1, 3, 3)
    kernel_y = kernel_y.view(1, 1, 3, 3)
    pred_x = F.conv2d(pred, kernel_x, padding=1)
    pred_y = F.conv2d(pred, kernel_y, padding=1)
    target_x = F.conv2d(target, kernel_x, padding=1)
    target_y = F.conv2d(target, kernel_y, padding=1)
    pred_mag = torch.sqrt(pred_x.square() + pred_y.square() + 1e-8)
    target_mag = torch.sqrt(target_x.square() + target_y.square() + 1e-8)
    return F.l1_loss(pred_mag, target_mag)


def inspection_safe_loss(
    outputs: dict[str, torch.Tensor],
    hr: torch.Tensor,
    defect_mask: torch.Tensor,
    clean_mask: torch.Tensor,
    weights: dict[str, float] | None = None,
) -> dict[str, torch.Tensor]:
    """v1 loss bundle: reconstruction, defect, hallucination, edge, calibration proxy."""
    weights = weights or {
        "recon": 1.0,
        "defect": 1.0,
        "hallucination": 1.0,
        "edge": 0.2,
        "calibration": 0.05,
        "positive": 8.0,
    }
    sr = outputs["sr"]
    defect_prob = outputs["defect_prob"]
    risk = outputs["risk"]

    recon = F.l1_loss(sr, hr)
    positive_weight = float(weights.get("positive", 8.0))
    defect_weights = 1.0 + (positive_weight - 1.0) * defect_mask
    defect = F.binary_cross_entropy(defect_prob.clamp(1e-5, 1 - 1e-5), defect_mask, weight=defect_weights)
    clean_prob = defect_prob.clamp(1e-5, 1 - 1e-5)[clean_mask > 0.5]
    if clean_prob.numel() == 0:
        hallucination = defect_prob.new_tensor(0.0)
    else:
        hallucination = F.binary_cross_entropy(clean_prob, torch.zeros_like(clean_prob))
    edges = edge_loss(defect_prob, defect_mask)
    calibration = F.mse_loss(risk, torch.abs(defect_prob.detach() - defect_mask))
    total = (
        weights["recon"] * recon
        + weights["defect"] * defect
        + weights["hallucination"] * hallucination
        + weights["edge"] * edges
        + weights["calibration"] * calibration
    )
    return {
        "total": total,
        "recon": recon,
        "defect": defect,
        "hallucination": hallucination,
        "edge": edges,
        "calibration": calibration,
    }
