"""Checkpoint-driven heavyweight SR baseline adapters."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SRBaselineSpec:
    """One pretrained SR baseline declaration from a JSON manifest."""

    name: str
    family: str
    architecture: str
    checkpoint: str
    scale: int = 2
    input_channels: int = 1
    output_channels: int = 1
    state_key: str | None = None
    strict: bool = False
    kwargs: dict[str, Any] = field(default_factory=dict)
    source_url: str | None = None
    source_note: str | None = None
    min_loaded_fraction: float = 0.95


@dataclass(frozen=True)
class SRPrediction:
    """SR prediction plus runtime metadata for downstream detector evaluation."""

    name: str
    sr_image: np.ndarray
    runtime_ms: float


class EDSR(nn.Module):
    """Dependency-free EDSR-compatible architecture for x2/x4 state dicts."""

    def __init__(
        self,
        scale: int = 2,
        in_channels: int = 1,
        out_channels: int = 1,
        channels: int = 64,
        blocks: int = 16,
        residual_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.head = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
        self.body = nn.Sequential(
            *[EDSRBlock(channels, residual_scale=residual_scale) for _ in range(blocks)],
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.tail = UpsampleHead(channels, out_channels, scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.head(x)
        body = self.body(features) + features
        return self.tail(body)


class EDSRBlock(nn.Module):
    """Two-convolution residual block used by EDSR-style baselines."""

    def __init__(self, channels: int, residual_scale: float = 0.1) -> None:
        super().__init__()
        self.residual_scale = residual_scale
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.residual_scale * self.net(x)


class ChannelAttention(nn.Module):
    """Channel-attention block used in RCAN."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class RCAB(nn.Module):
    """Residual channel-attention block."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            ChannelAttention(channels, reduction=reduction),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ResidualGroup(nn.Module):
    """RCAN residual group."""

    def __init__(self, channels: int, blocks: int, reduction: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            *[RCAB(channels, reduction=reduction) for _ in range(blocks)],
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class RCAN(nn.Module):
    """Dependency-free RCAN-compatible architecture."""

    def __init__(
        self,
        scale: int = 2,
        in_channels: int = 1,
        out_channels: int = 1,
        channels: int = 64,
        groups: int = 10,
        blocks_per_group: int = 10,
        reduction: int = 16,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.head = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
        self.body = nn.Sequential(
            *[ResidualGroup(channels, blocks_per_group, reduction=reduction) for _ in range(groups)],
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.tail = UpsampleHead(channels, out_channels, scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.head(x)
        body = self.body(features) + features
        return self.tail(body)


class Mlp(nn.Module):
    """Feed-forward block used inside SwinIR transformer blocks."""

    def __init__(self, in_features: int, hidden_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Partition BHWC features into non-overlapping windows."""
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return windows.view(-1, window_size, window_size, c)


def window_reverse(windows: torch.Tensor, window_size: int, h: int, w: int) -> torch.Tensor:
    """Reverse window partition back to BHWC features."""
    windows_per_image = (h // window_size) * (w // window_size)
    b = windows.shape[0] // windows_per_image
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(b, h, w, -1)


class WindowAttention(nn.Module):
    """Window multi-head attention compatible with SwinIR checkpoints."""

    def __init__(self, dim: int, window_size: int, num_heads: int) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        table_size = (2 * window_size - 1) * (2 * window_size - 1)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(table_size, num_heads))
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b_windows, n_tokens, channels = x.shape
        qkv = self.qkv(x).reshape(b_windows, n_tokens, 3, self.num_heads, channels // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        relative_bias = relative_bias.view(n_tokens, n_tokens, -1).permute(2, 0, 1).contiguous()
        attn = attn + relative_bias.unsqueeze(0)
        if mask is not None:
            n_windows = mask.shape[0]
            attn = attn.view(b_windows // n_windows, n_windows, self.num_heads, n_tokens, n_tokens)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n_tokens, n_tokens)
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b_windows, n_tokens, channels)
        return self.proj(out)


class SwinTransformerBlock(nn.Module):
    """One SwinIR residual-group transformer block."""

    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        num_heads: int,
        window_size: int,
        shift_size: int,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size if min(input_resolution) > window_size else 0
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=window_size, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        if self.shift_size > 0:
            self.register_buffer("attn_mask", self.calculate_mask(input_resolution))
        else:
            self.attn_mask = None

    def calculate_mask(self, x_size: tuple[int, int]) -> torch.Tensor:
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        count = 0
        for h_slice in h_slices:
            for w_slice in w_slices:
                img_mask[:, h_slice, w_slice, :] = count
                count += 1
        mask_windows = window_partition(img_mask, self.window_size).view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        return attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    def forward(self, x: torch.Tensor, x_size: tuple[int, int]) -> torch.Tensor:
        h, w = x_size
        shortcut = x
        x = self.norm1(x).view(-1, h, w, self.dim)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, self.dim)
        if self.input_resolution == x_size:
            attn_mask = self.attn_mask
        else:
            attn_mask = self.calculate_mask(x_size).to(x.device)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.dim)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(-1, h * w, self.dim)
        x = shortcut + x
        return x + self.mlp(self.norm2(x))


class SwinIRBasicLayer(nn.Module):
    """Sequence of shifted-window blocks inside a residual SwinIR group."""

    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if idx % 2 == 0 else window_size // 2,
                    mlp_ratio=mlp_ratio,
                )
                for idx in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor, x_size: tuple[int, int]) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, x_size)
        return x


class PatchEmbed(nn.Module):
    """Flatten image features to tokens with optional normalization."""

    def __init__(self, embed_dim: int, norm_layer: type[nn.Module] | None = nn.LayerNorm) -> None:
        super().__init__()
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    """Convert tokens back to BCHW image features."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor, x_size: tuple[int, int]) -> torch.Tensor:
        b, _tokens, _channels = x.shape
        h, w = x_size
        return x.transpose(1, 2).contiguous().view(b, self.embed_dim, h, w)


class RSTB(nn.Module):
    """Residual Swin Transformer Block group used by SwinIR."""

    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        self.residual_group = SwinIRBasicLayer(dim, input_resolution, depth, num_heads, window_size, mlp_ratio)
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.patch_embed = PatchEmbed(dim, norm_layer=None)
        self.patch_unembed = PatchUnEmbed(dim)

    def forward(self, x: torch.Tensor, x_size: tuple[int, int]) -> torch.Tensor:
        residual = self.residual_group(x, x_size)
        residual = self.patch_unembed(residual, x_size)
        residual = self.conv(residual)
        residual = self.patch_embed(residual)
        return residual + x


class SwinIR(nn.Module):
    """SwinIR x2 SR model compatible with OpenMMLab generator checkpoints."""

    def __init__(
        self,
        scale: int = 2,
        in_channels: int = 3,
        out_channels: int = 3,
        img_size: int = 64,
        embed_dim: int = 180,
        depths: tuple[int, ...] = (6, 6, 6, 6, 6, 6),
        num_heads: tuple[int, ...] = (6, 6, 6, 6, 6, 6),
        window_size: int = 8,
        mlp_ratio: float = 2.0,
        upscale_channels: int = 64,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.window_size = window_size
        self.conv_first = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1)
        self.patch_embed = PatchEmbed(embed_dim)
        self.patch_unembed = PatchUnEmbed(embed_dim)
        input_resolution = (img_size, img_size)
        self.layers = nn.ModuleList(
            [
                RSTB(embed_dim, input_resolution, depth, heads, window_size, mlp_ratio)
                for depth, heads in zip(depths, num_heads)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.conv_before_upsample = nn.Sequential(
            nn.Conv2d(embed_dim, upscale_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )
        self.upsample = nn.Sequential(
            nn.Conv2d(upscale_channels, upscale_channels * scale * scale, kernel_size=3, padding=1),
            nn.PixelShuffle(scale),
        )
        self.conv_last = nn.Conv2d(upscale_channels, out_channels, kernel_size=3, padding=1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x, x_size)
        x = self.norm(x)
        return self.patch_unembed(x, x_size)

    def check_image_size(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        _, _, h, w = x.shape
        pad_h = (self.window_size - h % self.window_size) % self.window_size
        pad_w = (self.window_size - w % self.window_size) % self.window_size
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x, (h * self.scale, w * self.scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, target_size = self.check_image_size(x)
        features = self.conv_first(x)
        body = self.conv_after_body(self.forward_features(features)) + features
        out = self.conv_before_upsample(body)
        out = self.conv_last(self.upsample(out))
        return out[:, :, : target_size[0], : target_size[1]]


class ResidualDenseBlock(nn.Module):
    """Residual dense block used by RRDB/ESRGAN-style models."""

    def __init__(self, channels: int, growth_channels: int = 32, residual_scale: float = 0.2) -> None:
        super().__init__()
        self.residual_scale = residual_scale
        self.conv1 = nn.Conv2d(channels, growth_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels + growth_channels, growth_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(channels + 2 * growth_channels, growth_channels, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(channels + 3 * growth_channels, growth_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(channels + 4 * growth_channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = F.leaky_relu(self.conv1(x), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(torch.cat([x, x1], dim=1)), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(torch.cat([x, x1, x2], dim=1)), negative_slope=0.2, inplace=True)
        x4 = F.leaky_relu(self.conv4(torch.cat([x, x1, x2, x3], dim=1)), negative_slope=0.2, inplace=True)
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], dim=1))
        return x + self.residual_scale * x5


class RRDB(nn.Module):
    """Residual-in-residual dense block."""

    def __init__(self, channels: int, growth_channels: int = 32, residual_scale: float = 0.2) -> None:
        super().__init__()
        self.residual_scale = residual_scale
        self.blocks = nn.Sequential(
            ResidualDenseBlock(channels, growth_channels, residual_scale=residual_scale),
            ResidualDenseBlock(channels, growth_channels, residual_scale=residual_scale),
            ResidualDenseBlock(channels, growth_channels, residual_scale=residual_scale),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.residual_scale * self.blocks(x)


class RRDBNet(nn.Module):
    """ESRGAN/Real-ESRGAN-style RRDB network for checkpoint evaluation."""

    def __init__(
        self,
        scale: int = 2,
        in_channels: int = 1,
        out_channels: int = 1,
        channels: int = 64,
        blocks: int = 23,
        growth_channels: int = 32,
        pixel_unshuffle_scale: int = 1,
        upsample_steps: int | None = None,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.pixel_unshuffle_scale = pixel_unshuffle_scale
        conv_in_channels = in_channels * pixel_unshuffle_scale * pixel_unshuffle_scale
        self.conv_first = nn.Conv2d(conv_in_channels, channels, kernel_size=3, padding=1)
        self.trunk = nn.Sequential(*[RRDB(channels, growth_channels) for _ in range(blocks)])
        self.trunk_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        up_layers = []
        if upsample_steps is None:
            upsample_steps = 0
            remaining = scale
            while remaining > 1:
                upsample_steps += 1
                remaining //= 2
        for _ in range(upsample_steps):
            up_layers.extend(
                [
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                ]
            )
        self.upsampler = nn.Sequential(*up_layers)
        self.conv_hr = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv_last = nn.Conv2d(channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pixel_unshuffle_scale > 1:
            x = F.pixel_unshuffle(x, self.pixel_unshuffle_scale)
        features = self.conv_first(x)
        trunk = self.trunk_conv(self.trunk(features))
        features = features + trunk
        features = self.upsampler(features)
        features = F.leaky_relu(self.conv_hr(features), negative_slope=0.2, inplace=True)
        return self.conv_last(features)


class UpsampleHead(nn.Module):
    """Pixel-shuffle upsampler for power-of-two SR scales."""

    def __init__(self, channels: int, out_channels: int, scale: int) -> None:
        super().__init__()
        layers = []
        remaining = scale
        while remaining > 1:
            layers.extend(
                [
                    nn.Conv2d(channels, channels * 4, kernel_size=3, padding=1),
                    nn.PixelShuffle(2),
                    nn.ReLU(inplace=True),
                ]
            )
            remaining //= 2
        layers.append(nn.Conv2d(channels, out_channels, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_manifest(path: str | Path) -> list[SRBaselineSpec]:
    """Load baseline specs from a JSON manifest."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = doc.get("baselines", doc) if isinstance(doc, dict) else doc
    if not isinstance(entries, list):
        raise ValueError("Pretrained SR manifest must be a list or contain a 'baselines' list.")
    specs = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Each pretrained SR manifest entry must be an object.")
        kwargs = dict(entry.get("kwargs", {}))
        specs.append(
            SRBaselineSpec(
                name=str(entry["name"]),
                family=str(entry.get("family", entry["name"])),
                architecture=str(entry["architecture"]).lower(),
                checkpoint=str(entry["checkpoint"]),
                source_url=entry.get("source_url"),
                source_note=entry.get("source_note"),
                scale=int(entry.get("scale", 2)),
                input_channels=int(entry.get("input_channels", kwargs.pop("in_channels", 1))),
                output_channels=int(entry.get("output_channels", kwargs.pop("out_channels", 1))),
                state_key=entry.get("state_key"),
                strict=bool(entry.get("strict", False)),
                kwargs=kwargs,
                min_loaded_fraction=float(entry.get("min_loaded_fraction", 0.95)),
            )
        )
    return specs


def resolve_checkpoint(spec: SRBaselineSpec, manifest_path: str | Path) -> Path:
    """Resolve checkpoint relative to the manifest, then the current workspace."""
    raw = Path(spec.checkpoint)
    if raw.is_absolute():
        return raw
    manifest_relative = Path(manifest_path).resolve().parent / raw
    if manifest_relative.exists():
        return manifest_relative
    return raw


def build_model(spec: SRBaselineSpec) -> nn.Module:
    """Instantiate a supported SR architecture."""
    kwargs = dict(spec.kwargs)
    common = {
        "scale": spec.scale,
        "in_channels": spec.input_channels,
        "out_channels": spec.output_channels,
    }
    if spec.architecture == "edsr":
        return EDSR(**common, **kwargs)
    if spec.architecture == "rcan":
        return RCAN(**common, **kwargs)
    if spec.architecture == "swinir":
        return SwinIR(**common, **kwargs)
    if spec.architecture in {"rrdb", "rrdbnet", "esrgan", "realesrgan"}:
        return RRDBNet(**common, **kwargs)
    if spec.architecture == "torchscript":
        raise ValueError("TorchScript models are loaded directly from checkpoints.")
    raise ValueError(f"Unsupported pretrained SR architecture: {spec.architecture}")


def _extract_state_dict(payload: Any, state_key: str | None) -> dict[str, torch.Tensor]:
    if state_key:
        payload = payload[state_key]
    elif isinstance(payload, dict):
        for key in ("params_ema", "params", "state_dict", "model", "net", "generator"):
            value = payload.get(key)
            if isinstance(value, dict):
                payload = value
                break
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload does not contain a state dict.")
    state = {}
    for key, value in payload.items():
        if isinstance(value, torch.Tensor):
            clean_key = str(key)
            for prefix in ("module.", "model.", "net.", "generator."):
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix) :]
            clean_key = _canonical_checkpoint_key(clean_key)
            state[clean_key] = value
    if not state:
        raise ValueError("No tensor parameters found in checkpoint state dict.")
    return state


def _canonical_checkpoint_key(key: str) -> str:
    """Map common BasicSR RRDB checkpoint keys onto the local RRDBNet names."""
    replacements = {
        "conv_body.": "trunk_conv.",
        "conv_up1.": "upsampler.1.",
        "conv_up2.": "upsampler.4.",
    }
    for old, new in replacements.items():
        if key.startswith(old):
            return new + key[len(old) :]
    match = re.match(r"body\.(\d+)\.rdb([123])\.conv([1-5])\.(.+)", key)
    if match:
        block_idx, rdb_idx, conv_idx, suffix = match.groups()
        return f"trunk.{block_idx}.blocks.{int(rdb_idx) - 1}.conv{conv_idx}.{suffix}"
    match = re.match(r"head\.0\.(.+)", key)
    if match:
        return f"head.{match.group(1)}"
    match = re.match(r"body\.(\d+)\.body\.(0|2)\.(.+)", key)
    if match:
        block_idx, conv_idx, suffix = match.groups()
        return f"body.{block_idx}.net.{conv_idx}.{suffix}"
    match = re.match(r"tail\.0\.0\.(.+)", key)
    if match:
        return f"tail.net.0.{match.group(1)}"
    match = re.match(r"tail\.1\.(.+)", key)
    if match:
        return f"tail.net.3.{match.group(1)}"
    if key.startswith("generator."):
        return key[len("generator.") :]
    return key


def _compatible_state_dict(
    model_state: dict[str, torch.Tensor],
    checkpoint_state: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], int, int, float]:
    compatible = {}
    for key, value in checkpoint_state.items():
        target = model_state.get(key)
        if target is not None and tuple(target.shape) == tuple(value.shape):
            compatible[key] = value
    expected = len(model_state)
    matched = len(compatible)
    fraction = matched / expected if expected else 0.0
    return compatible, matched, expected, fraction


def load_runner(
    spec: SRBaselineSpec,
    manifest_path: str | Path,
    device: torch.device,
) -> tuple["PretrainedSRRunner | None", dict[str, Any]]:
    """Load one SR runner, returning status metadata even when loading fails."""
    checkpoint = resolve_checkpoint(spec, manifest_path)
    status: dict[str, Any] = {
        "name": spec.name,
        "family": spec.family,
        "architecture": spec.architecture,
        "checkpoint": str(checkpoint),
        "source_url": spec.source_url,
        "source_note": spec.source_note,
        "available": False,
    }
    if not checkpoint.exists():
        status["reason"] = "checkpoint_missing"
        return None, status
    try:
        if spec.architecture == "onnx":
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise RuntimeError("onnxruntime is required for ONNX pretrained SR baselines.") from exc
            session = ort.InferenceSession(str(checkpoint), providers=["CPUExecutionProvider"])
            load_info = {
                "inputs": [item.name for item in session.get_inputs()],
                "outputs": [item.name for item in session.get_outputs()],
                "input_shapes": [list(getattr(item, "shape", [])) for item in session.get_inputs()],
                "output_shapes": [list(getattr(item, "shape", [])) for item in session.get_outputs()],
                "tile_fixed_input": bool(spec.kwargs.get("tile_fixed_input", False)),
                "tile_overlap": int(spec.kwargs.get("tile_overlap", 8)),
                "providers": session.get_providers(),
            }
            status["available"] = True
            status["load_info"] = load_info
            return OnnxSRRunner(spec, session), status
        if spec.architecture == "torchscript":
            model = torch.jit.load(str(checkpoint), map_location=device)
            load_info = {"missing_keys": [], "unexpected_keys": []}
        elif spec.architecture == "spandrel":
            try:
                from spandrel import ModelLoader
            except ImportError as exc:
                raise RuntimeError("spandrel is required for this pretrained SR baseline.") from exc
            model = ModelLoader(device=device).load_from_file(checkpoint)
            load_info = {
                "architecture": type(getattr(model, "architecture", None)).__name__,
                "scale": getattr(model, "scale", None),
                "input_channels": getattr(model, "input_channels", None),
                "output_channels": getattr(model, "output_channels", None),
                "size_requirements": str(getattr(model, "size_requirements", "")),
            }
        else:
            model = build_model(spec)
            try:
                payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            except TypeError:
                payload = torch.load(checkpoint, map_location="cpu")
            state = _extract_state_dict(payload, spec.state_key)
            compatible_state, matched, expected, loaded_fraction = _compatible_state_dict(model.state_dict(), state)
            if loaded_fraction < spec.min_loaded_fraction:
                raise ValueError(
                    "Checkpoint tensor match is too low "
                    f"({matched}/{expected}, fraction={loaded_fraction:.3f}, "
                    f"required={spec.min_loaded_fraction:.3f})."
                )
            load_result = model.load_state_dict(compatible_state, strict=spec.strict)
            load_info = {
                "missing_keys": list(load_result.missing_keys),
                "unexpected_keys": list(load_result.unexpected_keys),
                "matched_tensors": matched,
                "expected_tensors": expected,
                "checkpoint_tensors": len(state),
                "loaded_fraction": loaded_fraction,
            }
        model.to(device)
        model.eval()
    except Exception as exc:  # pragma: no cover - exercised by integration runs.
        status["reason"] = f"load_failed: {exc}"
        return None, status
    status["available"] = True
    status["load_info"] = load_info
    return PretrainedSRRunner(spec, model, device), status


class PretrainedSRRunner:
    """Run a loaded SR model on single-channel benchmark crops."""

    def __init__(self, spec: SRBaselineSpec, model: nn.Module, device: torch.device) -> None:
        self.spec = spec
        self.model = model
        self.device = device

    def predict(self, lr: np.ndarray, target_shape: tuple[int, int]) -> SRPrediction:
        start = time.perf_counter()
        arr = np.asarray(lr, dtype=np.float32)
        tensor = torch.from_numpy(arr).view(1, 1, arr.shape[0], arr.shape[1]).to(self.device)
        if self.spec.input_channels == 3:
            tensor = tensor.repeat(1, 3, 1, 1)
        with torch.no_grad():
            out = self.model(tensor)
            if isinstance(out, dict):
                out = out.get("sr", next(iter(out.values())))
            if isinstance(out, (list, tuple)):
                out = out[0]
            if not isinstance(out, torch.Tensor):
                raise TypeError(f"{self.spec.name} returned unsupported output type {type(out)!r}")
            if out.ndim == 3:
                out = out.unsqueeze(0)
            if out.shape[1] > 1:
                out = out.mean(dim=1, keepdim=True)
            if out.shape[-2:] != target_shape:
                out = F.interpolate(out, size=target_shape, mode="bicubic", align_corners=False)
            sr = out[0, 0].detach().cpu().numpy().astype(np.float32)
        runtime_ms = (time.perf_counter() - start) * 1000.0
        return SRPrediction(self.spec.name, np.clip(sr, 0.0, 1.0), runtime_ms)


class OnnxSRRunner:
    """Run an ONNX SR model with configurable NHWC/NCHW tensor layout."""

    def __init__(self, spec: SRBaselineSpec, session: Any) -> None:
        self.spec = spec
        self.session = session
        self.input_name = session.get_inputs()[0].name
        self.output_name = session.get_outputs()[0].name
        self.input_layout = str(spec.kwargs.get("input_layout", "nhwc")).lower()
        self.output_layout = str(spec.kwargs.get("output_layout", "nhwc")).lower()
        self.input_shape = list(getattr(session.get_inputs()[0], "shape", []))
        self.output_shape = list(getattr(session.get_outputs()[0], "shape", []))
        self.fixed_input_hw = self._fixed_hw(self.input_shape, self.input_layout)
        self.tile_fixed_input = bool(spec.kwargs.get("tile_fixed_input", False))
        self.tile_overlap = int(spec.kwargs.get("tile_overlap", 8))

    def predict(self, lr: np.ndarray, target_shape: tuple[int, int]) -> SRPrediction:
        start = time.perf_counter()
        arr = np.asarray(lr, dtype=np.float32)
        if self._should_tile(arr.shape):
            sr = self._predict_tiled(arr, target_shape)
        else:
            sr = self._predict_whole(arr, target_shape)
        runtime_ms = (time.perf_counter() - start) * 1000.0
        return SRPrediction(self.spec.name, np.clip(sr, 0.0, 1.0), runtime_ms)

    @staticmethod
    def _fixed_hw(shape: list[Any], layout: str) -> tuple[int | None, int | None]:
        if layout == "nhwc" and len(shape) >= 3:
            dims = (shape[1], shape[2])
        elif layout == "nchw" and len(shape) >= 4:
            dims = (shape[2], shape[3])
        else:
            return (None, None)
        fixed = []
        for dim in dims:
            fixed.append(dim if isinstance(dim, int) and dim > 0 else None)
        return (fixed[0], fixed[1])

    @staticmethod
    def _tile_starts(length: int, tile: int, stride: int) -> list[int]:
        if length <= tile:
            return [0]
        starts = list(range(0, length - tile + 1, stride))
        last = length - tile
        if not starts or starts[-1] != last:
            starts.append(last)
        return starts

    def _should_tile(self, shape: tuple[int, int]) -> bool:
        if not self.tile_fixed_input:
            return False
        fixed_h, fixed_w = self.fixed_input_hw
        return fixed_h is not None and fixed_w is not None and shape != (fixed_h, fixed_w)

    def _make_input_tensor(self, arr: np.ndarray) -> np.ndarray:
        tensor = arr[None, :, :, None]
        if self.spec.input_channels == 3:
            tensor = np.repeat(tensor, 3, axis=-1)
        if self.input_layout == "nchw":
            tensor = np.transpose(tensor, (0, 3, 1, 2))
        elif self.input_layout != "nhwc":
            raise ValueError(f"Unsupported ONNX input layout: {self.input_layout}")
        return tensor.astype(np.float32, copy=False)

    def _run_array(self, arr: np.ndarray) -> np.ndarray:
        tensor = self._make_input_tensor(arr)
        outputs = self.session.run([self.output_name], {self.input_name: tensor})
        out = np.asarray(outputs[0], dtype=np.float32)
        if self.output_layout == "nchw":
            out = np.transpose(out, (0, 2, 3, 1))
        elif self.output_layout != "nhwc":
            raise ValueError(f"Unsupported ONNX output layout: {self.output_layout}")
        if out.ndim != 4:
            raise ValueError(f"Expected ONNX SR output rank 4, got shape {out.shape}.")
        if out.shape[-1] > 1:
            out = out.mean(axis=-1, keepdims=True)
        return out[0, :, :, 0]

    def _predict_whole(self, arr: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
        sr = self._run_array(arr)
        if sr.shape != target_shape:
            sr_tensor = torch.from_numpy(sr).view(1, 1, sr.shape[0], sr.shape[1])
            sr = F.interpolate(sr_tensor, size=target_shape, mode="bicubic", align_corners=False)[0, 0].numpy()
        return sr

    def _predict_tiled(self, arr: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
        tile_h, tile_w = self.fixed_input_hw
        if tile_h is None or tile_w is None:
            raise ValueError("ONNX fixed-input tiling requires fixed spatial input dimensions.")
        height, width = arr.shape
        padded_h = max(height, tile_h)
        padded_w = max(width, tile_w)
        pad_h = padded_h - height
        pad_w = padded_w - width
        padded = np.pad(arr, ((0, pad_h), (0, pad_w)), mode="edge")
        overlap = max(0, min(self.tile_overlap, tile_h - 1, tile_w - 1))
        stride_h = max(1, tile_h - overlap)
        stride_w = max(1, tile_w - overlap)
        y_starts = self._tile_starts(padded_h, tile_h, stride_h)
        x_starts = self._tile_starts(padded_w, tile_w, stride_w)
        first = self._run_array(padded[y_starts[0] : y_starts[0] + tile_h, x_starts[0] : x_starts[0] + tile_w])
        scale_h = first.shape[0] / float(tile_h)
        scale_w = first.shape[1] / float(tile_w)
        canvas_h = int(round(padded_h * scale_h))
        canvas_w = int(round(padded_w * scale_w))
        canvas = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        weight = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        for y in y_starts:
            for x in x_starts:
                tile = padded[y : y + tile_h, x : x + tile_w]
                sr_tile = first if y == y_starts[0] and x == x_starts[0] else self._run_array(tile)
                y0 = int(round(y * scale_h))
                x0 = int(round(x * scale_w))
                y1 = min(canvas_h, y0 + sr_tile.shape[0])
                x1 = min(canvas_w, x0 + sr_tile.shape[1])
                canvas[y0:y1, x0:x1] += sr_tile[: y1 - y0, : x1 - x0]
                weight[y0:y1, x0:x1] += 1.0
        if np.any(weight <= 0.0):
            raise RuntimeError("ONNX tiled inference left uncovered output pixels.")
        sr = canvas / weight
        crop_h = int(round(height * scale_h))
        crop_w = int(round(width * scale_w))
        sr = sr[:crop_h, :crop_w]
        if sr.shape != target_shape:
            sr_tensor = torch.from_numpy(sr).view(1, 1, sr.shape[0], sr.shape[1])
            sr = F.interpolate(sr_tensor, size=target_shape, mode="bicubic", align_corners=False)[0, 0].numpy()
        return sr
