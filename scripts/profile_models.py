from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waferinspectsr.models import (
    CompactASPPDetector,
    CompactInspectionSafeSR,
    CompactNAFSR,
    CompactSRLite,
    CompactUNetDetector,
)

try:
    from train_detector_baseline import TorchvisionDeepLabDetector
except Exception:  # pragma: no cover - optional torchvision path
    TorchvisionDeepLabDetector = None  # type: ignore[assignment]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile v1 model parameter counts and approximate convolution MACs.")
    parser.add_argument("--lr-height", type=int, default=128)
    parser.add_argument("--lr-width", type=int, default=128)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--output-md", default="paper/tables/model_profile.md")
    parser.add_argument("--output-csv", default="paper/tables/model_profile.csv")
    parser.add_argument("--include-deeplabv3", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    return parser.parse_args()


def parameter_count(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def conv_macs(module: nn.Module, output: torch.Tensor) -> int:
    if not isinstance(module, nn.Conv2d):
        return 0
    batch, out_channels, out_h, out_w = output.shape
    in_channels = module.in_channels
    kernel_h, kernel_w = module.kernel_size
    groups = module.groups
    return int(batch * out_h * out_w * out_channels * (in_channels // groups) * kernel_h * kernel_w)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def profile_model(model: nn.Module, lr_shape: tuple[int, int, int, int], device: torch.device) -> tuple[int, int]:
    macs = 0
    hooks = []

    def hook(module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        nonlocal macs
        if isinstance(output, torch.Tensor):
            macs += conv_macs(module, output)

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(hook))
    model.to(device)
    model.eval()
    with torch.no_grad():
        model(torch.zeros(lr_shape, dtype=torch.float32, device=device))
    for handle in hooks:
        handle.remove()
    return parameter_count(model), macs


def write_outputs(rows: list[dict[str, object]], md_path: Path, csv_path: Path) -> None:
    md_path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["Model", "Role", "Parameters", "Approx MACs", "Input", "Output"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row[header]) for header in headers) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    lr_shape = (1, 1, args.lr_height, args.lr_width)
    hr_shape = (args.lr_height * args.scale, args.lr_width * args.scale)
    specs = [
        ("SR-lite", "residual neural SR baseline", CompactSRLite(scale=args.scale, channels=24, blocks=2)),
        ("NAFNet-lite", "recognized SR-style neural baseline", CompactNAFSR(scale=args.scale, channels=24, blocks=2)),
        ("U-Net detector", "detector-only learned baseline", CompactUNetDetector(scale=args.scale, channels=16)),
        ("ASPP detector", "dilated-context detector baseline", CompactASPPDetector(scale=args.scale, channels=24)),
        (
            "DPU-WaferSR",
            "proof-of-concept SR/defect/risk probe",
            CompactInspectionSafeSR(scale=args.scale, channels=24, blocks=2, dropout=0.05),
        ),
    ]
    if args.include_deeplabv3 and TorchvisionDeepLabDetector is not None:
        specs.insert(
            -1,
            (
                "DeepLabV3 detector",
                "high-capacity semantic detector",
                TorchvisionDeepLabDetector(scale=args.scale),
            ),
        )
    rows = []
    for name, role, model in specs:
        params, macs = profile_model(model, lr_shape, device)
        rows.append(
            {
                "Model": name,
                "Role": role,
                "Parameters": params,
                "Approx MACs": macs,
                "Input": f"{args.lr_height}x{args.lr_width}",
                "Output": f"{hr_shape[0]}x{hr_shape[1]}",
            }
        )
    write_outputs(rows, Path(args.output_md), Path(args.output_csv))
    print(f"wrote {args.output_md}")
    print(f"wrote {args.output_csv}")


if __name__ == "__main__":
    main()
