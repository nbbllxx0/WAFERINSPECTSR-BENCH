"""PyTorch device selection helpers."""

from __future__ import annotations

import torch


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve auto/cuda/cpu with an explicit error for unavailable CUDA."""
    normalized = requested.lower().strip()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but this Python environment has a CPU-only "
                f"PyTorch build ({torch.__version__}). Install a CUDA-enabled "
                "PyTorch build or use --device cpu."
            )
        return torch.device(normalized)
    if normalized == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device: {requested}. Use auto, cuda, cuda:0, or cpu.")


def device_report(device: torch.device) -> dict[str, object]:
    """Return a JSON-serializable runtime device report."""
    report: dict[str, object] = {
        "requested_runtime": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
    }
    if device.type == "cuda":
        report["cuda_device_name"] = torch.cuda.get_device_name(device)
        report["cuda_current_device"] = torch.cuda.current_device()
    return report
