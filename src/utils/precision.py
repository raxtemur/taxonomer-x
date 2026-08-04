"""One precision policy shared by training, caching, and inference."""

from __future__ import annotations

from contextlib import nullcontext

import torch


PRECISIONS = ("auto", "fp32", "bf16", "fp16")


def resolve_precision(device: torch.device, precision: str = "auto") -> torch.dtype | None:
    if precision not in PRECISIONS: raise ValueError(f"Unknown precision {precision!r}")
    if device.type != "cuda" or precision == "fp32": return None
    if precision == "bf16": return torch.bfloat16
    if precision == "fp16": return torch.float16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def autocast_context(device: torch.device, precision: str = "auto"):
    dtype = resolve_precision(device, precision)
    return torch.autocast(device_type="cuda", dtype=dtype) if dtype is not None else nullcontext()


def uses_fp16(device: torch.device, precision: str = "auto") -> bool:
    return resolve_precision(device, precision) == torch.float16
