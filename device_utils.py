from __future__ import annotations

import sys

import torch


def _mps_available() -> bool:
    mps_backend = getattr(torch.backends, "mps", None)
    return bool(mps_backend and mps_backend.is_available())


def best_available_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if _mps_available():
        return "mps"
    return "cpu"


def resolve_device(requested: str | None, *, warn: bool = True) -> str:
    raw = (requested or "auto").strip()
    lowered = raw.lower()

    if lowered == "auto":
        resolved = best_available_device()
        if warn:
            print(f"[DEVICE] Using {resolved}", file=sys.stderr)
        return resolved

    if lowered.startswith("cuda"):
        if torch.cuda.is_available():
            return raw
        fallback = "mps" if _mps_available() else "cpu"
        if warn:
            print(
                f"[DEVICE] Requested {raw!r} but CUDA is unavailable; falling back to {fallback!r}.",
                file=sys.stderr,
            )
        return fallback

    if lowered.startswith("mps"):
        if _mps_available():
            return "mps"
        fallback = "cuda" if torch.cuda.is_available() else "cpu"
        if warn:
            print(
                f"[DEVICE] Requested {raw!r} but MPS is unavailable; falling back to {fallback!r}.",
                file=sys.stderr,
            )
        return fallback

    if lowered == "cpu":
        return "cpu"

    return raw


def recommended_sam_device(base_device: str, *, warn: bool = True) -> str:
    resolved_base = resolve_device(base_device, warn=False)
    if resolved_base == "mps":
        if warn:
            print(
                "[DEVICE] Using 'cpu' for SAM on Apple MPS to avoid out-of-memory.",
                file=sys.stderr,
            )
        return "cpu"
    return resolved_base


def clear_device_cache(device: str) -> None:
    lowered = device.lower()
    if lowered.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        return

    mps_module = getattr(torch, "mps", None)
    if lowered.startswith("mps") and mps_module is not None and hasattr(mps_module, "empty_cache"):
        mps_module.empty_cache()
