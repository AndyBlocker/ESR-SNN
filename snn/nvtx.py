"""NVTX helpers with safe fallbacks."""

from __future__ import annotations

from contextlib import contextmanager

try:
    import torch
except Exception:  # pragma: no cover - torch may be unavailable in some environments
    torch = None


def _is_nvtx_available() -> bool:
    if torch is None:
        return False
    if not hasattr(torch, "cuda"):
        return False
    if not hasattr(torch.cuda, "nvtx"):
        return False
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


_NVTX_AVAILABLE = None


def _nvtx_available() -> bool:
    global _NVTX_AVAILABLE
    if _NVTX_AVAILABLE is None:
        _NVTX_AVAILABLE = _is_nvtx_available()
    return _NVTX_AVAILABLE


@contextmanager
def nvtx_range(message: str):
    pushed = False
    if _nvtx_available():
        try:
            torch.cuda.nvtx.range_push(message)
            pushed = True
        except Exception:
            pushed = False
    try:
        yield
    finally:
        if pushed:
            try:
                torch.cuda.nvtx.range_pop()
            except Exception:
                pass
