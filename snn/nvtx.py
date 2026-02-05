"""NVTX helpers with safe fallbacks."""

from __future__ import annotations

from contextlib import contextmanager

try:
    import torch
except Exception:  # pragma: no cover - torch may be unavailable in some environments
    torch = None

try:
    from torch.profiler import record_function as _record_function
except Exception:  # pragma: no cover - torch.profiler may be unavailable / older torch
    try:
        from torch.autograd.profiler import record_function as _record_function
    except Exception:  # pragma: no cover
        _record_function = None


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


def _is_compiling() -> bool:
    if torch is None:
        return False
    try:
        return torch.compiler.is_compiling()
    except Exception:
        try:
            return torch._dynamo.is_compiling()
        except Exception:
            return False


@contextmanager
def nvtx_range(message: str):
    if _is_compiling():
        yield
        return
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


@contextmanager
def profiler_range(message: str, enabled: bool = True):
    """Record a named region for torch.profiler (no-op unless enabled)."""
    if not enabled or torch is None or _record_function is None:
        yield
        return
    if _is_compiling():
        yield
        return
    ctx = None
    try:
        ctx = _record_function(message)
    except Exception:
        ctx = None
    if ctx is None:
        yield
        return
    with ctx:
        yield
