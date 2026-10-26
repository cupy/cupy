"""Backend resolver for the FFT library binding.

``cupy.fft._fft`` must work on both the CUDA backend (cuFFT via
``cupy.cuda.cufft``) and the Ascend backend (ops-fft / aclfft via
``cupy.backends.ascend.api.aclfft``). This module resolves the right binding
lazily -- at the first FFT call, not at import time -- so that ``import cupy``
never fails just because an *optional* FFT library is absent, and exposes
capability flags so ``_fft.py`` can degrade gracefully on backends with a
smaller feature set (see ``docs/ascend/ascend_fft.md``).

The Ascend binding is only compiled when ``libcann_ops_fft.so`` was found at
build time (see ``install/cupy_builder/features/ascend_fft.py``).
"""
from __future__ import annotations

import functools

import numpy as np

__all__ = ['get_cufft', 'is_hip', 'supports_nd_plan', 'supports_callbacks']


# ---------------------------------------------------------------------------
# Capability flags of the currently resolved backend. They are updated by
# :func:`get_cufft` on first use. Defaults describe the CUDA/cuFFT backend.
# ---------------------------------------------------------------------------
supports_nd_plan = True
"""Whether N-D plans with arbitrary strides are supported (cuFFT:
``cufftMakePlanMany``). aclfft: ``False`` -> ``cupy.fft`` degrades to
repeated 1-D transforms."""

supports_callbacks = True
"""Whether cuFFT-style callbacks are supported (CUDA/Linux only)."""


def _detect():
    """Return the FFT binding module, or ``None`` if not available."""
    # Ascend first: on an Ascend build ``cupy.cuda.cufft`` is not compiled,
    # while on a CUDA build the ``cupy.backends.ascend.api.aclfft`` module is
    # not compiled, so the try order is unambiguous.
    try:
        from cupy.backends.ascend.api import aclfft
        return aclfft
    except ImportError:
        pass
    try:
        from cupy.cuda import cufft
        return cufft
    except ImportError:
        return None


@functools.lru_cache(maxsize=None)
def get_cufft():
    """Return the backend FFT binding module.

    Raises ``RuntimeError`` with an actionable message if FFT support was not
    built. Call this lazily from inside functions (mirroring the upstream
    ``from cupy.cuda import cufft`` pattern) so that importing ``cupy.fft``
    stays possible even without FFT support.
    """
    global supports_nd_plan, supports_callbacks
    cufft = _detect()
    if cufft is None:
        raise RuntimeError(
            'FFT support is not available in this build.\n'
            '- CUDA backend: cupy.cuda.cufft was not compiled.\n'
            '- Ascend backend: ops-fft (libcann_ops_fft.so) was not found at '
            'build time; install it or set ASCEND_OPS_FFT_PATH, then rebuild '
            '(see docs/ascend/ascend_fft.md).')
    # record capabilities for _fft.py
    supports_nd_plan = getattr(cufft, 'supports_nd_plan', True)
    supports_callbacks = getattr(cufft, 'supports_callbacks', True)
    return cufft


def is_hip() -> bool:
    """Backend-agnostic replacement for ``cupy.cuda.runtime.is_hip``.

    ``cupy.cuda`` is not importable on the Ascend build, so FFT code must not
    reference it directly.
    """
    try:
        from cupy_backends.cuda.api import runtime
        return bool(runtime.is_hip)
    except ImportError:
        pass
    try:
        from cupy.backends.backend.api import runtime
        return bool(runtime.is_hip)
    except ImportError:
        return False
