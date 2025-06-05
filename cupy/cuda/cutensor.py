"""
cuTENSOR Wrapper

Use `cupy_backends.cuda.libs.cutensor` directly in CuPy codebase.
"""
from __future__ import annotations


available = True

try:
    from cupy_backends.cuda.libs.cutensor import *  # NOQA
except ImportError as e:
    available = False
    from cupy._environment import _preload_warning
    _preload_warning('cutensor', e)
