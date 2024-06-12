"""
NCCL Wrapper

Use `cupy_backends.cuda.libs.nccl` directly in CuPy codebase.
"""

from cupy import _environment


available = True

try:
    _environment._preload_library('nccl')
    from cupy_backends.cuda.libs.nccl import *  # NOQA
except ImportError as e:
    available = False
    _environment._preload_warning('nccl', e)
