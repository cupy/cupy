"""
cuDNN Wrapper

Use `cupy_backends.cuda.libs.cudnn` directly in CuPy codebase.
"""

from cupy import _environment


available = True

try:
    _environment._preload_library('cudnn')
    from cupy_backends.cuda.libs.cudnn import *  # NOQA
except ImportError as e:
    available = False
    _environment._preload_warning('cudnn', e)
