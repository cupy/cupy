"""
cuDNN Wrapper

Use `cupy_backends.cuda.libs.cudnn` directly in CuPy codebase.
"""

available = True

try:
    from cupy_backends.cuda.libs.cudnn import *  # NOQA
except ImportError as e:
    available = False
    from cupy._environment import _preload_warning
    _preload_warning('cudnn', e)
