import sys
from typing import Any, TypeVar

import numpy

import cupy


if numpy.lib.NumpyVersion(numpy.__version__) >= '1.20.0':
    from numpy.typing import ArrayLike  # NOQA
    from numpy.typing import DTypeLike  # NOQA
    from numpy.typing import NBitBase  # NOQA
else:
    ArrayLike = Any  # type: ignore
    DTypeLike = Any  # type: ignore
    NBitBase = Any  # type: ignore


if sys.version_info >= (3, 9):
    from types import GenericAlias
else:
    try:
        # NumPy 1.23+
        import numpy._typing as _numpy_typing
    except Exception:
        try:
            # NumPy 1.21 & 1.22
            import numpy.typing as _numpy_typing  # type: ignore
        except Exception:
            _numpy_typing = None  # type: ignore

    _GenericAlias = getattr(_numpy_typing, '_GenericAlias', None)

    if _GenericAlias is not None:
        GenericAlias = _GenericAlias
    else:
        def GenericAlias(*args):
            return Any


_ScalarType = TypeVar('_ScalarType', bound=numpy.generic, covariant=True)
_DType = GenericAlias(numpy.dtype, (_ScalarType,))
NDArray = GenericAlias(cupy.ndarray, (Any, _DType))
