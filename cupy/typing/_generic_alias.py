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
elif numpy.lib.NumpyVersion(numpy.__version__) >= '1.21.0':
    from numpy.typing import _GenericAlias as GenericAlias
else:
    def GenericAlias(*args):  # type: ignore
        return Any


_ScalarType = TypeVar('_ScalarType', bound=numpy.generic, covariant=True)
_DType = GenericAlias(numpy.dtype, (_ScalarType,))
NDArray = GenericAlias(cupy.ndarray, (Any, _DType))
