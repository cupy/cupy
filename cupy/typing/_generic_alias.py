import sys
from typing import Any, TypeVar

import numpy

import cupy


if sys.version_info >= (3, 9):
    from types import GenericAlias
elif numpy.lib.NumpyVersion(numpy.__version__) >= '1.21.0':
    from numpy.typing import _GenericAlias as GenericAlias
else:
    raise ImportError(
        'cupy.ndarray.__class_getitem__ requires Python>=3.9 or numpy>=1.21.')


_ScalarType = TypeVar("ScalarType", bound=numpy.generic, covariant=True)
_DType = GenericAlias(numpy.dtype, (_ScalarType))
NDArray = GenericAlias(cupy.ndarray, (Any, _DType))
