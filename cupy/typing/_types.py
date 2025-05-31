from __future__ import annotations

from typing import Any, TypeVar

import numpy
from numpy.typing import DTypeLike, NBitBase  # noqa: F401

from cupy._core import core
from cupy.typing._internal import _ScalarT
from cupy.typing._proto import _Buffer, _NestedSequence, _SupportsArray

# numpy.typing.ArrayLike minus str/bytes
ArrayLike = (
    _Buffer
    | complex
    | _NestedSequence[complex]
    | _SupportsArray[numpy.dtype[Any]]
    | _NestedSequence[_SupportsArray[numpy.dtype[Any]]]
)
NDArray = core.ndarray[Any, numpy.dtype[_ScalarT]]

_ArrayT = TypeVar("_ArrayT", bound=core.ndarray)
_NumpyArrayT = TypeVar("_NumpyArrayT", bound=numpy.ndarray)
_IntArrayT = TypeVar("_IntArrayT", bound=NDArray[numpy.integer])
_NumericArrayT = TypeVar("_NumericArrayT", bound=NDArray[numpy.number])
_RealArrayT = TypeVar(
    "_RealArrayT", bound=NDArray[numpy.floating | numpy.integer | numpy.bool]
)
_IntegralArrayT = TypeVar(
    "_IntegralArrayT", bound=NDArray[numpy.integer | numpy.bool]
)
