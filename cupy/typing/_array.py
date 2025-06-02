from __future__ import annotations

from typing import Any, TypeVar

import numpy

from cupy._core import core
from cupy.typing._internal import _DualArrayLike
from cupy.typing._standalone import _Buffer, _ScalarT

# numpy.typing.ArrayLike minus str/bytes
ArrayLike = _Buffer | _DualArrayLike[numpy.dtype[Any], complex]
NDArray = core.ndarray[Any, numpy.dtype[_ScalarT]]

_ArrayT = TypeVar("_ArrayT", bound=core.ndarray)
_IntArrayT = TypeVar("_IntArrayT", bound=NDArray[numpy.integer])
_NumericArrayT = TypeVar("_NumericArrayT", bound=NDArray[numpy.number])
_RealArrayT = TypeVar(
    "_RealArrayT", bound=NDArray[numpy.floating | numpy.integer | numpy.bool]
)
_IntegralArrayT = TypeVar(
    "_IntegralArrayT", bound=NDArray[numpy.integer | numpy.bool]
)
