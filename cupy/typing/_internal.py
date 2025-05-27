from __future__ import annotations

import numpy
from typing import Literal, Sequence, SupportsIndex, TypeVar
from cupy._core import core

_ScalarT = TypeVar("_ScalarT", bound=numpy.generic)
_ShapeT_co = TypeVar("_ShapeT_co", bound=tuple[int, ...], covariant=True)
_DTypeT_co = TypeVar("_DTypeT_co", bound=numpy.dtype, covariant=True)
_DTypeT = TypeVar("_DTypeT", bound=numpy.dtype)
_ShapeLike = SupportsIndex | Sequence[SupportsIndex]
_OrderKACF = Literal["K", "A", "C", "F"] | None
_DTypeLike = type[_ScalarT] | numpy.dtype[_ScalarT]
_ArrayT = TypeVar("_ArrayT", bound=core.ndarray)
