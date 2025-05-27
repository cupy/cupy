from __future__ import annotations

import numpy
from typing import Any, Literal, Sequence, SupportsIndex, TypeVar
from cupy._core import core

# MEMO
# Protocol types are currently not included for simplicity

# Miscellaneous types
_Index = int  # MEMO: SupportsIndex in numpy
_ScalarT = TypeVar("_ScalarT", bound=numpy.generic)
_DTypeT_co = TypeVar("_DTypeT_co", bound=numpy.dtype, covariant=True)
_DTypeT = TypeVar("_DTypeT", bound=numpy.dtype)
_ShapeLike = SupportsIndex | Sequence[SupportsIndex]
_OrderKACF = Literal["K", "A", "C", "F"] | None
_DTypeLike = type[_ScalarT] | numpy.dtype[_ScalarT]
_ArrayT = TypeVar("_ArrayT", bound=core.ndarray)
_ModeKind = Literal["raise", "wrap", "clip"]
_SortSide = Literal["left", "right"]

# Anything castable to ndarray with specified dtype (work in progress)
_ArrayLikeInt_co = Any
_ArrayLikeInt = Any
