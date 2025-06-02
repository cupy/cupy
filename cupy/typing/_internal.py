"""Type utilities dependent on cupy."""

from __future__ import annotations

import typing
from types import EllipsisType
from typing import TYPE_CHECKING, Any, Protocol

import numpy

from cupy._core import core
from cupy.typing._standalone import (
    _T,
    _DTypeT,
    _DTypeT_co,
    _Index,
    _NestedSequence,
)

if TYPE_CHECKING:
    from typing_extensions import TypeAlias


@typing.runtime_checkable
class _SupportsArray(Protocol[_DTypeT_co]):
    # Only covers default signature (no optional args provided)
    def __array__(self) -> core.ndarray[Any, _DTypeT_co]: ...


_DualArrayLike: TypeAlias = (
    _SupportsArray[_DTypeT]
    | _NestedSequence[_SupportsArray[_DTypeT]]
    | _T
    | _NestedSequence[_T]
)

# Anything castable to ndarray with specified dtype (work in progress)
_ArrayLikeBool_co = _DualArrayLike[numpy.dtype[numpy.bool], bool]
_ArrayLikeUInt_co = _DualArrayLike[
    numpy.dtype[numpy.bool | numpy.unsignedinteger], bool
]
_ArrayLikeInt_co = _DualArrayLike[numpy.dtype[numpy.bool | numpy.integer], int]
_ArrayLikeFloat_co = _DualArrayLike[
    numpy.dtype[numpy.bool | numpy.integer | numpy.floating], float
]
_ArrayLikeComplex_co = _DualArrayLike[
    numpy.dtype[numpy.bool | numpy.number], complex
]
_ArrayLikeNumber_co = _ArrayLikeComplex_co

# Index-like
_ToIndex = _Index | slice | EllipsisType | _ArrayLikeInt_co | None
_ToIndices = _ToIndex | tuple[_ToIndex, ...]
