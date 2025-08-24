"""Type utilities dependent on cupy."""

from __future__ import annotations

import typing
from types import EllipsisType
from typing import TYPE_CHECKING, Any, Protocol, SupportsIndex

import numpy

from cupy._core import core
from cupy.typing._standalone import (
    _T,
    _DTypeT,
    _DTypeT_co,
    _NestedSequence,
    _ScalarT,
)

if TYPE_CHECKING:
    from typing_extensions import TypeAlias


@typing.runtime_checkable
class _SupportsArrayD(Protocol[_DTypeT_co]):
    # Only covers default signature (no optional args provided)
    def __array__(self) -> core.ndarray[Any, _DTypeT_co]: ...


_ArrayLike: TypeAlias = (
    _SupportsArrayD[numpy.dtype[_ScalarT]]
    | _NestedSequence[_SupportsArrayD[numpy.dtype[_ScalarT]]]
)
_DualArrayLike: TypeAlias = (
    _SupportsArrayD[_DTypeT]
    | _NestedSequence[_SupportsArrayD[_DTypeT]]
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
_ToIndex = SupportsIndex | slice | EllipsisType | _ArrayLikeInt_co | None
_ToIndices = _ToIndex | tuple[_ToIndex, ...]
