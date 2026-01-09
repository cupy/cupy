"""Type utilities not dependent on cupy."""

from __future__ import annotations

import sys
import typing
from collections.abc import Iterator, Sequence
from typing import (
    Any,
    Literal,
    Protocol,
    SupportsIndex,
    TypeVar,
)

import numpy

if sys.version_info >= (3, 12):
    from collections.abc import Buffer as _Buffer
else:

    @typing.runtime_checkable
    class _Buffer(Protocol):
        def __buffer__(self, flags: int, /) -> memoryview: ...


class _SupportsFileMethods(Protocol):
    def flush(self) -> object: ...
    def fileno(self) -> SupportsIndex: ...
    def tell(self) -> SupportsIndex: ...
    def seek(self, offset: int, whence: int, /) -> object: ...


_T = TypeVar("_T")
_T_co = TypeVar("_T_co", covariant=True)


@typing.runtime_checkable
class _NestedSequence(Protocol[_T_co]):
    # Required methods to be collections.abc.Sequence
    def __getitem__(self, index: int, /) -> _T_co | _NestedSequence[_T_co]:
        raise NotImplementedError

    def __len__(self, /) -> int:
        raise NotImplementedError

    # Methods can be derived by collections.abc.Sequence
    def __contains__(self, x: object, /) -> bool:
        raise NotImplementedError

    def __iter__(self, /) -> Iterator[_T_co | _NestedSequence[_T_co]]:
        raise NotImplementedError

    def __reversed__(self, /) -> Iterator[_T_co | _NestedSequence[_T_co]]:
        raise NotImplementedError

    def count(self, value: Any, /) -> int:
        raise NotImplementedError

    def index(self, value: Any, /) -> int:
        raise NotImplementedError


# Miscellaneous types
_ScalarT = TypeVar("_ScalarT", bound=numpy.generic)
_ScalarLike_co = complex | numpy.generic
_DTypeT_co = TypeVar("_DTypeT_co", bound=numpy.dtype, covariant=True)
_ShapeT_co = TypeVar("_ShapeT_co", bound=tuple[int, ...], covariant=True)
_DTypeT = TypeVar("_DTypeT", bound=numpy.dtype)
_ShapeLike = SupportsIndex | Sequence[SupportsIndex]
_NonNullKACF = Literal["K", "A", "C", "F"]
_NonNullCAF = Literal["A", "C", "F"]
_NonNullCF = Literal["C", "F"]
_OrderKACF = _NonNullKACF | None
_OrderCAF = _NonNullCAF | None
_OrderCF = _NonNullCF | None
_DTypeLike = type[_ScalarT] | numpy.dtype[_ScalarT]
_ModeKind = Literal["raise", "wrap", "clip"]
_SortSide = Literal["left", "right"]
_NumpyArrayT = TypeVar("_NumpyArrayT", bound=numpy.ndarray)

# Typed scalars
_FloatT = TypeVar("_FloatT", bound=numpy.floating)
_InexactT = TypeVar("_InexactT", bound=numpy.inexact)
