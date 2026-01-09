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
    TypeAlias,
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
_NumpyArrayT = TypeVar("_NumpyArrayT", bound=numpy.ndarray)
_ScalarT = TypeVar("_ScalarT", bound=numpy.generic)
_DTypeT_co = TypeVar("_DTypeT_co", bound=numpy.dtype, covariant=True)
_ShapeT_co = TypeVar("_ShapeT_co", bound=tuple[int, ...], covariant=True)
_DTypeT = TypeVar("_DTypeT", bound=numpy.dtype)

_ScalarLike_co: TypeAlias = complex | numpy.generic
_ShapeLike: TypeAlias = SupportsIndex | Sequence[SupportsIndex]
_NonNullKACF: TypeAlias = Literal["K", "A", "C", "F"]
_NonNullCAF: TypeAlias = Literal["A", "C", "F"]
_NonNullCF: TypeAlias = Literal["C", "F"]
_OrderKACF: TypeAlias = _NonNullKACF | None
_OrderCAF: TypeAlias = _NonNullCAF | None
_OrderCF: TypeAlias = _NonNullCF | None
_DTypeLike: TypeAlias = type[_ScalarT] | numpy.dtype[_ScalarT]
_ModeKind: TypeAlias = Literal["raise", "wrap", "clip"]
_SortSide: TypeAlias = Literal["left", "right"]

# Typed scalars
_FloatT = TypeVar("_FloatT", bound=numpy.floating)
_InexactT = TypeVar("_InexactT", bound=numpy.inexact)
