from __future__ import annotations

from typing import Any, Protocol, TypeVar, overload

import numpy
import typing

from cupy._core import core
from cupy.typing._internal import _DualArrayLike
from cupy.typing._standalone import _Buffer, _ScalarT

# numpy.typing.ArrayLike minus str/bytes
ArrayLike = _Buffer | _DualArrayLike[numpy.dtype[Any], complex]
NDArray = core.ndarray[Any, numpy.dtype[_ScalarT]]

_ArrayT = TypeVar("_ArrayT", bound=core.ndarray)
_ArrayT_co = TypeVar("_ArrayT_co", bound=core.ndarray, covariant=True)
_IntArrayT = TypeVar("_IntArrayT", bound=NDArray[numpy.integer])
_NumericArrayT = TypeVar("_NumericArrayT", bound=NDArray[numpy.number])
_RealArrayT = TypeVar(
    "_RealArrayT", bound=NDArray[numpy.floating | numpy.integer | numpy.bool]
)
_IntegralArrayT = TypeVar(
    "_IntegralArrayT", bound=NDArray[numpy.integer | numpy.bool]
)

_ArrayUInt_co = NDArray[numpy.unsignedinteger | numpy.bool]
_ArrayInt_co = NDArray[numpy.integer | numpy.bool]
_ArrayFloat_co = NDArray[numpy.floating | numpy.integer | numpy.bool]
_ArrayComplex_co = NDArray[numpy.inexact | numpy.integer | numpy.bool]


class _SupportsRealImag(Protocol):
    @overload
    def __get__(
        self, instance: _RealArrayT, owner: type | None = ...
    ) -> _RealArrayT: ...

    @overload
    def __get__(
        self, instance: NDArray[numpy.complex64], owner: type | None = ...
    ) -> NDArray[numpy.float32]: ...

    @overload
    def __get__(
        self, instance: NDArray[numpy.complex128], owner: type | None = ...
    ) -> NDArray[numpy.float64]: ...

    @overload
    def __get__(
        self, instance: None, owner: type | None = ...
    ) -> _SupportsRealImag: ...
    def __set__(self, instance: NDArray[Any], value: ArrayLike) -> None: ...


@typing.runtime_checkable
class _SupportsArrayA(Protocol[_ArrayT_co]):
    def __array__(self) -> _ArrayT_co: ...
