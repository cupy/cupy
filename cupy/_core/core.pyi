from collections.abc import Iterator
from typing import Any, Generic, Literal

from _typeshed import StrOrBytesPath, SupportsWrite
from typing import overload
import numpy
from typing_extensions import Self

from cupy import dtype
from cupy._core.flags import Flags
from cupy.typing._types import NDArray, DTypeLike, ArrayLike
from cupy.typing._internal import (
    _ArrayLikeInt_co,
    _SortSide,
    _ArrayLikeInt,
    _DTypeT_co,
    _Index,
    _OrderKACF,
    _DTypeLike,
    _ScalarT,
    _ShapeLike,
    _ArrayT,
    _ModeKind,
)

class ndarray(Generic[_DTypeT_co]):
    def __cuda_array_interface__(self) -> dict[str, Any]: ...
    def is_host_accessible(self) -> bool: ...
    @property
    def flags(self) -> Flags: ...
    @property
    def shape(self) -> tuple[int, ...]: ...
    @shape.setter
    def shape(self, newshape: _ShapeLike) -> None: ...
    def strides(self) -> tuple[int, ...]: ...
    def ndim(self) -> int: ...
    def itemsize(self) -> int: ...
    def nbytes(self) -> int: ...
    @property
    def T(self) -> Self: ...
    @property
    def mT(self) -> Self: ...
    @property
    def flat(self: NDArray[_ScalarT]) -> Iterator[_ScalarT]: ...
    def item(self: NDArray[_ScalarT]) -> _ScalarT: ...
    def tolist(self: NDArray[_ScalarT]) -> list[Any]: ...
    def tobytes(self, order: _OrderKACF = ...) -> bytes: ...
    def tofile(
        self, fid: StrOrBytesPath, sep: str = ..., format: str = ...
    ) -> None: ...
    def dump(self, file: StrOrBytesPath | SupportsWrite[bytes]) -> None: ...
    def dumps(self) -> bytes: ...
    @overload
    def astype(
        self,
        dtype: _DTypeLike[_ScalarT],
        order: _OrderKACF = ...,
        copy: bool = ...,
    ) -> ndarray[dtype[_ScalarT]]: ...
    @overload
    def astype(
        self,
        dtype: DTypeLike,
        order: _OrderKACF = ...,
        copy: bool = ...,
    ) -> ndarray[dtype]: ...
    def copy(self, order: _OrderKACF = ...) -> Self: ...
    @overload
    def view(self) -> Self: ...
    @overload
    def view(self, dtype: DTypeLike) -> ndarray[dtype]: ...
    @overload
    def view(
        self, dtype: _DTypeLike[_ScalarT]
    ) -> ndarray[dtype[_ScalarT]]: ...
    @overload
    def view(self, type: type[_ArrayT]) -> _ArrayT: ...
    @overload
    def view(self, dtype: DTypeLike, type: type[_ArrayT]) -> _ArrayT: ...
    def fill(self, value: Any) -> None: ...
    def reshape(self, *shape: _Index, order: _OrderKACF = ...) -> Self: ...
    def transpose(self, *axes: _Index) -> Self: ...
    def swapaxes(self, axis1: _Index, axis2: _Index) -> Self: ...
    def flatten(self, order: _OrderKACF = ...) -> Self: ...
    def squeeze(
        self,
        axis: _Index | tuple[_Index, ...] | None = ...,
    ) -> Self: ...
    @overload
    def take(
        self: NDArray[_ScalarT],
        indices: int | numpy.integer,
        axis: _Index | None = ...,
        out: None = ...,
    ) -> _ScalarT: ...
    @overload
    def take(
        self,
        indices: _ArrayLikeInt_co,
        axis: _Index | None = ...,
        out: None = ...,
    ) -> Self: ...
    @overload
    def take(
        self,
        indices: _ArrayLikeInt_co,
        axis: _Index | None = ...,
        *out: _ArrayT,
    ) -> _ArrayT: ...
    def put(
        self,
        indices: _ArrayLikeInt_co,
        values: ArrayLike,
        mode: _ModeKind = ...,
    ) -> None: ...
    @overload
    def repeat(
        self,
        repeats: _ArrayLikeInt_co,
        axis: None = ...,
    ) -> Self: ...
    @overload
    def repeat(
        self,
        repeats: _ArrayLikeInt_co,
        axis: _Index,
    ) -> Self: ...
    @overload
    def choose(
        self, choices: ArrayLike, out: None = ..., mode: _ModeKind = ...
    ) -> NDArray[Any]: ...
    @overload
    def choose(
        self, choices: ArrayLike, out: _ArrayT, mode: _ModeKind = ...
    ) -> _ArrayT: ...
    def sort(
        self,
        axis: _Index = ...,
        kind: Literal["stable"] | None = ...,
    ) -> None: ...
    def argsort(
        self,
        axis: _Index | None = ...,
        kind: Literal["stable"] | None = ...,
    ) -> NDArray[Any]: ...
    def partition(
        self,
        kth: _ArrayLikeInt,
        axis: _Index = ...,
    ) -> None: ...
    def argpartition(
        self,
        kth: _ArrayLikeInt,
        axis: _Index | None = ...,
    ) -> NDArray[numpy.intp]: ...
    def searchsorted(
        self,
        v: ArrayLike,
        side: _SortSide = ...,
        sorter: _ArrayLikeInt_co | None = ...,
    ) -> NDArray[numpy.intp]: ...
    def nonzero(self) -> tuple[NDArray[numpy.intp], ...]: ...
    @overload
    def compress(
        self,
        condition: _ArrayLikeInt_co,
        axis: _Index | None = ...,
        out: None = ...,
    ) -> NDArray[Any]: ...
    @overload
    def compress(
        self, condition: _ArrayLikeInt_co, axis: _Index | None, out: _ArrayT
    ) -> _ArrayT: ...
    @overload
    def compress(
        self,
        condition: _ArrayLikeInt_co,
        axis: _Index | None = ...,
        *,
        out: _ArrayT,
    ) -> _ArrayT: ...
    def diagonal(
        self, offset: _Index = ..., axis1: _Index = ..., axis2: _Index = ...
    ) -> Self: ...
