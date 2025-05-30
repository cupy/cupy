from collections.abc import Iterator
from typing import Any, Generic, Literal

from _typeshed import StrOrBytesPath, SupportsWrite
from typing import overload
import numpy

from cupy import dtype
from cupy._core.flags import Flags
from cupy.cuda.device import Device
from cupy.cuda.stream import Stream
from cupy.typing._types import (
    NDArray,
    DTypeLike,
    ArrayLike,
    _IntArrayT,
    _NumericArrayT,
    _IntegralArrayT,
    _NumpyArrayT,
    _RealArrayT,
    _ArrayT,
)
from cupy.typing._proto import _SupportsFileMethods
from cupy.typing._internal import (
    _SortSide,
    _ArrayLikeInt,
    _DTypeT_co,
    _ScalarLike_co,
    _OrderCAF,
    _Index,
    _OrderKACF,
    _DTypeLike,
    _ScalarT,
    _ShapeLike,
    _ModeKind,
)

# MEMO: Some methods have special overloads for most-conventional use cases.

class ndarray(Generic[_DTypeT_co]):
    # TODO: Annotate dlpack interface
    def __cuda_array_interface__(self) -> dict[str, Any]: ...
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
    def T(self) -> ndarray[_DTypeT_co]: ...
    @property
    def mT(self) -> ndarray[_DTypeT_co]: ...
    @property
    def flat(self: NDArray[_ScalarT]) -> Iterator[_ScalarT]: ...
    def item(self: NDArray[_ScalarT]) -> _ScalarT: ...
    def tolist(self: NDArray[_ScalarT]) -> list[Any]: ...
    def tobytes(self, order: _OrderKACF = ...) -> bytes: ...
    def tofile(
        self,
        fid: StrOrBytesPath | _SupportsFileMethods,
        sep: str = ...,
        format: str = ...,
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
    def copy(self, order: _OrderKACF = ...) -> ndarray[_DTypeT_co]: ...
    @overload
    def view(self) -> ndarray[_DTypeT_co]: ...
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
    def reshape(
        self, *shape: _Index, order: _OrderKACF = ...
    ) -> ndarray[_DTypeT_co]: ...
    def transpose(self, *axes: _Index) -> ndarray[_DTypeT_co]: ...
    def swapaxes(
        self, axis1: _Index, axis2: _Index
    ) -> ndarray[_DTypeT_co]: ...
    def flatten(self, order: _OrderKACF = ...) -> ndarray[_DTypeT_co]: ...
    def squeeze(
        self,
        axis: _Index | tuple[_Index, ...] | None = ...,
    ) -> ndarray[_DTypeT_co]: ...
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
        indices: _ArrayLikeInt,
        axis: _Index | None = ...,
        out: None = ...,
    ) -> ndarray[_DTypeT_co]: ...
    @overload
    def take(
        self,
        indices: _ArrayLikeInt,
        axis: _Index | None = ...,
        *out: _ArrayT,
    ) -> _ArrayT: ...
    def put(
        self,
        indices: _ArrayLikeInt,
        values: ArrayLike,
        mode: _ModeKind = ...,
    ) -> None: ...
    @overload
    def repeat(
        self,
        repeats: _ArrayLikeInt,
        axis: None = ...,
    ) -> ndarray[_DTypeT_co]: ...
    @overload
    def repeat(
        self,
        repeats: _ArrayLikeInt,
        axis: _Index,
    ) -> ndarray[_DTypeT_co]: ...
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
        sorter: _ArrayLikeInt | None = ...,
    ) -> NDArray[numpy.intp]: ...
    def nonzero(self) -> tuple[NDArray[numpy.intp], ...]: ...
    @overload
    def compress(
        self,
        condition: _ArrayLikeInt,
        axis: _Index | None = ...,
        out: None = ...,
    ) -> NDArray[Any]: ...
    @overload
    def compress(
        self, condition: _ArrayLikeInt, axis: _Index | None, out: _ArrayT
    ) -> _ArrayT: ...
    @overload
    def compress(
        self,
        condition: _ArrayLikeInt,
        axis: _Index | None = ...,
        *,
        out: _ArrayT,
    ) -> _ArrayT: ...
    def diagonal(
        self, offset: _Index = ..., axis1: _Index = ..., axis2: _Index = ...
    ) -> ndarray[_DTypeT_co]: ...
    # SPECIAL
    @overload
    def max(self: NDArray[_ScalarT]) -> _ScalarT: ...
    @overload
    def max(
        self,
        axis: _ShapeLike | None = ...,
        out: None = ...,
        keepdims: bool = ...,
    ) -> Any: ...
    @overload
    def max(
        self,
        axis: _ShapeLike | None,
        out: _ArrayT,
        keepdims: bool = ...,
    ) -> _ArrayT: ...
    @overload
    def max(
        self,
        axis: _ShapeLike | None = ...,
        *,
        out: _ArrayT,
        keepdims: bool = ...,
    ) -> _ArrayT: ...
    # SPECIAL
    @overload
    def argmax(self) -> numpy.intp: ...
    @overload
    def argmax(
        self,
        axis: _Index,
        out: None = ...,
        dtype: DTypeLike = ...,
        keepdims: bool = ...,
    ) -> Any: ...
    @overload
    def argmax(
        self,
        axis: _Index | None,
        out: _IntArrayT,
        dtype: DTypeLike = ...,
        keepdims: bool = ...,
    ) -> _IntArrayT: ...
    @overload
    def argmax(
        self,
        axis: _Index | None = ...,
        *,
        out: _IntArrayT,
        dtype: DTypeLike = ...,
        keepdims: bool = ...,
    ) -> _IntArrayT: ...
    # SPECIAL
    @overload
    def min(self: NDArray[_ScalarT]) -> _ScalarT: ...
    @overload
    def min(
        self,
        axis: _ShapeLike | None = ...,
        out: None = ...,
        keepdims: bool = ...,
    ) -> Any: ...
    @overload
    def min(
        self,
        axis: _ShapeLike | None,
        out: _ArrayT,
        keepdims: bool = ...,
    ) -> _ArrayT: ...
    @overload
    def min(
        self,
        axis: _ShapeLike | None = ...,
        *,
        out: _ArrayT,
        keepdims: bool = ...,
    ) -> _ArrayT: ...
    # SPECIAL
    @overload
    def argmin(self) -> numpy.intp: ...
    @overload
    def argmin(
        self,
        axis: _Index,
        out: None = ...,
        dtype: DTypeLike = ...,
        keepdims: bool = ...,
    ) -> Any: ...
    @overload
    def argmin(
        self,
        axis: _Index | None,
        out: _IntArrayT,
        dtype: DTypeLike = ...,
        keepdims: bool = ...,
    ) -> _IntArrayT: ...
    @overload
    def argmin(
        self,
        axis: _Index | None = ...,
        *,
        out: _IntArrayT,
        dtype: DTypeLike = ...,
        keepdims: bool = ...,
    ) -> _IntArrayT: ...
    # SPECIAL
    @overload
    def ptp(self: NDArray[_ScalarT]) -> _ScalarT: ...
    @overload
    def ptp(
        self,
        axis: _ShapeLike | None = ...,
        out: None = ...,
        keepdims: bool = ...,
    ) -> Any: ...
    @overload
    def ptp(
        self,
        axis: _ShapeLike | None,
        out: _ArrayT,
        keepdims: bool = ...,
    ) -> _ArrayT: ...
    @overload
    def ptp(
        self,
        axis: _ShapeLike | None = ...,
        *,
        out: _ArrayT,
        keepdims: bool = ...,
    ) -> _ArrayT: ...
    @overload
    def clip(
        self,
        min: ArrayLike | None = ...,
        max: ArrayLike | None = ...,
        out: None = ...,
    ) -> NDArray[Any]: ...
    @overload
    def clip(
        self,
        min: ArrayLike | None,
        max: ArrayLike | None,
        out: _ArrayT,
    ) -> _ArrayT: ...
    @overload
    def clip(
        self,
        min: ArrayLike | None = ...,
        max: ArrayLike | None = ...,
        *,
        out: _ArrayT,
    ) -> _ArrayT: ...
    @overload
    def round(
        self, decimals: _Index = ..., out: None = ...
    ) -> ndarray[_DTypeT_co]: ...
    @overload
    def round(self, decimals: _Index, out: _ArrayT) -> _ArrayT: ...
    @overload
    def round(self, decimals: _Index = ..., *, out: _ArrayT) -> _ArrayT: ...
    @overload
    def trace(
        self,
        offset: _Index = ...,
        axis1: _Index = ...,
        axis2: _Index = ...,
        dtype: DTypeLike = ...,
        out: None = ...,
    ) -> Any: ...
    @overload
    def trace(
        self,
        offset: _Index,
        axis1: _Index,
        axis2: _Index,
        dtype: DTypeLike,
        out: _ArrayT,
    ) -> _ArrayT: ...
    @overload
    def trace(
        self,
        offset: _Index = ...,
        axis1: _Index = ...,
        axis2: _Index = ...,
        dtype: DTypeLike = ...,
        *,
        out: _ArrayT,
    ) -> _ArrayT: ...
    # SPECIAL
    @overload
    def sum(self: NDArray[_ScalarT]) -> _ScalarT: ...
    @overload
    def sum(
        self,
        axis: _ShapeLike | None = ...,
        dtype: DTypeLike | None = ...,
        out: None = ...,
        keepdims: bool = ...,
    ) -> Any: ...
    @overload
    def sum(
        self,
        axis: _ShapeLike | None,
        dtype: DTypeLike | None,
        out: _ArrayT,
        keepdims: bool = ...,
    ) -> _ArrayT: ...
    @overload
    def sum(
        self,
        axis: _ShapeLike | None = ...,
        dtype: DTypeLike | None = ...,
        *,
        out: _ArrayT,
        keepdims: bool = ...,
    ) -> _ArrayT: ...
    # SPECIAL
    @overload
    def cumsum(self: NDArray[_ScalarT]) -> NDArray[_ScalarT]: ...
    @overload
    def cumsum(
        self,
        axis: _Index | None = ...,
        dtype: DTypeLike | None = ...,
        out: None = ...,
    ) -> NDArray[Any]: ...
    @overload
    def cumsum(
        self, axis: _Index | None, dtype: DTypeLike | None, out: _ArrayT
    ) -> _ArrayT: ...
    @overload
    def cumsum(
        self,
        axis: _Index | None = ...,
        dtype: DTypeLike | None = ...,
        *,
        out: _ArrayT,
    ) -> _ArrayT: ...
    @overload
    def mean(
        self,
        axis: _ShapeLike | None = ...,
        dtype: DTypeLike | None = ...,
        out: None = ...,
        keepdims: bool = ...,
    ) -> Any: ...
    @overload
    def mean(
        self,
        axis: _ShapeLike | None,
        dtype: DTypeLike | None,
        out: _ArrayT,
        keepdims: bool = ...,
    ) -> _ArrayT: ...
    @overload
    def var(
        self,
        axis: _ShapeLike | None = ...,
        dtype: DTypeLike | None = ...,
        out: None = ...,
        ddof: float = ...,
        keepdims: bool = ...,
    ) -> Any: ...
    @overload
    def var(
        self,
        axis: _ShapeLike | None,
        dtype: DTypeLike | None,
        out: _ArrayT,
        ddof: float = ...,
        keepdims: bool = ...,
    ) -> _ArrayT: ...
    @overload
    def var(
        self,
        axis: _ShapeLike | None = ...,
        dtype: DTypeLike | None = ...,
        *,
        out: _ArrayT,
        ddof: float = ...,
        keepdims: bool = ...,
    ) -> _ArrayT: ...
    @overload
    def std(
        self,
        axis: _ShapeLike | None = ...,
        dtype: DTypeLike | None = ...,
        out: None = ...,
        ddof: float = ...,
        keepdims: bool = ...,
    ) -> Any: ...
    @overload
    def std(
        self,
        axis: _ShapeLike | None,
        dtype: DTypeLike | None,
        out: _ArrayT,
        ddof: float = ...,
        keepdims: bool = ...,
    ) -> _ArrayT: ...
    @overload
    def std(
        self,
        axis: _ShapeLike | None = ...,
        dtype: DTypeLike | None = ...,
        *,
        out: _ArrayT,
        ddof: float = ...,
        keepdims: bool = ...,
    ) -> _ArrayT: ...
    # SPECIAL
    @overload
    def prod(self: NDArray[_ScalarT]) -> _ScalarT: ...
    @overload
    def prod(
        self,
        axis: _ShapeLike | None = ...,
        dtype: DTypeLike | None = ...,
        out: None = ...,
        keepdims: bool = ...,
    ) -> Any: ...
    @overload
    def prod(
        self,
        axis: _ShapeLike | None,
        dtype: DTypeLike | None,
        out: _ArrayT,
        keepdims: bool = ...,
    ) -> _ArrayT: ...
    @overload
    def prod(
        self,
        axis: _ShapeLike | None = ...,
        dtype: DTypeLike | None = ...,
        *,
        out: _ArrayT,
        keepdims: bool = ...,
    ) -> _ArrayT: ...
    # SPECIAL
    @overload
    def cumprod(self: NDArray[_ScalarT]) -> NDArray[_ScalarT]: ...
    @overload
    def cumprod(
        self,
        axis: _Index | None = ...,
        dtype: DTypeLike | None = ...,
        out: None = ...,
    ) -> NDArray[Any]: ...
    @overload
    def cumprod(
        self, axis: _Index | None, dtype: DTypeLike | None, out: _ArrayT
    ) -> _ArrayT: ...
    @overload
    def cumprod(
        self,
        axis: _Index | None = ...,
        dtype: DTypeLike | None = ...,
        *,
        out: _ArrayT,
    ) -> _ArrayT: ...
    @overload
    def all(
        self,
        axis: None = ...,
        out: None = ...,
        keepdims: Literal[False, 0] = ...,
    ) -> numpy.bool: ...
    @overload
    def all(
        self,
        axis: int | tuple[int, ...] | None = ...,
        out: None = ...,
        keepdims: _Index = ...,
    ) -> numpy.bool | NDArray[numpy.bool]: ...
    @overload
    def all(
        self,
        axis: int | tuple[int, ...] | None,
        out: _ArrayT,
        keepdims: _Index = ...,
    ) -> _ArrayT: ...
    @overload
    def all(
        self,
        axis: int | tuple[int, ...] | None = ...,
        *,
        out: _ArrayT,
        keepdims: _Index = ...,
    ) -> _ArrayT: ...
    @overload
    def any(
        self,
        axis: None = ...,
        out: None = ...,
        keepdims: Literal[False, 0] = ...,
    ) -> numpy.bool: ...
    @overload
    def any(
        self,
        axis: int | tuple[int, ...] | None = ...,
        out: None = ...,
        keepdims: _Index = ...,
    ) -> numpy.bool | NDArray[numpy.bool]: ...
    @overload
    def any(
        self,
        axis: int | tuple[int, ...] | None,
        out: _ArrayT,
        keepdims: _Index = ...,
    ) -> _ArrayT: ...
    @overload
    def any(
        self,
        axis: int | tuple[int, ...] | None = ...,
        *,
        out: _ArrayT,
        keepdims: _Index = ...,
    ) -> _ArrayT: ...
    def __nonzero__(self) -> bool: ...
    def __neg__(self: _NumericArrayT) -> _NumericArrayT: ...
    def __pos__(self: _NumericArrayT) -> _NumericArrayT: ...
    @overload
    def __abs__(self: _RealArrayT) -> _RealArrayT: ...
    @overload
    def __abs__(self: NDArray[numpy.complex64]) -> NDArray[numpy.float64]: ...
    @overload
    def __abs__(self: NDArray[numpy.complex128]) -> NDArray[numpy.float64]: ...
    def __invert__(self: _IntegralArrayT) -> _IntegralArrayT: ...
    # TODO: Annotate binary operators
    def conj(self) -> ndarray[_DTypeT_co]: ...
    def conjugate(self) -> ndarray[_DTypeT_co]: ...
    # TODO: Annotate real/imag
    # TODO: Annotate remaining dunders
    def __copy__(self) -> ndarray[_DTypeT_co]: ...
    def __deepcopy__(
        self, memo: dict[int, Any] | None
    ) -> ndarray[_DTypeT_co]: ...
    def __iter__(self: NDArray[_ScalarT]) -> Iterator[_ScalarT]: ...
    def __len__(self) -> int: ...
    def __int__(self) -> int: ...
    def __float__(self) -> float: ...
    def __complex__(self) -> complex: ...
    def __oct__(self) -> str: ...
    def __hex__(self) -> str: ...
    def __bytes__(self) -> bytes: ...
    def __str__(self) -> str: ...  # noqa: PYI029
    def __repr__(self) -> str: ...  # noqa: PYI029
    def __format__(self, format_spec: str) -> str: ...
    @overload
    def dot(self, b: _ScalarLike_co, out: None = ...) -> NDArray[Any]: ...
    @overload
    def dot(self, b: ArrayLike, out: None = ...) -> Any: ...
    @overload
    def dot(self, b: ArrayLike, out: _ArrayT) -> _ArrayT: ...
    @property
    def device(self) -> Device: ...
    @overload
    def get(
        self: NDArray[_ScalarT],
        stream: Stream | None = ...,
        order: _OrderCAF = ...,
        out: None = ...,
        blocking: bool = ...,
    ) -> numpy.typing.NDArray[_ScalarT]: ...
    @overload
    def get(
        self,
        stream: Stream | None = ...,
        order: _OrderCAF = ...,
        *,
        out: _NumpyArrayT,
        blocking: bool = ...,
    ) -> _NumpyArrayT: ...
    def set(
        self: NDArray[_ScalarT],
        arr: numpy.typing.NDArray[_ScalarT],
        stream: Stream | None = ...,
    ) -> None: ...
    def reduced_view(self) -> ndarray[_DTypeT_co]: ...
