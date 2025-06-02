from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any, ClassVar, Generic, Literal, SupportsIndex, overload

import numpy
from _typeshed import StrOrBytesPath, SupportsWrite

from cupy._core.flags import Flags
from cupy.cuda.device import Device
from cupy.cuda.memory import MemoryPointer
from cupy.cuda.stream import Stream
from cupy.typing import DTypeLike
from cupy.typing._array import (
    ArrayLike,
    NDArray,
    _ArrayInt_co,
    _ArrayT,
    _IntArrayT,
    _IntegralArrayT,
    _NumericArrayT,
    _RealArrayT,
    _SupportsArrayA,
    _SupportsRealImag,
)
from cupy.typing._internal import _ArrayLike, _ArrayLikeInt_co, _ToIndices
from cupy.typing._standalone import (
    _DTypeLike,
    _DTypeT_co,
    _FloatT,
    _InexactT,
    _ModeKind,
    _NumpyArrayT,
    _OrderKACF,
    _ScalarLike_co,
    _ScalarT,
    _ShapeLike,
    _ShapeT_co,
    _SortSide,
    _SupportsFileMethods,
)

# MEMO: Some methods have special overloads for most-conventional use cases.

# TODO: Add shape support (currently Any)
class ndarray(Generic[_ShapeT_co, _DTypeT_co]):
    def __init__(
        self,
        shape: _ShapeLike,
        dtype: DTypeLike = ...,
        memptr: MemoryPointer | None = ...,
        strides: _ShapeLike | None = ...,
        order: Literal["C", "F"] = ...,
    ) -> None: ...
    # Attributes
    base: ndarray[Any, _DTypeT_co] | None
    dtype: _DTypeT_co
    memptr: MemoryPointer
    size: int
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
    def T(self) -> ndarray[Any, _DTypeT_co]: ...
    @property
    def mT(self) -> ndarray[Any, _DTypeT_co]: ...
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
    ) -> ndarray[Any, numpy.dtype[_ScalarT]]: ...
    @overload
    def astype(
        self,
        dtype: DTypeLike,
        order: _OrderKACF = ...,
        copy: bool = ...,
    ) -> ndarray[Any, numpy.dtype]: ...
    def copy(self, order: _OrderKACF = ...) -> ndarray[Any, _DTypeT_co]: ...
    @overload
    def view(self) -> ndarray[Any, _DTypeT_co]: ...
    @overload
    def view(self, dtype: DTypeLike) -> ndarray[Any, numpy.dtype]: ...
    @overload
    def view(
        self, dtype: _DTypeLike[_ScalarT]
    ) -> ndarray[Any, numpy.dtype[_ScalarT]]: ...
    @overload
    def view(self, type: type[_ArrayT]) -> _ArrayT: ...
    @overload
    def view(self, dtype: DTypeLike, type: type[_ArrayT]) -> _ArrayT: ...
    def fill(self, value: Any) -> None: ...
    def reshape(
        self, *shape: SupportsIndex, order: _OrderKACF = ...
    ) -> ndarray[Any, _DTypeT_co]: ...
    def transpose(self, *axes: SupportsIndex) -> ndarray[Any, _DTypeT_co]: ...
    def swapaxes(
        self, axis1: SupportsIndex, axis2: SupportsIndex
    ) -> ndarray[Any, _DTypeT_co]: ...
    def flatten(self, order: _OrderKACF = ...) -> ndarray[Any, _DTypeT_co]: ...
    def squeeze(
        self,
        axis: SupportsIndex | tuple[SupportsIndex, ...] | None = ...,
    ) -> ndarray[Any, _DTypeT_co]: ...
    @overload
    def take(
        self: NDArray[_ScalarT],
        indices: int | numpy.integer,
        axis: SupportsIndex | None = ...,
        out: None = ...,
    ) -> _ScalarT: ...
    @overload
    def take(
        self,
        indices: _ArrayLikeInt_co,
        axis: SupportsIndex | None = ...,
        out: None = ...,
    ) -> ndarray[Any, _DTypeT_co]: ...
    @overload
    def take(
        self,
        indices: _ArrayLikeInt_co,
        axis: SupportsIndex | None = ...,
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
    ) -> ndarray[Any, _DTypeT_co]: ...
    @overload
    def repeat(
        self,
        repeats: _ArrayLikeInt_co,
        axis: SupportsIndex,
    ) -> ndarray[Any, _DTypeT_co]: ...
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
        axis: SupportsIndex = ...,
        kind: Literal["stable"] | None = ...,
    ) -> None: ...
    def argsort(
        self,
        axis: SupportsIndex | None = ...,
        kind: Literal["stable"] | None = ...,
    ) -> NDArray[Any]: ...
    def partition(
        self,
        kth: _ArrayLikeInt_co,
        axis: SupportsIndex = ...,
    ) -> None: ...
    def argpartition(
        self,
        kth: _ArrayLikeInt_co,
        axis: SupportsIndex | None = ...,
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
        axis: SupportsIndex | None = ...,
        out: None = ...,
    ) -> NDArray[Any]: ...
    @overload
    def compress(
        self,
        condition: _ArrayLikeInt_co,
        axis: SupportsIndex | None,
        out: _ArrayT,
    ) -> _ArrayT: ...
    @overload
    def compress(
        self,
        condition: _ArrayLikeInt_co,
        axis: SupportsIndex | None = ...,
        *,
        out: _ArrayT,
    ) -> _ArrayT: ...
    def diagonal(
        self,
        offset: SupportsIndex = ...,
        axis1: SupportsIndex = ...,
        axis2: SupportsIndex = ...,
    ) -> ndarray[Any, _DTypeT_co]: ...
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
        axis: SupportsIndex,
        out: None = ...,
        dtype: DTypeLike = ...,
        keepdims: bool = ...,
    ) -> Any: ...
    @overload
    def argmax(
        self,
        axis: SupportsIndex | None,
        out: _IntArrayT,
        dtype: DTypeLike = ...,
        keepdims: bool = ...,
    ) -> _IntArrayT: ...
    @overload
    def argmax(
        self,
        axis: SupportsIndex | None = ...,
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
        axis: SupportsIndex,
        out: None = ...,
        dtype: DTypeLike = ...,
        keepdims: bool = ...,
    ) -> Any: ...
    @overload
    def argmin(
        self,
        axis: SupportsIndex | None,
        out: _IntArrayT,
        dtype: DTypeLike = ...,
        keepdims: bool = ...,
    ) -> _IntArrayT: ...
    @overload
    def argmin(
        self,
        axis: SupportsIndex | None = ...,
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
        self, decimals: SupportsIndex = ..., out: None = ...
    ) -> ndarray[Any, _DTypeT_co]: ...
    @overload
    def round(self, decimals: SupportsIndex, out: _ArrayT) -> _ArrayT: ...
    @overload
    def round(
        self, decimals: SupportsIndex = ..., *, out: _ArrayT
    ) -> _ArrayT: ...
    @overload
    def trace(
        self,
        offset: SupportsIndex = ...,
        axis1: SupportsIndex = ...,
        axis2: SupportsIndex = ...,
        dtype: DTypeLike = ...,
        out: None = ...,
    ) -> Any: ...
    @overload
    def trace(
        self,
        offset: SupportsIndex,
        axis1: SupportsIndex,
        axis2: SupportsIndex,
        dtype: DTypeLike,
        out: _ArrayT,
    ) -> _ArrayT: ...
    @overload
    def trace(
        self,
        offset: SupportsIndex = ...,
        axis1: SupportsIndex = ...,
        axis2: SupportsIndex = ...,
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
        axis: SupportsIndex | None = ...,
        dtype: DTypeLike | None = ...,
        out: None = ...,
    ) -> NDArray[Any]: ...
    @overload
    def cumsum(
        self, axis: SupportsIndex | None, dtype: DTypeLike | None, out: _ArrayT
    ) -> _ArrayT: ...
    @overload
    def cumsum(
        self,
        axis: SupportsIndex | None = ...,
        dtype: DTypeLike | None = ...,
        *,
        out: _ArrayT,
    ) -> _ArrayT: ...
    # SPECIAL
    @overload
    def mean(self: NDArray[_InexactT]) -> _InexactT: ...
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
    # SPECIAL
    @overload
    def var(self: NDArray[_FloatT], ddof: float = ...) -> _FloatT: ...
    @overload
    def var(
        self: NDArray[numpy.complex64], ddof: float = ...
    ) -> numpy.float32: ...
    @overload
    def var(
        self: NDArray[numpy.complex128], ddof: float = ...
    ) -> numpy.float64: ...
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
    # SPECIAL
    @overload
    def std(self: NDArray[_FloatT], ddof: float = ...) -> _FloatT: ...
    @overload
    def std(
        self: NDArray[numpy.complex64], ddof: float = ...
    ) -> numpy.float32: ...
    @overload
    def std(
        self: NDArray[numpy.complex128], ddof: float = ...
    ) -> numpy.float64: ...
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
        axis: SupportsIndex | None = ...,
        dtype: DTypeLike | None = ...,
        out: None = ...,
    ) -> NDArray[Any]: ...
    @overload
    def cumprod(
        self, axis: SupportsIndex | None, dtype: DTypeLike | None, out: _ArrayT
    ) -> _ArrayT: ...
    @overload
    def cumprod(
        self,
        axis: SupportsIndex | None = ...,
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
        keepdims: SupportsIndex = ...,
    ) -> numpy.bool | NDArray[numpy.bool]: ...
    @overload
    def all(
        self,
        axis: int | tuple[int, ...] | None,
        out: _ArrayT,
        keepdims: SupportsIndex = ...,
    ) -> _ArrayT: ...
    @overload
    def all(
        self,
        axis: int | tuple[int, ...] | None = ...,
        *,
        out: _ArrayT,
        keepdims: SupportsIndex = ...,
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
        keepdims: SupportsIndex = ...,
    ) -> numpy.bool | NDArray[numpy.bool]: ...
    @overload
    def any(
        self,
        axis: int | tuple[int, ...] | None,
        out: _ArrayT,
        keepdims: SupportsIndex = ...,
    ) -> _ArrayT: ...
    @overload
    def any(
        self,
        axis: int | tuple[int, ...] | None = ...,
        *,
        out: _ArrayT,
        keepdims: SupportsIndex = ...,
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
    def conj(self) -> ndarray[Any, _DTypeT_co]: ...
    def conjugate(self) -> ndarray[Any, _DTypeT_co]: ...
    real: ClassVar[_SupportsRealImag]
    imag: ClassVar[_SupportsRealImag]
    # TODO: Annotate remaining dunders
    def __copy__(self) -> ndarray[Any, _DTypeT_co]: ...
    def __deepcopy__(
        self, memo: dict[int, Any] | None
    ) -> ndarray[Any, _DTypeT_co]: ...
    def __reduce__(
        self: NDArray[_ScalarT],
    ) -> tuple[
        Callable[[numpy.typing.NDArray[_ScalarT]], NDArray[_ScalarT]],
        tuple[numpy.typing.NDArray[_ScalarT]],
    ]: ...
    def __iter__(self: NDArray[_ScalarT]) -> Iterator[_ScalarT]: ...
    def __len__(self) -> int: ...
    @overload
    def __getitem__(
        self, key: _ArrayInt_co | tuple[_ArrayInt_co, ...], /
    ) -> ndarray[Any, _DTypeT_co]: ...
    @overload
    def __getitem__(
        self, key: SupportsIndex | tuple[SupportsIndex, ...], /
    ) -> Any: ...
    @overload
    def __getitem__(self, key: _ToIndices, /) -> ndarray[Any, _DTypeT_co]: ...
    # MEMO: May be overloaded
    def __setitem__(self, key: _ToIndices, value: ArrayLike, /) -> None: ...
    def __array_function__(
        self,
        func: Callable[..., Any],
        types: Iterable[type],
        args: Iterable[Any],
        kwargs: Mapping[str, Any],
    ) -> Any: ...
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
        order: Literal["C", "A", "F"] = ...,
        out: None = ...,
        blocking: bool = ...,
    ) -> numpy.typing.NDArray[_ScalarT]: ...
    @overload
    def get(
        self,
        stream: Stream | None = ...,
        order: Literal["C", "A", "F"] = ...,
        *,
        out: _NumpyArrayT,
        blocking: bool = ...,
    ) -> _NumpyArrayT: ...
    def set(
        self: NDArray[_ScalarT],
        arr: numpy.typing.NDArray[_ScalarT],
        stream: Stream | None = ...,
    ) -> None: ...
    def reduced_view(self) -> ndarray[Any, _DTypeT_co]: ...

@overload
def array(
    obj: _ArrayT,
    dtype: None = ...,
    *,
    copy: bool = ...,
    order: _OrderKACF = ...,
    subok: bool = ...,
    ndmin: int = ...,
    blocking: bool = ...,
) -> _ArrayT: ...
@overload
def array(
    obj: _SupportsArrayA[_ArrayT],
    dtype: None = ...,
    *,
    copy: bool = ...,
    order: _OrderKACF = ...,
    subok: bool = ...,
    ndmin: Literal[0] = ...,
    blocking: bool = ...,
) -> _ArrayT: ...
@overload
def array(
    obj: _ArrayLike[_ScalarT],
    dtype: None = ...,
    *,
    copy: bool = ...,
    order: _OrderKACF = ...,
    subok: bool = ...,
    ndmin: int = ...,
    blocking: bool = ...,
) -> NDArray[_ScalarT]: ...
@overload
def array(
    obj: Any,
    dtype: _DTypeLike[_ScalarT],
    *,
    copy: bool = ...,
    order: _OrderKACF = ...,
    subok: bool = ...,
    ndmin: int = ...,
    blocking: bool = ...,
) -> NDArray[_ScalarT]: ...
@overload
def array(
    obj: Any,
    dtype: DTypeLike | None = ...,
    *,
    copy: bool = ...,
    order: _OrderKACF = ...,
    subok: bool = ...,
    ndmin: int = ...,
    blocking: bool = ...,
) -> NDArray[Any]: ...
@overload
def ascontiguousarray(a: _ArrayT, dtype: None = ...) -> _ArrayT: ...
@overload
def ascontiguousarray(
    a: NDArray[Any], dtype: _DTypeLike[_ScalarT]
) -> NDArray[_ScalarT]: ...
@overload
def ascontiguousarray(
    a: NDArray[Any], dtype: DTypeLike | None
) -> NDArray[Any]: ...
@overload
def asfortranarray(a: _ArrayT, dtype: None = ...) -> _ArrayT: ...
@overload
def asfortranarray(
    a: NDArray[Any], dtype: _DTypeLike[_ScalarT]
) -> NDArray[_ScalarT]: ...
@overload
def asfortranarray(
    a: NDArray[Any], dtype: DTypeLike | None
) -> NDArray[Any]: ...
def min_scalar_type(a: ArrayLike) -> numpy.dtype[Any]: ...
