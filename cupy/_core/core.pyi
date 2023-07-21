# https://github.com/numpy/numpy/blob/dabb12ec5c27846b1ded065b5e61eabb89d029ed/numpy/__init__.pyi

from enum import IntEnum
import os
from types import GenericAlias
from typing import (
    Literal as L,
    Any,
    Callable,
    Iterable,
    Generic,
    Mapping,
    Sequence,
    SupportsComplex,
    SupportsFloat,
    SupportsInt,
    TypeVar,
    Union,
    SupportsIndex,
    overload,
)

from numpy import (
    # Arrays
    _ArrayLikeBool_co,
    _ArrayLikeInt_co,

    # __init__.pyi
    _SupportsItem,
    _CastingKind,
    _IOProtocol,
    _SupportsArray,
    _SupportsWrite,
    _PyCapsule,
    dtype,
    _ctypes,

    bool_,
    intp,
    integer,
    unsignedinteger,
    signedinteger,
    floating,
    complexfloating,
)

import cupy.cuda
import cupy._core.flags

from cupy.typing._types import (
    _IntLike_co,
    _ScalarLike_co,
    _ComplexType_co,
    _NumberLike_co,
    _ArrayScalar,
    ArrayLike,
    DTypeLike,
    NBitBase,
    NDArray,

    # Literals
    _OrderKACF,
    _OrderACF,
    _ModeKind,
    _PartitionKind,
    _SortKind,
    _SortSide,

    # Shapes
    _Shape,
    _ShapeLike,
)


_NdArraySubClass = TypeVar("_NdArraySubClass", bound=ndarray[Any, Any])

_DType_co = TypeVar("_DType_co", covariant=True, bound=dtype[_ComplexType_co])
_ShapeType = TypeVar("_ShapeType", bound=Any)
_ShapeType2 = TypeVar("_ShapeType2", bound=Any)

_ScalarType = TypeVar("_ScalarType", bound=_ComplexType_co)
_NBit1 = TypeVar("_NBit1", bound=NBitBase)
_NBit2 = TypeVar("_NBit2", bound=NBitBase)

_T = TypeVar("_T")
_2Tuple = tuple[_T, _T]


_ArraySelf = TypeVar("_ArraySelf", bound="ndarray")
ufunc = Callable[..., Any]


class ndarray(Generic[_ShapeType, _DType_co]):

    @property
    def T(self: _ArraySelf) -> _ArraySelf: ...

    @property
    def data(self) -> cupy.cuda.MemoryPointer: ...

    @property
    def flags(self) -> cupy._core.flags.Flags: ...

    @property
    def itemsize(self) -> int: ...

    @property
    def nbytes(self) -> int: ...

    def __bool__(self) -> bool: ...

    def __bytes__(self) -> bytes: ...

    def __str__(self) -> str: ...

    def __repr__(self) -> str: ...

    def __copy__(self: _ArraySelf) -> _ArraySelf: ...

    def __deepcopy__(self: _ArraySelf, memo: None | dict[int, Any], /) -> _ArraySelf: ...

    def __eq__(self, other: NDArray[Any]) -> NDArray[bool_]: ...  # type: ignore[override]

    def __ne__(self, other: NDArray[Any]) -> NDArray[bool_]: ...  # type: ignore[override]

    def copy(self: _ArraySelf, order: _OrderKACF = ...) -> _ArraySelf: ...

    def dump(self, file: str | bytes | os.PathLike[str] | os.PathLike[bytes] | _SupportsWrite[bytes]) -> None: ...

    def dumps(self) -> bytes: ...

    def tobytes(self, order: _OrderKACF = ...) -> bytes: ...

    def tofile(
        self,
        fid: str | bytes | os.PathLike[str] | os.PathLike[bytes] | _IOProtocol,
        sep: str = ...,
        format: str = ...,
    ) -> None: ...

    def tolist(self) -> Any: ...

    @property
    def __array_interface__(self) -> dict[str, Any]: ...

    @property
    def __array_priority__(self) -> float: ...

    @overload  # type: ignore[misc]
    def all(
        self,
        axis: None | _ShapeLike = ...,
        out: None = ...,
        keepdims: bool = ...,
        *,
        where: _ArrayLikeBool_co = ...,
    ) -> NDArray[bool_]: ...
    @overload
    def all(
        self,
        axis: None | _ShapeLike = ...,
        out: _NdArraySubClass = ...,
        keepdims: bool = ...,
        *,
        where: _ArrayLikeBool_co = ...,
    ) -> _NdArraySubClass: ...

    @overload  # type: ignore[misc]
    def any(
        self,
        axis: None | _ShapeLike = ...,
        out: None = ...,
        keepdims: bool = ...,
        *,
        where: _ArrayLikeBool_co = ...,
    ) -> NDArray[bool_]: ...
    @overload
    def any(
        self,
        axis: None | _ShapeLike = ...,
        out: _NdArraySubClass = ...,
        keepdims: bool = ...,
        *,
        where: _ArrayLikeBool_co = ...,
    ) -> _NdArraySubClass: ...

    @overload  # type: ignore[misc]
    def argmax(
        self,
        axis: SupportsIndex = ...,
        out: None = ...,
        *,
        keepdims: bool = ...,
    ) -> NDArray[intp]: ...
    @overload
    def argmax(
        self,
        axis: None | SupportsIndex = ...,
        out: _NdArraySubClass = ...,
        *,
        keepdims: bool = ...,
    ) -> _NdArraySubClass: ...

    @overload  # type: ignore[misc]
    def argmin(
        self,
        axis: SupportsIndex = ...,
        out: None = ...,
        *,
        keepdims: bool = ...,
    ) -> NDArray[intp]: ...
    @overload
    def argmin(
        self,
        axis: None | SupportsIndex = ...,
        out: _NdArraySubClass = ...,
        *,
        keepdims: bool = ...,
    ) -> _NdArraySubClass: ...

    def argsort(
        self,
        axis: None | SupportsIndex = ...,
        kind: None | _SortKind = ...,
        order: None | str | Sequence[str] = ...,
    ) -> ndarray[Any, Any]: ...

    @overload  # type: ignore[misc]
    def choose(
        self,
        choices: ArrayLike,
        out: None = ...,
        mode: _ModeKind = ...,
    ) -> ndarray[Any, Any]: ...
    @overload
    def choose(
        self,
        choices: ArrayLike,
        out: _NdArraySubClass = ...,
        mode: _ModeKind = ...,
    ) -> _NdArraySubClass: ...

    @overload  # type: ignore[misc]
    def clip(
        self,
        min: None | ArrayLike = ...,
        max: None | ArrayLike = ...,
        out: None = ...,
        **kwargs: Any,
    ) -> ndarray[Any, Any]: ...
    @overload
    def clip(
        self,
        min: None | ArrayLike = ...,
        max: None | ArrayLike = ...,
        out: _NdArraySubClass = ...,
        **kwargs: Any,
    ) -> _NdArraySubClass: ...

    @overload  # type: ignore[misc]
    def compress(
        self,
        a: ArrayLike,
        axis: None | SupportsIndex = ...,
        out: None = ...,
    ) -> ndarray[Any, Any]: ...
    @overload
    def compress(
        self,
        a: ArrayLike,
        axis: None | SupportsIndex = ...,
        out: _NdArraySubClass = ...,
    ) -> _NdArraySubClass: ...

    def conj(self: _ArraySelf) -> _ArraySelf: ...

    def conjugate(self: _ArraySelf) -> _ArraySelf: ...

    @overload  # type: ignore[misc]
    def cumprod(
        self,
        axis: None | SupportsIndex = ...,
        dtype: DTypeLike = ...,
        out: None = ...,
    ) -> ndarray[Any, Any]: ...
    @overload
    def cumprod(
        self,
        axis: None | SupportsIndex = ...,
        dtype: DTypeLike = ...,
        out: _NdArraySubClass = ...,
    ) -> _NdArraySubClass: ...

    @overload  # type: ignore[misc]
    def cumsum(
        self,
        axis: None | SupportsIndex = ...,
        dtype: DTypeLike = ...,
        out: None = ...,
    ) -> ndarray[Any, Any]: ...
    @overload
    def cumsum(
        self,
        axis: None | SupportsIndex = ...,
        dtype: DTypeLike = ...,
        out: _NdArraySubClass = ...,
    ) -> _NdArraySubClass: ...

    @overload
    def max(
        self,
        axis: None | _ShapeLike = ...,
        out: None = ...,
        keepdims: bool = ...,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool_co = ...,
    ) -> Any: ...
    @overload
    def max(
        self,
        axis: None | _ShapeLike = ...,
        out: _NdArraySubClass = ...,
        keepdims: bool = ...,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool_co = ...,
    ) -> _NdArraySubClass: ...

    @overload
    def mean(
        self,
        axis: None | _ShapeLike = ...,
        dtype: DTypeLike = ...,
        out: None = ...,
        keepdims: bool = ...,
        *,
        where: _ArrayLikeBool_co = ...,
    ) -> Any: ...
    @overload
    def mean(
        self,
        axis: None | _ShapeLike = ...,
        dtype: DTypeLike = ...,
        out: _NdArraySubClass = ...,
        keepdims: bool = ...,
        *,
        where: _ArrayLikeBool_co = ...,
    ) -> _NdArraySubClass: ...

    @overload
    def min(
        self,
        axis: None | _ShapeLike = ...,
        out: None = ...,
        keepdims: bool = ...,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool_co = ...,
    ) -> Any: ...
    @overload
    def min(
        self,
        axis: None | _ShapeLike = ...,
        out: _NdArraySubClass = ...,
        keepdims: bool = ...,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool_co = ...,
    ) -> _NdArraySubClass: ...

    @overload
    def prod(
        self,
        axis: None | _ShapeLike = ...,
        dtype: DTypeLike = ...,
        out: None = ...,
        keepdims: bool = ...,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool_co = ...,
    ) -> Any: ...
    @overload
    def prod(
        self,
        axis: None | _ShapeLike = ...,
        dtype: DTypeLike = ...,
        out: _NdArraySubClass = ...,
        keepdims: bool = ...,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool_co = ...,
    ) -> _NdArraySubClass: ...

    @overload
    def ptp(
        self,
        axis: None | _ShapeLike = ...,
        out: None = ...,
        keepdims: bool = ...,
    ) -> Any: ...
    @overload
    def ptp(
        self,
        axis: None | _ShapeLike = ...,
        out: _NdArraySubClass = ...,
        keepdims: bool = ...,
    ) -> _NdArraySubClass: ...

    @overload
    def round(
        self: _ArraySelf,
        decimals: SupportsIndex = ...,
        out: None = ...,
    ) -> _ArraySelf: ...
    @overload
    def round(
        self,
        decimals: SupportsIndex = ...,
        out: _NdArraySubClass = ...,
    ) -> _NdArraySubClass: ...

    @overload
    def std(
        self,
        axis: None | _ShapeLike = ...,
        dtype: DTypeLike = ...,
        out: None = ...,
        ddof: float = ...,
        keepdims: bool = ...,
        *,
        where: _ArrayLikeBool_co = ...,
    ) -> Any: ...
    @overload
    def std(
        self,
        axis: None | _ShapeLike = ...,
        dtype: DTypeLike = ...,
        out: _NdArraySubClass = ...,
        ddof: float = ...,
        keepdims: bool = ...,
        *,
        where: _ArrayLikeBool_co = ...,
    ) -> _NdArraySubClass: ...

    @overload
    def sum(
        self,
        axis: None | _ShapeLike = ...,
        dtype: DTypeLike = ...,
        out: None = ...,
        keepdims: bool = ...,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool_co = ...,
    ) -> Any: ...
    @overload
    def sum(
        self,
        axis: None | _ShapeLike = ...,
        dtype: DTypeLike = ...,
        out: _NdArraySubClass = ...,
        keepdims: bool = ...,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool_co = ...,
    ) -> _NdArraySubClass: ...

    @overload
    def var(
        self,
        axis: None | _ShapeLike = ...,
        dtype: DTypeLike = ...,
        out: None = ...,
        ddof: float = ...,
        keepdims: bool = ...,
        *,
        where: _ArrayLikeBool_co = ...,
    ) -> Any: ...
    @overload
    def var(
        self,
        axis: None | _ShapeLike = ...,
        dtype: DTypeLike = ...,
        out: _NdArraySubClass = ...,
        ddof: float = ...,
        keepdims: bool = ...,
        *,
        where: _ArrayLikeBool_co = ...,
    ) -> _NdArraySubClass: ...

    @property
    def base(self) -> None | ndarray[Any, Any]: ...

    @property
    def ndim(self) -> int: ...

    @property
    def size(self) -> int: ...

    @property
    def real(self: ndarray[_ShapeType, Any]) -> ndarray[_ShapeType, Any]: ...

    @real.setter
    def real(self, value: _ArrayScalar) -> None: ...

    @property
    def imag(self: ndarray[_ShapeType, Any]) -> ndarray[_ShapeType, Any]: ...

    @imag.setter
    def imag(self, value: _ArrayScalar) -> None: ...

    def __new__(
        cls: type[_ArraySelf],
        shape: _ShapeLike,
        dtype: DTypeLike = ...,
        memptr: cupy.cuda.MemoryPointer = ...,
        strides: None | _ShapeLike = ...,
        order: _OrderKACF = ...,
        *,
        _obj: Any = ...,
        _not_init: bool = ...,
    ) -> _ArraySelf: ...

    def __class_getitem__(self, item: Any) -> GenericAlias: ...

    def __array_ufunc__(
        self,
        ufunc: ufunc,
        method: L["__call__", "reduce", "reduceat", "accumulate", "outer", "inner"],
        *inputs: Any,
        **kwargs: Any,
    ) -> Any: ...

    def __array_function__(
        self,
        func: Callable[..., Any],
        types: Iterable[type],
        args: Iterable[Any],
        kwargs: Mapping[str, Any],
    ) -> Any: ...

    def __array_finalize__(self, obj: None | NDArray[Any], /) -> None: ...

    @overload
    def __getitem__(self, key: (
        NDArray[integer[Any]]
        | NDArray[bool_]
        | tuple[NDArray[integer[Any]] | NDArray[bool_], ...]
    )) -> ndarray[Any, _DType_co]: ...
    @overload
    def __getitem__(self, key: SupportsIndex | tuple[SupportsIndex, ...]) -> Any: ...
    @overload
    def __getitem__(self, key: (
        None
        | slice
        | ellipsis
        | SupportsIndex
        | _ArrayLikeInt_co
        | tuple[None | slice | ellipsis | _ArrayLikeInt_co | SupportsIndex, ...]
    )) -> ndarray[Any, _DType_co]: ...

    @property
    def shape(self) -> _Shape: ...

    @shape.setter
    def shape(self, value: _ShapeLike) -> None: ...

    @property
    def strides(self) -> _Shape: ...

    @strides.setter
    def strides(self, value: _ShapeLike) -> None: ...

    def fill(self, value: Any) -> None: ...

    @property
    def flat(self: _NdArraySubClass) -> cupy.flatiter: ...

    # Use the same output type as that of the underlying `generic`
    @overload
    def item(
        self: ndarray[Any, dtype[_SupportsItem[_T]]],  # type: ignore[type-var]
        *args: SupportsIndex,
    ) -> _T: ...
    @overload
    def item(
        self: ndarray[Any, dtype[_SupportsItem[_T]]],  # type: ignore[type-var]
        args: tuple[SupportsIndex, ...],
        /,
    ) -> _T: ...

    # @overload
    # def resize(self, new_shape: _ShapeLike, /, *, refcheck: bool = ...) -> None: ...
    # @overload
    # def resize(self, *new_shape: SupportsIndex, refcheck: bool = ...) -> None: ...

    # def setflags(
    #     self, write: bool = ..., align: bool = ..., uic: bool = ...
    # ) -> None: ...

    def squeeze(
        self,
        axis: None | SupportsIndex | tuple[SupportsIndex, ...] = ...,
    ) -> ndarray[Any, _DType_co]: ...

    def swapaxes(
        self,
        axis1: SupportsIndex,
        axis2: SupportsIndex,
    ) -> ndarray[Any, _DType_co]: ...

    @overload
    def transpose(self: _ArraySelf, axes: None | _ShapeLike, /) -> _ArraySelf: ...
    @overload
    def transpose(self: _ArraySelf, *axes: SupportsIndex) -> _ArraySelf: ...

    def argpartition(
        self,
        kth: _ArrayLikeInt_co,
        axis: None | SupportsIndex = ...,
        kind: _PartitionKind = ...,
        order: None | str | Sequence[str] = ...,
    ) -> ndarray[Any, dtype[intp]]: ...

    def diagonal(
        self,
        offset: SupportsIndex = ...,
        axis1: SupportsIndex = ...,
        axis2: SupportsIndex = ...,
    ) -> ndarray[Any, _DType_co]: ...

    @overload
    def dot(self, b: _ArrayScalar, out: None = ...) -> ndarray[Any, Any]: ...
    @overload
    def dot(self, b: _ArrayScalar, out: _NdArraySubClass) -> _NdArraySubClass: ...

    def nonzero(self) -> tuple[ndarray[Any, dtype[intp]], ...]: ...

    def partition(
        self,
        kth: _ArrayLikeInt_co,
        axis: SupportsIndex = ...,
        kind: _PartitionKind = ...,
        order: None | str | Sequence[str] = ...,
    ) -> None: ...

    # `put` is technically available to `generic`,
    # but is pointless as `generic`s are immutable
    def put(
        self,
        ind: _ArrayLikeInt_co,
        v: ArrayLike,
        mode: _ModeKind = ...,
    ) -> None: ...

    def searchsorted(
        self,  # >= 1D array
        v: ArrayLike,
        side: _SortSide = ...,
        sorter: None | _ArrayLikeInt_co = ...,
    ) -> ndarray[Any, dtype[intp]]: ...

    # def setfield(
    #     self,
    #     val: ArrayLike,
    #     dtype: DTypeLike,
    #     offset: SupportsIndex = ...,
    # ) -> None: ...

    def sort(
        self,
        axis: SupportsIndex = ...,
        kind: None | _SortKind = ...,
        order: None | str | Sequence[str] = ...,
    ) -> None: ...

    @overload
    def trace(
        self,  # >= 2D array
        offset: SupportsIndex = ...,
        axis1: SupportsIndex = ...,
        axis2: SupportsIndex = ...,
        dtype: DTypeLike = ...,
        out: None = ...,
    ) -> Any: ...
    @overload
    def trace(
        self,  # >= 2D array
        offset: SupportsIndex = ...,
        axis1: SupportsIndex = ...,
        axis2: SupportsIndex = ...,
        dtype: DTypeLike = ...,
        out: _NdArraySubClass = ...,
    ) -> _NdArraySubClass: ...

    @overload
    def take(  # type: ignore[misc]
        self,
        indices: _ArrayLikeInt_co,
        axis: None | SupportsIndex = ...,
        out: None = ...,
        mode: _ModeKind = ...,
    ) -> ndarray[Any, _DType_co]: ...
    @overload
    def take(
        self,
        indices: _ArrayLikeInt_co,
        axis: None | SupportsIndex = ...,
        out: _NdArraySubClass = ...,
        mode: _ModeKind = ...,
    ) -> _NdArraySubClass: ...

    def repeat(
        self,
        repeats: _ArrayLikeInt_co,
        axis: None | SupportsIndex = ...,
    ) -> ndarray[Any, _DType_co]: ...

    def flatten(
        self,
        order: _OrderKACF = ...,
    ) -> ndarray[Any, _DType_co]: ...

    def ravel(
        self,
        order: _OrderKACF = ...,
    ) -> ndarray[Any, _DType_co]: ...

    @overload
    def reshape(
        self, shape: _ShapeLike, /, *, order: _OrderACF = ...
    ) -> ndarray[Any, _DType_co]: ...
    @overload
    def reshape(
        self, *shape: SupportsIndex, order: _OrderACF = ...
    ) -> ndarray[Any, _DType_co]: ...

    def astype(
        self,
        dtype: DTypeLike,
        order: _OrderKACF = ...,
        casting: _CastingKind = ...,
        subok: bool = ...,
        copy: bool = ...,
    ) -> NDArray[Any]: ...

    @overload
    def view(self: _ArraySelf) -> _ArraySelf: ...
    @overload
    def view(self, type: type[_NdArraySubClass]) -> _NdArraySubClass: ...
    @overload
    def view(self, dtype: DTypeLike) -> NDArray[Any]: ...
    @overload
    def view(
        self,
        dtype: DTypeLike,
        type: type[_NdArraySubClass],
    ) -> _NdArraySubClass: ...

    def __int__(
        self: ndarray[Any, dtype[SupportsInt]],  # type: ignore[type-var]
    ) -> int: ...

    def __float__(
        self: ndarray[Any, dtype[SupportsFloat]],  # type: ignore[type-var]
    ) -> float: ...

    def __complex__(
        self: ndarray[Any, dtype[SupportsComplex]],  # type: ignore[type-var]
    ) -> complex: ...

    def __index__(
        self: ndarray[Any, dtype[SupportsIndex]],  # type: ignore[type-var]
    ) -> int: ...

    def __len__(self) -> int: ...
    def __setitem__(self, key, value): ...
    def __iter__(self) -> Any: ...
    # def __contains__(self, key) -> bool: ...

    def __lt__(self: _ArrayScalar, other: _ArrayScalar) -> NDArray[bool_]: ...

    def __le__(self: _ArrayScalar, other: _ArrayScalar) -> NDArray[bool_]: ...

    def __gt__(self: _ArrayScalar, other: _ArrayScalar) -> NDArray[bool_]: ...

    def __ge__(self: _ArrayScalar, other: _ArrayScalar) -> NDArray[bool_]: ...

    # Unary ops
    def __abs__(self: NDArray[Any]) -> NDArray[Any]: ...

    def __invert__(self: NDArray[Any]) -> NDArray[Any]: ...

    def __pos__(self: NDArray[Any]): ...

    def __neg__(self: NDArray[Any]) -> NDArray[Any]: ...

    # Binary ops
    def __matmul__(self: NDArray[Any], other: _ArrayScalar) -> NDArray[Any]: ...

    def __rmatmul__(self: NDArray[Any], other: _ArrayScalar) -> NDArray[Any]: ...

    def __mod__(self: NDArray[Any], other: _ArrayScalar) -> NDArray[Any]: ...

    def __rmod__(self: NDArray[Any], other: _ArrayScalar) -> NDArray[Any]: ...

    def __divmod__(self: NDArray[Any], other: _ArrayScalar) -> _2Tuple[NDArray[Any]]: ...

    def __rdivmod__(self: NDArray[Any], other: _ArrayScalar) -> _2Tuple[NDArray[Any]]: ...

    def __add__(self: NDArray[Any], other: _ArrayScalar) -> NDArray[Any]: ...

    def __radd__(self: NDArray[Any], other: _ArrayScalar) -> NDArray[Any]: ...  # type: ignore[misc]

    def __sub__(self: NDArray[Any], other: _ArrayScalar) -> NDArray[Any]: ...

    def __rsub__(self: NDArray[Any], other: _ArrayScalar) -> NDArray[Any]: ...  # type: ignore[misc]

    def __mul__(self: NDArray[Any], other: _ArrayScalar) -> NDArray[Any]: ...

    def __rmul__(self: NDArray[Any], other: _ArrayScalar) -> NDArray[Any]: ...  # type: ignore[misc]

    def __floordiv__(self: NDArray[Any], other: _ArrayScalar) -> NDArray[Any]: ...

    def __rfloordiv__(self: NDArray[Any], other: _ArrayScalar) -> NDArray[Any]: ...

    def __pow__(self: NDArray[Any], other: _ArrayScalar) -> NDArray[Any]: ...

    def __rpow__(self: NDArray[Any], other: _ArrayScalar) -> NDArray[Any]: ...  # type: ignore[misc]

    def __truediv__(self: NDArray[Any], other: _ArrayScalar) -> NDArray[Any]: ...

    def __rtruediv__(self: NDArray[Any], other: _ArrayScalar) -> NDArray[Any]: ...  # type: ignore[misc]

    def __lshift__(self: NDArray[Any], other: _ArrayScalar) -> NDArray[Any]: ...

    def __rlshift__(self: NDArray[Any], other: _ArrayScalar) -> NDArray[Any]: ...

    def __rshift__(self: NDArray[Any], other: _ArrayScalar) -> NDArray[Any]: ...

    def __rrshift__(self: NDArray[Any], other: _ArrayScalar) -> NDArray[Any]: ...

    def __and__(self: NDArray[Any], other: _ArrayScalar) -> NDArray[Any]: ...

    def __rand__(self: NDArray[Any], other: _ArrayScalar) -> NDArray[Any]: ...

    def __xor__(self: NDArray[Any], other: _ArrayScalar) -> NDArray[Any]: ...

    def __rxor__(self: NDArray[Any], other: _ArrayScalar) -> NDArray[Any]: ...

    def __or__(self: NDArray[Any], other: _ArrayScalar) -> NDArray[Any]: ...

    def __ror__(self: NDArray[Any], other: _ArrayScalar) -> NDArray[Any]: ...

    def __iadd__(self: NDArray[Any], other: _ArrayScalar) -> NDArray[Any]: ...

    def __isub__(self: NDArray[Any], other: _ArrayScalar) -> NDArray[Any]: ...

    def __imul__(self: NDArray[Any], other: _ArrayScalar) -> NDArray[Any]: ...

    def __itruediv__(self: NDArray[Any], other: _ArrayScalar) -> NDArray[Any]: ...

    def __ifloordiv__(self: NDArray[Any], other: _ArrayScalar) -> NDArray[Any]: ...

    def __ipow__(self: NDArray[Any], other: _ArrayScalar) -> NDArray[Any]: ...

    def __imod__(self: NDArray[Any], other: _ArrayScalar) -> NDArray[Any]: ...

    def __ilshift__(self: NDArray[Any], other: _ArrayScalar) -> NDArray[Any]: ...

    def __irshift__(self: NDArray[Any], other: _ArrayScalar) -> NDArray[Any]: ...

    def __iand__(self: NDArray[Any], other: _ArrayScalar) -> NDArray[Any]: ...

    def __ixor__(self: NDArray[Any], other: _ArrayScalar) -> NDArray[Any]: ...

    def __ior__(self: NDArray[Any], other: _ArrayScalar) -> NDArray[Any]: ...

    def __imatmul__(self: NDArray[Any], other: _ArrayScalar) -> NDArray[Any]: ...

    def __dlpack__(self: NDArray[Any], *, stream: None = ...) -> _PyCapsule: ...

    def __dlpack_device__(self) -> tuple[IntEnum, int]: ...

    @property
    def dtype(self) -> _DType_co: ...
