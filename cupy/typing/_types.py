from __future__ import annotations

from collections.abc import Sequence

from typing import Any, Literal, SupportsIndex, TypeVar

import numpy
import numpy as np
from cupy._core import core

from numpy.typing import ArrayLike  # NOQA
from numpy.typing import DTypeLike  # NOQA
from numpy.typing import NBitBase  # NOQA


# Shapes
_Shape = tuple[int, ...]
_ShapeLike = SupportsIndex | Sequence[SupportsIndex]


# Supported array dtypes
_BoolType_co = np.bool_
_UIntType_co = Union[_BoolType_co, np.unsignedinteger[Any]]
_IntType_co = Union[_BoolType_co, np.integer[Any]]
_FloatType_co = Union[_IntType_co, np.floating[Any]]
_ComplexType_co = Union[_FloatType_co, np.complexfloating[Any, Any]]

# Arrays
_ScalarType_co = TypeVar(
    "_ScalarType_co", bound=_ComplexType_co, covariant=True)
NDArray = core.ndarray[Any, np.dtype[_ScalarType_co]]

_ArrayBool_co = NDArray[_BoolType_co]
_ArrayUInt_co = NDArray[_UIntType_co]
_ArrayInt_co = NDArray[_IntType_co]
_ArrayFloat_co = NDArray[_FloatType_co]
_ArrayComplex_co = NDArray[_ComplexType_co]
_Array = NDArray[Any]

# Scalars
_BoolLike_co = Union[bool, np.bool_]
_UIntLike_co = Union[_BoolLike_co, np.unsignedinteger[Any]]
_IntLike_co = Union[_BoolLike_co, int, np.integer[Any]]
_FloatLike_co = Union[_IntLike_co, float, np.floating[Any]]
_ComplexLike_co = Union[_FloatLike_co, complex, np.complexfloating[Any, Any]]
_ScalarLike_co = _ComplexLike_co
_NumberLike_co = Union[int, float, complex, np.number[Any], np.bool_]

# ArrayScalar
_ArrayScalarBool_co = Union[_ArrayBool_co, _BoolLike_co]
_ArrayScalarUInt_co = Union[_ArrayUInt_co, _UIntLike_co]
_ArrayScalarInt_co = Union[_ArrayInt_co, _IntLike_co]
_ArrayScalarFloat_co = Union[_ArrayFloat_co, _FloatLike_co]
_ArrayScalarComplex_co = Union[_ArrayComplex_co, _ComplexLike_co]
_ArrayScalar = Union[_Array, _ScalarLike_co]

# Literals
_OrderKACF = Literal[None, "K", "A", "C", "F"]
_OrderACF = Literal[None, "A", "C", "F"]
_OrderCF = Literal[None, "C", "F"]
_ModeKind = Literal["raise", "wrap", "clip"]
_PartitionKind = Literal["introselect"]
_SortKind = Literal["quicksort", "mergesort", "heapsort", "stable"]
_SortSide = Literal["left", "right"]
