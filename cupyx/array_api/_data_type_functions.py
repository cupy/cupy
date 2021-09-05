from __future__ import annotations

from ._array_object import Array
from ._dtypes import _all_dtypes, _result_type

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Tuple, Union

if TYPE_CHECKING:
    from ._typing import Dtype
    from collections.abc import Sequence

import cupy as cp


def broadcast_arrays(*arrays: Sequence[Array]) -> List[Array]:
    """
    Array API compatible wrapper for :py:func:`cp.broadcast_arrays <cupy.broadcast_arrays>`.

    See its docstring for more information.
    """
    from ._array_object import Array

    return [
        Array._new(array) for array in cp.broadcast_arrays(*[a._array for a in arrays])
    ]


def broadcast_to(x: Array, /, shape: Tuple[int, ...]) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.broadcast_to <cupy.broadcast_to>`.

    See its docstring for more information.
    """
    from ._array_object import Array

    return Array._new(cp.broadcast_to(x._array, shape))


def can_cast(from_: Union[Dtype, Array], to: Dtype, /) -> bool:
    """
    Array API compatible wrapper for :py:func:`cp.can_cast <cupy.can_cast>`.

    See its docstring for more information.
    """
    from ._array_object import Array

    if isinstance(from_, Array):
        from_ = from_._array
    return cp.can_cast(from_, to)


# These are internal objects for the return types of finfo and iinfo, since
# the NumPy versions contain extra data that isn't part of the spec.
@dataclass
class finfo_object:
    bits: int
    # Note: The types of the float data here are float, whereas in NumPy they
    # are scalars of the corresponding float dtype.
    eps: float
    max: float
    min: float
    smallest_normal: float


@dataclass
class iinfo_object:
    bits: int
    max: int
    min: int


def finfo(type: Union[Dtype, Array], /) -> finfo_object:
    """
    Array API compatible wrapper for :py:func:`cp.finfo <cupy.finfo>`.

    See its docstring for more information.
    """
    fi = cp.finfo(type)
    # Note: The types of the float data here are float, whereas in NumPy they
    # are scalars of the corresponding float dtype.
    return finfo_object(
        fi.bits,
        float(fi.eps),
        float(fi.max),
        float(fi.min),
        float(fi.smallest_normal),
    )


def iinfo(type: Union[Dtype, Array], /) -> iinfo_object:
    """
    Array API compatible wrapper for :py:func:`cp.iinfo <cupy.iinfo>`.

    See its docstring for more information.
    """
    ii = cp.iinfo(type)
    return iinfo_object(ii.bits, ii.max, ii.min)


def result_type(*arrays_and_dtypes: Sequence[Union[Array, Dtype]]) -> Dtype:
    """
    Array API compatible wrapper for :py:func:`cp.result_type <cupy.result_type>`.

    See its docstring for more information.
    """
    # Note: we use a custom implementation that gives only the type promotions
    # required by the spec rather than using cp.result_type. NumPy implements
    # too many extra type promotions like int64 + uint64 -> float64, and does
    # value-based casting on scalar arrays.
    A = []
    for a in arrays_and_dtypes:
        if isinstance(a, Array):
            a = a.dtype
        elif isinstance(a, cp.ndarray) or a not in _all_dtypes:
            raise TypeError("result_type() inputs must be array_api arrays or dtypes")
        A.append(a)

    if len(A) == 0:
        raise ValueError("at least one array or dtype is required")
    elif len(A) == 1:
        return A[0]
    else:
        t = A[0]
        for t2 in A[1:]:
            t = _result_type(t, t2)
        return t
