from __future__ import annotations

from ._array_object import Array
from ._dtypes import _numeric_dtypes, _result_type

from typing import Optional, Sequence, Tuple, Union

import cupy as cp

# einsum is not yet implemented in the array API spec.

# def einsum():
#     """
#     Array API compatible wrapper for :py:func:`cp.einsum <cupy.einsum>`.
#
#     See its docstring for more information.
#     """
#     return cp.einsum()


def matmul(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.matmul <cupy.matmul>`.

    See its docstring for more information.
    """
    # Note: the restriction to numeric dtypes only is different from
    # cp.matmul.
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in matmul")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)

    return Array._new(cp.matmul(x1._array, x2._array))


# Note: axes must be a tuple, unlike cp.tensordot where it can be an array or array-like.
def tensordot(
    x1: Array,
    x2: Array,
    /,
    *,
    axes: Union[int, Tuple[Sequence[int], Sequence[int]]] = 2,
) -> Array:
    # Note: the restriction to numeric dtypes only is different from
    # cp.tensordot.
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in tensordot")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)

    return Array._new(cp.tensordot(x1._array, x2._array, axes=axes))


def transpose(x: Array, /, *, axes: Optional[Tuple[int, ...]] = None) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.transpose <cupy.transpose>`.

    See its docstring for more information.
    """
    return Array._new(cp.transpose(x._array, axes=axes))


# Note: vecdot is not in NumPy
def vecdot(x1: Array, x2: Array, /, *, axis: Optional[int] = None) -> Array:
    if axis is None:
        axis = -1
    return tensordot(x1, x2, axes=((axis,), (axis,)))
