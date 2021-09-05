from __future__ import annotations

from ._array_object import Array
from ._dtypes import _result_type

from typing import Optional, Tuple

import cupy as cp


def argmax(x: Array, /, *, axis: Optional[int] = None, keepdims: bool = False) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.argmax <cupy.argmax>`.

    See its docstring for more information.
    """
    return Array._new(cp.asarray(cp.argmax(x._array, axis=axis, keepdims=keepdims)))


def argmin(x: Array, /, *, axis: Optional[int] = None, keepdims: bool = False) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.argmin <cupy.argmin>`.

    See its docstring for more information.
    """
    return Array._new(cp.asarray(cp.argmin(x._array, axis=axis, keepdims=keepdims)))


def nonzero(x: Array, /) -> Tuple[Array, ...]:
    """
    Array API compatible wrapper for :py:func:`cp.nonzero <cupy.nonzero>`.

    See its docstring for more information.
    """
    return tuple(Array._new(i) for i in cp.nonzero(x._array))


def where(condition: Array, x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.where <cupy.where>`.

    See its docstring for more information.
    """
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    return Array._new(cp.where(condition._array, x1._array, x2._array))
