from __future__ import annotations

from ._array_object import Array

from typing import Optional, Tuple, Union

import cupy as cp


def all(
    x: Array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.all <cupy.all>`.

    See its docstring for more information.
    """
    return Array._new(cp.asarray(cp.all(x._array, axis=axis, keepdims=keepdims)))


def any(
    x: Array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.any <cupy.any>`.

    See its docstring for more information.
    """
    return Array._new(cp.asarray(cp.any(x._array, axis=axis, keepdims=keepdims)))
