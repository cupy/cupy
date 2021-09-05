from __future__ import annotations

from ._array_object import Array

import cupy as cp


def argsort(
    x: Array, /, *, axis: int = -1, descending: bool = False, stable: bool = True
) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.argsort <cupy.argsort>`.

    See its docstring for more information.
    """
    # Note: this keyword argument is different, and the default is different.
    kind = "stable" if stable else "quicksort"
    res = cp.argsort(x._array, axis=axis, kind=kind)
    if descending:
        res = cp.flip(res, axis=axis)
    return Array._new(res)


def sort(
    x: Array, /, *, axis: int = -1, descending: bool = False, stable: bool = True
) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.sort <cupy.sort>`.

    See its docstring for more information.
    """
    # Note: this keyword argument is different, and the default is different.
    kind = "stable" if stable else "quicksort"
    res = cp.sort(x._array, axis=axis, kind=kind)
    if descending:
        res = cp.flip(res, axis=axis)
    return Array._new(res)
