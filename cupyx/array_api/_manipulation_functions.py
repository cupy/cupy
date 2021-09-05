from __future__ import annotations

from ._array_object import Array
from ._data_type_functions import result_type

from typing import List, Optional, Tuple, Union

import cupy as cp

# Note: the function name is different here
def concat(
    arrays: Union[Tuple[Array, ...], List[Array]], /, *, axis: Optional[int] = 0
) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.concatenate <cupy.concatenate>`.

    See its docstring for more information.
    """
    # Note: Casting rules here are different from the cp.concatenate default
    # (no for scalars with axis=None, no cross-kind casting)
    dtype = result_type(*arrays)
    arrays = tuple(a._array for a in arrays)
    return Array._new(cp.concatenate(arrays, axis=axis, dtype=dtype))


def expand_dims(x: Array, /, *, axis: int) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.expand_dims <cupy.expand_dims>`.

    See its docstring for more information.
    """
    return Array._new(cp.expand_dims(x._array, axis))


def flip(x: Array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.flip <cupy.flip>`.

    See its docstring for more information.
    """
    return Array._new(cp.flip(x._array, axis=axis))


def reshape(x: Array, /, shape: Tuple[int, ...]) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.reshape <cupy.reshape>`.

    See its docstring for more information.
    """
    return Array._new(cp.reshape(x._array, shape))


def roll(
    x: Array,
    /,
    shift: Union[int, Tuple[int, ...]],
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.roll <cupy.roll>`.

    See its docstring for more information.
    """
    return Array._new(cp.roll(x._array, shift, axis=axis))


def squeeze(x: Array, /, axis: Union[int, Tuple[int, ...]]) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.squeeze <cupy.squeeze>`.

    See its docstring for more information.
    """
    return Array._new(cp.squeeze(x._array, axis=axis))


def stack(arrays: Union[Tuple[Array, ...], List[Array]], /, *, axis: int = 0) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.stack <cupy.stack>`.

    See its docstring for more information.
    """
    # Call result type here just to raise on disallowed type combinations
    result_type(*arrays)
    arrays = tuple(a._array for a in arrays)
    return Array._new(cp.stack(arrays, axis=axis))
