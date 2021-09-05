from __future__ import annotations


from typing import TYPE_CHECKING, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from ._typing import (
        Array,
        Device,
        Dtype,
        NestedSequence,
        SupportsDLPack,
        SupportsBufferProtocol,
    )
    from collections.abc import Sequence
from ._dtypes import _all_dtypes

import cupy as cp


def _check_valid_dtype(dtype):
    # Note: Only spelling dtypes as the dtype objects is supported.

    # We use this instead of "dtype in _all_dtypes" because the dtype objects
    # define equality with the sorts of things we want to disallw.
    for d in (None,) + _all_dtypes:
        if dtype is d:
            return
    raise ValueError("dtype must be one of the supported dtypes")


def asarray(
    obj: Union[
        Array,
        bool,
        int,
        float,
        NestedSequence[bool | int | float],
        SupportsDLPack,
        SupportsBufferProtocol,
    ],
    /,
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
    copy: Optional[bool] = None,
) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.asarray <cupy.asarray>`.

    See its docstring for more information.
    """
    # _array_object imports in this file are inside the functions to avoid
    # circular imports
    from ._array_object import Array

    _check_valid_dtype(dtype)
    if device not in ["cpu", None]:
        raise ValueError(f"Unsupported device {device!r}")
    if copy is False:
        # Note: copy=False is not yet implemented in cp.asarray
        raise NotImplementedError("copy=False is not yet implemented")
    if isinstance(obj, Array) and (dtype is None or obj.dtype == dtype):
        if copy is True:
            return Array._new(cp.array(obj._array, copy=True, dtype=dtype))
        return obj
    if dtype is None and isinstance(obj, int) and (obj > 2 ** 64 or obj < -(2 ** 63)):
        # Give a better error message in this case. NumPy would convert this
        # to an object array. TODO: This won't handle large integers in lists.
        raise OverflowError("Integer out of bounds for array dtypes")
    res = cp.asarray(obj, dtype=dtype)
    return Array._new(res)


def arange(
    start: Union[int, float],
    /,
    stop: Optional[Union[int, float]] = None,
    step: Union[int, float] = 1,
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.arange <cupy.arange>`.

    See its docstring for more information.
    """
    from ._array_object import Array

    _check_valid_dtype(dtype)
    if device not in ["cpu", None]:
        raise ValueError(f"Unsupported device {device!r}")
    return Array._new(cp.arange(start, stop=stop, step=step, dtype=dtype))


def empty(
    shape: Union[int, Tuple[int, ...]],
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.empty <cupy.empty>`.

    See its docstring for more information.
    """
    from ._array_object import Array

    _check_valid_dtype(dtype)
    if device not in ["cpu", None]:
        raise ValueError(f"Unsupported device {device!r}")
    return Array._new(cp.empty(shape, dtype=dtype))


def empty_like(
    x: Array, /, *, dtype: Optional[Dtype] = None, device: Optional[Device] = None
) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.empty_like <cupy.empty_like>`.

    See its docstring for more information.
    """
    from ._array_object import Array

    _check_valid_dtype(dtype)
    if device not in ["cpu", None]:
        raise ValueError(f"Unsupported device {device!r}")
    return Array._new(cp.empty_like(x._array, dtype=dtype))


def eye(
    n_rows: int,
    n_cols: Optional[int] = None,
    /,
    *,
    k: Optional[int] = 0,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.eye <cupy.eye>`.

    See its docstring for more information.
    """
    from ._array_object import Array

    _check_valid_dtype(dtype)
    if device not in ["cpu", None]:
        raise ValueError(f"Unsupported device {device!r}")
    return Array._new(cp.eye(n_rows, M=n_cols, k=k, dtype=dtype))


def from_dlpack(x: object, /) -> Array:
    # Note: dlpack support is not yet implemented on Array
    raise NotImplementedError("DLPack support is not yet implemented")


def full(
    shape: Union[int, Tuple[int, ...]],
    fill_value: Union[int, float],
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.full <cupy.full>`.

    See its docstring for more information.
    """
    from ._array_object import Array

    _check_valid_dtype(dtype)
    if device not in ["cpu", None]:
        raise ValueError(f"Unsupported device {device!r}")
    if isinstance(fill_value, Array) and fill_value.ndim == 0:
        fill_value = fill_value._array
    res = cp.full(shape, fill_value, dtype=dtype)
    if res.dtype not in _all_dtypes:
        # This will happen if the fill value is not something that NumPy
        # coerces to one of the acceptable dtypes.
        raise TypeError("Invalid input to full")
    return Array._new(res)


def full_like(
    x: Array,
    /,
    fill_value: Union[int, float],
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.full_like <cupy.full_like>`.

    See its docstring for more information.
    """
    from ._array_object import Array

    _check_valid_dtype(dtype)
    if device not in ["cpu", None]:
        raise ValueError(f"Unsupported device {device!r}")
    res = cp.full_like(x._array, fill_value, dtype=dtype)
    if res.dtype not in _all_dtypes:
        # This will happen if the fill value is not something that NumPy
        # coerces to one of the acceptable dtypes.
        raise TypeError("Invalid input to full_like")
    return Array._new(res)


def linspace(
    start: Union[int, float],
    stop: Union[int, float],
    /,
    num: int,
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
    endpoint: bool = True,
) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.linspace <cupy.linspace>`.

    See its docstring for more information.
    """
    from ._array_object import Array

    _check_valid_dtype(dtype)
    if device not in ["cpu", None]:
        raise ValueError(f"Unsupported device {device!r}")
    return Array._new(cp.linspace(start, stop, num, dtype=dtype, endpoint=endpoint))


def meshgrid(*arrays: Sequence[Array], indexing: str = "xy") -> List[Array, ...]:
    """
    Array API compatible wrapper for :py:func:`cp.meshgrid <cupy.meshgrid>`.

    See its docstring for more information.
    """
    from ._array_object import Array

    return [
        Array._new(array)
        for array in cp.meshgrid(*[a._array for a in arrays], indexing=indexing)
    ]


def ones(
    shape: Union[int, Tuple[int, ...]],
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.ones <cupy.ones>`.

    See its docstring for more information.
    """
    from ._array_object import Array

    _check_valid_dtype(dtype)
    if device not in ["cpu", None]:
        raise ValueError(f"Unsupported device {device!r}")
    return Array._new(cp.ones(shape, dtype=dtype))


def ones_like(
    x: Array, /, *, dtype: Optional[Dtype] = None, device: Optional[Device] = None
) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.ones_like <cupy.ones_like>`.

    See its docstring for more information.
    """
    from ._array_object import Array

    _check_valid_dtype(dtype)
    if device not in ["cpu", None]:
        raise ValueError(f"Unsupported device {device!r}")
    return Array._new(cp.ones_like(x._array, dtype=dtype))


def zeros(
    shape: Union[int, Tuple[int, ...]],
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.zeros <cupy.zeros>`.

    See its docstring for more information.
    """
    from ._array_object import Array

    _check_valid_dtype(dtype)
    if device not in ["cpu", None]:
        raise ValueError(f"Unsupported device {device!r}")
    return Array._new(cp.zeros(shape, dtype=dtype))


def zeros_like(
    x: Array, /, *, dtype: Optional[Dtype] = None, device: Optional[Device] = None
) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.zeros_like <cupy.zeros_like>`.

    See its docstring for more information.
    """
    from ._array_object import Array

    _check_valid_dtype(dtype)
    if device not in ["cpu", None]:
        raise ValueError(f"Unsupported device {device!r}")
    return Array._new(cp.zeros_like(x._array, dtype=dtype))
