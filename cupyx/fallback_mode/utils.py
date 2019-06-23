"""
Utilities needed for fallback_mode.
"""
import cupy as cp
from cupyx.fallback_mode import data_transfer
from cupyx.fallback_mode import ndarray


def _call_cupy(func, args, kwargs):
    """
    Calls cupy function with *args and **kwargs.

    Args:
        func: A cupy function that needs to be called.
        args (tuple): Arguments.
        kwargs (dict): Keyword arguments.

    Returns:
        Result after calling func.
    """
    args = _convert_ndarrays(args)

    res = func(*args, **kwargs)

    return _get_fallback_ndarray(res)


def _call_numpy(func, args, kwargs):
    """
    Calls numpy function with *args and **kwargs.

    Args:
        func: A numpy function that needs to be called.
        args (tuple): Arguments.
        kwargs (dict): Keyword arguments.

    Returns:
        Result after calling func.
    """
    args = _convert_ndarrays(args)

    numpy_args, numpy_kwargs = data_transfer._get_numpy_args(args, kwargs)

    numpy_res = func(*numpy_args, **numpy_kwargs)

    cupy_res = data_transfer._get_cupy_result(numpy_res)

    return _get_fallback_ndarray(cupy_res)


def _convert_ndarrays(args):
    return tuple(
        [i._array if isinstance(i, ndarray.ndarray) else i for i in args])


def _get_fallback_ndarray(res):

    if isinstance(res, cp.ndarray):
        return ndarray.ndarray(res)

    return res
