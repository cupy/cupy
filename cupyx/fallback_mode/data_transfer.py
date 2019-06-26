"""
Data transfer methods
"""

import numpy as np

import cupy as cp


def _get_xp_args(ndarray_instance, to_xp, arg):
    """
    Converts org_module.ndarray object using to_xp.

    Args:
        (org_module.ndarray, tuple, list, dict): These will be returned by
        either converting the object or it's elements if object is iterable.
        (int, float, str, numpy.ScalarType (constant)): Returned as it is.

    Returns:
        Return data structure will be same as before after converting ndarrays.
    """

    if isinstance(arg, ndarray_instance):
        return to_xp(arg)

    if isinstance(arg, tuple):
        return tuple([_get_xp_args(ndarray_instance, to_xp, x) for x in arg])

    if isinstance(arg, dict):
        return {x_name: _get_xp_args(ndarray_instance, to_xp, x)
                for x_name, x in arg.items()}

    if isinstance(arg, list):
        return [_get_xp_args(ndarray_instance, to_xp, x) for x in arg]

    return arg


def _get_cupy_result(numpy_res):
    return _get_xp_args(np.ndarray, cp.array, numpy_res)


def _get_numpy_args(args, kwargs):
    return _get_xp_args(cp.ndarray, cp.asnumpy, (args, kwargs))


def _get_cupy_ndarray(ndarray, args, kwargs):
    return _get_xp_args(ndarray, ndarray._get_array, (args, kwargs))


def _get_fallback_ndarray(ndarray, cupy_res):
    return _get_xp_args(cp.ndarray, ndarray, cupy_res)
