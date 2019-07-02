"""
Data transfer methods
"""

import numpy as np

import cupy as cp


def _get_xp_args(ndarray_instance, to_xp, arg):
    """
    Converts ndarray_instance type object to target object using to_xp.
    ndarray_instance: numpy.ndarray, cupy.ndarray or utils.ndarray

    Args:
        (ndarray_instance, tuple, list, dict): These will be returned by
        either converting the object or it's elements if object is iterable.
        Everything else is returned as it is.

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
