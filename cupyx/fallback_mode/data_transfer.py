"""
Data transfer methods
"""

import cupy as cp
import numpy as np


def _get_xp_args(org_module, to_xp, arg):
    """
    Converts org_module.ndarray object using to_xp.

    Args:
        (org_module.ndarray, tuple, list, dict): These will be returned by
        either converting the object or it's elements if object is iterable.
        (int, float, str, numpy.ScalarType (constant)): Returned as it is.

    Returns:
        Return data structure will be same as before after converting ndarrays.
    """

    if isinstance(arg, org_module.ndarray):
        return to_xp(arg)

    if isinstance(arg, tuple):
        return tuple([_get_xp_args(org_module, to_xp, x) for x in arg])

    if isinstance(arg, dict):
        return {x_name: _get_xp_args(org_module, to_xp, x)
                for x_name, x in arg.items()}

    if isinstance(arg, list):
        return [_get_xp_args(org_module, to_xp, x) for x in arg]

    if isinstance(arg, np.ScalarType) or callable(arg):
        return arg

    raise NotImplementedError


def _get_cupy_result(numpy_res):
    return _get_xp_args(np, cp.array, numpy_res)


def _get_numpy_args(args, kwargs):
    return _get_xp_args(cp, cp.asnumpy, (args, kwargs))
