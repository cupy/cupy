import cupy
from cupy.core import core


def isdense(x):
    return isinstance(x, core.ndarray)


def isintlike(x):
    try:
        return bool(int(x) == x)
    except (TypeError, ValueError):
        return False


def isscalarlike(x):
    return cupy.isscalar(x) or (isdense(x) and x.ndim == 0)


def isshape(x):
    if not isinstance(x, tuple) or len(x) != 2:
        return False
    m, n = x
    return isintlike(m) and isintlike(n)


def prune_array(array):
    """Return an array equivalent to the input array. If the input
    array is a view of a much larger array, copy its contents to a
    newly allocated array. Otherwise, return the input unchanged.
    """

    # Scipy includes this in a util shared between dense and sparse
    # even though it's only used in compressed.py.
    if array.base is not None and array.size < array.base.size // 2:
        return array.copy()
    return array
