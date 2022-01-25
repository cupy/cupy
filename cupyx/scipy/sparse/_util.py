import cupy
from cupy._core import core


def isdense(x):
    # TODO: Use get_namespace here.
    if hasattr(x, '__array_namespace__') and x.__array_namespace__():
        return isinstance(x._array, core.ndarray)
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
