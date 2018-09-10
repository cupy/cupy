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


def validateaxis(axis):
    if axis is not None:
        axis_type = type(axis)

        if axis_type == tuple:
            raise TypeError(
                'Tuples are not accepted for the \'axis\' '
                'parameter. Please pass in one of the '
                'following: {-2, -1, 0, 1, None}.')

        if not cupy.issubdtype(cupy.dtype(axis_type), cupy.integer):
            raise TypeError('axis must be an integer, not {name}'
                            .format(name=axis_type.__name__))

        if not (-2 <= axis <= 1):
            raise ValueError('axis out of range')
