import cupy


def isdense(x):
    return isinstance(x, cupy.ndarray)


def isscalarlike(x):
    """Is x either a scalar, an array scalar, or a 0-dim array?"""
    return cupy.isscalar(x) or (isdense(x) and x.ndim == 0)
