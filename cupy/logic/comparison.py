import cupy
from cupy import core


# TODO(okuta): Implement allclose


def isclose(a, b, rtol=1.e-5, atol=1.e-8, equal_nan=False):
    """Returns a boolean array where two arrays are equal within a tolerance.

    Two values in ``a`` and ``b`` are  considiered equal when the following
    equation is satisfied.

    .. math::

       |a - b| \le \mathrm{atol} + \mathrm{rtol} |b|

    Args:
        a (cupy.ndarray): Input array to compare.
        b (cupy.ndarray): Input array to compare.
        rtol (float): The relative tolerance.
        atol (float): The absolute tolerance.
        equal_nan (bool): If ``True``, NaN's in ``a`` will be considered equal
            to NaN's in ``b``.

    Returns:
        cupy.ndarray: A boolean array storing where ``a`` and ``b`` are equal.

    .. seealso:: :func:`numpy.isclose`

    """
    def within_tol(x, y):
        return abs(x - y) <= atol + rtol * abs(y)

    # When we use integer type, abs(MIN_INT) causes overflow.
    x = cupy.asarray(a, dtype='d')
    y = cupy.asarray(b, dtype='d')

    xfin = cupy.isfinite(x)
    yfin = cupy.isfinite(y)
    if all(xfin) and all(yfin):
        cond = within_tol(x, y)
    else:
        finite = xfin & yfin
        cond = cupy.zeros_like(finite)
        # For boolean indexing, we need to broadcast arrays.
        x, y = cupy.broadcast_arrays(x, y)
        cond[finite] = within_tol(x[finite], y[finite])
        cond[~finite] = (x[~finite] == y[~finite])
        if equal_nan:
            both_nan = cupy.isnan(x) & cupy.isnan(y)
            cond[both_nan] = both_nan[both_nan]

    return cond


# TODO(okuta): Implement array_equal


# TODO(okuta): Implement array_equiv


greater = core.greater


greater_equal = core.greater_equal


less = core.less


less_equal = core.less_equal


equal = core.equal


not_equal = core.not_equal
