from cupy import core


_is_close = core.create_ufunc(
    'cupy_is_close',
    ('eeee?->?', 'ffff?->?', 'dddd?->?'),
    '''
    bool equal_nan = in4;
    if (isfinite(in0) && isfinite(in1)) {
      out0 = fabs(in0 - in1) <= in3 + in2 * fabs(in1);
    } else if (equal_nan) {
      out0 = (in0 == in1) || (isnan(in0) && isnan(in1));
    } else {
      out0 = (in0 == in1);
    }
    '''
)


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
    return _is_close(a, b, rtol, atol, equal_nan)


# TODO(okuta): Implement array_equal


# TODO(okuta): Implement array_equiv


greater = core.greater


greater_equal = core.greater_equal


less = core.less


less_equal = core.less_equal


equal = core.equal


not_equal = core.not_equal
