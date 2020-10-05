import numpy

import cupy
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

# Note that in cupy/core/include/cupy/complex.cuh, we already got isfinite and
# isnan working for complex numbers, so just replace fabs above by abs (from
# thrust) and we are ready to go
_is_close_complex = core.create_ufunc(
    'cupy_is_close_complex',
    ('FFff?->?', 'DDdd?->?'),
    '''
    bool equal_nan = in4;
    if (isfinite(in0) && isfinite(in1)) {
      out0 = abs(in0 - in1) <= in3 + in2 * abs(in1);
    } else if (equal_nan) {
      out0 = (in0 == in1) || (isnan(in0) && isnan(in1));
    } else {
      out0 = (in0 == in1);
    }
    '''
)


def array_equal(a1, a2):
    """Returns ``True`` if two arrays are element-wise exactly equal.

    Args:
        a1 (cupy.ndarray): Input array to compare.
        a2 (cupy.ndarray): Input array to compare.

    Returns:
        cupy.ndarray: A boolean 0-dim array.
            If its value is ``True``, two arrays are element-wise equal.

    .. seealso:: :func:`numpy.array_equal`

    """
    if a1.shape != a2.shape:
        return cupy.array(False)
    return (a1 == a2).all()


def allclose(a, b, rtol=1.e-5, atol=1.e-8, equal_nan=False):
    """Returns True if two arrays are element-wise equal within a tolerance.

    Two values in ``a`` and ``b`` are  considiered equal when the following
    equation is satisfied.

    .. math::

       |a - b| \\le \\mathrm{atol} + \\mathrm{rtol} |b|

    Args:
        a (cupy.ndarray): Input array to compare.
        b (cupy.ndarray): Input array to compare.
        rtol (float): The relative tolerance.
        atol (float): The absolute tolerance.
        equal_nan (bool): If ``True``, NaN's in ``a`` will be considered equal
            to NaN's in ``b``.

    Returns:
        cupy.ndarray: A boolean 0-dim array.
            If its value is ``True``, two arrays are element-wise equal within
            a tolerance.

    .. seealso:: :func:`numpy.allclose`

    """
    return isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan).all()


def isclose(a, b, rtol=1.e-5, atol=1.e-8, equal_nan=False):
    """Returns a boolean array where two arrays are equal within a tolerance.

    Two values in ``a`` and ``b`` are  considiered equal when the following
    equation is satisfied.

    .. math::

       |a - b| \\le \\mathrm{atol} + \\mathrm{rtol} |b|

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
    a = cupy.asanyarray(a)
    b = cupy.asanyarray(b)
    if (a.dtype in [numpy.complex64, numpy.complex128]) or \
       (b.dtype in [numpy.complex64, numpy.complex128]):
        return _is_close_complex(a, b, rtol, atol, equal_nan)
    else:
        return _is_close(a, b, rtol, atol, equal_nan)


# TODO(okuta): Implement array_equal


# TODO(okuta): Implement array_equiv


greater = core.greater


greater_equal = core.greater_equal


less = core.less


less_equal = core.less_equal


equal = core.equal


not_equal = core.not_equal
