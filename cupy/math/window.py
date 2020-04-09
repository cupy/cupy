import numpy

from cupy import array
from cupy import empty
from cupy import float64
from cupy import util
from cupy.creation import basic
from cupy.creation import from_data
from cupy.creation import ranges
from cupy.math import trigonometric
from cupy.core import ElementwiseKernel


def blackman(M):
    """Returns the Blackman window.

    The Blackman window is defined as

    .. math::
        w(n) = 0.42 - 0.5 \\cos\\left(\\frac{2\\pi{n}}{M-1}\\right)
        + 0.08 \\cos\\left(\\frac{4\\pi{n}}{M-1}\\right)
        \\qquad 0 \\leq n \\leq M-1

    Args:
        M (:class:`~int`):
            Number of points in the output window. If zero or less, an empty
            array is returned.

    Returns:
        ~cupy.ndarray: Output ndarray.

    .. seealso:: :func:`numpy.blackman`
    """
    if M < 1:
        return from_data.array([])
    if M == 1:
        return basic.ones(1, float)
    n = ranges.arange(0, M)
    return 0.42 - 0.5 * trigonometric.cos(2.0 * numpy.pi * n / (M - 1))\
        + 0.08 * trigonometric.cos(4.0 * numpy.pi * n / (M - 1))


def hamming(M):
    """Returns the Hamming window.

    The Hamming window is defined as

    .. math::
        w(n) = 0.54 - 0.46\\cos\\left(\\frac{2\\pi{n}}{M-1}\\right)
        \\qquad 0 \\leq n \\leq M-1

    Args:
        M (:class:`~int`):
            Number of points in the output window. If zero or less, an empty
            array is returned.

    Returns:
        ~cupy.ndarray: Output ndarray.

    .. seealso:: :func:`numpy.hamming`
    """
    if M < 1:
        return from_data.array([])
    if M == 1:
        return basic.ones(1, float)
    n = ranges.arange(0, M)
    return 0.54 - 0.46 * trigonometric.cos(2.0 * numpy.pi * n / (M - 1))


def hanning(M):
    """Returns the Hanning window.

    The Hanning window is defined as

    .. math::
        w(n) = 0.5 - 0.5\\cos\\left(\\frac{2\\pi{n}}{M-1}\\right)
        \\qquad 0 \\leq n \\leq M-1

    Args:
        M (:class:`~int`):
            Number of points in the output window. If zero or less, an empty
            array is returned.

    Returns:
        ~cupy.ndarray: Output ndarray.

    .. seealso:: :func:`numpy.hanning`
    """
    if M < 1:
        return from_data.array([])
    if M == 1:
        return basic.ones(1, float)
    n = ranges.arange(0, M)
    return 0.5 - 0.5 * trigonometric.cos(2.0 * numpy.pi * n / (M - 1))


@util.memoize()
def _kaiser_kernel():
    input_arguments = "float32 beta, float32 alpha"
    output_arguments = "T arr"
    return ElementwiseKernel(input_arguments,
                             output_arguments,
                             """
                             float temp = (i - alpha)/alpha;
                             arr = cyl_bessel_i0(beta * sqrt(1-(temp * temp)));
                             arr /= cyl_bessel_i0(beta);
                             """)


def kaiser(M, beta):
    """Return the Kaiser window.
    The Kaiser window is a taper formed by using a Bessel function.

    .. math::  w(n) = I_0\\left( \\beta \\sqrt{1-\\frac{4n^2}{(M-1)^2}}
               \\right)/I_0(\\beta)

    with

    .. math:: \\quad -\\frac{M-1}{2} \\leq n \\leq \\frac{M-1}{2}

    where :math:`I_0` is the modified zeroth-order Bessel function.

     Args:
        M (:class:`~int`):
            Number of points in the output window. If zero or less, an empty
            array is returned.
        beta (:class:`~float`):
            Shape parameter for window

    Returns:
        ~cupy.ndarray:  The window, with the maximum value normalized to one
        (the value one appears only if the number of samples is odd).

    .. seealso:: :func:`numpy.kaiser`
    """
    if M == 1:
        return array([1.])
    if M < 0:
        return array([])
    alpha = (M-1)/2.0
    out = empty(M, dtype=float64)
    return _kaiser_kernel()(beta, alpha, out)
