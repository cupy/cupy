import numpy

from cupy.creation import basic
from cupy.creation import from_data
from cupy.creation import ranges
from cupy.math import trigonometric


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
