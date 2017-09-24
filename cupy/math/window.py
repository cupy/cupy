import numpy

from cupy.creation.basic import ones
from cupy.creation.from_data import array
from cupy.creation.ranges import arange
from cupy.math.trigonometric import cos


def blackman(M):
    """.. seealso:: :func:`numpy.blackman`"""
    if M < 1:
        return array([])
    if M == 1:
        return ones(1, float)
    n = arange(0, M)
    return 0.42 - 0.5 * cos(2.0 * numpy.pi * n / (M - 1))\
        + 0.08 * cos(4.0 * numpy.pi * n / (M - 1))


def hamming(M):
    """.. seealso:: :func:`numpy.hamming`"""
    if M < 1:
        return array([])
    if M == 1:
        return ones(1, float)
    n = arange(0, M)
    return 0.54 - 0.46 * cos(2.0 * numpy.pi * n / (M - 1))


def hanning(M):
    """.. seealso:: :func:`numpy.hanning`"""
    if M < 1:
        return array([])
    if M == 1:
        return ones(1, float)
    n = arange(0, M)
    return 0.5 - 0.5 * cos(2.0 * numpy.pi * n / (M - 1))
