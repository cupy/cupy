import numpy

from cupy.creation import basic
from cupy.creation import from_data
from cupy.creation import ranges
from cupy.math import trigonometric


def blackman(M):
    """.. seealso:: :func:`numpy.blackman`"""
    if M < 1:
        return from_data.array([])
    if M == 1:
        return basic.ones(1, float)
    n = ranges.arange(0, M)
    return 0.42 - 0.5 * trigonometric.cos(2.0 * numpy.pi * n / (M - 1))\
        + 0.08 * trigonometric.cos(4.0 * numpy.pi * n / (M - 1))


def hamming(M):
    """.. seealso:: :func:`numpy.hamming`"""
    if M < 1:
        return from_data.array([])
    if M == 1:
        return basic.ones(1, float)
    n = ranges.arange(0, M)
    return 0.54 - 0.46 * trigonometric.cos(2.0 * numpy.pi * n / (M - 1))


def hanning(M):
    """.. seealso:: :func:`numpy.hanning`"""
    if M < 1:
        return from_data.array([])
    if M == 1:
        return basic.ones(1, float)
    n = ranges.arange(0, M)
    return 0.5 - 0.5 * trigonometric.cos(2.0 * numpy.pi * n / (M - 1))
