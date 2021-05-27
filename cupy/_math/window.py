import numpy

import cupy
from cupy import _core

_blackman_kernel = _core.ElementwiseKernel(
    "float32 alpha",
    "float64 out",
    """
    out = 0.42 - 0.5 * cos(i * alpha) + 0.08 * cos(2 * alpha * i);
    """, name="cupy_blackman")


_bartlett_kernel = _core.ElementwiseKernel(
    "float32 alpha",
    "T arr",
    """
    if (i < alpha)
        arr = i / alpha;
    else
        arr = 2.0 - i / alpha;
    """, name="cupy_bartlett")


def bartlett(M):
    """Returns the Bartlett window.

    The Bartlett window is defined as

    .. math::
            w(n) = \\frac{2}{M-1} \\left(
            \\frac{M-1}{2} - \\left|n - \\frac{M-1}{2}\\right|
            \\right)

    Args:
        M (int):
            Number of points in the output window. If zero or less, an empty
            array is returned.

    Returns:
        ~cupy.ndarray: Output ndarray.

    .. seealso:: :func:`numpy.bartlett`
    """
    if M == 1:
        return cupy.ones(1, dtype=cupy.float64)
    if M <= 0:
        return cupy.array([])
    alpha = (M - 1) / 2.0
    out = cupy.empty(M, dtype=cupy.float64)
    return _bartlett_kernel(alpha, out)


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
    if M == 1:
        return cupy.ones(1, dtype=cupy.float64)
    if M <= 0:
        return cupy.array([])
    alpha = numpy.pi * 2 / (M - 1)
    out = cupy.empty(M, dtype=cupy.float64)
    return _blackman_kernel(alpha, out)


_hamming_kernel = _core.ElementwiseKernel(
    "float32 alpha",
    "float64 out",
    """
    out = 0.54 - 0.46 * cos(i * alpha);
    """, name="cupy_hamming")


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
    if M == 1:
        return cupy.ones(1, dtype=cupy.float64)
    if M <= 0:
        return cupy.array([])
    alpha = numpy.pi * 2 / (M - 1)
    out = cupy.empty(M, dtype=cupy.float64)
    return _hamming_kernel(alpha, out)


_hanning_kernel = _core.ElementwiseKernel(
    "float32 alpha",
    "float64 out",
    """
    out = 0.5 - 0.5 * cos(i * alpha);
    """, name="cupy_hanning")


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
    if M == 1:
        return cupy.ones(1, dtype=cupy.float64)
    if M <= 0:
        return cupy.array([])
    alpha = numpy.pi * 2 / (M - 1)
    out = cupy.empty(M, dtype=cupy.float64)
    return _hanning_kernel(alpha, out)


_kaiser_kernel = _core.ElementwiseKernel(
    "float32 beta, float32 alpha",
    "T arr",
    """
    float temp = (i - alpha) / alpha;
    arr = cyl_bessel_i0(beta * sqrt(1 - (temp * temp)));
    arr /= cyl_bessel_i0(beta);
    """, name="cupy_kaiser")


def kaiser(M, beta):
    """Return the Kaiser window.
    The Kaiser window is a taper formed by using a Bessel function.

    .. math::  w(n) = I_0\\left( \\beta \\sqrt{1-\\frac{4n^2}{(M-1)^2}}
               \\right)/I_0(\\beta)

    with

    .. math:: \\quad -\\frac{M-1}{2} \\leq n \\leq \\frac{M-1}{2}

    where :math:`I_0` is the modified zeroth-order Bessel function.

     Args:
        M (int):
            Number of points in the output window. If zero or less, an empty
            array is returned.
        beta (float):
            Shape parameter for window

    Returns:
        ~cupy.ndarray:  The window, with the maximum value normalized to one
        (the value one appears only if the number of samples is odd).

    .. seealso:: :func:`numpy.kaiser`
    """
    if M == 1:
        return cupy.array([1.])
    if M <= 0:
        return cupy.array([])
    alpha = (M - 1) / 2.0
    out = cupy.empty(M, dtype=cupy.float64)
    return _kaiser_kernel(beta, alpha, out)
