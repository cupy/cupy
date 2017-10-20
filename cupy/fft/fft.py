import six

import numpy as np

import cupy
from cupy import cufft


def fft(a, n=None, axis=-1, norm=None):
    return cufft.fft(a, (n,), (axis,), norm, cupy.cuda.cufft.CUFFT_FORWARD)


def ifft(a, n=None, axis=-1, norm=None):
    return cufft.fft(a, (n,), (axis,), norm, cupy.cuda.cufft.CUFFT_INVERSE)


def fft2(a, s=None, axes=(-2, -1), norm=None):
    return cufft.fft(a, s, axes, norm, cupy.cuda.cufft.CUFFT_FORWARD)


def ifft2(a, s=None, axes=(-2, -1), norm=None):
    return cufft.fft(a, s, axes, norm, cupy.cuda.cufft.CUFFT_INVERSE)


def fftn(a, s=None, axes=None, norm=None):
    return cufft.fft(a, s, axes, norm, cupy.cuda.cufft.CUFFT_FORWARD)


def ifftn(a, s=None, axes=None, norm=None):
    return cufft.fft(a, s, axes, norm, cupy.cuda.cufft.CUFFT_INVERSE)


def rfft(a, n=None, axis=-1, norm=None):
    return cufft.fft(a, (n,), (axis,), norm, cupy.cuda.cufft.CUFFT_FORWARD,
                     'R2C')


def irfft(a, n=None, axis=-1, norm=None):
    return cufft.fft(a, (n,), (axis,), norm, cupy.cuda.cufft.CUFFT_INVERSE,
                     'C2R')


def rfft2(a, s=None, axes=(-2, -1), norm=None):
    return cufft.fft(a, s, axes, norm, cupy.cuda.cufft.CUFFT_FORWARD, 'R2C')


def irfft2(a, s=None, axes=(-2, -1), norm=None):
    return cufft.fft(a, s, axes, norm, cupy.cuda.cufft.CUFFT_INVERSE, 'C2R')


def rfftn(a, s=None, axes=None, norm=None):
    return cufft.fft(a, s, axes, norm, cupy.cuda.cufft.CUFFT_FORWARD, 'R2C')


def irfftn(a, s=None, axes=None, norm=None):
    return cufft.fft(a, s, axes, norm, cupy.cuda.cufft.CUFFT_INVERSE, 'C2R')


def hfft(a, n=None, axis=-1, norm=None):
    a = irfft(a.conj(), n, axis)
    return a * (a.shape[axis] if norm is None else
                cupy.sqrt(a.shape[axis], dtype=a.dtype))


def ihfft(a, n=None, axis=-1, norm=None):
    if n is None:
        n = a.shape[axis]
    return rfft(a, n, axis, norm).conj() / (n if norm is None else 1)


def fftfreq(n, d=1.0):
    return cupy.hstack((cupy.arange(0, (n - 1) // 2 + 1),
                        cupy.arange(-(n // 2), 0))) / n / d


def rfftfreq(n, d=1.0):
    return cupy.arange(0, n // 2 + 1) / n / d


def fftshift(x, axes=None):
    x = cupy.asarray(x)
    if axes is None:
        axes = list(six.moves.range(x.ndim))
    elif isinstance(axes, np.compat.integer_types):
        axes = (axes,)
    for axis in axes:
        x = cupy.roll(x, x.shape[axis] // 2, axis)
    return x


def ifftshift(x, axes=None):
    x = cupy.asarray(x)
    if axes is None:
        axes = list(six.moves.range(x.ndim))
    elif isinstance(axes, np.compat.integer_types):
        axes = (axes,)
    for axis in axes:
        x = cupy.roll(x, -(x.shape[axis] // 2), axis)
    return x
