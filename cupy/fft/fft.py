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
