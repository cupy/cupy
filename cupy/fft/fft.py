import numpy as np

import cupy
from cupy import cufft


def fft(a, n=None, axis=-1, norm=None):
    if a.dtype in [np.float16, np.float32, np.complex64]:
        value_type = cupy.cuda.cufft.CUFFT_C2C
    else:
        value_type = cupy.cuda.cufft.CUFFT_Z2Z

    return cufft.fft(a, n, axis, norm, value_type, cupy.cuda.cufft.CUFFT_FORWARD)


def ifft(a, n=None, axis=-1, norm=None):
    if a.dtype in [np.float16, np.float32, np.complex64]:
        value_type = cupy.cuda.cufft.CUFFT_C2C
    else:
        value_type = cupy.cuda.cufft.CUFFT_Z2Z

    return cufft.fft(a, n, axis, norm, value_type, cupy.cuda.cufft.CUFFT_INVERSE)
