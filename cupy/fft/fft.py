import numpy as np

import cupy
from cupy import cufft


def fft(a, n=None, axis=-1, norm=None):
    if a.dtype == np.float16:
        a = a.astype(np.complex64)
    elif a.dtype == np.float32:
        a = a.astype(np.complex64)
    elif a.dtype == np.float64:
        a = a.astype(np.complex128)
    elif a.dtype != np.complex64 and a.dtype != np.complex128:
        a = a.astype(np.complex128)

    if a.dtype == np.complex64:
        value_type = cupy.cuda.cufft.CUFFT_C2C
    if a.dtype == np.complex128:
        value_type = cupy.cuda.cufft.CUFFT_Z2Z

    return cufft.fft(a, n, axis, norm, value_type, cupy.cuda.cufft.CUFFT_FORWARD)
