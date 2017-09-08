import numpy as np

import cupy
from cupy.cuda import cufft


def fft(a, n=None, axis=-1, norm=None, value_type=cufft.CUFFT_C2C, direction=cufft.CUFFT_FORWARD):
    if a.dtype == np.float16:
        a = a.astype(np.complex64)
    elif a.dtype == np.float32:
        a = a.astype(np.complex64)
    elif a.dtype == np.float64:
        a = a.astype(np.complex128)
    elif a.dtype != np.complex64 and a.dtype != np.complex128:
        a = a.astype(np.complex128)

    if n is None:
        n = a.shape[axis]
    if a.shape[axis] != n:
        s = list(a.shape)
        if s[axis] > n:
            index = [slice(None)]*len(s)
            index[axis] = slice(0, n)
            a = a[index]
        else:
            index = [slice(None)]*len(s)
            index[axis] = slice(0, s[axis])
            s[axis] = n
            z = cupy.zeros(s, a.dtype.char)
            z[index] = a
            a = z

    plan = cufft.plan1d(a.shape[axis], value_type, a.size // a.shape[axis])
    out = cupy.empty_like(a)

    if value_type == cufft.CUFFT_C2C:
        cufft.execC2C(plan, a.data, out.data, direction)
    if value_type == cufft.CUFFT_R2C:
        cufft.execR2C(plan, a.data, out.data)
    if value_type == cufft.CUFFT_C2R:
        cufft.execC2R(plan, a.data, out.data)
    if value_type == cufft.CUFFT_Z2Z:
        cufft.execZ2Z(plan, a.data, out.data, direction)
    if value_type == cufft.CUFFT_D2Z:
        cufft.execD2Z(plan, a.data, out.data)
    if value_type == cufft.CUFFT_Z2D:
        cufft.execZ2D(plan, a.data, out.data)

    if norm is None and direction == cufft.CUFFT_INVERSE:
        out /= n
    if norm is not None:
        out /= cupy.sqrt(n)
    cufft.destroy(plan)
    return out
