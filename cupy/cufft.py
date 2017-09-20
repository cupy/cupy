import numpy as np

import cupy
from cupy.cuda import cufft


def fft(a, s, axes, norm, direction):
    if a.dtype in [np.float16, np.float32]:
        a = a.astype(np.complex64)
    elif a.dtype not in [np.complex64, np.complex128]:
        a = a.astype(np.complex128)

    if a.dtype == np.complex64:
        value_type = cupy.cuda.cufft.CUFFT_C2C
    else:
        value_type = cupy.cuda.cufft.CUFFT_Z2Z

    if s is None:
        s = [a.shape[i] for i in axes]
    else:
        s = list(s)
    for i, n in enumerate(s):
        if n is None:
            s[i] = a.shape[axes[i]]

    for n, i in zip(s, axes):
        if a.shape[i] != n:
            shape = list(a.shape)
            if shape[i] > n:
                index = [slice(None)]*a.ndim
                index[i] = slice(0, n)
                a = a[index]
            else:
                index = [slice(None)]*a.ndim
                index[i] = slice(0, shape[i])
                shape[i] = n
                z = cupy.zeros(shape, a.dtype.char)
                z[index] = a
                a = z
    if a.base is not None:
        a = a.copy()

    if len(axes) == 1:
        plan = cufft.plan1d(s[0], value_type, a.size // a.shape[axes[0]])
    elif len(axes) == 2:
        plan = cufft.plan2d(s[0], s[1], value_type)
    elif len(axes) == 3:
        plan = cufft.plan3d(s[0], s[1], s[2], value_type)
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

    if norm is None:
        if direction == cufft.CUFFT_INVERSE:
            for i in axes:
                out /= a.shape[i]
    else:
        for i in axes:
            out /= cupy.sqrt(a.shape[i])

    cufft.destroy(plan)

    return out
