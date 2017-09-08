import numpy as np

import cupy
from cupy.cuda import cufft


def fft(a, n=None, axis=-1, norm=None, value_type=cufft.CUFFT_C2C, direction=cufft.CUFFT_FORWARD):
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

    cufft.destroy(plan)
    return out
