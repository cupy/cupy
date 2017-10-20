import six

import numpy as np

import cupy
from cupy.cuda import cufft


def _convert_dtype(a, value_type):
    if value_type != 'R2C':
        if a.dtype in [np.float16, np.float32]:
            return a.astype(np.complex64)
        elif a.dtype not in [np.complex64, np.complex128]:
            return a.astype(np.complex128)
    else:
        if a.dtype in [np.complex64, np.complex128]:
            return a.real
        elif a.dtype == np.float16:
            return a.astype(np.float32)
        elif a.dtype not in [np.float32, np.float64]:
            return a.astype(np.float64)
    return a


def _cook_shape(a, s, axes, value_type):
    if s is None:
        return a
    if not hasattr(s, '__iter__'):
        s = (s,)
    if (value_type == 'C2R') and (s[-1] is not None):
        s = list(s)
        s[-1] = s[-1] // 2 + 1
    for sz, axis in zip(s, axes):
        if (sz is not None) and (sz != a.shape[axis]):
            shape = list(a.shape)
            if shape[axis] > sz:
                index = [slice(None)] * a.ndim
                index[axis] = slice(0, sz)
                a = a[index]
            else:
                index = [slice(None)] * a.ndim
                index[axis] = slice(0, shape[axis])
                shape[axis] = sz
                z = cupy.zeros(shape, a.dtype.char)
                z[index] = a
                a = z
    return a


def _convert_fft_type(a, value_type):
    if value_type == 'C2C' and a.dtype == np.complex64:
        return cufft.CUFFT_C2C
    elif value_type == 'R2C' and a.dtype == np.float32:
        return cufft.CUFFT_R2C
    elif value_type == 'C2R' and a.dtype == np.complex64:
        return cufft.CUFFT_C2R
    elif value_type == 'C2C' and a.dtype == np.complex128:
        return cufft.CUFFT_Z2Z
    elif value_type == 'R2C' and a.dtype == np.float64:
        return cufft.CUFFT_D2Z
    else:
        return cufft.CUFFT_Z2D


def _execFft(a, direction, value_type, norm, axis, out_size=None):
    fft_type = _convert_fft_type(a, value_type)

    if axis % a.ndim != a.ndim - 1:
        a = a.swapaxes(axis, -1)

    plan = cufft.plan1d(a.shape[-1] if out_size is None else out_size,
                        fft_type, a.size // a.shape[-1])

    if a.base is not None:
        a = a.copy()

    shape = list(a.shape)

    if fft_type == cufft.CUFFT_C2C:
        out = cupy.empty(shape, np.complex64)
        cufft.execC2C(plan, a.data, out.data, direction)
    elif fft_type == cufft.CUFFT_R2C:
        shape[-1] = shape[-1] // 2 + 1
        out = cupy.empty(shape, np.complex64)
        cufft.execR2C(plan, a.data, out.data)
    elif fft_type == cufft.CUFFT_C2R:
        shape[-1] = out_size
        out = cupy.empty(shape, np.float32)
        cufft.execC2R(plan, a.data, out.data)
    elif fft_type == cufft.CUFFT_Z2Z:
        out = cupy.empty(shape, np.complex128)
        cufft.execZ2Z(plan, a.data, out.data, direction)
    elif fft_type == cufft.CUFFT_D2Z:
        shape[-1] = shape[-1] // 2 + 1
        out = cupy.empty(shape, np.complex128)
        cufft.execD2Z(plan, a.data, out.data)
    elif fft_type == cufft.CUFFT_Z2D:
        shape[-1] = out_size
        out = cupy.empty(shape, np.float64)
        cufft.execZ2D(plan, a.data, out.data)

    cufft.destroy(plan)

    sz = out.shape[-1]
    if fft_type == cufft.CUFFT_R2C or fft_type == cufft.CUFFT_D2Z:
        sz = a.shape[-1]
    if norm is None:
        if direction == cufft.CUFFT_INVERSE:
            out /= sz
    else:
        out /= cupy.sqrt(sz)

    if axis % a.ndim != a.ndim - 1:
        out = out.swapaxes(axis, -1)

    return out


def _fft_c2c(a, direction, norm, axes):
    for axis in axes:
        a = _execFft(a, direction, 'C2C', norm, axis)
    return a


def fft(a, s, axes, norm, direction, value_type='C2C'):
    if norm not in (None, 'ortho'):
        raise ValueError('Invalid norm value %s, should be None or \"ortho\".'
                         % norm)

    if s is not None:
        for n in s:
            if (n is not None) and (n < 1):
                raise ValueError(
                    "Invalid number of FFT data points (%d) specified." % n)

    if (s is not None) and (axes is not None) and len(s) != len(axes):
        raise ValueError("Shape and axes have different lengths.")

    a = _convert_dtype(a, value_type)
    if axes is None:
        if s is None:
            dim = a.ndim
        else:
            dim = len(s)
        axes = [i for i in six.moves.range(-dim, 0)]
    elif isinstance(axes, np.compat.integer_types):
        axes = (axes,)
    a = _cook_shape(a, s, axes, value_type)

    if value_type == 'C2C':
        a = _fft_c2c(a, direction, norm, axes)
    elif value_type == 'R2C':
        a = _execFft(a, direction, value_type, norm, axes[-1])
        a = _fft_c2c(a, direction, norm, axes[:-1])
    else:
        a = _fft_c2c(a, direction, norm, axes[:-1])
        if isinstance(s, np.compat.integer_types):
            out_size = s
        elif (s is None) or (s[-1] is None):
            out_size = a.shape[axes[-1]] * 2 - 2
        else:
            out_size = s[-1]
        a = _execFft(a, direction, value_type, norm, axes[-1], out_size)

    return a
