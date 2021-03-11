import numpy

from cupy import cuda
from cupy.core import internal


def empty_pinned(shape, dtype=float, order='C', *, like=None):
    if like is not None:
        raise ValueError('like is not supported')
    shape = tuple(shape)
    nbytes = internal.prod(shape) * numpy.dtype(dtype).itemsize
    mem = cuda.alloc_pinned_memory(nbytes)
    out = numpy.ndarray(shape, dtype=dtype, buffer=mem, order=order)
    return out


def empty_like_pinned(a, dtype=None, order='K', shape=None):
    if dtype is None:
        dtype = a.dtype
    if shape is None:
        shape = a.shape
    return empty_pinned(shape, dtype, order)


def zeros_pinned(shape, dtype=float, order='C', *, like=None):
    out = empty_pinned(shape, dtype, order, like=like)
    out[...] = 0
    return out


def zeros_like_pinned(a, dtype=None, order='K', shape=None):
    if dtype is None:
        dtype = a.dtype
    if shape is None:
        shape = a.shape
    return zeros_pinned(shape, dtype, order)
