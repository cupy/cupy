import numpy

import cupy
from cupy import cuda


def empty(shape, dtype=numpy.float32, allocator=cuda.alloc):
    # TODO(beam2d): Support ordering option
    return cupy.ndarray(shape, dtype=dtype, allocator=allocator)


def empty_like(a, dtype=None, allocator=None):
    # TODO(beam2d): Support ordering option
    if dtype is None:
        dtype = a.dtype
    if allocator is None:
        allocator = a.allocator
    return empty(a.shape, dtype=dtype, allocator=allocator)


def eye(N, M=None, k=0, dtype=float, allocator=cuda.alloc):
    if M is None:
        M = N
    ret = zeros((N, M), dtype, allocator)
    ret.diagonal(k)[:] = 1
    return ret


def identity(n, dtype=float, allocator=cuda.alloc):
    return eye(n, dtype=dtype, allocator=allocator)


def ones(shape, dtype=numpy.float32, allocator=cuda.alloc):
    # TODO(beam2d): Support ordering option
    return full(shape, 1, dtype, allocator)


def ones_like(a, dtype=None, allocator=None):
    # TODO(beam2d): Support ordering option
    if dtype is None:
        dtype = a.dtype
    if allocator is None:
        allocator = a.allocator
    return ones(a.shape, dtype, allocator)


def zeros(shape, dtype=numpy.float32, allocator=cuda.alloc):
    # TODO(beam2d): Support ordering option
    a = empty(shape, dtype, allocator)
    a.data.memset(0, a.nbytes)
    return a


def zeros_like(a, dtype=None, allocator=None):
    # TODO(beam2d): Support ordering option
    if dtype is None:
        dtype = a.dtype
    if allocator is None:
        allocator = a.allocator
    return zeros(a.shape, dtype=dtype, allocator=allocator)


def full(shape, fill_value, dtype=None, allocator=cuda.alloc):
    # TODO(beam2d): Support ordering option
    a = empty(shape, dtype, allocator)
    a.fill(fill_value)
    return a


def full_like(a, fill_value, dtype=None, allocator=None):
    # TODO(beam2d): Support ordering option
    if dtype is None:
        dtype = a.dtype
    if allocator is None:
        allocator = a.allocator
    return full(a.shape, fill_value, dtype, allocator)
