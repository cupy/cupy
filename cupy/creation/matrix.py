import numpy

import cupy


def diag(v, k=0):
    if v.ndim == 1:
        size = v.size + abs(k)
        ret = cupy.zeros((size, size), dtype=v.dtype, allocator=v.allocator)
        ret.diagonal(k)[:] = v
    else:
        ret = v.diagonal(k)
    return ret


def diagflat(v, k=0):
    return cupy.diag(v.ravel(), k)


def tri(N, M=None, k=0, dtype=numpy.float64, allocator=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def tril(m, k=0, allocator=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def triu(m, k=0, allocator=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def vander(x, N=None, increasing=False, allocator=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def mat(data, dtype=None, allocator=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def bmat(obj, ldict=None, gdict=None, allocator=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError
