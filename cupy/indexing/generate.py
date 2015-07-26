import numpy

from cupy import cuda


# TODO(beam2d): Implement these
# class c_(object):
# class r_(object):
# class s_(object):


def indices(dimensions, dtype=numpy.int_):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def ix_(*args, **kwargs):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def ravel_multi_index(multi_index, dims, mode='raise', order='C',
                      allocator=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def unravel_index(indices, dims, order='C', allocator=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def diag_indices(n, ndim=2, allocator=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def diag_indices_from(arr, allocator=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def mask_indices(n, mask_func, k=0, allocator=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def tril_indices(n, k=0, m=None, allocator=cuda.alloc):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def tril_indices_from(arr, k=0, allocator=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def triu_indices(n, k=0, m=None, allocator=cuda.alloc):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def triu_indices_from(arr, k=0, allocator=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError
