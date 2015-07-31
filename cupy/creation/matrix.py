import numpy

import cupy
from cupy import cuda


def diag(v, k=0, allocator=None):
    """Returns a diagonal or a diagonal array.

    Args:
        v (array-like): Array or array-like object.
        k (int): Index of diagonals. Zero indicates the main diagonal, a
            positive value an upper diagonal, and a negative value a lower
            diagonal.
        allocator (function): CuPy memory allocator. If ``v`` is a cupy.ndarray
            object, its allocator is used by default. Otherwise,
            :func:`cupy.cuda.alloc` is used by default.

    Returns:
        cupy.ndarray: If ``v`` indicates a 1-D array, then it returns a 2-D
        array with the specified diagonal filled by ``v``. If ``v`` indicates a
        2-D array, then it returns the specified diagonal of ``v``. In latter
        case, if ``v`` is a cupy.ndarray object, then its view is returned.

    .. seealso:: :func:`numpy.diag`

    """
    if isinstance(v, cupy.ndarray):
        if v.ndim == 1:
            size = v.size + abs(k)
            if allocator is None:
                allocator = v.allocator
            ret = cupy.zeros((size, size), dtype=v.dtype, allocator=allocator)
            ret.diagonal(k)[:] = v
            return ret
        else:
            return v.diagonal(k)
    else:
        if allocator is None:
            allocator = cuda.alloc
        return cupy.array(numpy.diag(v, k), allocator=allocator)


def diagflat(v, k=0, allocator=None):
    """Creates a diagonal array from the flattened input.

    Args:
        v (array-like): Array or array-like object.
        k (int): Index of diagonals. See :func:`cupy.diag` for detail.
        allocator (function): CuPy memory allocator. If ``v`` is a cupy.ndarray
            object, its allocator is used by default. Otherwise,
            :func:`cupy.cuda.alloc` is used by default.

    Returns:
        cupy.ndarray: A 2-D diagonal array with the diagonal copied from ``v``.

    """
    if isinstance(v, cupy.ndarray):
        return cupy.diag(v.ravel(), k, allocator)
    else:
        return cupy.diag(numpy.ndarray(v).ravel(), k, allocator)


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
