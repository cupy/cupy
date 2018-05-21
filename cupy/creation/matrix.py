import numpy

import cupy


def diag(v, k=0):
    """Returns a diagonal or a diagonal array.

    Args:
        v (array-like): Array or array-like object.
        k (int): Index of diagonals. Zero indicates the main diagonal, a
            positive value an upper diagonal, and a negative value a lower
            diagonal.

    Returns:
        cupy.ndarray: If ``v`` indicates a 1-D array, then it returns a 2-D
        array with the specified diagonal filled by ``v``. If ``v`` indicates a
        2-D array, then it returns the specified diagonal of ``v``. In latter
        case, if ``v`` is a :class:`cupy.ndarray` object, then its view is
        returned.

    .. seealso:: :func:`numpy.diag`

    """
    if numpy.isscalar(v):
        return cupy.array(numpy.diag(v, k))

    if isinstance(v, cupy.ndarray):
        ndim = v.ndim
    else:
        ndim = numpy.ndim(v)
        # to save bandwidth, don't copy non-diag elements to GPU
        xp = cupy if ndim == 1 else numpy
        v = xp.array(v)

    if ndim == 1:
        size = v.size + abs(k)
        ret = cupy.zeros((size, size), dtype=v.dtype)
        ret.diagonal(k)[:] = v
        return ret
    else:
        return cupy.array(v.diagonal(k))


def diagflat(v, k=0):
    """Creates a diagonal array from the flattened input.

    Args:
        v (array-like): Array or array-like object.
        k (int): Index of diagonals. See :func:`cupy.diag` for detail.

    Returns:
        cupy.ndarray: A 2-D diagonal array with the diagonal copied from ``v``.

    """
    if numpy.isscalar(v):
        return cupy.array(numpy.diagflat(v, k))

    return cupy.diag(v.ravel(), k)


def tri(N, M=None, k=0, dtype=float):
    """Creates an array with ones at and below the given diagonal.

    Args:
        N (int): Number of rows.
        M (int): Number of columns. M == N by default.
        k (int): The sub-diagonal at and below which the array is filled. Zero
            is the main diagonal, a positive value is above it, and a negative
            value is below.
        dtype: Data type specifier.

    Returns:
        cupy.ndarray: An array with ones at and below the given diagonal.

    .. seealso:: :func:`numpy.tri`

    """
    if M is None:
        M = N
    out = cupy.empty((N, M), dtype=dtype)

    return cupy.ElementwiseKernel(
        'int32 m, int32 k',
        'T out',
        '''
        int row = i % m;
        int col = i / m;
        out = (row <= col + k);
        ''',
        'tri',
    )(M, k, out)


def tril(m, k=0):
    """Returns a lower triangle of an array.

    Args:
        m (array-like): Array or array-like object.
        k (int): The diagonal above which to zero elements. Zero is the main
            diagonal, a positive value is above it, and a negative value is
            below.

    Returns:
        cupy.ndarray: A lower triangle of an array.

    .. seealso:: :func:`numpy.tril`

    """
    m = cupy.asarray(m)
    mask = tri(*m.shape[-2:], k=k, dtype=bool)

    return cupy.where(mask, m, m.dtype.type(0))


def triu(m, k=0):
    """Returns an upper triangle of an array.

    Args:
        m (array-like): Array or array-like object.
        k (int): The diagonal below which to zero elements. Zero is the main
            diagonal, a positive value is above it, and a negative value is
            below.

    Returns:
        cupy.ndarray: An upper triangle of an array.

    .. seealso:: :func:`numpy.triu`

    """
    m = cupy.asarray(m)
    mask = tri(*m.shape[-2:], k=k-1, dtype=bool)

    return cupy.where(mask, m.dtype.type(0), m)


# TODO(okuta): Implement vander


# TODO(okuta): Implement mat


# TODO(okuta): Implement bmat
