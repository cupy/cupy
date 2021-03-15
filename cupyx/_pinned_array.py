import numpy

from cupy import cuda
from cupy.core import internal


def empty_pinned(shape, dtype=float, order='C'):
    """Returns an array without initializing the elements.

    This is a convenience function which is just :func:`numpy.empty`,
    except that the underlying memory is allocated from CuPy's pinned
    memory pool.

    Args:
        shape (int or tuple of ints): Dimensionalities of the array.
        dtype: Data type specifier.
        order ({'C', 'F'}): Row-major (C-style) or column-major
            (Fortran-style) order.

    Returns:
        numpy.ndarray: A new array with elements not initialized.

    .. seealso:: :func:`numpy.empty`

    """
    shape = (shape,) if isinstance(shape, int) else tuple(shape)
    nbytes = internal.prod(shape) * numpy.dtype(dtype).itemsize
    mem = cuda.alloc_pinned_memory(nbytes)
    out = numpy.ndarray(shape, dtype=dtype, buffer=mem, order=order)
    return out


def empty_like_pinned(a, dtype=None, order='K', subok=None, shape=None):
    """Returns a new array with same shape and dtype of a given array.

    This is a convenience function which is just :func:`numpy.empty_like`,
    except that the underlying memory is allocated from CuPy's pinned
    memory pool.

    This function currently does not support ``subok`` option.

    Args:
        a (numpy.ndarray or cupy.ndarray): Base array.
        dtype: Data type specifier. The data type of ``a`` is used by default.
        order ({'C', 'F', 'A', or 'K'}): Overrides the memory layout of the
            result. ``'C'`` means C-order, ``'F'`` means F-order, ``'A'`` means
            ``'F'`` if ``a`` is Fortran contiguous, ``'C'`` otherwise.
            ``'K'`` means match the layout of ``a`` as closely as possible.
        subok: Not supported yet, must be None.
        shape (int or tuple of ints): Overrides the shape of the result. If
            ``order='K'`` and the number of dimensions is unchanged, will try
            to keep order, otherwise, ``order='C'`` is implied.

    Returns:
        numpy.ndarray: A new array with same shape and dtype of ``a`` with
        elements not initialized.

    .. seealso:: :func:`numpy.empty_like`

    """
    if subok is not None:
        raise TypeError('subok is not supported yet')
    if dtype is None:
        dtype = a.dtype
    if shape is None:
        shape = a.shape
    return empty_pinned(shape, dtype, order)


def zeros_pinned(shape, dtype=float, order='C'):
    """Returns a new array of given shape and dtype, filled with zeros.

    This is a convenience function which is just :func:`numpy.zeros`,
    except that the underlying memory is allocated from CuPy's pinned
    memory pool.

    Args:
        shape (int or tuple of ints): Dimensionalities of the array.
        dtype: Data type specifier.
        order ({'C', 'F'}): Row-major (C-style) or column-major
            (Fortran-style) order.

    Returns:
        numpy.ndarray: An array filled with zeros.

    .. seealso:: :func:`numpy.zeros`

    """
    out = empty_pinned(shape, dtype, order)
    out[...] = 0
    return out


def zeros_like_pinned(a, dtype=None, order='K', subok=None, shape=None):
    """Returns an array of zeros with same shape and dtype as a given array.

    This is a convenience function which is just :func:`numpy.zeros_like`,
    except that the underlying memory is allocated from CuPy's pinned
    memory pool.

    This function currently does not support ``subok`` option.

    Args:
        a (numpy.ndarray or cupy.ndarray): Base array.
        dtype: Data type specifier. The dtype of ``a`` is used by default.
        order ({'C', 'F', 'A', or 'K'}): Overrides the memory layout of the
            result. ``'C'`` means C-order, ``'F'`` means F-order, ``'A'`` means
            ``'F'`` if ``a`` is Fortran contiguous, ``'C'`` otherwise.
            ``'K'`` means match the layout of ``a`` as closely as possible.
        subok: Not supported yet, must be None.
        shape (int or tuple of ints): Overrides the shape of the result. If
            ``order='K'`` and the number of dimensions is unchanged, will try
            to keep order, otherwise, ``order='C'`` is implied.

    Returns:
        numpy.ndarray: An array filled with zeros.

    .. seealso:: :func:`numpy.zeros_like`

    """
    if subok is not None:
        raise TypeError('subok is not supported yet')
    if dtype is None:
        dtype = a.dtype
    if shape is None:
        shape = a.shape
    return zeros_pinned(shape, dtype, order)
