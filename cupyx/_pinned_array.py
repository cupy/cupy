import numpy

from cupy import cuda
from cupy._creation.basic import _new_like_order_and_strides
from cupy.core import internal


def _update_shape(a, shape):
    if shape is None and a is not None:
        shape = a.shape
    elif isinstance(shape, int):
        shape = (shape,)
    else:
        shape = tuple(shape)
    return shape


def _update_order_char(x, order_char):
    # This is a pure Python version of cupy.core.core._update_order_char()
    # update order_char based on array contiguity
    if order_char == ord('A'):
        if x.flags.f_contiguous:
            order_char = ord('F')
        else:
            order_char = ord('C')
    elif order_char == ord('K'):
        if x.flags.f_contiguous:
            order_char = ord('F')
        elif x.flags.c_contiguous:
            order_char = ord('C')
    return order_char


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
    shape = _update_shape(None, shape)
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
    # We're kinda duplicating the code here because order='K' needs special
    # treatment: strides need to be computed
    if subok is not None:
        raise TypeError('subok is not supported yet')
    if dtype is None:
        dtype = a.dtype
    shape = _update_shape(a, shape)
    order, strides, _ = _new_like_order_and_strides(
        a, dtype, order, shape, get_memptr=False, get_char=_update_order_char)
    nbytes = internal.prod(shape) * numpy.dtype(dtype).itemsize
    mem = cuda.alloc_pinned_memory(nbytes)
    out = numpy.ndarray(shape, dtype=dtype, buffer=mem,
                        strides=strides, order=order)
    return out


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
    numpy.copyto(out, 0, casting='unsafe')
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
    out = empty_like_pinned(a, dtype, order, subok, shape)
    numpy.copyto(out, 0, casting='unsafe')
    return out
