import numpy

import cupy


def shape(a):
    """Returns the shape of an array

    Args:
        a (array_like): Input array

    Returns:
        tuple of ints: The elements of the shape tuple give the lengths of the
        corresponding array dimensions.

    """
    if isinstance(a, cupy.ndarray):
        return a.shape
    else:
        return numpy.shape(a)


def reshape(a, newshape, order='C'):
    """Returns an array with new shape and same elements.

    It tries to return a view if possible, otherwise returns a copy.

    Args:
        a (cupy.ndarray): Array to be reshaped.
        newshape (int or tuple of ints): The new shape of the array to return.
            If it is an integer, then it is treated as a tuple of length one.
            It should be compatible with ``a.size``. One of the elements can be
            -1, which is automatically replaced with the appropriate value to
            make the shape compatible with ``a.size``.
        order ({'C', 'F', 'A'}):
            Read the elements of ``a`` using this index order, and place the
            elements into the reshaped array using this index order.
            'C' means to read / write the elements using C-like index order,
            with the last axis index changing fastest, back to the first axis
            index changing slowest. 'F' means to read / write the elements
            using Fortran-like index order, with the first index changing
            fastest, and the last index changing slowest. Note that the 'C'
            and 'F' options take no account of the memory layout of the
            underlying array, and only refer to the order of indexing. 'A'
            means to read / write the elements in Fortran-like index order if
            a is Fortran contiguous in memory, C-like order otherwise.

    Returns:
        cupy.ndarray: A reshaped view of ``a`` if possible, otherwise a copy.

    .. seealso:: :func:`numpy.reshape`

    """
    # TODO(okuta): check type
    return a.reshape(newshape, order=order)


def ravel(a, order='C'):
    """Returns a flattened array.

    It tries to return a view if possible, otherwise returns a copy.

    This function currently does not support the ``order = 'K'`` option.

    Args:
        a (cupy.ndarray): Array to be flattened.
        order ({'C', 'F', 'A'}):
            Read the elements of ``a`` using this index order, and place the
            elements into the reshaped array using this index order.
            'C' means to read / write the elements using C-like index order,
            with the last axis index changing fastest, back to the first axis
            index changing slowest. 'F' means to read / write the elements
            using Fortran-like index order, with the first index changing
            fastest, and the last index changing slowest. Note that the 'C'
            and 'F' options take no account of the memory layout of the
            underlying array, and only refer to the order of indexing. 'A'
            means to read / write the elements in Fortran-like index order if
            a is Fortran contiguous in memory, C-like order otherwise.

    Returns:
        cupy.ndarray: A flattened view of ``a`` if possible, otherwise a copy.

    .. seealso:: :func:`numpy.ravel`

    """
    # TODO(beam2d, grlee77): Support ordering option K
    # TODO(okuta): check type
    return a.ravel(order)
