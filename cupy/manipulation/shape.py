def reshape(a, newshape):
    """Returns an array with new shape and same elements.

    It tries to return a view if possible, otherwise returns a copy.

    This function currently does not support ``order`` option.

    Args:
        a (cupy.ndarray): Array to be reshaped.
        newshape (int or tuple of ints): The new shape of the array to return.
            If it is an integer, then it is treated as a tuple of length one.
            It should be compatible with ``a.size``. One of the elements can be
            -1, which is automatically replaced with the appropriate value to
            make the shape compatible with ``a.size``.

    Returns:
        cupy.ndarray: A reshaped view of ``a`` if possible, otherwise a copy.

    .. seealso:: :func:`numpy.reshape`

    """
    # TODO(beam2d): Support ordering option
    # TODO(okuta): check type
    return a.reshape(newshape)


def ravel(a):
    """Returns a flattened array.

    It tries to return a view if possible, otherwise returns a copy.

    This function currently does not support ``order`` option.

    Args:
        a (cupy.ndarray): Array to be flattened.

    Returns:
        cupy.ndarray: A flattened view of ``a`` if possible, otherwise a copy.

    .. seealso:: :func:`numpy.ravel`

    """
    # TODO(beam2d): Support ordering option
    # TODO(okuta): check type
    return a.ravel()
