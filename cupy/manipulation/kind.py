import cupy


# TODO(okuta): Implement asfarray


def asfortranarray(a, dtype=None):
    """Return an array laid out in Fortran order in memory.

    Args:
        a (~cupy.ndarray): The input array.
        dtype (str or dtype object, optional): By default, the data-type is
            inferred from the input data.

    Returns:
        ~cupy.ndarray: The input `a` in Fortran, or column-major, order.

    .. seealso:: :func:`numpy.asfortranarray`

    """
    ret = cupy.empty(a.shape[::-1], a.dtype if dtype is None else dtype).T
    ret[...] = a
    return ret


# TODO(okuta): Implement asarray_chkfinite


# TODO(okuta): Implement asscalar


# TODO(okuta): Implement require
