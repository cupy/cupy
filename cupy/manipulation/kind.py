from cupy import core


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
    return core.asfortranarray(a, dtype)


# TODO(okuta): Implement asarray_chkfinite


# TODO(okuta): Implement asscalar


# TODO(okuta): Implement require
