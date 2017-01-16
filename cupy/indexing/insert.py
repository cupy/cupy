import numpy

# TODO(okuta): Implement place


# TODO(okuta): Implement put


# TODO(okuta): Implement putmask


def fill_diagonal(a, val, wrap=False):
    """Fill the main diagonal of the given array of any dimensionality.

    For an array `a` with ``a.ndim > 2``, the diagonal is the list of
    locations with indices ``a[i, i, ..., i]`` all identical. This function
    modifies the input array in-place, it does not return a value.

    Args:
        a (cupy.ndarray): The array, at least 2-D.
        val (scalar): The value to be written on the diagonal.
            Its type must be compatible with that of the array a.
        wrap (bool): If specified, the diagonal is "wrapped" after N columns.
            This affects only tall matrices.

    Examples
    --------
    >>> a = cupy.zeros((3, 3), int)
    >>> cupy.fill_diagonal(a, 5)
    >>> a
    array([[5, 0, 0],
           [0, 5, 0],
           [0, 0, 5]])

     .. seealso:: :func:`numpy.fill_diagonal`
    """
    # The followings are imported from the original numpy
    if a.ndim < 2:
        raise ValueError("array must be at least 2-d")
    end = None
    if a.ndim == 2:
        step = a.shape[1] + 1
        if not wrap:
            end = a.shape[1] * a.shape[1]
    else:
        if not numpy.alltrue(numpy.diff(a.shape) == 0):
            raise ValueError("All dimensions of input must be of equal length")
        step = 1 + numpy.cumprod(a.shape[:-1]).sum()

    # Since the current cupy does not support a.flat,
    # we use a.ravel() instead of a.flat
    a.ravel()[:end:step] = val
