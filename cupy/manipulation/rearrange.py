import cupy

# TODO(okuta): Implement fliplr


# TODO(okuta): Implement flipud


def roll(a, shift, axis=None):
    """Roll array elements along a given axis.

    Args:
        a (~cupy.ndarray): Array to be rolled.
        shift (int): The number of places by which elements are shifted.
        axis (int or None): The axis along which elements are shifted.
            If ``axis`` is ``None``, the array is flattend before shifting,
            and afther that it is reshaped to the original shape.

    Returns:
        ~cupy.ndarray: Output array.

    .. seealso:: :func:`numpy.roll`

    """
    if axis is None:
        size = a.size
    else:
        axis = int(axis)
        if axis >= a.ndim:
            raise ValueError('axis must be >= 0 and < %d' % a.ndim)
        size = a.shape[axis]
    if size == 0:
        return a
    shift %= size
    indexes = cupy.concatenate(
        (cupy.arange(size - shift, size), cupy.arange(size - shift)))
    res = a.take(indexes, axis)

    if axis is None:
        res = res.reshape(a.shape)
    return res

# TODO(okuta): Implement rot90
