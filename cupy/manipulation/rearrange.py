import cupy

# TODO(okuta): Implement fliplr


# TODO(okuta): Implement flipud


def roll(a, shift, axis=None):
    """Roll array elements along a given axis.

    Args:
        a (~cupy.ndarray): Array to be rolled.
        shift (int): The number of places by which elements are shifted.
        axis (int or None): The axis along which elements are shifted.
            If ``axis`` is ``None``, the array is flattened before shifting,
            and after that it is reshaped to the original shape.

    Returns:
        ~cupy.ndarray: Output array.

    .. seealso:: :func:`numpy.roll`

    """
    if axis is None:
        if a.size == 0:
            return a
        size = a.size
        ra = a.ravel()
        shift %= size
        res = cupy.empty((size,), a.dtype)
        res[:shift] = ra[size - shift:]
        res[shift:] = ra[:size - shift]
        return res.reshape(a.shape)
    else:
        axis = int(axis)
        if axis < 0:
            axis += a.ndim
        if not 0 <= axis < a.ndim:
            raise ValueError('axis must be >= %d and < %d' % (-a.ndim, a.ndim))
        size = a.shape[axis]
        if size == 0:
            return a
        shift %= size
        prev = (slice(None),) * axis
        rest = (slice(None),) * (a.ndim - axis - 1)
        # Roll only the dimensiont at the given axis
        # ind1 is [:, ..., size-shift:, ..., :]
        # ind2 is [:, ..., :size-shift, ..., :]
        ind1 = prev + (slice(size - shift, None, None),) + rest
        ind2 = prev + (slice(None, size - shift, None),) + rest
        r_ind1 = prev + (slice(None, shift, None),) + rest
        r_ind2 = prev + (slice(shift, None, None),) + rest
        res = cupy.empty_like(a)
        res[r_ind1] = a[ind1]
        res[r_ind2] = a[ind2]
        return res

# TODO(okuta): Implement rot90
