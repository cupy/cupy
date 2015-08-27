import six


def rollaxis(a, axis, start=0):
    """Moves the specified axis backwards to the given place.

    Args:
        a (cupy.ndarray): Array to move the axis.
        axis (int): The axis to move.
        start (int): The place to which the axis is moved.

    Returns:
        cupy.ndarray: A view of ``a`` that the axis is moved to ``start``.

    .. seealso:: :func:`numpy.rollaxis`

    """
    ndim = a.ndim
    if axis < 0:
        axis += ndim
    if start < 0:
        start += ndim
    if not (0 <= axis < ndim and 0 <= start <= ndim):
        raise ValueError('Axis out of range')
    if axis < start:
        start -= 1
    if axis == start:
        return a
    if ndim == 2:
        return transpose(a, None)

    axes = list(six.moves.range(ndim))
    del axes[axis]
    axes.insert(start, axis)
    return transpose(a, axes)


def swapaxes(a, axis1, axis2):
    """Swaps the two axes.

    Args:
        a (cupy.ndarray): Array to swap the axes.
        axis1 (int): The first axis to swap.
        axis2 (int): The second axis to swap.

    Returns:
        cupy.ndarray: A view of ``a`` that the two axes are swapped.

    .. seealso:: :func:`numpy.swapaxes`

    """
    ndim = a.ndim
    if axis1 >= ndim or axis2 >= ndim:
        raise ValueError('Axis out of range')
    axes = list(six.moves.range(ndim))
    axes[axis1], axes[axis2] = axes[axis2], axes[axis1]
    return transpose(a, axes)


def transpose(a, axes=None):
    """Permutes the dimensions of an array.

    Args:
        a (cupy.ndarray): Array to permute the dimensions.
        axes (tuple of ints): Permutation of the dimensions. This function
            reverses the shape by default.

    Returns:
        cupy.ndarray: A view of ``a`` that the dimensions are permuted.

    .. seealso:: :func:`numpy.transpose`

    """
    ndim = a.ndim
    a_shape = a._shape
    a_strides = a._strides
    ret = a.view()

    if not axes:
        if ndim > 1:
            ret._shape = a_shape[::-1]
            ret._strides = a_strides[::-1]
            ret._c_contiguous, ret._f_contiguous = \
                a._f_contiguous, a._c_contiguous
        return ret

    if ndim != len(axes):
        raise ValueError('Invalid axes value: %s' % str(axes))

    if ndim <= 2:
        if ndim == 0:
            return ret
        elif ndim == 1:
            if axes[0] == 0:
                return ret
        else:
            axis0, axis1 = axes
            if axis0 == 0 and axis1 == 1:
                return ret
            elif axis0 == 1 and axis1 == 0:
                ret._shape = a_shape[::-1]
                ret._strides = a_strides[::-1]
                ret._c_contiguous, ret._f_contiguous = \
                    a._f_contiguous, a._c_contiguous
                return ret
        raise ValueError('Invalid axes value: %s' % str(axes))

    for axis in axes:
        if axis < -ndim or axis >= ndim:
            raise IndexError('Axes overrun')
    axes = [axis % ndim for axis in axes]

    a_axes = list(six.moves.range(ndim))

    if a_axes == axes:
        return ret

    if a_axes == axes[::-1]:
        ret._shape = a_shape[::-1]
        ret._strides = a_strides[::-1]
        ret._c_contiguous, ret._f_contiguous = \
            a._f_contiguous, a._c_contiguous
        return ret

    if a_axes != sorted(axes):
        raise ValueError('Invalid axes value: %s' % str(axes))

    ret._shape = tuple([a_shape[axis] for axis in axes])
    ret._strides = tuple([a_strides[axis] for axis in axes])
    ret._c_contiguous = -1
    ret._f_contiguous = -1

    return ret
