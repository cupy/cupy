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
    if axis >= a.ndim:
        raise ValueError('Axis out of range')
    tr = list(six.moves.range(a.ndim))
    del tr[axis]
    tr.insert(start, axis)
    return transpose(a, tr)


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
    if axis1 >= a.ndim or axis2 >= a.ndim:
        raise ValueError('Axis out of range')
    tr = list(six.moves.range(a.ndim))
    tr[axis1], tr[axis2] = tr[axis2], tr[axis1]
    return transpose(a, tr)


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
    newarray = a.view()

    if not axes:
        if ndim > 1:
            newarray._shape = a_shape[::-1]
            newarray._strides = a_strides[::-1]
            newarray._c_contiguous, newarray._f_contiguous = \
                newarray._f_contiguous, newarray._c_contiguous
        return newarray

    for axis in axes:
        if axis < -ndim or axis >= ndim:
            raise IndexError('Axes overrun')
    axes = [axis % ndim for axis in axes]

    a_axes = list(six.moves.range(ndim))
    if a_axes != sorted(axes):
        raise ValueError('Invalid axes value: %s' % str(axes))

    if ndim <= 1 or a_axes == axes:
        return newarray

    if a_axes == axes[::-1]:
        newarray._shape = a_shape[::-1]
        newarray._strides = a_strides[::-1]
        newarray._c_contiguous, newarray._f_contiguous = \
            newarray._f_contiguous, newarray._c_contiguous
    else:
        newarray._shape = tuple([a_shape[axis] for axis in axes])
        newarray._strides = tuple([a_strides[axis] for axis in axes])
        newarray._c_contiguous = -1
        newarray._f_contiguous = -1

    return newarray
