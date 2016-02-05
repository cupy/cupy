from cupy import core


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
    return core.rollaxis(a, axis, start)


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
    # TODO(okuta): check type
    return a.swapaxes(axis1, axis2)


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
    # TODO(okuta): check type
    return a.transpose(axes)
