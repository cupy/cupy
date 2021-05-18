from cupy import _core
from cupy._core import _routines_manipulation


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
    return _core.rollaxis(a, axis, start)


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


def moveaxis(a, source, destination):
    """Moves axes of an array to new positions.

    Other axes remain in their original order.

    Args:
        a (cupy.ndarray): Array whose axes should be reordered.
        source (int or sequence of int):
            Original positions of the axes to move. These must be unique.
        destination (int or sequence of int):
            Destination positions for each of the original axes. These must
            also be unique.

    Returns:
        cupy.ndarray:
        Array with moved axes. This array is a view of the input array.

    .. seealso:: :func:`numpy.moveaxis`

    """
    # TODO(fukatani): check type
    # checking __len__ attribute is faster than cupy.isscalar or isinstance
    if not (hasattr(source, '__len__') or hasattr(destination, '__len__')):
        return _routines_manipulation._move_single_axis(a, source, destination)
    return _core.moveaxis(a, source, destination)


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
