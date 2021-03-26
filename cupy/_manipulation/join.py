import cupy
from cupy import _core


def column_stack(tup):
    """Stacks 1-D and 2-D arrays as columns into a 2-D array.

    A 1-D array is first converted to a 2-D column array. Then, the 2-D arrays
    are concatenated along the second axis.

    Args:
        tup (sequence of arrays): 1-D or 2-D arrays to be stacked.

    Returns:
        cupy.ndarray: A new 2-D array of stacked columns.

    .. seealso:: :func:`numpy.column_stack`

    """
    if any(not isinstance(a, cupy.ndarray) for a in tup):
        raise TypeError('Only cupy arrays can be column stacked')

    lst = list(tup)
    for i, a in enumerate(lst):
        if a.ndim == 1:
            a = a[:, cupy.newaxis]
            lst[i] = a
        elif a.ndim != 2:
            raise ValueError(
                'Only 1 or 2 dimensional arrays can be column stacked')

    return concatenate(lst, axis=1)


def concatenate(tup, axis=0, out=None):
    """Joins arrays along an axis.

    Args:
        tup (sequence of arrays): Arrays to be joined. All of these should have
            same dimensionalities except the specified axis.
        axis (int or None): The axis to join arrays along.
            If axis is None, arrays are flattened before use.
            Default is 0.
        out (cupy.ndarray): Output array.

    Returns:
        cupy.ndarray: Joined array.

    .. seealso:: :func:`numpy.concatenate`

    """
    if axis is None:
        tup = [m.ravel() for m in tup]
        axis = 0
    return _core.concatenate_method(tup, axis, out)


def dstack(tup):
    """Stacks arrays along the third axis.

    Args:
        tup (sequence of arrays): Arrays to be stacked. Each array is converted
            by :func:`cupy.atleast_3d` before stacking.

    Returns:
        cupy.ndarray: Stacked array.

    .. seealso:: :func:`numpy.dstack`

    """
    return concatenate([cupy.atleast_3d(m) for m in tup], 2)


def hstack(tup):
    """Stacks arrays horizontally.

    If an input array has one dimension, then the array is treated as a
    horizontal vector and stacked along the first axis. Otherwise, the array is
    stacked along the second axis.

    Args:
        tup (sequence of arrays): Arrays to be stacked.

    Returns:
        cupy.ndarray: Stacked array.

    .. seealso:: :func:`numpy.hstack`

    """
    arrs = [cupy.atleast_1d(a) for a in tup]
    axis = 1
    if arrs[0].ndim == 1:
        axis = 0
    return concatenate(arrs, axis)


def vstack(tup):
    """Stacks arrays vertically.

    If an input array has one dimension, then the array is treated as a
    horizontal vector and stacked along the additional axis at the head.
    Otherwise, the array is stacked along the first axis.

    Args:
        tup (sequence of arrays): Arrays to be stacked. Each array is converted
            by :func:`cupy.atleast_2d` before stacking.

    Returns:
        cupy.ndarray: Stacked array.

    .. seealso:: :func:`numpy.dstack`

    """
    return concatenate([cupy.atleast_2d(m) for m in tup], 0)


def stack(tup, axis=0, out=None):
    """Stacks arrays along a new axis.

    Args:
        tup (sequence of arrays): Arrays to be stacked.
        axis (int): Axis along which the arrays are stacked.
        out (cupy.ndarray): Output array.

    Returns:
        cupy.ndarray: Stacked array.

    .. seealso:: :func:`numpy.stack`
    """
    return concatenate([cupy.expand_dims(x, axis) for x in tup], axis, out)
