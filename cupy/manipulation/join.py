import numpy
import six

import cupy


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


def concatenate(tup, axis=0):
    """Joins arrays along an axis.

    Args:
        tup (sequence of arrays): Arrays to be joined. All of these should have
            same dimensionalities except the specified axis.
        axis (int): The axis to join arrays along.

    Returns:
        cupy.ndarray: Joined array.

    .. seealso:: :func:`numpy.concatenate`

    """
    ndim = None
    shape = None
    for a in tup:
        if not isinstance(a, cupy.ndarray):
            raise TypeError('Only cupy arrays can be concatenated')
        if a.ndim == 0:
            raise TypeError('zero-dimensional arrays cannot be concatenated')
        if ndim is None:
            ndim = a.ndim
            shape = list(a.shape)
            axis = _get_positive_axis(a.ndim, axis)
            continue

        if a.ndim != ndim:
            raise ValueError(
                'All arrays to concatenate must have the same ndim')
        if any(i != axis and shape[i] != a.shape[i]
               for i in six.moves.range(ndim)):
            raise ValueError(
                'All arrays must have same shape except the axis to '
                'concatenate')
        shape[axis] += a.shape[axis]

    if ndim is None:
        raise ValueError('Cannot concatenate from empty tuple')

    dtype = numpy.find_common_type([a.dtype for a in tup], [])
    ret = cupy.empty(shape, dtype=dtype)

    skip = (slice(None),) * axis
    i = 0
    for a in tup:
        aw = a.shape[axis]
        ret[skip + (slice(i, i + aw),)] = a
        i += aw

    return ret


def dstack(tup):
    """Stacks arrays along the third axis.

    Args:
        tup (sequence of arrays): Arrays to be stacked. Each array is converted
            by :func:`cupy.atleast_3d` before stacking.

    Returns:
        cupy.ndarray: Stacked array.

    .. seealso:: :func:`numpy.dstack`

    """
    return concatenate(cupy.atleast_3d(*tup), 2)


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
    return concatenate(tup, axis)


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
    return concatenate(cupy.atleast_2d(*tup), 0)


def _get_positive_axis(ndim, axis):
    a = axis
    if a < 0:
        a += ndim
    if a < 0 or a >= ndim:
        raise IndexError('axis {} out of bounds [0, {})'.format(axis, ndim))
    return a
