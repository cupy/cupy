import itertools

import numpy

import cupy
from cupy import core
from cupy.core._kernel import _get_axis


def flip(a, axis):
    """Reverse the order of elements in an array along the given axis.

    Note that ``flip`` function has been introduced since NumPy v1.12.
    The contents of this document is the same as the original one.

    Args:
        a (~cupy.ndarray): Input array.
        axis (int): Axis in array, which entries are reversed.

    Returns:
        ~cupy.ndarray: Output array.

    .. seealso:: :func:`numpy.flip`

    """
    a_ndim = a.ndim
    if a_ndim < 1:
        raise core._AxisError('Input must be >= 1-d')

    axis = int(axis)
    if not -a_ndim <= axis < a_ndim:
        raise core._AxisError(
            'axis must be >= %d and < %d' % (-a_ndim, a_ndim))

    return _flip(a, axis)


def fliplr(a):
    """Flip array in the left/right direction.

    Flip the entries in each row in the left/right direction. Columns
    are preserved, but appear in a different order than before.

    Args:
        a (~cupy.ndarray): Input array.

    Returns:
        ~cupy.ndarray: Output array.

    .. seealso:: :func:`numpy.fliplr`

    """
    if a.ndim < 2:
        raise ValueError('Input must be >= 2-d')
    return a[::, ::-1]


def flipud(a):
    """Flip array in the up/down direction.

    Flip the entries in each column in the up/down direction. Rows are
    preserved, but appear in a different order than before.

    Args:
        a (~cupy.ndarray): Input array.

    Returns:
        ~cupy.ndarray: Output array.

    .. seealso:: :func:`numpy.flipud`

    """
    if a.ndim < 1:
        raise ValueError('Input must be >= 1-d')
    return a[::-1]


def roll(a, shift, axis=None):
    """Roll array elements along a given axis.

    Elements that roll beyond the last position are re-introduced at the first.

    Args:
        a (~cupy.ndarray): Array to be rolled.
        shift (int or tuple of int): The number of places by which elements are
            shifted. If a tuple, then `axis` must be a tuple of the same size,
            and each of the given axes is shifted by the corresponding number.
            If an int while `axis` is a tuple of ints, then the same value is
            used for all given axes.
        axis (int or tuple of int or None): The axis along which elements are
            shifted. By default, the array is flattened before shifting, after
            which the original shape is restored.

    Returns:
        ~cupy.ndarray: Output array.

    .. seealso:: :func:`numpy.roll`

    """
    if axis is None:
        return roll(a.ravel(), shift, 0).reshape(a.shape)
    else:
        axis = _get_axis(axis, a.ndim)[0]

        broadcasted = numpy.broadcast(shift, axis)
        if broadcasted.nd > 1:
            raise ValueError(
                '\'shift\' and \'axis\' should be scalars or 1D sequences')
        shifts = {ax: 0 for ax in range(a.ndim)}
        for sh, ax in broadcasted:
            shifts[ax] += sh

        rolls = [((slice(None), slice(None)),)] * a.ndim
        for ax, offset in shifts.items():
            offset %= a.shape[ax] or 1  # If `a` is empty, nothing matters.
            if offset:
                # (original, result), (original, result)
                rolls[ax] = ((slice(None, -offset), slice(offset, None)),
                             (slice(-offset, None), slice(None, offset)))

        result = cupy.empty_like(a)
        for indices in itertools.product(*rolls):
            arr_index, res_index = zip(*indices)
            result[res_index] = a[arr_index]

        return result


def rot90(a, k=1, axes=(0, 1)):
    """Rotate an array by 90 degrees in the plane specified by axes.

    Note that ``axes`` argument has been introduced since NumPy v1.12.
    The contents of this document is the same as the original one.

    Args:
        a (~cupy.ndarray): Array of two or more dimensions.
        k (int): Number of times the array is rotated by 90 degrees.
        axes: (tuple of ints): The array is rotated in the plane defined by
            the axes. Axes must be different.

    Returns:
        ~cupy.ndarray: Output array.

    .. seealso:: :func:`numpy.rot90`

    """
    a_ndim = a.ndim
    if a_ndim < 2:
        raise ValueError('Input must be >= 2-d')

    axes = tuple(axes)
    if len(axes) != 2:
        raise ValueError('len(axes) must be 2')
    if axes[0] == axes[1] or abs(axes[0] - axes[1]) == a_ndim:
        raise ValueError('axes must be different')
    if not (-a_ndim <= axes[0] < a_ndim and -a_ndim <= axes[1] < a_ndim):
        raise ValueError('axes must be >= %d and < %d' % (-a_ndim, a_ndim))

    k = k % 4

    if k == 0:
        return a[:]
    if k == 2:
        return _flip(_flip(a, axes[0]), axes[1])

    axes_t = list(range(0, a_ndim))
    axes_t[axes[0]], axes_t[axes[1]] = axes_t[axes[1]], axes_t[axes[0]]

    if k == 1:
        return cupy.transpose(_flip(a, axes[1]), axes_t)
    else:
        return _flip(cupy.transpose(a, axes_t), axes[1])


def _flip(a, axis):
    # This function flips array without checking args.
    indexer = [slice(None)] * a.ndim
    indexer[axis] = slice(None, None, -1)

    return a[tuple(indexer)]
