import itertools
import numpy


import cupy
from cupy import core

try:
    from numpy.core.numeric import normalize_axis_tuple
except ImportError:
    # TODO: remove this copy once minimum supported is numpy >= 1.13.


    def normalize_axis_tuple(axis, ndim, argname=None, allow_duplicate=False):
        """
        Normalizes an axis argument into a tuple of non-negative integer axes.

        This handles shorthands such as ``1`` and converts them to ``(1,)``,
        as well as performing the handling of negative indices covered by
        `normalize_axis_index`.

        By default, this forbids axes from being specified multiple times.

        Used internally by multi-axis-checking logic.

        .. versionadded:: 1.13.0

        Parameters
        ----------
        axis : int, iterable of int
            The un-normalized index or indices of the axis.
        ndim : int
            The number of dimensions of the array that `axis` should be
            normalized against.
        argname : str, optional
            A prefix to put before the error message, typically the name of the
            argument.
        allow_duplicate : bool, optional
            If False, the default, disallow an axis from being specified twice.

        Returns
        -------
        normalized_axes : tuple of int
            The normalized axis index, such that `0 <= normalized_axis < ndim`

        Raises
        ------
        AxisError
            If any axis provided is out of range
        ValueError
            If an axis is repeated

        See also
        --------
        normalize_axis_index : normalizing a single scalar axis
        """
        import operator
        from numpy.core.multiarray import normalize_axis_index
        # Optimization to speed-up the most common cases.
        if type(axis) not in (tuple, list):
            try:
                axis = [operator.index(axis)]
            except TypeError:
                pass
        # Going via an iterator directly is slower than via list comprehension.
        axis = tuple([normalize_axis_index(ax, ndim, argname) for ax in axis])
        if not allow_duplicate and len(set(axis)) != len(axis):
            if argname:
                raise ValueError(
                    'repeated axis in `{}` argument'.format(argname))
            else:
                raise ValueError('repeated axis')
        return axis


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
        raise core.core._AxisError('Input must be >= 1-d')

    axis = int(axis)
    if not -a_ndim <= axis < a_ndim:
        raise core.core._AxisError(
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
        axis = normalize_axis_tuple(axis, a.ndim, allow_duplicate=True)
        broadcasted = numpy.broadcast(shift, axis)
        if broadcasted.ndim > 1:
            raise ValueError(
                "'shift' and 'axis' should be scalars or 1D sequences")
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
