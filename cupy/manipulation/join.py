import numpy
import six

import cupy


def column_stack(tup, allocator=None):
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

    return concatenate(lst, axis=1, allocator=None)


def concatenate(tup, axis=0, allocator=None):
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
    if allocator is None:
        allocator = tup[0].allocator
    ret = cupy.empty(shape, dtype=dtype, allocator=allocator)

    skip = (slice(None),) * axis
    i = 0
    for a in tup:
        aw = a.shape[axis]
        ret[skip + (slice(i, i + aw),)] = a
        i += aw

    return ret


def dstack(tup, allocator=None):
    lst = list(tup)
    for i, a in enumerate(lst):
        if not isinstance(a, cupy.ndarray):
            raise TypeError('Only cupy arrays can be dstacked')
        if a.ndim == 1:
            lst[i] = a[cupy.newaxis, :, cupy.newaxis]
        elif a.ndim == 2:
            lst[i] = a[:, :, cupy.newaxis]

    return concatenate(lst, 2, allocator)


def hstack(tup, allocator=None):
    arrs = [cupy.atleast_1d(a) for a in tup]
    axis = 1
    if arrs[0].ndim == 1:
        axis = 0
    return concatenate(tup, axis, allocator)


def vstack(tup, allocator=None):
    return concatenate([cupy.atleast_2d(a) for a in tup], 0, allocator)


def _get_positive_axis(ndim, axis):
    a = axis
    if a < 0:
        a += ndim
    if a < 0 or a >= ndim:
        raise IndexError('axis {} out of bounds [0, {})'.format(axis, ndim))
    return a
