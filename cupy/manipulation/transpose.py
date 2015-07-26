import collections

import six


def rollaxis(a, axis, start=0):
    if axis >= a.ndim:
        raise ValueError('Axis out of range')
    tr = list(six.moves.range(a.ndim))
    del tr[axis]
    tr.insert(start, axis)
    return a.transpose(*tr)


def swapaxes(a, axis1, axis2):
    if axis1 >= a.ndim or axis2 >= a.ndim:
        raise ValueError('Axis out of range')
    tr = list(six.moves.range(a.ndim))
    tr[axis1], tr[axis2] = tr[axis2], tr[axis1]
    return a.transpose(*tr)


def transpose(a, axes=None):
    if not axes:
        axes = tuple(reversed(six.moves.range(a.ndim)))
    elif len(axes) == 1 and isinstance(axes[0], collections.Iterable):
        axes = tuple(axes[0])

    if any(axis < -a.ndim or axis >= a.ndim for axis in axes):
        raise IndexError('Axes overrun')

    axes = tuple(axis % a.ndim for axis in axes)

    if list(six.moves.range(a.ndim)) != sorted(axes):
        raise ValueError('Invalid axes value: %s' % str(axes))

    newarray = a.view()
    newarray._shape = tuple(a._shape[axis] for axis in axes)
    newarray._strides = tuple(a._strides[axis] for axis in axes)
    newarray._update_contiguity()
    return newarray
