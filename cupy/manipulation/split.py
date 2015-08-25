import numpy
import six


def array_split(ary, indices_or_sections, axis=0):
    """Splits an array into multiple sub arrays along a given axis.

    This function is almost equivalent to :func:`cupy.split`. The only
    difference is that this function allows an integer sections that does not
    evenly divide the axis.

    .. seealso:: :func:`cupy.split` for more detail, :func:`numpy.array_split`

    """
    if ary.ndim <= axis:
        raise IndexError('Axis exceeds ndim')
    size = ary.shape[axis]

    if numpy.isscalar(indices_or_sections):
        each_size = (size - 1) // indices_or_sections + 1
        indices = [i * each_size
                   for i in six.moves.range(1, indices_or_sections)]
    else:
        indices = indices_or_sections

    skip = (slice(None),) * axis
    ret = []
    i = 0
    for index in indices:
        ret.append(ary[skip + (slice(i, index),)])
        i = index
    ret.append(ary[skip + (slice(index, size),)])

    return ret


def dsplit(ary, indices_or_sections):
    """Splits an array into multiple sub arrays along the third axis.

    This is equivalent to ``split`` with ``axis=2``.

    .. seealso:: :func:`cupy.split` for more detail, :func:`numpy.dsplit`

    """
    if ary.ndim <= 2:
        raise ValueError('Cannot dsplit an array with less than 3 dimensions')
    return split(ary, indices_or_sections, 2)


def hsplit(ary, indices_or_sections):
    """Splits an array into multiple sub arrays horizontally.

    This is equivalent to ``split`` with ``axis=0`` if ``ary`` has one
    dimension, and otherwise that with ``axis=1``.

    .. seealso:: :func:`cupy.split` for more detail, :func:`numpy.hsplit`

    """
    if ary.ndim == 0:
        raise ValueError('Cannot hsplit a zero-dimensional array')
    if ary.ndim == 1:
        return split(ary, indices_or_sections, 0)
    else:
        return split(ary, indices_or_sections, 1)


def split(ary, indices_or_sections, axis=0):
    """Splits an array into multiple sub arrays along a given axis.

    Args:
        ary (cupy.ndarray): Array to split.
        indices_or_sections (int or sequence of ints): A value indicating how
            to divide the axis. If it is an integer, then is treated as the
            number of sections, and the axis is evenly divided. Otherwise,
            the integers indicate indices to split at. Note that the sequence
            on the device memory is not allowed.
        axis (int): Axis along which the array is split.

    Returns:
        A list of sub arrays. Eacy array is a view of the corresponding input
        array.

    .. seealso:: :func:`numpy.split`

    """
    if ary.ndim <= axis:
        raise IndexError('Axis exceeds ndim')
    size = ary.shape[axis]

    if numpy.isscalar(indices_or_sections):
        if size % indices_or_sections != 0:
            raise ValueError(
                'indices_or_sections must divide the size along the axes.\n'
                'If you want to split the array into non-equally-sized '
                'arrays, use array_split instead.')
    return array_split(ary, indices_or_sections, axis)


def vsplit(ary, indices_or_sections):
    """Splits an array into multiple sub arrays along the first axis.

    This is equivalent to ``split`` with ``axis=0``.

    .. seealso:: :func:`cupy.split` for more detail, :func:`numpy.dsplit`

    """
    if ary.ndim <= 1:
        raise ValueError('Cannot vsplit an array with less than 2 dimensions')
    return split(ary, indices_or_sections, 0)
