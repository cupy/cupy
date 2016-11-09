import six

import cupy
from cupy import core


zip_longest = six.moves.zip_longest
six_zip = six.moves.zip


def atleast_1d(*arys):
    """Converts arrays to arrays with dimensions >= 1.

    Args:
        arys (tuple of arrays): Arrays to be converted. All arguments must be
            :class:`cupy.ndarray` objects. Only zero-dimensional array is
            affected.

    Returns:
        If there are only one input, then it returns its converted version.
        Otherwise, it returns a list of converted arrays.

    .. seealso:: :func:`numpy.atleast_1d`

    """
    res = []
    for a in arys:
        if not isinstance(a, cupy.ndarray):
            raise TypeError('Only cupy arrays can be atleast_1d')
        if a.ndim == 0:
            a = a.reshape(1)
        res.append(a)
    if len(res) == 1:
        res = res[0]
    return res


def atleast_2d(*arys):
    """Converts arrays to arrays with dimensions >= 2.

    If an input array has dimensions less than two, then this function inserts
    new axes at the head of dimensions to make it have two dimensions.

    Args:
        arys (tuple of arrays): Arrays to be converted. All arguments must be
            :class:`cupy.ndarray` objects.

    Returns:
        If there are only one input, then it returns its converted version.
        Otherwise, it returns a list of converted arrays.

    .. seealso:: :func:`numpy.atleast_2d`

    """
    res = []
    for a in arys:
        if not isinstance(a, cupy.ndarray):
            raise TypeError('Only cupy arrays can be atleast_2d')
        if a.ndim == 0:
            a = a.reshape(1, 1)
        elif a.ndim == 1:
            a = a[None, :]
        res.append(a)
    if len(res) == 1:
        res = res[0]
    return res


def atleast_3d(*arys):
    """Converts arrays to arrays with dimensions >= 3.

    If an input array has dimensions less than three, then this function
    inserts new axes to make it have three dimensions. The place of the new
    axes are following:

    - If its shape is ``()``, then the shape of output is ``(1, 1, 1)``.
    - If its shape is ``(N,)``, then the shape of output is ``(1, N, 1)``.
    - If its shape is ``(M, N)``, then the shape of output is ``(M, N, 1)``.
    - Otherwise, the output is the input array itself.

    Args:
        arys (tuple of arrays): Arrays to be converted. All arguments must be
            :class:`cupy.ndarray` objects.

    Returns:
        If there are only one input, then it returns its converted version.
        Otherwise, it returns a list of converted arrays.

    .. seealso:: :func:`numpy.atleast_3d`

    """
    res = []
    for a in arys:
        if not isinstance(a, cupy.ndarray):
            raise TypeError('Only cupy arrays can be atleast_3d')
        if a.ndim == 0:
            a = a.reshape(1, 1, 1)
        elif a.ndim == 1:
            a = a[None, :, None]
        elif a.ndim == 2:
            a = a[:, :, None]
        res.append(a)
    if len(res) == 1:
        res = res[0]
    return res


broadcast = core.broadcast


def broadcast_arrays(*args):
    """Broadcasts given arrays.

    Args:
        args (tuple of arrays): Arrays to broadcast for each other.

    Returns:
        list: A list of broadcasted arrays.

    .. seealso:: :func:`numpy.broadcast_arrays`

    """
    return broadcast(*args).values


def broadcast_to(array, shape):
    """Broadcast an array to a given shape.

    Args:
        array (cupy.ndarray): Array to broadcast.
        shape (tuple of int): The shape of the desired array.

    Returns:
        cupy.ndarray: Broadcasted view.

    .. seealso:: :func:`numpy.broadcast_to`

    """
    return core.broadcast_to(array, shape)


def expand_dims(a, axis):
    """Expands given arrays.

    Args:
        a (cupy.ndarray): Array to be expanded.
        axis (int): Position where new axis is to be inserted.

    Returns:
        cupy.ndarray: The number of dimensions is one greater than that of
            the input array.

    .. seealso:: :func:`numpy.expand_dims`

    """
    # TODO(okuta): check type
    shape = a.shape
    if axis < 0:
        axis = axis + len(shape) + 1
    return a.reshape(shape[:axis] + (1,) + shape[axis:])


def squeeze(a, axis=None):
    """Removes size-one axes from the shape of an array.

    Args:
        a (cupy.ndarray): Array to be reshaped.
        axis (int or tuple of ints): Axes to be removed. This function removes
            all size-one axes by default. If one of the specified axes is not
            of size one, an exception is raised.

    Returns:
        cupy.ndarray: An array without (specified) size-one axes.

    .. seealso:: :func:`numpy.squeeze`

    """
    # TODO(okuta): check type
    return a.squeeze(axis)
