import six
import warnings

import cupy
from cupy import core


zip_longest = six.moves.zip_longest
six_zip = six.moves.zip


# Shape map for atleast_nd functions
# (minimum dimension, input dimension) -> (output shape)
_atleast_nd_shape_map = {
    (1, 0): lambda shape: (1,),
    (2, 0): lambda shape: (1, 1),
    (2, 1): lambda shape: (1,) + shape,
    (3, 0): lambda shape: (1, 1, 1),
    (3, 1): lambda shape: (1,) + shape + (1,),
    (3, 2): lambda shape: shape + (1,),
}


def _atleast_nd_helper(n, arys):
    """Helper function for atleast_nd functions."""

    res = []
    for a in arys:
        a = cupy.asarray(a)
        if a.ndim < n:
            new_shape = _atleast_nd_shape_map[(n, a.ndim)](a.shape)
            a = a.reshape(*new_shape)
        res.append(a)

    if len(res) == 1:
        res, = res
    return res


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
    return _atleast_nd_helper(1, arys)


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
    return _atleast_nd_helper(2, arys)


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
    return _atleast_nd_helper(3, arys)


broadcast = core.broadcast


def broadcast_arrays(*args):
    """Broadcasts given arrays.

    Args:
        args (tuple of arrays): Arrays to broadcast for each other.

    Returns:
        list: A list of broadcasted arrays.

    .. seealso:: :func:`numpy.broadcast_arrays`

    """
    return list(broadcast(*args).values)


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
    if axis > a.ndim or axis < 0:
        # TODO(unno): Too large and too small axis is deprecated in NumPy 1.13
        # We need to fix this behavior after NumPy forbids it.
        warnings.warn(
            'Both axis > a.ndim and axis < -a.ndim - 1 are deprecated and '
            'will raise an AxisError in the future.',
            DeprecationWarning)
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
