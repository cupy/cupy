import six

import cupy
from cupy import internal


zip_longest = six.moves.zip_longest
six_zip = six.moves.zip


def atleast_1d(*arys):
    """Converts arrays to arrays with dimensions >= 1.

    Args:
        arys (tuple of arrays): Arrays to be converted. All arguments must be
            cupy.ndarray objects. Only zero-dimensional array is affected.

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
            cupy.ndarray objects.

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
            a = a[cupy.newaxis, :]
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
            cupy.ndarray objects.

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
            a = a[cupy.newaxis, :, cupy.newaxis]
        elif a.ndim == 2:
            a = a[:, :, cupy.newaxis]
        res.append(a)
    if len(res) == 1:
        res = res[0]
    return res


class broadcast(object):
    """Object that performs broadcasting.

    CuPy actually uses this class to support broadcasting in various
    operations. Note that this class does not provide an iterator.

    Args:
        arrays (tuple of arrays): Arrays to be broadcasted.

    Attributes:
        shape (tuple of ints): The broadcasted shape.
        nd (int): Number of dimensions of the broadcasted shape.
        size (int): Total size of the broadcasted shape.
        values (list of arrays): The broadcasted arrays.

    .. seealso:: :class:`numpy.broadcast`

    """

    def __init__(self, *arrays):
        ndarray = cupy.ndarray
        rev = slice(None, None, -1)
        shape_arr = [a._shape[rev] for a in arrays
                     if isinstance(a, ndarray)]
        r_shape = [max(ss) for ss in zip_longest(*shape_arr, fillvalue=0)]

        self.shape = shape = tuple(r_shape[rev])
        self.size = size = internal.prod(shape)
        self.nd = ndim = len(shape)

        broadcasted = list(arrays)
        for i, a in enumerate(broadcasted):
            if not isinstance(a, ndarray):
                continue

            a_shape = a.shape
            if a_shape == shape:
                continue

            r_strides = [
                a_st if sh == a_sh else (0 if a_sh == 1 else None)
                for sh, a_sh, a_st
                in six_zip(r_shape, a._shape[rev], a._strides[rev])]

            if None in r_strides:
                raise RuntimeError('Broadcasting failed')

            offset = (0,) * (ndim - len(r_strides))

            broadcasted[i] = view = a.view()
            view._shape = shape
            view._strides = offset + tuple(r_strides[rev])
            view._size = size
            view._c_contiguous = -1
            view._f_contiguous = -1

        self.values = tuple(broadcasted)


def broadcast_arrays(*args):
    """Broadcasts given arrays.

    Args:
        args (tuple of arrays): Arrays to broadcast for each other.

    Returns:
        list: A list of broadcasted arrays.

    .. seealso:: :func:`numpy.broadcast_arrays`

    """
    return broadcast(*args).values


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
    if axis is None:
        axis = tuple(i for i, n in enumerate(a._shape) if n == 1)
    elif isinstance(axis, int):
        axis = axis,

    new_shape = []
    new_strides = []
    j = 0
    for i, n in enumerate(a._shape):
        if j < len(axis) and i == axis[j]:
            if n != 1:
                raise RuntimeError('Cannot squeeze dimension of size > 1')
            j += 1
        else:
            new_shape.append(n)
            new_strides.append(a._strides[i])

    v = a.view()
    v._shape = tuple(new_shape)
    v._strides = tuple(new_strides)
    v._c_contiguous = -1
    v._f_contiguous = -1
    return v
