import cupy
from cupy import internal


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
        arr = [a for a in arrays if isinstance(a, cupy.ndarray)]
        ndim = 0
        for a in arr:
            ndim = max(ndim, a.ndim)

        shape = [1] * ndim
        for a in arr:
            offset = len(shape) - a.ndim
            for i, dim in enumerate(a.shape):
                if dim != 1 and shape[i + offset] != dim:
                    if shape[i + offset] != 1:
                        raise RuntimeError('Broadcasting failed')
                    else:
                        shape[i + offset] = dim

        self.shape = shape = tuple(shape)
        self.size = internal.prod(shape)
        self.nd = len(shape)

        broadcasted = []
        for a in arrays:
            if not isinstance(a, cupy.ndarray) or a.shape == shape:
                broadcasted.append(a)
                continue

            off = self.nd - a.ndim
            a_sh = a.shape
            a_st = a._strides
            strides = [0 if i < off or a_sh[i - off] != dim else a_st[i - off]
                       for i, dim in enumerate(shape)]

            view = a.view()
            view._shape = shape
            view._strides = tuple(strides)
            view._size = self.size
            view._c_contiguous = -1
            view._f_contiguous = -1
            broadcasted.append(view)

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
    v._mark_dirty()
    return v
