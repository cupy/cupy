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
    """Convers arrays to arrays with dimensions >= 2.

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
    """Convers arrays to arrays with dimensions >= 3.

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
    """Object that mimisc broadcasting.

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
        ndim = 0
        for array in arrays:
            if isinstance(array, cupy.ndarray):
                ndim = max(ndim, array.ndim)

        shape = [1] * ndim
        for array in arrays:
            if isinstance(array, cupy.ndarray):
                offset = len(shape) - array.ndim
                for i, dim in enumerate(array.shape):
                    if dim != 1 and shape[i + offset] != dim:
                        if shape[i + offset] != 1:
                            raise RuntimeError('Broadcasting failed')
                        else:
                            shape[i + offset] = dim

        self.shape = tuple(shape)
        self.size = internal.prod(self.shape)
        self.nd = len(shape)

        broadcasted = []
        for array in arrays:
            if not isinstance(array, cupy.ndarray):
                broadcasted.append(array)
                continue
            if array.shape == self.shape:
                broadcasted.append(array)
                continue

            offset = self.nd - array.ndim
            strides = []
            for i, dim in enumerate(shape):
                if i < offset:
                    # TODO(okuta) fix if `dim` == 1
                    strides.append(0)
                elif array.shape[i - offset] != dim:
                    strides.append(0)
                else:
                    strides.append(array._strides[i - offset])

            view = array.view()
            view._shape = self.shape
            view._strides = tuple(strides)
            view._mark_dirty()
            broadcasted.append(view)

        self.values = broadcasted


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
    """Removes single-dimensional axes from the shape of an array.

    Args:
        a (cupy.ndarray): Array to be reshaped.
        axis (int or tuple of ints): Axes to be removed. This function removes
            all single-dimensional axes by default. If one of the specified
            axes is not single-dimensional, an exception is raised.

    Returns:
        cupy.ndarray: An array without (specified) single-dimensional axes.

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
