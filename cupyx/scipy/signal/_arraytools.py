"""
Functions for acting on a axis of an array.
"""
import cupy


def axis_slice(a, start=None, stop=None, step=None, axis=-1):
    """Take a slice along axis 'axis' from 'a'.

    Parameters
    ----------
    a : cupy.ndarray
        The array to be sliced.
    start, stop, step : int or None
        The slice parameters.
    axis : int, optional
        The axis of `a` to be sliced.

    Examples
    --------
    >>> a = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> axis_slice(a, start=0, stop=1, axis=1)
    array([[1],
           [4],
           [7]])
    >>> axis_slice(a, start=1, axis=0)
    array([[4, 5, 6],
           [7, 8, 9]])

    Notes
    -----
    The keyword arguments start, stop and step are used by calling
    slice(start, stop, step). This implies axis_slice() does not
    handle its arguments the exactly the same as indexing. To select
    a single index k, for example, use
        axis_slice(a, start=k, stop=k+1)
    In this case, the length of the axis 'axis' in the result will
    be 1; the trivial dimension is not removed. (Use cupy.squeeze()
    to remove trivial axes.)
    """
    a_slice = [slice(None)] * a.ndim
    a_slice[axis] = slice(start, stop, step)
    b = a[tuple(a_slice)]
    return b


def axis_assign(a, b, start=None, stop=None, step=None, axis=-1):
    """Take a slice along axis 'axis' from 'a' and set it to 'b' in-place.

    Parameters
    ----------
    a : numpy.ndarray
        The array to be sliced.
    b : cupy.ndarray
        The array to be assigned.
    start, stop, step : int or None
        The slice parameters.
    axis : int, optional
        The axis of `a` to be sliced.

    Examples
    --------
    >>> a = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> b1 = array([[-1], [-4], [-7]])
    >>> axis_assign(a, b1, start=0, stop=1, axis=1)
    array([[-1, 2, 3],
           [-4, 5, 6],
           [-7, 8, 9]])

    Notes
    -----
    The keyword arguments start, stop and step are used by calling
    slice(start, stop, step). This implies axis_assign() does not
    handle its arguments the exactly the same as indexing. To assign
    a single index k, for example, use
        axis_assign(a, start=k, stop=k+1)
    In this case, the length of the axis 'axis' in the result will
    be 1; the trivial dimension is not removed. (Use numpy.squeeze()
    to remove trivial axes.)

    This function works in-place and will modify the values contained in `a`
    """
    a_slice = [slice(None)] * a.ndim
    a_slice[axis] = slice(start, stop, step)
    a[tuple(a_slice)] = b
    return a


def axis_reverse(a, axis=-1):
    """Reverse the 1-D slices of `a` along axis `axis`.

    Returns axis_slice(a, step=-1, axis=axis).
    """
    return axis_slice(a, step=-1, axis=axis)


def odd_ext(x, n, axis=-1):
    """
    Odd extension at the boundaries of an array

    Generate a new ndarray by making an odd extension of `x` along an axis.

    Parameters
    ----------
    x : ndarray
        The array to be extended.
    n : int
        The number of elements by which to extend `x` at each end of the axis.
    axis : int, optional
        The axis along which to extend `x`. Default is -1.

    Examples
    --------
    >>> from cupyx.scipy.signal._arraytools import odd_ext
    >>> a = cupy.array([[1, 2, 3, 4, 5], [0, 1, 4, 9, 16]])
    >>> odd_ext(a, 2)
    array([[-1,  0,  1,  2,  3,  4,  5,  6,  7],
           [-4, -1,  0,  1,  4,  9, 16, 23, 28]])
    """
    if n < 1:
        return x
    if n > x.shape[axis] - 1:
        raise ValueError(("The extension length n (%d) is too big. " +
                          "It must not exceed x.shape[axis]-1, which is %d.")
                         % (n, x.shape[axis] - 1))
    left_end = axis_slice(x, start=0, stop=1, axis=axis)
    left_ext = axis_slice(x, start=n, stop=0, step=-1, axis=axis)
    right_end = axis_slice(x, start=-1, axis=axis)
    right_ext = axis_slice(x, start=-2, stop=-(n + 2), step=-1, axis=axis)
    ext = cupy.concatenate((2 * left_end - left_ext, x,
                            2 * right_end - right_ext), axis=axis)
    return ext


def even_ext(x, n, axis=-1):
    """
    Even extension at the boundaries of an array

    Generate a new ndarray by making an even extension of `x` along an axis.

    Parameters
    ----------
    x : ndarray
        The array to be extended.
    n : int
        The number of elements by which to extend `x` at each end of the axis.
    axis : int, optional
        The axis along which to extend `x`. Default is -1.

    Examples
    --------
    >>> from cupyx.scipy.signal._arraytools import even_ext
    >>> a = cupy.array([[1, 2, 3, 4, 5], [0, 1, 4, 9, 16]])
    >>> even_ext(a, 2)
    array([[ 3,  2,  1,  2,  3,  4,  5,  4,  3],
           [ 4,  1,  0,  1,  4,  9, 16,  9,  4]])

    """
    if n < 1:
        return x
    if n > x.shape[axis] - 1:
        raise ValueError(("The extension length n (%d) is too big. " +
                          "It must not exceed x.shape[axis]-1, which is %d.")
                         % (n, x.shape[axis] - 1))
    left_ext = axis_slice(x, start=n, stop=0, step=-1, axis=axis)
    right_ext = axis_slice(x, start=-2, stop=-(n + 2), step=-1, axis=axis)
    ext = cupy.concatenate((left_ext, x, right_ext), axis=axis)
    return ext


def const_ext(x, n, axis=-1):
    """
    Constant extension at the boundaries of an array

    Generate a new ndarray that is a constant extension of `x` along an axis.
    The extension repeats the values at the first and last element of
    the axis.

    Parameters
    ----------
    x : ndarray
        The array to be extended.
    n : int
        The number of elements by which to extend `x` at each end of the axis.
    axis : int, optional
        The axis along which to extend `x`. Default is -1.

    Examples
    --------
    >>> from cupyx.scipy.signal._arraytools import const_ext
    >>> a = cupy.array([[1, 2, 3, 4, 5], [0, 1, 4, 9, 16]])
    >>> const_ext(a, 2)
    array([[ 1,  1,  1,  2,  3,  4,  5,  5,  5],
           [ 0,  0,  0,  1,  4,  9, 16, 16, 16]])
    """
    if n < 1:
        return x
    left_end = axis_slice(x, start=0, stop=1, axis=axis)
    ones_shape = [1] * x.ndim
    ones_shape[axis] = n
    ones = cupy.ones(ones_shape, dtype=x.dtype)
    left_ext = ones * left_end
    right_end = axis_slice(x, start=-1, axis=axis)
    right_ext = ones * right_end
    ext = cupy.concatenate((left_ext, x, right_ext), axis=axis)
    return ext


def zero_ext(x, n, axis=-1):
    """
    Zero padding at the boundaries of an array

    Generate a new ndarray that is a zero-padded extension of `x` along
    an axis.

    Parameters
    ----------
    x : ndarray
        The array to be extended.
    n : int
        The number of elements by which to extend `x` at each end of the
        axis.
    axis : int, optional
        The axis along which to extend `x`. Default is -1.

    Examples
    --------
    >>> from cupyx.scipy.signal._arraytools import zero_ext
    >>> a = cupy.array([[1, 2, 3, 4, 5], [0, 1, 4, 9, 16]])
    >>> zero_ext(a, 2)
    array([[ 0,  0,  1,  2,  3,  4,  5,  0,  0],
           [ 0,  0,  0,  1,  4,  9, 16,  0,  0]])
    """
    if n < 1:
        return x
    zeros_shape = list(x.shape)
    zeros_shape[axis] = n
    zeros = cupy.zeros(zeros_shape, dtype=x.dtype)
    ext = cupy.concatenate((zeros, x, zeros), axis=axis)
    return ext


def _as_strided(x, shape=None, strides=None):
    """
    Create a view into the array with the given shape and strides.
    .. warning:: This function has to be used with extreme care, see notes.

    Parameters
    ----------
    x : ndarray
        Array to create a new.
    shape : sequence of int, optional
        The shape of the new array. Defaults to ``x.shape``.
    strides : sequence of int, optional
        The strides of the new array. Defaults to ``x.strides``.

    Returns
    -------
    view : ndarray

    Notes
    -----
    ``as_strided`` creates a view into the array given the exact strides
    and shape. This means it manipulates the internal data structure of
    ndarray and, if done incorrectly, the array elements can point to
    invalid memory and can corrupt results or crash your program.
    """
    shape = x.shape if shape is None else tuple(shape)
    strides = x.strides if strides is None else tuple(strides)

    return cupy.ndarray(
        shape=shape, dtype=x.dtype, memptr=x.data, strides=strides)
