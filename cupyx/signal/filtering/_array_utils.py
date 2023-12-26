import cupy


def _odd_ext(x, n, axis=-1):
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
        The axis along which to extend `x`.  Default is -1.

    Examples
    --------
    >>> from cusignal.utils.arraytools import _odd_ext
    >>> import cupy as cupy
    >>> a = cupy.array([[1, 2, 3, 4, 5], [0, 1, 4, 9, 16]])
    >>> _odd_ext(a, 2)
    array([[-1,  0,  1,  2,  3,  4,  5,  6,  7],
           [-4, -1,  0,  1,  4,  9, 16, 23, 28]])

    Odd extension is a "180 degree rotation" at the endpoints of the original
    array:

    >>> t = cupy.linspace(0, 1.5, 100)
    >>> a = 0.9 * cupy.sin(2 * cupy.pi * t**2)
    >>> b = _odd_ext(a, 40)
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(arange(-40, 140), cupy.asnumpy(b), 'b', lw=1, \
                 label='odd extension')
    >>> plt.plot(arange(100), cupy.asnumpy(a), 'r', lw=2, label='original')
    >>> plt.legend(loc='best')
    >>> plt.show()
    """
    x = cupy.asarray(x)
    if n < 1:
        return x
    if n > x.shape[axis] - 1:
        raise ValueError(
            (
                "The extension length n (%d) is too big. "
                + "It must not exceed x.shape[axis]-1, which is %d."
            )
            % (n, x.shape[axis] - 1)
        )
    left_end = _axis_slice(x, start=0, stop=1, axis=axis)
    left_ext = _axis_slice(x, start=n, stop=0, step=-1, axis=axis)
    right_end = _axis_slice(x, start=-1, axis=axis)
    right_ext = _axis_slice(x, start=-2, stop=-(n + 2), step=-1, axis=axis)
    ext = cupy.concatenate(
        (2 * left_end - left_ext, x, 2 * right_end - right_ext), axis=axis
    )
    return ext


def _even_ext(x, n, axis=-1):
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
        The axis along which to extend `x`.  Default is -1.

    Examples
    --------
    >>> from cusignal.utils.arraytools import _even_ext
    >>> from cupy import cupy
    >>> a = cupy.array([[1, 2, 3, 4, 5], [0, 1, 4, 9, 16]])
    >>> _even_ext(a, 2)
    array([[ 3,  2,  1,  2,  3,  4,  5,  4,  3],
           [ 4,  1,  0,  1,  4,  9, 16,  9,  4]])

    Even extension is a "mirror image" at the boundaries of the original array:

    >>> t = cupy.linspace(0, 1.5, 100)
    >>> a = 0.9 * cupy.sin(2 * cupy.pi * t**2)
    >>> b = _even_ext(a, 40)
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(arange(-40, 140), cupy.asnumpy(b), 'b', lw=1, \
                 label='even extension')
    >>> plt.plot(arange(100), cupy.asnumpy(a), 'r', lw=2, label='original')
    >>> plt.legend(loc='best')
    >>> plt.show()
    """
    x = cupy.asarray(x)
    if n < 1:
        return x
    if n > x.shape[axis] - 1:
        raise ValueError(
            (
                "The extension length n (%d) is too big. "
                + "It must not exceed x.shape[axis]-1, which is %d."
            )
            % (n, x.shape[axis] - 1)
        )
    left_ext = _axis_slice(x, start=n, stop=0, step=-1, axis=axis)
    right_ext = _axis_slice(x, start=-2, stop=-(n + 2), step=-1, axis=axis)
    ext = cupy.concatenate((left_ext, x, right_ext), axis=axis)
    return ext


def _const_ext(x, n, axis=-1):
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
        The axis along which to extend `x`.  Default is -1.

    Examples
    --------
    >>> from cusignal.utils.arraytools import _const_ext
    >>> import cupy as cupy
    >>> a = cupy.array([[1, 2, 3, 4, 5], [0, 1, 4, 9, 16]])
    >>> _const_ext(a, 2)
    array([[ 1,  1,  1,  2,  3,  4,  5,  5,  5],
           [ 0,  0,  0,  1,  4,  9, 16, 16, 16]])

    Constant extension continues with the same values as the endpoints of the
    array:

    >>> t = cupy.linspace(0, 1.5, 100)
    >>> a = 0.9 * cupy.sin(2 * cupy.pi * t**2)
    >>> b = _const_ext(a, 40)
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(arange(-40, 140), cupy.asnumpy(b), 'b', lw=1, \
                 label='constant extension')
    >>> plt.plot(arange(100), cupy.asnumpy(a), 'r', lw=2, label='original')
    >>> plt.legend(loc='best')
    >>> plt.show()
    """
    x = cupy.asarray(x)
    if n < 1:
        return x
    left_end = _axis_slice(x, start=0, stop=1, axis=axis)
    ones_shape = [1] * x.ndim
    ones_shape[axis] = n
    ones = cupy.ones(ones_shape, dtype=x.dtype)
    left_ext = ones * left_end
    right_end = _axis_slice(x, start=-1, axis=axis)
    right_ext = ones * right_end
    ext = cupy.concatenate((left_ext, x, right_ext), axis=axis)
    return ext


def _zero_ext(x, n, axis=-1):
    """
    Zero padding at the boundaries of an array

    Generate a new ndarray that is a zero padded extension of `x` along
    an axis.

    Parameters
    ----------
    x : ndarray
        The array to be extended.
    n : int
        The number of elements by which to extend `x` at each end of the
        axis.
    axis : int, optional
        The axis along which to extend `x`.  Default is -1.

    Examples
    --------
    >>> from cusignal.utils.arraytools import _zero_ext
    >>> import cupy as cupy
    >>> a = cupy.array([[1, 2, 3, 4, 5], [0, 1, 4, 9, 16]])
    >>> _zero_ext(a, 2)
    array([[ 0,  0,  1,  2,  3,  4,  5,  0,  0],
           [ 0,  0,  0,  1,  4,  9, 16,  0,  0]])
    """
    x = cupy.asarray(x)
    if n < 1:
        return x
    zeros_shape = list(x.shape)
    zeros_shape[axis] = n
    zeros = cupy.zeros(zeros_shape, dtype=x.dtype)
    ext = cupy.concatenate((zeros, x, zeros), axis=axis)
    return ext


def _axis_slice(a, start=None, stop=None, step=None, axis=-1):
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
    >>> from cupy import array
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
    slice(start, stop, step).  This implies axis_slice() does not
    handle its arguments the exactly the same as indexing.  To select
    a single index k, for example, use
        axis_slice(a, start=k, stop=k+1)
    In this case, the length of the axis 'axis' in the result will
    be 1; the trivial dimension is not removed. (Use numpy.squeeze()
    to remove trivial axes.)
    """
    a = cupy.asarray(a)
    a_slice = [slice(None)] * a.ndim
    a_slice[axis] = slice(start, stop, step)
    b = a[tuple(a_slice)]
    return b


def _axis_reverse(a, axis=-1):
    """Reverse the 1-d slices of `a` along axis `axis`.

    Returns axis_slice(a, step=-1, axis=axis).
    """
    return _axis_slice(a, step=-1, axis=axis)
