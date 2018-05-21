import cupy
import numpy


def empty(shape, dtype=float, order='C'):
    """Returns an array without initializing the elements.

    Args:
        shape (tuple of ints): Dimensionalities of the array.
        dtype: Data type specifier.
        order ({'C', 'F'}): Row-major (C-style) or column-major
            (Fortran-style) order.

    Returns:
        cupy.ndarray: A new array with elements not initialized.

    .. seealso:: :func:`numpy.empty`

    """
    return cupy.ndarray(shape, dtype=dtype, order=order)


def empty_like(a, dtype=None):
    """Returns a new array with same shape and dtype of a given array.

    This function currently does not support ``order`` and ``subok`` options.

    Args:
        a (cupy.ndarray): Base array.
        dtype: Data type specifier. The data type of ``a`` is used by default.

    Returns:
        cupy.ndarray: A new array with same shape and dtype of ``a`` with
        elements not initialized.

    .. seealso:: :func:`numpy.empty_like`

    """
    # TODO(beam2d): Support ordering option
    if dtype is None:
        dtype = a.dtype
    return cupy.ndarray(a.shape, dtype=dtype)


def eye(N, M=None, k=0, dtype=float):
    """Returns a 2-D array with ones on the diagonals and zeros elsewhere.

    Args:
        N (int): Number of rows.
        M (int): Number of columns. M == N by default.
        k (int): Index of the diagonal. Zero indicates the main diagonal,
            a positive index an upper diagonal, and a negative index a lower
            diagonal.
        dtype: Data type specifier.

    Returns:
        cupy.ndarray: A 2-D array with given diagonals filled with ones and
        zeros elsewhere.

    .. seealso:: :func:`numpy.eye`

    """
    if M is None:
        M = N
    ret = zeros((N, M), dtype)
    ret.diagonal(k)[:] = 1
    return ret


def identity(n, dtype=float):
    """Returns a 2-D identity array.

    It is equivalent to ``eye(n, n, dtype)``.

    Args:
        n (int): Number of rows and columns.
        dtype: Data type specifier.

    Returns:
        cupy.ndarray: A 2-D identity array.

    .. seealso:: :func:`numpy.identity`

    """
    return eye(n, dtype=dtype)


def ones(shape, dtype=float):
    """Returns a new array of given shape and dtype, filled with ones.

    This function currently does not support ``order`` option.

    Args:
        shape (tuple of ints): Dimensionalities of the array.
        dtype: Data type specifier.

    Returns:
        cupy.ndarray: An array filled with ones.

    .. seealso:: :func:`numpy.ones`

    """
    # TODO(beam2d): Support ordering option
    a = cupy.ndarray(shape, dtype=dtype)
    a.fill(1)
    return a


def ones_like(a, dtype=None):
    """Returns an array of ones with same shape and dtype as a given array.

    This function currently does not support ``order`` and ``subok`` options.

    Args:
        a (cupy.ndarray): Base array.
        dtype: Data type specifier. The dtype of ``a`` is used by default.

    Returns:
        cupy.ndarray: An array filled with ones.

    .. seealso:: :func:`numpy.ones_like`

    """
    # TODO(beam2d): Support ordering option
    if dtype is None:
        dtype = a.dtype
    a = cupy.ndarray(a.shape, dtype=dtype)
    a.fill(1)
    return a


def zeros(shape, dtype=float, order='C'):
    """Returns a new array of given shape and dtype, filled with zeros.

    Args:
        shape (tuple of ints): Dimensionalities of the array.
        dtype: Data type specifier.
        order ({'C', 'F'}): Row-major (C-style) or column-major
            (Fortran-style) order.

    Returns:
        cupy.ndarray: An array filled with ones.

    .. seealso:: :func:`numpy.zeros`

    """
    a = cupy.ndarray(shape, dtype, order=order)
    a.data.memset_async(0, a.nbytes)
    return a


def zeros_like(a, dtype=None):
    """Returns an array of zeros with same shape and dtype as a given array.

    This function currently does not support ``order`` and ``subok`` options.

    Args:
        a (cupy.ndarray): Base array.
        dtype: Data type specifier. The dtype of ``a`` is used by default.

    Returns:
        cupy.ndarray: An array filled with ones.

    .. seealso:: :func:`numpy.zeros_like`

    """
    # TODO(beam2d): Support ordering option
    if dtype is None:
        dtype = a.dtype
    a = cupy.ndarray(a.shape, dtype)
    a.data.memset_async(0, a.nbytes)
    return a


def full(shape, fill_value, dtype=None):
    """Returns a new array of given shape and dtype, filled with a given value.

    This function currently does not support ``order`` option.

    Args:
        shape (tuple of ints): Dimensionalities of the array.
        fill_value: A scalar value to fill a new array.
        dtype: Data type specifier.

    Returns:
        cupy.ndarray: An array filled with ``fill_value``.

    .. seealso:: :func:`numpy.full`

    """
    # TODO(beam2d): Support ordering option
    if dtype is None:
        if isinstance(fill_value, cupy.ndarray):
            dtype = fill_value.dtype
        else:
            dtype = numpy.array(fill_value).dtype
    a = cupy.ndarray(shape, dtype=dtype)
    a.fill(fill_value)
    return a


def full_like(a, fill_value, dtype=None):
    """Returns a full array with same shape and dtype as a given array.

    This function currently does not support ``order`` and ``subok`` options.

    Args:
        a (cupy.ndarray): Base array.
        fill_value: A scalar value to fill a new array.
        dtype: Data type specifier. The dtype of ``a`` is used by default.

    Returns:
        cupy.ndarray: An array filled with ``fill_value``.

    .. seealso:: :func:`numpy.full_like`

    """
    # TODO(beam2d): Support ordering option
    if dtype is None:
        dtype = a.dtype
    a = cupy.ndarray(a.shape, dtype=dtype)
    a.fill(fill_value)
    return a
