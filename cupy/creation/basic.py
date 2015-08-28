import cupy


def empty(shape, dtype=float):
    """Returns an array without initializing the elements.

    This function currently does not support ``order`` option.

    Args:
        shape (tuple of ints): Dimensionalities of the array.
        dtype: Data type specifier.

    Returns:
        cupy.ndarray: A new array with elements not initialized.

    .. seealso:: :func:`numpy.empty`

    """
    # TODO(beam2d): Support ordering option
    return cupy.ndarray(shape, dtype=dtype)


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
    return empty(a.shape, dtype=dtype)


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
    return full(shape, 1, dtype)


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
    return ones(a.shape, dtype)


def zeros(shape, dtype=float):
    """Returns a new array of given shape and dtype, filled with zeros.

    This function currently does not support ``order`` option.

    Args:
        shape (tuple of ints): Dimensionalities of the array.
        dtype: Data type specifier.

    Returns:
        cupy.ndarray: An array filled with ones.

    .. seealso:: :func:`numpy.zeros`

    """
    # TODO(beam2d): Support ordering option
    a = empty(shape, dtype)
    a.data.memset(0, a.nbytes)
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
    return zeros(a.shape, dtype=dtype)


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
    a = empty(shape, dtype)
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
    return full(a.shape, fill_value, dtype)
