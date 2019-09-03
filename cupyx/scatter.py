def scatter_add(a, slices, value):
    """Adds given values to specified elements of an array.

    It adds ``value`` to the specified elements of ``a``.
    If all of the indices target different locations, the operation of
    :func:`scatter_add` is equivalent to ``a[slices] = a[slices] + value``.
    If there are multiple elements targeting the same location,
    :func:`scatter_add` uses all of these values for addition. On the other
    hand, ``a[slices] = a[slices] + value`` only adds the contribution from one
    of the indices targeting the same location.

    Note that just like an array indexing, negative indices are interpreted as
    counting from the end of an array.

    Also note that :func:`scatter_add` behaves identically
    to :func:`numpy.add.at`.

    Example
    -------
    >>> import numpy
    >>> import cupy
    >>> a = cupy.zeros((6,), dtype=numpy.float32)
    >>> i = cupy.array([1, 0, 1])
    >>> v = cupy.array([1., 1., 1.])
    >>> cupyx.scatter_add(a, i, v);
    >>> a
    array([1., 2., 0., 0., 0., 0.], dtype=float32)

    Args:
        a (ndarray): An array that gets added.
        slices: It is integer, slices, ellipsis, numpy.newaxis,
            integer array-like, boolean array-like or tuple of them.
            It works for slices used for
            :func:`cupy.ndarray.__getitem__` and
            :func:`cupy.ndarray.__setitem__`.
        v (array-like): Values to increment ``a`` at referenced locations.

    .. note::
        It only supports types that are supported by CUDA's atomicAdd when
        an integer array is included in ``slices``.
        The supported types are ``numpy.float32``, ``numpy.int32``,
        ``numpy.uint32``, ``numpy.uint64`` and ``numpy.ulonglong``.

    .. note::
        :func:`scatter_add` does not raise an error when indices exceed size of
        axes. Instead, it wraps indices.

    .. note::
        As of v4, this function is moved from ``cupy`` package to ``cupyx``
        package.
        ``cupy.scatter_add`` is still available for backward compatibility.

    .. seealso:: :meth:`numpy.ufunc.at`.

    """
    a.scatter_add(slices, value)


def scatter_max(a, slices, value):
    """Stores a maximum value of elements specified by indices to an array.

    It stores the maximum value of elements in ``value`` array indexed by
    ``slices`` to ``a``. If all of the indices target different locations,
    the operation of :func:`scatter_max` is equivalent to
    ``a[slices] = cupy.maximum(a[slices], value)``.
    If there are multiple elements targeting the same location,
    :func:`scatter_max` stores the maximum of all of these values to the given
    index of ``a``, the initial element of ``a`` is also taken in account.

    Note that just like an array indexing, negative indices are interpreted as
    counting from the end of an array.

    Also note that :func:`scatter_max` behaves identically
    to :func:`numpy.maximum.at`.

    Example
    -------
    >>> import numpy
    >>> import cupy
    >>> a = cupy.zeros((6,), dtype=numpy.float32)
    >>> i = cupy.array([1, 0, 1, 2])
    >>> v = cupy.array([1., 2., 3., -1.])
    >>> cupyx.scatter_max(a, i, v);
    >>> a
    array([2., 3., 0., 0., 0., 0.], dtype=float32)

    Args:
        a (ndarray): An array to store the results.
        slices: It is integer, slices, ellipsis, numpy.newaxis,
            integer array-like, boolean array-like or tuple of them.
            It works for slices used for
            :func:`cupy.ndarray.__getitem__` and
            :func:`cupy.ndarray.__setitem__`.
        v (array-like): An array used for reference.
    """
    a.scatter_max(slices, value)


def scatter_min(a, slices, value):
    """Stores a minimum value of elements specified by indices to an array.

    It stores the minimum value of elements in ``value`` array indexed by
    ``slices`` to ``a``. If all of the indices target different locations,
    the operation of :func:`scatter_min` is equivalent to
    ``a[slices] = cupy.minimum(a[slices], value)``.
    If there are multiple elements targeting the same location,
    :func:`scatter_min` stores the minimum of all of these values to the given
    index of ``a``, the initial element of ``a`` is also taken in account.

    Note that just like an array indexing, negative indices are interpreted as
    counting from the end of an array.

    Also note that :func:`scatter_min` behaves identically
    to :func:`numpy.minimum.at`.

    Example
    -------
    >>> import numpy
    >>> import cupy
    >>> a = cupy.zeros((6,), dtype=numpy.float32)
    >>> i = cupy.array([1, 0, 1, 2])
    >>> v = cupy.array([1., 2., 3., -1.])
    >>> cupyx.scatter_min(a, i, v);
    >>> a
    array([ 0.,  0., -1.,  0.,  0.,  0.], dtype=float32)

    Args:
        a (ndarray): An array to store the results.
        slices: It is integer, slices, ellipsis, numpy.newaxis,
            integer array-like, boolean array-like or tuple of them.
            It works for slices used for
            :func:`cupy.ndarray.__getitem__` and
            :func:`cupy.ndarray.__setitem__`.
        v (array-like): An array used for reference.
    """
    a.scatter_min(slices, value)
