def scatter_add(a, slices, value):
    """Adds given values to specified elements of an array.

    This function adds given ``value`` to specified elements of ``a``.
    When all of the indices target different locations, :func:`scatter_add`
    does operation equivalent to ``a[slices] = a[slices] + value``.
    When there are indices targeting the same location, :func:`scatter_add`
    uses all of the values for addition. On the other hand,
    ``a[slices] = a[slices] + value`` only adds the contribution from one of
    the indices targeting the same location.

    Similarly to array indexing, negative indices are interpreted as counting
    from the end of an axis.

    Example
    -------
    >>> import cupy
    >>> a = cupy.zeros((6,))
    >>> i = cupy.array([1, 0, 1])
    >>> v = cupy.array([1., 1., 1.])
    >>> cupy.scatter_add(a, i, v);
    >>> a
    array([ 1.,  2.,  0.,  0.,  0.,  0.])

    Args:
        a (ndarray): An array that gets added.
        slices: It is integer, slices, ellipsis, numpy.newaxis,
            integer array-like or tuple of them.
            It works for slices used for
            :func:`cupy.ndarray.__getitem__` and
            :func:`cupy.ndarray.__setitem__`.
        v (array-like): Values to increment ``a`` at referenced locations.

    .. note::
        Supports only arrays of type ``numpy.float32`` and ``numpy.int32``.

    .. note::
        :func:`scatter_add` does not raise an error when indices exceed size of
        axes. Instead, it wraps indices.

    """
    a.scatter_add(slices, value)
