def byte_bounds(a):
    """Returns pointers to the end-points of an array.

    Args:
        a: ndarray
    Returns:
        Tuple[int, int]: pointers to the end-points of an array

    .. seealso:: :func:`numpy.byte_bounds`
    """
    a_low = a_high = a.data.ptr
    astrides = a.strides
    ashape = a.shape
    bytes_a = a.dtype.itemsize

    if astrides is None:
        # contiguous case
        a_high += a.size * bytes_a
    else:
        # non-contiguous case
        for shape, stride in zip(ashape, astrides):
            if stride < 0:
                a_low += (shape - 1) * stride
            else:
                a_high += (shape - 1) * stride
        a_high += bytes_a
    return a_low, a_high
