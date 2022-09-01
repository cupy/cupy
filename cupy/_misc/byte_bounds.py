def byte_bounds(a):
    """Returns pointers to the end-points of an array.

    Args:
        a: ndarray
    Returns:
        Tuple[int, int]: pointers to the end-points of an array

    .. seealso:: :func:`numpy.byte_bounds`
    """
    a_low = a_high = a.data.ptr
    a_strides = a.strides
    a_shape = a.shape
    a_item_bytes = a.itemsize

    for shape, stride in zip(a_shape, a_strides):
        if stride < 0:
            a_low += (shape - 1) * stride
        else:
            a_high += (shape - 1) * stride

    a_high += a_item_bytes
    return a_low, a_high
