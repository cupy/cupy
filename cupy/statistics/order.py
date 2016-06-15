def amin(a, axis=None, out=None, keepdims=False, dtype=None):
    """Returns the minimum of an array or the minimum along an axis.

    Args:
        a (cupy.ndarray): Array to take the minimum.
        axis (int): Along which axis to take the minimum. The flattened array
            is used by default.
        out (cupy.ndarray): Output array.
        keepdims (bool): If ``True``, the axis is remained as an axis of
            size one.
        dtype: Data type specifier.

    Returns:
        cupy.ndarray: The minimum of ``a``, along the axis if specified.

    .. seealso:: :func:`numpy.amin`

    """
    # TODO(okuta): check type
    return a.min(axis=axis, dtype=dtype, out=out, keepdims=keepdims)


def amax(a, axis=None, out=None, keepdims=False, dtype=None):
    """Returns the maximum of an array or the maximum along an axis.

    Args:
        a (cupy.ndarray): Array to take the maximum.
        axis (int): Along which axis to take the maximum. The flattened array
            is used by default.
        out (cupy.ndarray): Output array.
        keepdims (bool): If ``True``, the axis is remained as an axis of
            size one.
        dtype: Data type specifier.

    Returns:
        cupy.ndarray: The maximum of ``a``, along the axis if specified.

    .. seealso:: :func:`numpy.amax`

    """
    # TODO(okuta): check type
    return a.max(axis=axis, dtype=dtype, out=out, keepdims=keepdims)


# TODO(okuta): Implement nanmin


# TODO(okuta): Implement nanmax


# TODO(okuta): Implement ptp


# TODO(okuta): Implement percentile
