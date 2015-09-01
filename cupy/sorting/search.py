from cupy import reduction


def argmax(a, axis=None, dtype=None, out=None, keepdims=False):
    """Returns the indices of the maximum along an axis.

    Args:
        a (cupy.ndarray): Array to take argmax.
        axis (int): Along which axis to find the maximum. ``a`` is flattened by
            default.
        dtype: Data type specifier.
        out (cupy.ndarray): Output array.
        keepdims (bool): If True, the axis ``axis`` is preserved as an axis of
            length one.

    Returns:
        cupy.ndarray: The indices of the maximum of ``a`` along an axis.

    .. seealso:: :func:`numpy.argmax`

    """
    return reduction.argmax(a, axis=axis, dtype=dtype, out=out,
                            keepdims=keepdims)


# TODO(okuta): Implement nanargmax


def argmin(a, axis=None, dtype=None, out=None, keepdims=False):
    """Returns the indices of the minimum along an axis.

    Args:
        a (cupy.ndarray): Array to take argmin.
        axis (int): Along which axis to find the minimum. ``a`` is flattened by
            default.
        dtype: Data type specifier.
        out (cupy.ndarray): Output array.
        keepdims (bool): If True, the axis ``axis`` is preserved as an axis of
            length one.

    Returns:
        cupy.ndarray: The indices of the minimum of ``a`` along an axis.

    .. seealso:: :func:`numpy.argmin`

    """
    return reduction.argmin(a, axis=axis, dtype=dtype, out=out,
                            keepdims=keepdims)


# TODO(okuta): Implement nanargmin


# TODO(okuta): Implement argwhere


# TODO(okuta): Implement nonzero


# TODO(okuta): Implement flatnonzero


# TODO(okuta): Implement where


# TODO(okuta): Implement searchsorted


# TODO(okuta): Implement extract
