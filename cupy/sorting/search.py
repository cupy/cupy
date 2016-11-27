from cupy import core


def argmax(a, axis=None, dtype=None, out=None, keepdims=False):
    """Returns the indices of the maximum along an axis.

    Args:
        a (cupy.ndarray): Array to take argmax.
        axis (int): Along which axis to find the maximum. ``a`` is flattened by
            default.
        dtype: Data type specifier.
        out (cupy.ndarray): Output array.
        keepdims (bool): If ``True``, the axis ``axis`` is preserved as an axis
            of length one.

    Returns:
        cupy.ndarray: The indices of the maximum of ``a`` along an axis.

    .. seealso:: :func:`numpy.argmax`

    """
    # TODO(okuta): check type
    return a.argmax(axis=axis, dtype=dtype, out=out, keepdims=keepdims)


# TODO(okuta): Implement nanargmax


def argmin(a, axis=None, dtype=None, out=None, keepdims=False):
    """Returns the indices of the minimum along an axis.

    Args:
        a (cupy.ndarray): Array to take argmin.
        axis (int): Along which axis to find the minimum. ``a`` is flattened by
            default.
        dtype: Data type specifier.
        out (cupy.ndarray): Output array.
        keepdims (bool): If ``True``, the axis ``axis`` is preserved as an axis
            of length one.

    Returns:
        cupy.ndarray: The indices of the minimum of ``a`` along an axis.

    .. seealso:: :func:`numpy.argmin`

    """
    # TODO(okuta): check type
    return a.argmin(axis=axis, dtype=dtype, out=out, keepdims=keepdims)


# TODO(okuta): Implement nanargmin


# TODO(okuta): Implement argwhere


def nonzero(a):
    """Return the indices of the elements that are non-zero.

    Returns a tuple of arrays, one for each dimension of a,
    containing the indices of the non-zero elements in that dimension.

    Args:
        a (cupy.ndarray): array

    Returns:
        tuple of arrays: Indices of elements that are non-zero.

    .. seealso:: :func:`numpy.nonzero`

    """
    return a.nonzero()


def flatnonzero(a):
    """Return indices that are non-zero in the flattened version of a.

    This is equivalent to a.ravel().nonzero()[0].

    Args:
        a (cupy.ndarray): input array

    Returns:
        cupy.ndarray: Output array,
        containing the indices of the elements of a.ravel() that are non-zero.

    .. seealso:: :func:`numpy.flatnonzero`
    """
    return a.ravel().nonzero()[0]


def where(condition, x=None, y=None):
    """Return elements, either from x or y, depending on condition.

    If only condition is given, return ``condition.nonzero()``.

    Args:
        condition (cupy.ndarray): When True, take x, otherwise take y.
        x (cupy.ndarray): Values from which to choose on ``True``.
        y (cupy.ndarray): Values from which to choose on ``False``.

    Returns:
        cupy.ndarray: Each element of output contains elements of ``x`` when
            ``condition`` is ``True``, otherwise elements of ``y``. If only
            ``condition`` is given, return the tuple ``condition.nonzero()``,
            the indices where ``condition`` is True.

    .. seealso:: :func:`numpy.where`

    """

    missing = (x is None, y is None).count(True)

    if missing == 1:
        raise ValueError("Must provide both 'x' and 'y' or neither.")
    if missing == 2:
        return nonzero(condition)

    return _where_ufunc(condition.astype('?'), x, y)

_where_ufunc = core.create_ufunc(
    'cupy_where',
    ('???->?', '?bb->b', '?BB->B', '?hh->h', '?HH->H', '?ii->i', '?II->I',
     '?ll->l', '?LL->L', '?qq->q', '?QQ->Q', '?ee->e', '?ff->f',
     # On CUDA 6.5 these combinations don't work correctly (on CUDA >=7.0, it
     # works).
     # See issue #551.
     '?hd->d', '?Hd->d',
     '?dd->d'),
    'out0 = in0 ? in1 : in2')


# TODO(okuta): Implement searchsorted


# TODO(okuta): Implement extract
