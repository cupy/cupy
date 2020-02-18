import numpy

import cupy
from cupy.core import _routines_math as _math
from cupy.core import fusion


def sum(a, axis=None, dtype=None, out=None, keepdims=False):
    """Returns the sum of an array along given axes.

    Args:
        a (cupy.ndarray): Array to take sum.
        axis (int or sequence of ints): Axes along which the sum is taken.
        dtype: Data type specifier.
        out (cupy.ndarray): Output array.
        keepdims (bool): If ``True``, the specified axes are remained as axes
            of length one.

    Returns:
        cupy.ndarray: The result array.

    .. seealso:: :func:`numpy.sum`

    """
    if fusion._is_fusing():
        if keepdims:
            raise NotImplementedError(
                'cupy.sum does not support `keepdims` in fusion yet.')
        return fusion._call_reduction(_math.sum_auto_dtype,
                                      a, axis=axis, dtype=dtype, out=out)

    # TODO(okuta): check type
    return a.sum(axis, dtype, out, keepdims)


def prod(a, axis=None, dtype=None, out=None, keepdims=False):
    """Returns the product of an array along given axes.

    Args:
        a (cupy.ndarray): Array to take product.
        axis (int or sequence of ints): Axes along which the product is taken.
        dtype: Data type specifier.
        out (cupy.ndarray): Output array.
        keepdims (bool): If ``True``, the specified axes are remained as axes
            of length one.

    Returns:
        cupy.ndarray: The result array.

    .. seealso:: :func:`numpy.prod`

    """
    if fusion._is_fusing():
        if keepdims:
            raise NotImplementedError(
                'cupy.prod does not support `keepdims` in fusion yet.')
        return fusion._call_reduction(_math.prod_auto_dtype,
                                      a, axis=axis, dtype=dtype, out=out)

    # TODO(okuta): check type
    return a.prod(axis, dtype, out, keepdims)


def nansum(a, axis=None, dtype=None, out=None, keepdims=False):
    """Returns the sum of an array along given axes treating Not a Numbers
    (NaNs) as zero.

    Args:
        a (cupy.ndarray): Array to take sum.
        axis (int or sequence of ints): Axes along which the sum is taken.
        dtype: Data type specifier.
        out (cupy.ndarray): Output array.
        keepdims (bool): If ``True``, the specified axes are remained as axes
            of length one.

    Returns:
        cupy.ndarray: The result array.

    .. seealso:: :func:`numpy.nansum`

    """
    if fusion._is_fusing():
        if keepdims:
            raise NotImplementedError(
                'cupy.nansum does not support `keepdims` in fusion yet.')
        return fusion._call_reduction(_math.nansum_auto_dtype,
                                      a, axis=axis, dtype=dtype, out=out)

    # TODO(okuta): check type
    return _math._nansum(a, axis, dtype, out, keepdims)


def nanprod(a, axis=None, dtype=None, out=None, keepdims=False):
    """Returns the product of an array along given axes treating Not a Numbers
    (NaNs) as zero.

    Args:
        a (cupy.ndarray): Array to take product.
        axis (int or sequence of ints): Axes along which the product is taken.
        dtype: Data type specifier.
        out (cupy.ndarray): Output array.
        keepdims (bool): If ``True``, the specified axes are remained as axes
            of length one.

    Returns:
        cupy.ndarray: The result array.

    .. seealso:: :func:`numpy.nanprod`

    """
    if fusion._is_fusing():
        if keepdims:
            raise NotImplementedError(
                'cupy.nanprod does not support `keepdims` in fusion yet.')
        return fusion._call_reduction(_math.nanprod_auto_dtype,
                                      a, axis=axis, dtype=dtype, out=out)

    # TODO(okuta): check type
    return _math._nanprod(a, axis, dtype, out, keepdims)


def cumsum(a, axis=None, dtype=None, out=None):
    """Returns the cumulative sum of an array along a given axis.

    Args:
        a (cupy.ndarray): Input array.
        axis (int): Axis along which the cumulative sum is taken. If it is not
            specified, the input is flattened.
        dtype: Data type specifier.
        out (cupy.ndarray): Output array.

    Returns:
        cupy.ndarray: The result array.

    .. seealso:: :func:`numpy.cumsum`

    """
    return _math.scan_core(a, axis, _math.scan_op.SCAN_SUM, dtype, out)


def cumprod(a, axis=None, dtype=None, out=None):
    """Returns the cumulative product of an array along a given axis.

    Args:
        a (cupy.ndarray): Input array.
        axis (int): Axis along which the cumulative product is taken. If it is
            not specified, the input is flattened.
        dtype: Data type specifier.
        out (cupy.ndarray): Output array.

    Returns:
        cupy.ndarray: The result array.

    .. seealso:: :func:`numpy.cumprod`

    """
    return _math.scan_core(a, axis, _math.scan_op.SCAN_PROD, dtype, out)


def diff(a, n=1, axis=-1, prepend=None, append=None):
    """Calculate the n-th discrete difference along the given axis.

    Args:
        a (cupy.ndarray): Input array.
        n (int): The number of times values are differenced. If zero, the input
            is returned as-is.
        axis (int): The axis along which the difference is taken, default is
            the last axis.
        prepend (int, float, cupy.ndarray): Value to prepend to ``a``.
        append (int, float, cupy.ndarray): Value to append to ``a``.

    Returns:
        cupy.ndarray: The result array.

    .. seealso:: :func:`numpy.diff`
    """

    if n == 0:
        return a
    if n < 0:
        raise ValueError(
            "order must be non-negative but got " + repr(n))

    a = cupy.asanyarray(a)
    nd = a.ndim

    combined = []

    if prepend is not None:
        prepend = cupy.asanyarray(prepend)
        if prepend.ndim == 0:
            shape = list(a.shape)
            shape[axis] = 1
            prepend = cupy.broadcast_to(prepend, tuple(shape))
        combined.append(prepend)

    combined.append(a)

    if append is not None:
        append = cupy.asanyarray(append)
        if append.ndim == 0:
            shape = list(a.shape)
            shape[axis] = 1
            append = cupy.broadcast_to(append, tuple(shape))
        combined.append(append)

    if len(combined) > 1:
        a = cupy.concatenate(combined, axis)

    slice1 = [slice(None)] * nd
    slice2 = [slice(None)] * nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    slice1 = tuple(slice1)
    slice2 = tuple(slice2)

    op = cupy.not_equal if a.dtype == numpy.bool_ else cupy.subtract
    for _ in range(n):
        a = op(a[slice1], a[slice2])

    return a


# TODO(okuta): Implement ediff1d


# TODO(okuta): Implement gradient


# TODO(okuta): Implement cross


# TODO(okuta): Implement trapz
