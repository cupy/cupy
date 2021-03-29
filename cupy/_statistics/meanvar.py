import functools
import numpy

import cupy
from cupy._core import _routines_statistics as _statistics


def median(a, axis=None, out=None, overwrite_input=False, keepdims=False):
    """Compute the median along the specified axis.

    Returns the median of the array elements.

    Args:
        a (cupy.ndarray): Array to compute the median.
        axis (int): Axis along which the medians are computed. The flattened
            array is used by default.
        out (cupy.ndarray): Output array.
        overwrite_input (bool): If ``True``, then allow use of memory of input
            array a for calculations. The input array will be modified by the
            call to median. This will save memory when you do not need to
            preserve the contents of the input array. Treat the input as
            undefined, but it will probably be fully or partially sorted.
            Default is ``False``. If ``overwrite_input`` is ``True`` and ``a``
            is not already an ndarray, an error will be raised.
        keepdims (bool): If ``True``, the axis is remained as an axis of size
            one.

    Returns:
        cupy.ndarray: The median of ``a``, along the axis if specified.

    .. seealso:: :func:`numpy.median`

    """
    return _statistics._median(a, axis, out, overwrite_input, keepdims)


def nanmedian(a, axis=None, out=None, overwrite_input=False, keepdims=False):
    """Compute the median along the specified axis, while ignoring NaNs.

    Returns the median of the array elements.

    Args:
        a (cupy.ndarray): Array to compute the median.
        axis (int): Axis along which the medians are computed. The flattened
            array is used by default.
        out (cupy.ndarray): Output array.
        overwrite_input (bool): If ``True``, then allow use of memory of input
            array a for calculations. The input array will be modified by the
            call to median. This will save memory when you do not need to
            preserve the contents of the input array. Treat the input as
            undefined, but it will probably be fully or partially sorted.
            Default is ``False``. If ``overwrite_input`` is ``True`` and ``a``
            is not already an ndarray, an error will be raised.
        keepdims (bool): If ``True``, the axis is remained as an axis of size
            one.

    Returns:
        cupy.ndarray: The median of ``a``, along the axis if specified.

    .. seealso:: :func:`numpy.nanmedian`

    """
    if a.dtype.char in 'efdFD':
        return _statistics._nanmedian(a, axis, out, overwrite_input, keepdims)
    else:
        return median(a, axis=axis, out=out, overwrite_input=overwrite_input,
                      keepdims=keepdims)


def average(a, axis=None, weights=None, returned=False):
    """Returns the weighted average along an axis.

    Args:
        a (cupy.ndarray): Array to compute average.
        axis (int): Along which axis to compute average. The flattened array
            is used by default.
        weights (cupy.ndarray): Array of weights where each element
            corresponds to the value in ``a``. If ``None``, all the values
            in ``a`` have a weight equal to one.
        returned (bool): If ``True``, a tuple of the average and the sum
            of weights is returned, otherwise only the average is returned.

    Returns:
        cupy.ndarray or tuple of cupy.ndarray: The average of the input array
            along the axis and the sum of weights.

    .. warning::

        This function may synchronize the device if ``weight`` is given.

    .. seealso:: :func:`numpy.average`
    """
    # TODO(niboshi): Avoid synchronization.
    a = cupy.asarray(a)

    if weights is None:
        avg = a.mean(axis)
        scl = avg.dtype.type(a.size / avg.size)
    else:
        wgt = cupy.asarray(weights)

        if issubclass(a.dtype.type, (numpy.integer, numpy.bool_)):
            result_dtype = functools.reduce(numpy.promote_types,
                                            (a.dtype, wgt.dtype, 'f8'))
        else:
            result_dtype = numpy.promote_types(a.dtype, wgt.dtype)

        # Sanity checks
        if a.shape != wgt.shape:
            if axis is None:
                raise TypeError(
                    'Axis must be specified when shapes of a and weights '
                    'differ.')
            if wgt.ndim != 1:
                raise TypeError(
                    '1D weights expected when shapes of a and weights differ.')
            if wgt.shape[0] != a.shape[axis]:
                raise ValueError(
                    'Length of weights not compatible with specified axis.')

            # setup wgt to broadcast along axis
            wgt = cupy.broadcast_to(wgt, (a.ndim - 1) * (1,) + wgt.shape)
            wgt = wgt.swapaxes(-1, axis)

        scl = wgt.sum(axis=axis, dtype=result_dtype)
        if cupy.any(scl == 0.0):  # synchronize!
            raise ZeroDivisionError(
                'Weights sum to zero, can\'t be normalized')

        avg = cupy.multiply(a, wgt, dtype=result_dtype).sum(axis) / scl

    if returned:
        if scl.shape != avg.shape:
            scl = cupy.broadcast_to(cupy.array(scl), avg.shape).copy()
        return avg, scl
    else:
        return avg


def mean(a, axis=None, dtype=None, out=None, keepdims=False):
    """Returns the arithmetic mean along an axis.

    Args:
        a (cupy.ndarray): Array to compute mean.
        axis (int): Along which axis to compute mean. The flattened array is
            used by default.
        dtype: Data type specifier.
        out (cupy.ndarray): Output array.
        keepdims (bool): If ``True``, the axis is remained as an axis of
            size one.

    Returns:
        cupy.ndarray: The mean of the input array along the axis.

    .. seealso:: :func:`numpy.mean`

    """
    # TODO(okuta): check type
    return a.mean(axis=axis, dtype=dtype, out=out, keepdims=keepdims)


def var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    """Returns the variance along an axis.

    Args:
        a (cupy.ndarray): Array to compute variance.
        axis (int): Along which axis to compute variance. The flattened array
            is used by default.
        dtype: Data type specifier.
        out (cupy.ndarray): Output array.
        keepdims (bool): If ``True``, the axis is remained as an axis of
            size one.

    Returns:
        cupy.ndarray: The variance of the input array along the axis.

    .. seealso:: :func:`numpy.var`

    """
    # TODO(okuta): check type
    return a.var(axis=axis, dtype=dtype, out=out, ddof=ddof,
                 keepdims=keepdims)


def std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    """Returns the standard deviation along an axis.

    Args:
        a (cupy.ndarray): Array to compute standard deviation.
        axis (int): Along which axis to compute standard deviation. The
            flattened array is used by default.
        dtype: Data type specifier.
        out (cupy.ndarray): Output array.
        keepdims (bool): If ``True``, the axis is remained as an axis of
            size one.

    Returns:
        cupy.ndarray: The standard deviation of the input array along the axis.

    .. seealso:: :func:`numpy.std`

    """
    # TODO(okuta): check type
    return a.std(axis=axis, dtype=dtype, out=out, ddof=ddof,
                 keepdims=keepdims)


def nanmean(a, axis=None, dtype=None, out=None, keepdims=False):
    """Returns the arithmetic mean along an axis ignoring NaN values.

    Args:
        a (cupy.ndarray): Array to compute mean.
        axis (int): Along which axis to compute mean. The flattened array is
            used by default.
        dtype: Data type specifier.
        out (cupy.ndarray): Output array.
        keepdims (bool): If ``True``, the axis is remained as an axis of
            size one.

    Returns:
        cupy.ndarray: The mean of the input array along the axis ignoring NaNs.

    .. seealso:: :func:`numpy.nanmean`

    """
    if a.dtype.kind in 'biu':
        return a.mean(axis=axis, dtype=dtype, out=out, keepdims=keepdims)

    # TODO(okuta): check type
    return _statistics._nanmean(
        a, axis=axis, dtype=dtype, out=out, keepdims=keepdims)


def nanvar(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    """Returns the variance along an axis ignoring NaN values.

    Args:
        a (cupy.ndarray): Array to compute variance.
        axis (int): Along which axis to compute variance. The flattened array
            is used by default.
        dtype: Data type specifier.
        out (cupy.ndarray): Output array.
        keepdims (bool): If ``True``, the axis is remained as an axis of
            size one.

    Returns:
        cupy.ndarray: The variance of the input array along the axis.

    .. seealso:: :func:`numpy.nanvar`

    """
    if a.dtype.kind in 'biu':
        return a.var(axis=axis, dtype=dtype, out=out, ddof=ddof,
                     keepdims=keepdims)

    # TODO(okuta): check type
    return _statistics._nanvar(
        a, axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims)


def nanstd(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    """Returns the standard deviation along an axis ignoring NaN values.

    Args:
        a (cupy.ndarray): Array to compute standard deviation.
        axis (int): Along which axis to compute standard deviation. The
            flattened array is used by default.
        dtype: Data type specifier.
        out (cupy.ndarray): Output array.
        keepdims (bool): If ``True``, the axis is remained as an axis of
            size one.

    Returns:
        cupy.ndarray: The standard deviation of the input array along the axis.

    .. seealso:: :func:`numpy.nanstd`

    """
    if a.dtype.kind in 'biu':
        return a.std(axis=axis, dtype=dtype, out=out, ddof=ddof,
                     keepdims=keepdims)

    # TODO(okuta): check type
    return _statistics._nanstd(
        a, axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims)
