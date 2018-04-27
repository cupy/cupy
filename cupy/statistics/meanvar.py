import numpy

import cupy


# TODO(okuta): Implement median


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

    .. seealso:: :func:`numpy.average`
    """
    a = cupy.asarray(a)

    if weights is None:
        avg = a.mean(axis)
        scl = avg.dtype.type(a.size / avg.size)
    else:
        wgt = cupy.asarray(weights)

        if issubclass(a.dtype.type, (numpy.integer, numpy.bool_)):
            result_dtype = numpy.result_type(a.dtype, wgt.dtype, 'f8')
        else:
            result_dtype = numpy.result_type(a.dtype, wgt.dtype)

        # Sanity checks
        if a.shape != wgt.shape:
            if axis is None:
                raise TypeError(
                    "Axis must be specified when shapes of a and weights "
                    "differ.")
            if wgt.ndim != 1:
                raise TypeError(
                    "1D weights expected when shapes of a and weights differ.")
            if wgt.shape[0] != a.shape[axis]:
                raise ValueError(
                    "Length of weights not compatible with specified axis.")

            # setup wgt to broadcast along axis
            wgt = cupy.broadcast_to(wgt, (a.ndim - 1) * (1,) + wgt.shape)
            wgt = wgt.swapaxes(-1, axis)

        scl = wgt.sum(axis=axis, dtype=result_dtype)
        if cupy.any(scl == 0.0):
            raise ZeroDivisionError(
                "Weights sum to zero, can't be normalized")

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


# TODO(okuta): Implement nanmean


# TODO(okuta): Implement nanstd


# TODO(okuta): Implement nanvar
