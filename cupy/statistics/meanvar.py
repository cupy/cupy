import numpy

from cupy import math
from cupy import reduction


# TODO(okuta): Implement median


# TODO(okuta): Implement average


def mean(a, axis=None, dtype=None, out=None, keepdims=False):
    """Returns the arithmetic mean along an axis.

    Args:
        a (cupy.ndarray): Array to compute mean.
        axis (int): Along which axis to compute mean. The flattened array is
            used by default.
        dtype: Data type specifier.
        out (cupy.ndarray): Output array.
        keepdims (bool): If True, the axis is remained as an axis of size one.

    Returns:
        cupy.ndarray: The mean of the input array along the axis.

    .. seealso:: :func:`numpy.mean`

    """
    return _mean(a, axis=axis, dtype=dtype, out=out, keepdims=keepdims)


def var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    """Returns the variance along an axis.

    Args:
        a (cupy.ndarray): Array to compute variance.
        axis (int): Along which axis to compute variance. The flattened array
            is used by default.
        dtype: Data type specifier.
        out (cupy.ndarray): Output array.
        keepdims (bool): If True, the axis is remained as an axis of size one.

    Returns:
        cupy.ndarray: The variance of the input array along the axis.

    .. seealso:: :func:`numpy.var`

    """
    if axis is None:
        axis = tuple(range(a.ndim))
    if not isinstance(axis, tuple):
        axis = (axis,)

    if dtype is None and issubclass(a.dtype.type,
                                    (numpy.integer, numpy.bool_)):
        dtype = 'd'

    shape = a.shape
    items = 1
    for ax in axis:
        items *= shape[ax]
    alpha = 1. / max(items - ddof, 0)
    arrmean = mean(a, axis=axis, dtype=dtype, keepdims=True)
    if out is None:
        return _var_core(a, arrmean, alpha, axis=axis, keepdims=keepdims)
    else:
        return _var_core_out(
            a, arrmean, alpha, out, axis=axis, keepdims=keepdims)


def std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    """Returns the standard deviation along an axis.

    Args:
        a (cupy.ndarray): Array to compute standard deviation.
        axis (int): Along which axis to compute standard deviation. The
            flattened array is used by default.
        dtype: Data type specifier.
        out (cupy.ndarray): Output array.
        keepdims (bool): If True, the axis is remained as an axis of size one.

    Returns:
        cupy.ndarray: The standard deviation of the input array along the axis.

    .. seealso:: :func:`numpy.std`

    """
    ret = var(a, axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims)
    return math.misc.sqrt_fixed(ret, dtype=dtype, out=out)


# TODO(okuta): Implement nanmean


# TODO(okuta): Implement nanstd


# TODO(okuta): Implement nanvar


_var_core = reduction.ReductionKernel(
    'S x, T mean, T alpha', 'T out',
    '(x - mean) * (x - mean)',
    'a + b', 'out = alpha * a', '0', '_var_core')
_var_core_out = reduction.ReductionKernel(
    'S x, T mean, T alpha', 'U out',
    '(x - mean) * (x - mean)',
    'a + b', 'out = alpha * a', '0', '_var_core')

# TODO(okuta) needs cast
_mean = reduction.create_reduction_func(
    'cupy_mean',
    ('?->d', 'B->d', 'h->d', 'H->d', 'i->d', 'I->d', 'l->d', 'L->d',
     'q->d', 'Q->d',
     ('e->e', (None, None, None, 'float')),
     'f->f', 'd->d'),
    ('in0', 'a + b', 'out0 = a / (_in_ind.size() / _out_ind.size())', None))
