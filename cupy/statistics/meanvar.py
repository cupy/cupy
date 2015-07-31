import numpy

import cupy
from cupy import math
from cupy import reduction


def median(a, axis=None, out=None, overwrite_input=False, keepdims=False,
           allocator=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def average(a, axis=None, weights=None, returned=False, allocator=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def mean(a, axis=None, dtype=None, out=None, keepdims=False, allocator=None):
    """Returns the arithmetic mean along an axis.

    Args:
        a (cupy.ndarray): Array to compute mean.
        axis (int): Along which axis to compute mean. The flattened array is
            used by default.
        dtype: Data type specifier.
        out (cupy.ndarray): Output array.
        keepdims (bool): If True, the axis is remained as an axis of size one.
        allocator (function): CuPy memory allocator. The allocator of ``a`` is
            used by default.

    Returns:
        cupy.ndarray: The mean of the input array along the axis.

    .. seealso:: :func:`numpy.mean`

    """
    return _mean(a, axis=axis, dtype=dtype, out=out, keepdims=keepdims,
                 allocator=allocator)


def var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False,
        allocator=None):
    """Returns the variance along an axis.

    Args:
        a (cupy.ndarray): Array to compute variance.
        axis (int): Along which axis to compute variance. The flattened array
            is used by default.
        dtype: Data type specifier.
        out (cupy.ndarray): Output array.
        keepdims (bool): If True, the axis is remained as an axis of size one.
        allocator (function): CuPy memory allocator. The allocator of ``a`` is
            used by default.

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
        dtype = numpy.dtype(numpy.float64)

    arrmean = mean(a, axis=axis, dtype=dtype, keepdims=True,
                   allocator=allocator)

    x = cupy.subtract(a, arrmean, dtype=dtype, allocator=allocator)
    cupy.square(x, x)
    ret = cupy.sum(x, axis=axis, dtype=dtype, out=out, keepdims=keepdims,
                   allocator=allocator)
    rcount = max(_count_reduce_items(a, axis) - ddof, 0)
    return cupy.multiply(ret, ret.dtype.type(1.0 / rcount), out=ret)


def std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False,
        allocator=None):
    """Returns the standard deviation along an axis.

    Args:
        a (cupy.ndarray): Array to compute standard deviation.
        axis (int): Along which axis to compute standard deviation. The
            flattened array is used by default.
        dtype: Data type specifier.
        out (cupy.ndarray): Output array.
        keepdims (bool): If True, the axis is remained as an axis of size one.
        allocator (function): CuPy memory allocator. The allocator of ``a`` is
            used by default.

    Returns:
        cupy.ndarray: The standard deviation of the input array along the axis.

    .. seealso:: :func:`numpy.std`

    """
    ret = var(a, axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims,
              allocator=allocator)
    return math.misc.sqrt_fixed(ret, dtype=dtype, out=out, allocator=allocator)


def nanmean(a, axis=None, dtype=None, out=None, keepdims=False,
            allocator=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def nanstd(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False,
           allocator=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def nanvar(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False,
           allocator=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def _count_reduce_items(arr, axis):
    items = 1
    for ax in axis:
        items *= arr.shape[ax]
    return items


# TODO(okuta) needs cast
_mean = reduction.create_reduction_func(
    'cupy_mean',
    ['?->d', 'B->d', 'h->d', 'H->d', 'i->d', 'I->d', 'l->d', 'L->d',
     'q->q', 'Q->Q', 'e->e', 'f->f', 'd->d'],
    ('a + b', 'in[i]', 'a / (in_size / out_size)'))
