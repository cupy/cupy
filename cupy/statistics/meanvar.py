import numpy

import cupy
from cupy import reduction
from cupy import math


def median(a, axis=None, out=None, overwrite_input=False, keepdims=False,
           allocator=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def average(a, axis=None, weights=None, returned=False, allocator=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError


# TODO(okuta) needs cast
mean = reduction.create_reduction_func(
    'cupy_mean',
    ['?->d', 'B->d', 'h->d', 'H->d', 'i->d', 'I->d', 'l->d', 'L->d',
     'q->q', 'Q->Q', 'e->e', 'f->f', 'd->d'],
    ('a + b', 'in[j]', 'out[i] = a / (in_size / out_size)'))


def var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False,
        allocator=None):
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
