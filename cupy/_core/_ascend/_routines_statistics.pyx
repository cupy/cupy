# Ascend-only implementation of cupy._core._routines_statistics.
#
# This file replaces cupy/_core/_routines_statistics.pyx when the Ascend
# backend is selected (see install/cupy_builder/features/ascend.py).  It is
# the upstream file with:
#
# * all CUDA kernel preambles removed (the reduction routine bodies are
#   ignored on Ascend -- dispatch goes through the registered aclnn ops,
#   only the ufunc name and dtype signature list matter);
# * all `IF CUPY_CANN_VERSION` conditional compilation resolved to the
#   Ascend branch;
# * the CUB / cuTENSOR accelerator paths dropped (CUDA-only);
# * the CUDA-only reduction kernels (_var_core_*, _nanvar_core*,
#   _exists_nan) removed; the operations they provided are composed from
#   already-registered aclnn ops at the Python level instead.

from cpython cimport sequence

import numpy
from numpy import nan

import cupy
from cupy.exceptions import AxisError
from cupy._core import _reduction
from cupy._core._reduction import create_reduction_func
from cupy._core._kernel import ElementwiseKernel
from cupy._core._ufuncs import elementwise_copy

from cupy._core cimport _routines_math as _math
from cupy._core.core cimport _ndarray_base


cdef _ndarray_base _ndarray_max(
        _ndarray_base self, axis, out, dtype, keepdims):
    return _amax(self, axis=axis, out=out, dtype=dtype, keepdims=keepdims)


cdef _ndarray_base _ndarray_min(
        _ndarray_base self, axis, out, dtype, keepdims):
    return _amin(self, axis=axis, out=out, dtype=dtype, keepdims=keepdims)


cdef _ndarray_base _ndarray_ptp(_ndarray_base self, axis, out, keepdims):
    result = _amax(self, axis=axis, out=out, keepdims=keepdims)
    result -= _amin(self, axis=axis, out=None, keepdims=keepdims)
    return result


# TODO(leofang): this signature is incompatible with NumPy!
# difference: cupy has one extra dtype arg
cdef _ndarray_base _ndarray_argmax(
        _ndarray_base self, axis, out, dtype, keepdims):
    return _argmax(self, axis=axis, out=out, dtype=dtype, keepdims=keepdims)


# TODO(leofang): this signature is incompatible with NumPy!
cdef _ndarray_base _ndarray_argmin(
        _ndarray_base self, axis, out, dtype, keepdims):
    return _argmin(self, axis=axis, out=out, dtype=dtype, keepdims=keepdims)


cdef _ndarray_base _ndarray_mean(
        _ndarray_base self, axis, dtype, out, keepdims):
    dtype_sum = dtype_out = dtype
    if dtype is None:
        if self.dtype.kind in 'iub':
            dtype_out = numpy.float64
            dtype_sum = numpy.float64
        elif self.dtype.char == 'e':
            dtype_sum = numpy.float32
            dtype_out = numpy.float16
        # ASCEND: aclnnMean() does not promote the integer sum result to
        # float before division.  Cast the input to the accumulation dtype
        # first, then reduce with dtype=None (slower but robust).
        result = _mean(
            self.asdtype(dtype_sum), axis=axis, dtype=None, out=out,
            keepdims=keepdims)
    elif numpy.dtype(dtype).kind in 'iub':
        # output will be the requested type, but compute the mean using float
        dtype_out = dtype
        dtype_sum = numpy.float64
        result = _mean(
            self, axis=axis, dtype=dtype_sum, out=out, keepdims=keepdims)
    else:
        result = _mean(
            self, axis=axis, dtype=dtype_sum, out=out, keepdims=keepdims)

    if dtype_out is not None and out is None:
        result = result.astype(dtype_out)
    return result


cdef _ndarray_base _ndarray_var(
        _ndarray_base self, axis, dtype, out, ddof, keepdims):
    return _var(
        self, axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims)


cdef _ndarray_base _ndarray_std(
        _ndarray_base self, axis, dtype, out, ddof, keepdims):
    return _std(
        self, axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims)


# NOTE: the routine strings below reference the upstream CUDA preamble types
# (min_max_st etc.) and are kept verbatim from upstream for diff-friendliness;
# on Ascend they are ignored because dispatch goes to the registered aclnn op.

cdef _amin = create_reduction_func(
    'cupy_min',
    ('?->?', 'b->b', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I', 'l->l', 'L->L',
     'q->q', 'Q->Q',
     ('e->e', (None, 'my_min_float(a, b)', None, None)),
     ('f->f', (None, 'my_min_float(a, b)', None, None)),
     ('d->d', (None, 'my_min_float(a, b)', None, None)),
     ('F->F', (None, 'my_min_float(a, b)', None, None)),
     ('D->D', (None, 'my_min_float(a, b)', None, None))),
    ('min_max_st<type_in0_raw>(in0)', 'my_min(a, b)', 'out0 = a.value',
     'min_max_st<type_in0_raw>'))


cdef _amax = create_reduction_func(
    'cupy_max',
    ('?->?', 'b->b', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I', 'l->l', 'L->L',
     'q->q', 'Q->Q',
     ('e->e', (None, 'my_max_float(a, b)', None, None)),
     ('f->f', (None, 'my_max_float(a, b)', None, None)),
     ('d->d', (None, 'my_max_float(a, b)', None, None)),
     ('F->F', (None, 'my_max_float(a, b)', None, None)),
     ('D->D', (None, 'my_max_float(a, b)', None, None)),
     ),
    ('min_max_st<type_in0_raw>(in0)', 'my_max(a, b)', 'out0 = a.value',
     'min_max_st<type_in0_raw>'))


nanmin = create_reduction_func(
    'cupy_nanmin',
    ('?->?', 'b->b', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I', 'l->l', 'L->L',
     'q->q', 'Q->Q', 'e->e', 'f->f', 'd->d', 'F->F', 'D->D'),
    ('min_max_st<type_in0_raw>(in0)', 'my_min(a, b)', 'out0 = a.value',
     'min_max_st<type_in0_raw>'))


nanmax = create_reduction_func(
    'cupy_nanmax',
    ('?->?', 'b->b', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I', 'l->l', 'L->L',
     'q->q', 'Q->Q', 'e->e', 'f->f', 'd->d', 'F->F', 'D->D'),
    ('min_max_st<type_in0_raw>(in0)', 'my_max(a, b)', 'out0 = a.value',
     'min_max_st<type_in0_raw>'))


cdef _argmin = create_reduction_func(
    'cupy_argmin',
    tuple(['{}->{}'.format(d, r) for r in 'qlihb' for d in '?BhHiIlLqQ'])
    + (
        ('e->q', (None, 'my_argmin_float(a, b)', None, None)),
        ('f->q', (None, 'my_argmin_float(a, b)', None, None)),
        ('d->q', (None, 'my_argmin_float(a, b)', None, None)),
        ('F->q', (None, 'my_argmin_float(a, b)', None, None)),
        ('D->q', (None, 'my_argmin_float(a, b)', None, None))),
    ('min_max_st<type_in0_raw>(in0, _J)', 'my_argmin(a, b)', 'out0 = a.index',
     'min_max_st<type_in0_raw>'),
    None, None, sort_reduce_axis=False)


cdef _argmax = create_reduction_func(
    'cupy_argmax',
    tuple(['{}->{}'.format(d, r) for r in 'qlihb' for d in '?BhHiIlLqQ'])
    + (
        ('e->q', (None, 'my_argmax_float(a, b)', None, None)),
        ('f->q', (None, 'my_argmax_float(a, b)', None, None)),
        ('d->q', (None, 'my_argmax_float(a, b)', None, None)),
        ('F->q', (None, 'my_argmax_float(a, b)', None, None)),
        ('D->q', (None, 'my_argmax_float(a, b)', None, None))),
    ('min_max_st<type_in0_raw>(in0, _J)', 'my_argmax(a, b)', 'out0 = a.index',
     'min_max_st<type_in0_raw>'),
    None, None, sort_reduce_axis=False)


cpdef _ndarray_base _nanargmax(_ndarray_base a, axis, out, dtype, keepdims):
    return _nanargmax_func(
        a, axis=axis, out=out, dtype=dtype, keepdims=keepdims)


cpdef _ndarray_base _nanargmin(_ndarray_base a, axis, out, dtype, keepdims):
    return _nanargmin_func(
        a, axis=axis, out=out, dtype=dtype, keepdims=keepdims)


cdef _nanargmin_func = create_reduction_func(
    'cupy_nanargmin',
    ('?->q', 'B->q', 'h->q', 'H->q', 'i->q', 'I->q', 'l->q', 'L->q',
     'q->q', 'Q->q',
     ('e->q', (None, 'my_argmin_float(a, b)', None, None)),
     ('f->q', (None, 'my_argmin_float(a, b)', None, None)),
     ('d->q', (None, 'my_argmin_float(a, b)', None, None)),
     ('F->q', (None, 'my_argmin_float(a, b)', None, None)),
     ('D->q', (None, 'my_argmin_float(a, b)', None, None))),
    ('min_max_st<type_in0_raw>(in0, isnan(in0) ? -1 : _J)',
     'my_argmin(a, b)', 'out0 = a.index', 'min_max_st<type_in0_raw>'),
    None, None, sort_reduce_axis=False)


cdef _nanargmax_func = create_reduction_func(
    'cupy_nanargmax',
    ('?->q', 'B->q', 'h->q', 'H->q', 'i->q', 'I->q', 'l->q', 'L->q',
     'q->q', 'Q->q',
     ('e->q', (None, 'my_argmax_float(a, b)', None, None)),
     ('f->q', (None, 'my_argmax_float(a, b)', None, None)),
     ('d->q', (None, 'my_argmax_float(a, b)', None, None)),
     ('F->q', (None, 'my_argmax_float(a, b)', None, None)),
     ('D->q', (None, 'my_argmax_float(a, b)', None, None))),
    ('min_max_st<type_in0_raw>(in0, isnan(in0) ? -1 : _J)',
     'my_argmax(a, b)', 'out0 = a.index', 'min_max_st<type_in0_raw>'),
    None, None, sort_reduce_axis=False)


cpdef _ndarray_base _median(
        _ndarray_base a, axis, out, overwrite_input, keepdims):

    keep_ndim = a.ndim

    out_shape = None
    if sequence.PySequence_Check(axis):
        # cupy.sort and cupy.partition only support integer axis, so move
        # all reduced dimensions to the end and reshape them into a single
        # reduction axis.
        reduce_axis, out_axis = _reduction._get_axis(axis, keep_ndim)
        out_shape = _reduction._get_out_shape(a.shape, reduce_axis, out_axis,
                                              keepdims)
        a = a.transpose(out_axis + reduce_axis)
        sort_shape = tuple([a.shape[n] for n in range(len(out_axis))]) + (-1,)
        a = a.reshape(sort_shape)
        if not a.flags.c_contiguous:
            a = cupy.ascontiguousarray(a)
        axis = -1

    if axis is None:
        sz = a.size
    else:
        if axis < -keep_ndim or axis >= keep_ndim:
            raise AxisError('Axis overrun')
        sz = a.shape[axis]
    if sz % 2 == 0:
        szh = sz // 2
        kth = [szh - 1, szh]
    else:
        kth = [(sz - 1) // 2]

    if overwrite_input:
        part = a
    else:
        part = a.copy()

    if axis is None:
        part = part.ravel()
        part.partition(kth)
    else:
        part.partition(kth, axis=axis)

    if part.shape == ():
        return part
    if axis is None:
        axis = 0

    indexer = [slice(None)] * part.ndim

    if keepdims and out_shape is None:
        _indexer = [None] * (keep_ndim - part.ndim)
        indexer.extend(_indexer)

    index = part.shape[axis] // 2
    if part.shape[axis] % 2 == 1:
        indexer[axis] = slice(index, index+1)
    else:
        indexer[axis] = slice(index-1, index+1)
    indexer = tuple(indexer)

    sel = part[indexer]
    # ASCEND: `_mean` has no dtype promotion for create_reduction_func()
    # int should be cast to float, after view is made contiguous
    if sel.dtype.kind in 'iub':
        sel = cupy.ascontiguousarray(sel).astype(numpy.float64)
    out = _mean(
        sel, axis=axis, dtype=None, out=out, keepdims=keepdims)

    if part.dtype.kind in 'fc':
        # ASCEND TODO: `_exists_nan` cannot be exported as a reduction kernel
        # here, so the NaN propagation of the median result (upstream:
        # isnan(part) + any, then cupy.where) is still missing.
        pass
    if out_shape is not None:
        out = out.reshape(out_shape)
    return out


cpdef _ndarray_base _nanmedian(
        _ndarray_base a, axis, out, overwrite_input, keepdims):

    if axis is None:
        axis = tuple(range(a.ndim))
    if not sequence.PySequence_Check(axis):
        axis = (axis,)

    reduce_axis = []
    reduce_shape = []
    out_axis = []
    out_shape = []
    for i in range(a.ndim):
        if axis is None or i in axis or i - a.ndim in axis:
            reduce_axis.append(i)
            reduce_shape.append(a.shape[i])
        else:
            out_axis.append(i)
            out_shape.append(a.shape[i])

    a_data_ptr = a.data.ptr
    a = a.transpose(out_axis + reduce_axis)
    a = a.reshape(out_shape + [-1, ])
    a = cupy.ascontiguousarray(a)

    n_reduce = numpy.prod(reduce_shape)
    n_reduce_each = cupy.full(out_shape, n_reduce, dtype='int32')
    if a_data_ptr == a.data.ptr and overwrite_input is False:
        a = a.copy()

    from cupy.backends.backend import is_ascend
    if is_ascend:
        # ASCEND has no such kernel, impl by cupy APIs.
        # cupy.where three args not working, use mask setitem to replace nan
        # (replaces `_replace_nan_kernel`: NaN -> finfo.max and per-row valid
        # count decrement).
        mask = cupy.isnan(a)
        a[mask] = a.dtype.type(numpy.finfo(a.dtype).max)
        n_valid_each = n_reduce_each - mask.sum(axis=-1).astype('int32')
        a = cupy.sort(a, axis=-1)

        # Pickup the median of each reduction row (replaces
        # `_pickup_median_kernel`): gather the l-th / h-th element of every
        # row with a flat take, then average the two middle values.
        n_out = int(numpy.prod(out_shape))
        rows = cupy.arange(n_out, dtype='int64')
        l_idx = cupy.maximum((n_valid_each - 1) // 2, 0).reshape(n_out)
        h_idx = (n_valid_each // 2).reshape(n_out)
        flat = a.reshape(n_out, n_reduce).ravel()
        al = flat.take(rows * n_reduce + l_idx.astype('int64'))
        ah = flat.take(rows * n_reduce + h_idx.astype('int64'))
        b = (al + ah) / 2
        # Rows without any valid value evaluate to NaN
        b[(n_valid_each == 0).reshape(n_out)] = b.dtype.type(nan)
        b = b.reshape(out_shape)
    else:
        _replace_nan_kernel(n_reduce, numpy.finfo(a.dtype).max, a, n_reduce_each)
        a = cupy.sort(a, axis=-1)

        b = cupy.full(out_shape, cupy.nan, dtype=a.dtype)
        _pickup_median_kernel(n_reduce, n_reduce_each, a, b)

    if keepdims:
        b = b.reshape(out_shape + [1, ] * len(reduce_axis))
        axes = [-1, ] * b.ndim
        for i, j in enumerate(out_axis + reduce_axis):
            axes[j] = i
        b = b.transpose(axes)

    if out is None:
        out = b
    else:
        elementwise_copy(b, out)
    return out


cdef _replace_nan_kernel = ElementwiseKernel(
    'I n_reduce, T val', 'T a, raw I n_reduce_each',
    '''
    if (a != a) {
        a = val;
        atomicAdd(&(n_reduce_each[i / n_reduce]), -1);
    }
    ''',
    'cupy_replace_nan'
)

cdef _pickup_median_kernel = ElementwiseKernel(
    'I n_reduce, I n_reduce_each, raw T a', 'T b',
    '''
    if (n_reduce_each > 0) {
        int l = (n_reduce_each - 1) / 2;
        int h = (n_reduce_each    ) / 2;
        if (l == h) {
            b = a[l + n_reduce * i];
        } else {
            b = (a[l + n_reduce * i] + a[h + n_reduce * i])
                / static_cast<T>(2.0);
        }
    }
    ''',
    'cupy_pickup_median'
)


cdef _ndarray_base _mean(
        _ndarray_base a, axis=None, dtype=None, out=None, keepdims=False):
    if a.size == 0:
        # Return nan; see also https://github.com/numpy/numpy/issues/13582
        return _mean_core_empty(a, axis, dtype, out, keepdims)
    return _mean_core(a, axis, dtype, out, keepdims)

cdef _ndarray_base _var(
        _ndarray_base a, axis=None, dtype=None, out=None, ddof=0,
        keepdims=False):

    if axis is None:
        axis = tuple(range(a.ndim))
    if not isinstance(axis, tuple):
        axis = (axis,)

    dtype_mean = a.dtype
    dtype_out = numpy.dtype(dtype)
    if dtype is None:
        if a.dtype.kind in 'biu':
            dtype_mean = 'float64'
            dtype_out = 'float64'
        else:
            dtype_mean = a.dtype
            dtype_out = a.dtype
            if a.dtype.kind == 'c':
                dtype_out = numpy.dtype(a.dtype.char.lower())

    shape = a.shape
    cdef Py_ssize_t items = 1
    for ax in axis:
        items *= shape[ax]

    # Make alpha NaN when array is empty, mimics NumPy behavior, resulting in
    # NaN. See https://github.com/numpy/numpy/issues/13582 for an explanation
    # on why NaN is the result.
    div = max(items - ddof, 0)
    alpha = 1. / div if div != 0 else nan

    arrmean = a.mean(axis=axis, dtype=dtype_mean, out=None, keepdims=True)
    # ASCEND: `_var_core_*` is a ReductionKernel with 3 inputs and 1 output,
    # which launch_reduction_op does not support.  Compose the variance from
    # already-registered ops instead: subtract -> in-place square -> sum
    # -> alpha (the C++ aclop_VarCore general op is kept as a fallback).
    d = a - arrmean
    d *= d
    if out is None:
        out = d.sum(axis=axis, dtype=dtype_out, keepdims=keepdims)
    else:
        d.sum(axis=axis, dtype=dtype_out, keepdims=keepdims, out=out)
    out *= alpha
    return out.astype(dtype_out, copy=False)


cdef _ndarray_base _std(
        _ndarray_base a, axis=None, dtype=None, out=None, ddof=0,
        keepdims=False):
    ret = _var(
        a, axis=axis, dtype=dtype, out=None, ddof=ddof, keepdims=keepdims)
    return _math._sqrt(ret, dtype=dtype, out=out)


# TODO(okuta) needs cast
cdef _mean_core = create_reduction_func(
    'cupy_mean',
    ('?->d', 'B->d', 'h->d', 'H->d', 'i->d', 'I->d', 'l->d', 'L->d',
     'q->d', 'Q->d',
     ('e->e', (None, None, None, 'float')),
     'f->f', 'd->d', 'F->F', 'D->D'),
    ('in0', 'a + b',
     'out0 = a / _type_reduce(_in_ind.size() / _out_ind.size())', None))

cdef _mean_core_empty = create_reduction_func(
    'cupy_mean_empty',
    ('?->d', 'B->d', 'h->d', 'H->d', 'i->d', 'I->d', 'l->d', 'L->d',
     'q->d', 'Q->d',
     ('e->e', (None, None, None, 'float')),
     'f->f', 'd->d', 'F->F', 'D->D'),
    ('in0', 'a + b',
     'out0 = a / _type_reduce(_in_ind.size() / _out_ind.size())', None), 0)


cdef _nanmean_func = create_reduction_func(
    'cupy_nanmean',
    ('e->e', 'f->f', 'd->d', 'F->F', 'D->D'),
    ('in0', 'my_nanmean(a, b)',
     'out0 = a.value / type_out0_raw(a.count)', 'nanmean_st<type_out0_raw>'))


_count_non_nan = create_reduction_func(
    'cupy_count_non_nan',
    ('e->q', 'f->q', 'd->q', 'F->q', 'D->q'),
    ('isnan(in0) ? 0 : 1', 'a + b', 'out0 = a', None), 0)


cpdef _ndarray_base _nanmean(_ndarray_base a, axis, dtype, out, keepdims):
    return _nanmean_func(a, axis=axis, dtype=dtype, out=out, keepdims=keepdims)


cpdef _ndarray_base _nanstd(_ndarray_base a, axis, dtype, out, ddof, keepdims):
    var = _nanvar(a, axis, dtype, None, ddof, keepdims)
    return _math._sqrt(var, dtype=dtype, out=out)


cpdef _ndarray_base _nanvar(_ndarray_base a, axis, dtype, out, ddof, keepdims):
    # ASCEND: `_nanvar_core` / `_count_non_nan` / `_math._nansum(dtype=None)`
    # do not work with the aclnn reductions yet.  Compute with public cupy
    # APIs using keepdims=False, then reshape afterwards to emulate keepdims.
    # TODO: this temporary solution may be slow; a custom kernel may be better
    arrsum = cupy.nansum(a, axis=axis)
    _count = cupy.sum((~cupy.isnan(a)).astype(numpy.float32), axis=axis)
    nanmean = arrsum / _count
    sq = a - nanmean
    sq = sq * sq
    sq_sum = cupy.nansum(sq, axis=axis)
    result = sq_sum / (_count - ddof)
    if keepdims:
        result = cupy.reshape(result, (1,) * a.ndim)
    if out is not None:
        out[...] = result
        result = out
    return result


# Variables to expose to Python
# (cythonized data cannot be exposed to Python, even with cpdef.)


amax = _amax
amin = _amin
