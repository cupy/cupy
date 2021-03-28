from cpython cimport sequence

import numpy
from numpy import nan

import cupy
from cupy._core import _reduction
from cupy._core._reduction import create_reduction_func
from cupy._core._reduction import ReductionKernel
from cupy._core._kernel import ElementwiseKernel

from cupy._core cimport _accelerator
from cupy._core cimport _routines_math as _math
from cupy._core.core cimport ndarray

from cupy.cuda import cub

try:
    import cupy_backends.cuda.libs.cutensor as cuda_cutensor
    from cupy import cutensor
except ImportError:
    cuda_cutensor = None
    cutensor = None


cdef ndarray _ndarray_max(ndarray self, axis, out, dtype, keepdims):
    for accelerator in _accelerator._routine_accelerators:
        result = None
        if accelerator == _accelerator.ACCELERATOR_CUB:
            # result will be None if the reduction is not compatible with CUB
            result = cub.cub_reduction(
                self, cub.CUPY_CUB_MAX, axis, dtype, out, keepdims)
        if accelerator == _accelerator.ACCELERATOR_CUTENSOR:
            if self.dtype.kind == 'c' or dtype in ('F', 'D'):
                # Complex dtype is not supported
                continue
            result = cutensor._try_reduction_routine(
                self, axis, dtype, out, keepdims, cuda_cutensor.OP_MAX, 1, 0)
        if result is not None:
            return result
    return _amax(self, axis=axis, out=out, dtype=dtype, keepdims=keepdims)


cdef ndarray _ndarray_min(ndarray self, axis, out, dtype, keepdims):
    for accelerator in _accelerator._routine_accelerators:
        result = None
        if accelerator == _accelerator.ACCELERATOR_CUB:
            # result will be None if the reduction is not compatible with CUB
            result = cub.cub_reduction(
                self, cub.CUPY_CUB_MIN, axis, out, dtype, keepdims)
        if accelerator == _accelerator.ACCELERATOR_CUTENSOR:
            if self.dtype.kind == 'c' or dtype in ('F', 'D'):
                # Complex dtype is not supported
                continue
            result = cutensor._try_reduction_routine(
                self, axis, dtype, out, keepdims, cuda_cutensor.OP_MIN, 1, 0)
        if result is not None:
            return result
    return _amin(self, axis=axis, out=out, dtype=dtype, keepdims=keepdims)


cdef ndarray _ndarray_ptp(ndarray self, axis, out, keepdims):
    for accelerator in _accelerator._routine_accelerators:
        if accelerator == _accelerator.ACCELERATOR_CUB:
            # result will be None if the reduction is not compatible with CUB
            result = cub.cub_reduction(
                self, cub.CUPY_CUB_MAX, axis, out, None, keepdims)
            if result is not None:
                result -= cub.cub_reduction(
                    self, cub.CUPY_CUB_MIN, axis, None, None, keepdims)
                return result
        if accelerator == _accelerator.ACCELERATOR_CUTENSOR:
            if self.dtype.kind == 'c':
                # Complex dtype is not supported
                continue
            maxv = cutensor._try_reduction_routine(
                self, axis, None, out, keepdims, cuda_cutensor.OP_MAX, 1, 0)
            if maxv is None:
                continue
            return cutensor._try_reduction_routine(
                self, axis, None, maxv, keepdims, cuda_cutensor.OP_MIN, -1, 1)

    result = _amax(self, axis=axis, out=out, keepdims=keepdims)
    result -= _amin(self, axis=axis, out=None, keepdims=keepdims)
    return result


# TODO(leofang): this signature is incompatible with NumPy!
cdef ndarray _ndarray_argmax(ndarray self, axis, out, dtype, keepdims):
    for accelerator in _accelerator._routine_accelerators:
        if accelerator == _accelerator.ACCELERATOR_CUB:
            # result will be None if the reduction is not compatible with CUB
            if self._f_contiguous and self.dtype == numpy.bool_:
                # temporary workaround casting the inputs to int8
                # CUB argmax seems to return different values to
                # NumPy for F-order bool array inputs
                self = self.astype(numpy.int8)
            result = cub.cub_reduction(
                self, cub.CUPY_CUB_ARGMAX, axis, dtype, out, keepdims)
            if result is not None:
                return result
    return _argmax(self, axis=axis, out=out, dtype=dtype, keepdims=keepdims)


# TODO(leofang): this signature is incompatible with NumPy!
cdef ndarray _ndarray_argmin(ndarray self, axis, out, dtype, keepdims):
    for accelerator in _accelerator._routine_accelerators:
        if accelerator == _accelerator.ACCELERATOR_CUB:
            # result will be None if the reduction is not compatible with CUB
            result = cub.cub_reduction(
                self, cub.CUPY_CUB_ARGMIN, axis, dtype, out, keepdims)
            if result is not None:
                return result
    return _argmin(self, axis=axis, out=out, dtype=dtype, keepdims=keepdims)


cdef ndarray _ndarray_mean(ndarray self, axis, dtype, out, keepdims):
    cdef Py_ssize_t n

    dtype_sum = dtype_out = dtype
    if dtype is None:
        if self.dtype.kind in 'iub':
            dtype_out = numpy.float64
            dtype_sum = numpy.float64
        elif self.dtype.char == 'e':
            dtype_sum = numpy.float32
            dtype_out = numpy.float16
    elif numpy.dtype(dtype).kind in 'iub':
        # output will be the requested type, but compute the mean using float
        dtype_out = dtype
        dtype_sum = numpy.float64

    for accelerator in _accelerator._routine_accelerators:
        if accelerator == _accelerator.ACCELERATOR_CUB and self.size != 0:
            result = cub.cub_reduction(
                self, cub.CUPY_CUB_SUM, axis, dtype_sum, out, keepdims)
            if result is not None:
                n = self.size // result.size
                cupy.true_divide(result, n, out=result, casting='unsafe')
                break
        if accelerator == _accelerator.ACCELERATOR_CUTENSOR:
            reduce_axis, _ = _reduction._get_axis(axis, self._shape.size())
            n = 1
            for i in reduce_axis:
                n *= self._shape[i]
            n = max(n, 1)
            result = cutensor._try_reduction_routine(
                self, axis, dtype_sum, out, keepdims,
                cuda_cutensor.OP_ADD, 1.0 / n, 0)
            if result is not None:
                break
    else:
        result = _mean(
            self, axis=axis, dtype=dtype_sum, out=out, keepdims=keepdims)

    if dtype_out is not None and out is None:
        result = result.astype(dtype_out)
    return result


cdef ndarray _ndarray_var(ndarray self, axis, dtype, out, ddof, keepdims):
    return _var(
        self, axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims)


cdef ndarray _ndarray_std(ndarray self, axis, dtype, out, ddof, keepdims):
    return _std(
        self, axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims)


cdef _min_max_preamble = '''
template <typename T>
struct min_max_st{
    T value;
    int index;
    __device__ min_max_st() : index(-1) { }
    __device__ min_max_st(T v) : value(v), index(0) { }
    __device__ min_max_st(T v, int i) : value(v), index(i) { }
};

template <typename T>
__device__ min_max_st<T> my_min(
        const min_max_st<T>& a, const min_max_st<T>& b) {
    if (a.index == -1) return b;
    if (b.index == -1) return a;
    return min_max_st<T>(min(a.value, b.value));
}
template <typename T>
__device__ min_max_st<T> my_min_float(
        const min_max_st<T>& a, const min_max_st<T>& b) {
    if (a.index == -1) return b;
    if (b.index == -1) return a;
    if (isnan(a.value)) return a;
    if (isnan(b.value)) return b;
    return min_max_st<T>(min(a.value, b.value));
}

template <typename T>
__device__ min_max_st<T> my_max(
        const min_max_st<T>& a, const min_max_st<T>& b) {
    if (a.index == -1) return b;
    if (b.index == -1) return a;
    return min_max_st<T>(max(a.value, b.value));
}
template <typename T>
__device__ min_max_st<T> my_max_float(
        const min_max_st<T>& a, const min_max_st<T>& b) {
    if (a.index == -1) return b;
    if (b.index == -1) return a;
    if (isnan(a.value)) return a;
    if (isnan(b.value)) return b;
    return min_max_st<T>(max(a.value, b.value));
}

template <typename T>
__device__ min_max_st<T> my_argmin(
        const min_max_st<T>& a, const min_max_st<T>& b) {
    if (a.index == -1) return b;
    if (b.index == -1) return a;
    if (a.value == b.value)
        return min_max_st<T>(a.value, min(a.index, b.index));
    return (a.value <= b.value) ? a : b;
}
template <typename T>
__device__ min_max_st<T> my_argmin_float(
        const min_max_st<T>& a, const min_max_st<T>& b) {
    if (a.index == -1) return b;
    if (b.index == -1) return a;
    if (a.value == b.value)
        return min_max_st<T>(a.value, min(a.index, b.index));
    if (isnan(a.value)) return a;
    if (isnan(b.value)) return b;
    return (a.value <= b.value) ? a : b;
}

template <typename T>
__device__ min_max_st<T> my_argmax(
        const min_max_st<T>& a, const min_max_st<T>& b) {
    if (a.index == -1) return b;
    if (b.index == -1) return a;
    if (a.value == b.value)
        return min_max_st<T>(a.value, min(a.index, b.index));
    return (a.value >= b.value) ? a : b;
}
template <typename T>
__device__ min_max_st<T> my_argmax_float(
        const min_max_st<T>& a, const min_max_st<T>& b) {
    if (a.index == -1) return b;
    if (b.index == -1) return a;
    if (a.value == b.value)
        return min_max_st<T>(a.value, min(a.index, b.index));
    if (isnan(a.value)) return a;
    if (isnan(b.value)) return b;
    return (a.value >= b.value) ? a : b;
}

'''


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
     'min_max_st<type_in0_raw>'),
    None, _min_max_preamble)


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
     'min_max_st<type_in0_raw>'),
    None, _min_max_preamble)


nanmin = create_reduction_func(
    'cupy_nanmin',
    ('?->?', 'b->b', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I', 'l->l', 'L->L',
     'q->q', 'Q->Q', 'e->e', 'f->f', 'd->d', 'F->F', 'D->D'),
    ('min_max_st<type_in0_raw>(in0)', 'my_min(a, b)', 'out0 = a.value',
     'min_max_st<type_in0_raw>'),
    None, _min_max_preamble)


nanmax = create_reduction_func(
    'cupy_nanmax',
    ('?->?', 'b->b', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I', 'l->l', 'L->L',
     'q->q', 'Q->Q', 'e->e', 'f->f', 'd->d', 'F->F', 'D->D'),
    ('min_max_st<type_in0_raw>(in0)', 'my_max(a, b)', 'out0 = a.value',
     'min_max_st<type_in0_raw>'),
    None, _min_max_preamble)


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
    None, _min_max_preamble, sort_reduce_axis=False)


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
    None, _min_max_preamble, sort_reduce_axis=False)


cpdef ndarray _nanargmax(ndarray a, axis, out, dtype, keepdims):
    return _nanargmax_func(
        a, axis=axis, out=out, dtype=dtype, keepdims=keepdims)


cpdef ndarray _nanargmin(ndarray a, axis, out, dtype, keepdims):
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
    None, _min_max_preamble, sort_reduce_axis=False)


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
    None, _min_max_preamble, sort_reduce_axis=False)


cpdef ndarray _median(
        ndarray a, axis, out, overwrite_input, keepdims):

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
            raise numpy.AxisError('Axis overrun')
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

    out = _mean(
        part[indexer], axis=axis, dtype=None, out=out, keepdims=keepdims)
    if out_shape is not None:
        out = out.reshape(out_shape)
    return out


cpdef ndarray _nanmedian(
        ndarray a, axis, out, overwrite_input, keepdims):

    if axis is None:
        axis = tuple(range(a.ndim))
    if not isinstance(axis, tuple):
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
    n_reduce_each = cupy.empty(out_shape, dtype='int32')
    n_reduce_each[...] = n_reduce
    if a_data_ptr == a.data.ptr and overwrite_input is False:
        a = a.copy()
    _replace_nan_kernel(n_reduce, numpy.finfo(a.dtype).max, a, n_reduce_each)
    a = cupy.sort(a, axis=-1)

    b = cupy.empty(out_shape, dtype=a.dtype)
    b[...] = cupy.nan
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
        out[...] = b
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


cdef ndarray _mean(
        ndarray a, axis=None, dtype=None, out=None, keepdims=False):
    if a.size == 0:
        # Return nan; see also https://github.com/numpy/numpy/issues/13582
        return _mean_core_empty(a, axis, dtype, out, keepdims)
    return _mean_core(a, axis, dtype, out, keepdims)

cdef ndarray _var(
        ndarray a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):

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

    if out is None:
        if dtype_out == 'float16':
            var_core = _var_core_float16
        elif dtype_out == 'float32':
            var_core = _var_core_float32
        else:
            var_core = _var_core_float64
        return var_core(a, arrmean, alpha, axis=axis, keepdims=keepdims)

    out = _var_core_out(a, arrmean, alpha, out, axis=axis, keepdims=keepdims)
    return out.astype(dtype_out, copy=False)


cdef ndarray _std(
        ndarray a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    ret = _var(
        a, axis=axis, dtype=dtype, out=None, ddof=ddof, keepdims=keepdims)
    return _math._sqrt(ret, dtype=dtype, out=out)


cdef _norm_preamble = '''
template <typename T> __device__ T my_norm(T x) { return x * x; }
__device__ float my_norm(const complex<float>& x) { return norm(x); }
__device__ double my_norm(const complex<double>& x) { return norm(x); }
'''


cdef _var_core_float16 = ReductionKernel(
    'S x, T mean, float32 alpha', 'float16 out',
    'my_norm(x - mean)',
    'a + b', 'out = alpha * a', '0', '_var_core', preamble=_norm_preamble)


cdef _var_core_float32 = ReductionKernel(
    'S x, T mean, float32 alpha', 'float32 out',
    'my_norm(x - mean)',
    'a + b', 'out = alpha * a', '0', '_var_core', preamble=_norm_preamble)


cdef _var_core_float64 = ReductionKernel(
    'S x, T mean, float64 alpha', 'float64 out',
    'my_norm(x - mean)',
    'a + b', 'out = alpha * a', '0', '_var_core', preamble=_norm_preamble)


cdef _var_core_out = ReductionKernel(
    'S x, T mean, U alpha', 'U out',
    'my_norm(x - mean)',
    'a + b', 'out = alpha * a', '0', '_var_core', preamble=_norm_preamble)


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
    'cupy_mean',
    ('?->d', 'B->d', 'h->d', 'H->d', 'i->d', 'I->d', 'l->d', 'L->d',
     'q->d', 'Q->d',
     ('e->e', (None, None, None, 'float')),
     'f->f', 'd->d', 'F->F', 'D->D'),
    ('in0', 'a + b',
     'out0 = a / _type_reduce(_in_ind.size() / _out_ind.size())', None), 0)

cdef _nanmean_preamble = '''
template <typename T>
struct nanmean_st{
    typedef long long ll;
    T value;
    ll count;
    __device__ nanmean_st() : value(0), count(0) { }
    __device__ nanmean_st(T v) :
        value(isnan(v) ? T(0) : v), count(isnan(v) ? 0 : 1) { }
    __device__ nanmean_st(T v, ll c) : value(v), count(c) { }
};

template <typename T>
__device__ nanmean_st<T> my_nanmean(
        const nanmean_st<T>& a, const nanmean_st<T>& b) {
    return nanmean_st<T>(a.value + b.value, a.count + b.count);
}
'''


cdef _nanmean_func = create_reduction_func(
    'cupy_nanmean',
    ('e->e', 'f->f', 'd->d', 'F->F', 'D->D'),
    ('in0', 'my_nanmean(a, b)',
     'out0 = a.value / type_out0_raw(a.count)', 'nanmean_st<type_out0_raw>'),
    None, _nanmean_preamble)


_count_non_nan = create_reduction_func(
    'cupy_count_non_nan',
    ('e->q', 'f->q', 'd->q'),
    ('isnan(in0) ? 0 : 1', 'a + b', 'out0 = a', None), 0)


cpdef ndarray _nanmean(ndarray a, axis, dtype, out, keepdims):
    return _nanmean_func(a, axis=axis, dtype=dtype, out=out, keepdims=keepdims)


cpdef ndarray _nanstd(ndarray a, axis, dtype, out, ddof, keepdims):
    var = _nanvar(a, axis, dtype, None, ddof, keepdims)
    return _math._sqrt(var, dtype=dtype, out=out)


cpdef ndarray _nanvar(ndarray a, axis, dtype, out, ddof, keepdims):
    assert a.dtype.kind != 'c', 'Variance for complex numbers is not ' \
                                'implemented. Current implemention does not ' \
                                'convert the dtype'

    _count = _count_non_nan(a, axis=axis, keepdims=True)
    arrsum = _math._nansum(a, axis=axis, dtype=dtype, out=None, keepdims=True)

    if out is None:
        return _nanvar_core(
            a, arrsum, _count, ddof, axis=axis, keepdims=keepdims)
    else:
        return _nanvar_core_out(
            a, arrsum, _count, ddof, out, axis=axis, keepdims=keepdims)


cdef _nanvar_preamble = '''
template <typename S, typename T>
__device__ T nanvar_impl(S x, T mean, long long alpha) {
    return (isnan(x) ? T(0) : T((x - mean) * (x - mean))) / alpha;
}
'''


cdef _nanvar_core = ReductionKernel(
    'S x, T sum, int64 _count, int64 ddof', 'S out',
    'nanvar_impl<S, T>(x, sum / _count, max(_count - ddof, 0LL))',
    'a + b', 'out = a', '0', '_nanvar_core', preamble=_nanvar_preamble)


cdef _nanvar_core_out = ReductionKernel(
    'S x, T sum, int64 _count, int64 ddof', 'U out',
    'nanvar_impl<S, T>(x, sum / _count, max(_count - ddof, 0LL))',
    'a + b', 'out = a', '0', '_nanvar_core', preamble=_nanvar_preamble)


# Variables to expose to Python
# (cythonized data cannot be exposed to Python, even with cpdef.)


amax = _amax
amin = _amin
