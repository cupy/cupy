import numpy
from numpy import nan

from cupy.core._reduction import create_reduction_func
from cupy.core._reduction import ReductionKernel

from cupy.core cimport _routines_math as _math
from cupy.core.core cimport ndarray

import cupy
if cupy.cuda.cub_enabled:
    from cupy.cuda import cub


cdef ndarray _ndarray_max(ndarray self, axis, out, dtype, keepdims):
    if cupy.cuda.cub_enabled:
        # result will be None if the reduction is not compatible with CUB
        result = cub.cub_reduction(self, cub.CUPY_CUB_MAX, axis, dtype, out,
                                   keepdims)
        if result is not None:
            return result
    return _amax(self, axis=axis, out=out, dtype=dtype, keepdims=keepdims)


cdef ndarray _ndarray_min(ndarray self, axis, out, dtype, keepdims):
    if cupy.cuda.cub_enabled:
        # result will be None if the reduction is not compatible with CUB
        result = cub.cub_reduction(self, cub.CUPY_CUB_MIN, axis, out, dtype,
                                   keepdims)
        if result is not None:
            return result
    return _amin(self, axis=axis, out=out, dtype=dtype, keepdims=keepdims)


cdef ndarray _ndarray_ptp(ndarray self, axis, out, keepdims):
    if cupy.cuda.cub_enabled:
        # result will be None if the reduction is not compatible with CUB
        result = cub.cub_reduction(self, cub.CUPY_CUB_MAX, axis, out, None,
                                   keepdims)
        if result is not None:
            result -= cub.cub_reduction(self, cub.CUPY_CUB_MIN, axis, None,
                                        None, keepdims)
            return result

    result = _amax(self, axis=axis, out=out, keepdims=keepdims)
    result -= _amin(self, axis=axis, out=None, keepdims=keepdims)
    return result


# TODO(leofang): this signature is incompatible with NumPy!
cdef ndarray _ndarray_argmax(ndarray self, axis, out, dtype, keepdims):
    if cupy.cuda.cub_enabled:
        # result will be None if the reduction is not compatible with CUB
        result = cub.cub_reduction(self, cub.CUPY_CUB_ARGMAX, axis, dtype, out,
                                   keepdims)
        if result is not None:
            return result
    return _argmax(self, axis=axis, out=out, dtype=dtype, keepdims=keepdims)


# TODO(leofang): this signature is incompatible with NumPy!
cdef ndarray _ndarray_argmin(ndarray self, axis, out, dtype, keepdims):
    if cupy.cuda.cub_enabled:
        # result will be None if the reduction is not compatible with CUB
        result = cub.cub_reduction(self, cub.CUPY_CUB_ARGMIN, axis, dtype, out,
                                   keepdims)
        if result is not None:
            return result
    return _argmin(self, axis=axis, out=out, dtype=dtype, keepdims=keepdims)


cdef ndarray _ndarray_mean(ndarray self, axis, dtype, out, keepdims):
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

    result = None
    if (cupy.cuda.cub_enabled and self.size != 0):
        result = cub.cub_reduction(self, cub.CUPY_CUB_SUM, axis, dtype_sum,
                                   out, keepdims)
    if result is not None:
        n = self.size // result.size
        cupy.true_divide(result, n, out=result, casting='unsafe')
    else:
        result = _mean(self, axis=axis, dtype=dtype_sum, out=out,
                       keepdims=keepdims)
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
     'q->q', 'Q->Q', 'e->e', 'f->f', 'd->d'),
    ('min_max_st<type_in0_raw>(in0)', 'my_min(a, b)', 'out0 = a.value',
     'min_max_st<type_in0_raw>'),
    None, _min_max_preamble)


nanmax = create_reduction_func(
    'cupy_nanmax',
    ('?->?', 'b->b', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I', 'l->l', 'L->L',
     'q->q', 'Q->Q', 'e->e', 'f->f', 'd->d'),
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
    None, _min_max_preamble)


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
    None, _min_max_preamble)


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
    None, _min_max_preamble)


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
    None, _min_max_preamble)


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
    ('e->l', 'f->l', 'd->l'),
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
