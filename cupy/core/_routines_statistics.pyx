import string

import numpy
from numpy import nan

from cupy.core._reduction import create_reduction_func
from cupy.core._reduction import ReductionKernel
from cupy.core._scalar import get_typename as _get_typename
from cupy import util

try:
    from cupy.cuda import thrust
except ImportError:
    pass

from cupy.core cimport _routines_math as _math
from cupy.core cimport _routines_manipulation as _manipulation
from cupy.core cimport _routines_sorting as _sorting
from cupy.core.core cimport compile_with_cache
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


@util.memoize(for_each_device=True)
def _median_partition_kernel(dtype):
    name = 'median_partition_kernel'
    dtype = _get_typename(dtype)
    source = string.Template('''
    template<typename T>
    __device__ void bitonic_sort_step(CArray<T, 1> a,
            ptrdiff_t x, ptrdiff_t y, int i, ptrdiff_t s, ptrdiff_t w) {
        for (ptrdiff_t j = i; j < (y - x) / 2; j += 32) {
            ptrdiff_t n = j + (j & -w);
            T v = a[n + x], u = a[n + w + x];
            if (n & s ? v < u : v > u) {
                a[n + x] = u;
                a[n + w + x] = v;
            }
        }
    }

    // Sort a[x:y].
    template<typename T>
    __device__ void bitonic_sort(
            CArray<T, 1> a, ptrdiff_t x, ptrdiff_t y, int i) {
        for (ptrdiff_t s = 2; s <= y - x; s *= 2) {
            for (ptrdiff_t w = s / 2; w >= 1; w /= 2) {
                bitonic_sort_step<T>(a, x, y, i, s, w);
            }
        }
    }

    // Merge first k elements and the next 32 times t elements.
    template<typename T>
    __device__ void merge(
            CArray<T, 1> a, int k, int i, ptrdiff_t x, ptrdiff_t z, int u) {
        for (int s = i; s < u; s += 32) {
            if (a[x + k - s - 1] > a[z + s]) {
                T tmp = a[x + k - s - 1];
                a[x + k - s - 1] = a[z + s];
                a[z + s] = tmp;
            }
        }

        // After merge step, the first k elements are already bitonic.
        // Therefore, we do not need to fully sort.
        for (int w = k / 2; w >= 1; w /= 2) {
            bitonic_sort_step<T>(a, x, k + x, i, k, w);
        }
    }

    extern "C" {
    // In this function, 32 threads handle one subarray. This number equals to
    // the warp size. The first k elements are always sorted and the next 32
    // times t elements stored values that have possibilities to be selected.
    __global__ void ${name}(
            CArray<${dtype}, 1> a, CArray<${dtype}, 1> out,
            int k, ptrdiff_t n, int t, ptrdiff_t sz, ptrdiff_t len) {

        // This thread handles a[z:m].
        ptrdiff_t i = static_cast<ptrdiff_t>(blockIdx.x) * blockDim.x
            + threadIdx.x;
        ptrdiff_t z = i / 32 * len;
        ptrdiff_t m = (i / 32 + 1) * len;
        int id = i % 32;
        int x = 0;

        bitonic_sort<${dtype}>(a, z, k + z, id);
        ptrdiff_t j;
        for (j = k + id + z; j < m - (m - z) % 32; j += 32) {
            if (a[j] < a[k - 1 + z]) {
                ${dtype} tmp = a[k + 32 * x + id + z];
                a[k + 32 * x + id + z] = a[j];
                a[j] = tmp;
                ++x;
            }

            // If at least one thread in the warp has found t values that
            // can be selected, we update the first k elements.
    #if __CUDACC_VER_MAJOR__ >= 9
            if (__any_sync(0xffffffff, x >= t)) {
    #else
            if (__any(x >= t)) {
    #endif
                bitonic_sort<${dtype}>(a, k + z, 32 * t + k + z, id);
                merge<${dtype}>(a, k, id, z, k + z, min(k, 32 * t));
                x = 0;
            }
        }
        if (j < m && a[j] < a[k - 1 + z]) {
            ${dtype} tmp = a[k + 32 * x + id + z];
            a[k + 32 * x + id + z] = a[j];
            a[j] = tmp;
        }

        // Finally, we merge the first k elements and the remainders to be
        // stored.
        bitonic_sort<${dtype}>(a, k + z, 32 * t + k + z, id);
        merge<${dtype}>(a, k, id, z, k + z, min(k, 32 * t));

        int kth = len / 2;
        if (len % 2)
            out[i / 32] = a[z + kth];
        else
            out[i / 32] = (a[z + kth - 1] + a[z + kth]) / 2;
    }
    }
    ''').substitute(name=name, dtype=dtype)
    module = compile_with_cache(source)
    return module.get_function(name)


cdef _median_partition_core(ndarray data, max_k, ndarray out, int axis):

    if data.dtype.kind == 'c':
        raise NotImplementedError('Sorting arrays with dtype \'{}\' is '
                                  'not supported'.format(data.dtype))

    cdef int ndim = data._shape.size()
    cdef Py_ssize_t length, s, sz, t

    if not data._c_contiguous:
        raise NotImplementedError('Sorting non-contiguous array is not '
                                  'supported.')

    # if out.size != self.size // self.shape[axis]:
    #     raise ValueError('Out size is mismatched')

    if axis < 0:
        axis += ndim
    if not (0 <= axis < ndim):
        raise numpy.AxisError('Axis out of range')

    if axis != ndim - 1:
        data = _manipulation.rollaxis(data, axis, ndim).copy()

    length = data._shape[ndim-1]

    # For simplicity, max_k is round up to the power of 2. If max_k is
    # already the power of 2, it is round up to the next power of 2 because
    # we need to collect the first max(kth)+1 elements.
    max_k = max(32, 1 << max_k.bit_length())

    # The parameter t is the length of the list that stores elements to be
    # selected for each thread. We divide the array into sz subarrays.
    # These parameters are determined from the measurement on TITAN X.
    t = 4
    sz = 1
    while sz > 0 and length // sz < max_k + 32 * t:
        sz //= 2
    sz *= data.size // length

    # If the array size is small or k is large, we simply sort the array.
    if length < 32 or sz < 1 or max_k >= 1024:
        # kth is ignored.
        data.sort(axis=-1)

        indexer = [slice(None)] * ndim

        index = length // 2
        if length % 2 == 1:
            indexer[ndim-1] = slice(index, index+1)
        else:
            indexer[ndim-1] = slice(index-1, index+1)
        indexer = tuple(indexer)

        out = _mean(
            data[indexer], axis=-1, dtype=None, out=None, keepdims=False)

    else:
        shape = data.shape
        data = data.ravel()

        # For each subarray, we collect first k elements to the head.
        kern = _median_partition_kernel(data.dtype)
        block_size = 32
        grid_size = sz
        kern(grid=(grid_size,), block=(block_size,), args=(
            data, out, max_k, data.size, t, sz, length))

    return out


cpdef ndarray _median(
        ndarray a, axis, out, overwrite_input, keepdims):

    if not isinstance(a, ndarray):
        raise TypeError('Array is not cupy.ndarray')

    keep_shape = a.shape

    if a.ndim == 0:
        return a

    if overwrite_input:
        data = a
    else:
        data = a.copy()

    if axis is None:
        sz = data.size
        data = data.ravel()
        axis = -1
        if keepdims:
            out_shape = tuple([1, ] * len(keep_shape))
        else:
            out_shape = ()
    else:
        sz = data.shape[axis]
        out_shape = data.shape
        if keepdims:
            out_shape = out_shape[:axis] + (1,) + out_shape[axis+1:]
        else:
            out_shape = out_shape[:axis] + () + out_shape[axis+1:]

    if out is None:
        out = cupy.empty(out_shape, dtype=data.dtype)
    elif out.shape != out_shape:
        raise ValueError('Out shape is mismatched')

    if sz % 2 == 0:
        szh = sz // 2
        max_k = szh
    else:
        max_k = (sz - 1) // 2

    out = out.ravel()
    out = _median_partition_core(data, max_k, out, axis=axis)

    out = out.reshape(out_shape)

    return out

    # if not isinstance(a, ndarray):
    #     raise TypeError('Array is not cupy.ndarray')

    # keep_ndim = a.ndim

    # if axis is None:
    #     sz = a.size
    # else:
    #     sz = a.shape[axis]
    # if sz % 2 == 0:
    #     szh = sz // 2
    #     kth = [szh - 1, szh]
    # else:
    #     kth = [(sz - 1) // 2]

    # if overwrite_input:
    #     part = a
    # else:
    #     part = a.copy()

    # if axis is None:
    #     part = part.ravel()
    #     part.partition(kth)
    # else:
    #     part.partition(kth, axis=axis)

    # if part.shape == ():
    #     return part
    # if axis is None:
    #     axis = 0

    # indexer = [slice(None)] * part.ndim

    # if keepdims:
    #     _indexer = [None] * (keep_ndim - part.ndim)
    #     indexer.extend(_indexer)

    # index = part.shape[axis] // 2
    # if part.shape[axis] % 2 == 1:
    #     indexer[axis] = slice(index, index+1)
    # else:
    #     indexer[axis] = slice(index-1, index+1)
    # indexer = tuple(indexer)

    # return _mean(
    #     part[indexer], axis=axis, dtype=None, out=out, keepdims=keepdims)


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
