import string

import numpy

import cupy
from cupy._core._scalar import get_typename as _get_typename
from cupy._core._ufuncs import elementwise_copy
from cupy import _util
from cupy.cuda import thrust

from cupy._core cimport _routines_manipulation as _manipulation
from cupy._core.core cimport compile_with_cache
from cupy._core.core cimport ndarray
from cupy._core cimport internal


cdef _ndarray_sort(ndarray self, int axis):
    cdef int ndim = self._shape.size()
    cdef ndarray data

    if not cupy.cuda.thrust.available:
        raise RuntimeError('Thrust is needed to use cupy.sort. Please '
                           'install CUDA Toolkit with Thrust then '
                           'reinstall CuPy after uninstalling it.')

    if ndim == 0:
        raise numpy.AxisError('Sorting arrays with the rank of zero is not '
                              'supported')  # as numpy.sort() raises

    # TODO(takagi): Support sorting views
    if not self._c_contiguous:
        raise NotImplementedError('Sorting non-contiguous array is not '
                                  'supported.')

    axis = internal._normalize_axis_index(axis, ndim)

    if axis == ndim - 1:
        data = self
    else:
        data = _manipulation.rollaxis(self, axis, ndim).copy()

    if ndim == 1:
        thrust.sort(self.dtype, data.data.ptr, 0, self.shape)
    else:
        keys_array = ndarray(data.shape, dtype=numpy.intp)
        thrust.sort(
            self.dtype, data.data.ptr, keys_array.data.ptr, data.shape)

    if axis == ndim - 1:
        pass
    else:
        data = _manipulation.rollaxis(data, -1, axis)
        elementwise_copy(data, self)


cdef ndarray _ndarray_argsort(ndarray self, axis):
    cdef int _axis, ndim
    cdef ndarray data

    if not cupy.cuda.thrust.available:
        raise RuntimeError('Thrust is needed to use cupy.argsort. Please '
                           'install CUDA Toolkit with Thrust then '
                           'reinstall CuPy after uninstalling it.')

    self = cupy.atleast_1d(self)
    ndim = self._shape.size()

    if axis is None:
        data = self.ravel()
        _axis = -1
    else:
        data = self
        _axis = axis

    _axis = internal._normalize_axis_index(_axis, ndim)

    if _axis == ndim - 1:
        data = data.copy()
    else:
        data = _manipulation.rollaxis(data, _axis, ndim).copy()
    shape = data.shape

    idx_array = ndarray(shape, dtype=numpy.intp)

    if ndim == 1:
        thrust.argsort(self.dtype, idx_array.data.ptr, data.data.ptr, 0,
                       shape)
    else:
        keys_array = ndarray(shape, dtype=numpy.intp)
        thrust.argsort(self.dtype, idx_array.data.ptr, data.data.ptr,
                       keys_array.data.ptr, shape)

    if _axis == ndim - 1:
        return idx_array
    else:
        return _manipulation.rollaxis(idx_array, -1, _axis)


cdef _ndarray_partition(ndarray self, kth, int axis):
    """Partitions an array.

    Args:
        kth (int or sequence of ints): Element index to partition by. If
            supplied with a sequence of k-th it will partition all elements
            indexed by k-th of them into their sorted position at once.

        axis (int): Axis along which to sort. Default is -1, which means
            sort along the last axis.

    .. seealso::
        :func:`cupy.partition` for full documentation,
        :meth:`numpy.ndarray.partition`

    """

    cdef int ndim = self._shape.size()
    cdef Py_ssize_t k, max_k, length, s, sz, t
    cdef ndarray data

    if ndim == 0:
        raise numpy.AxisError('Sorting arrays with the rank of zero is not '
                              'supported')

    if not self._c_contiguous:
        raise NotImplementedError('Sorting non-contiguous array is not '
                                  'supported.')

    axis = internal._normalize_axis_index(axis, ndim)

    if axis == ndim - 1:
        data = self
    else:
        data = _manipulation.rollaxis(self, axis, ndim).copy()

    length = self._shape[axis]
    if isinstance(kth, int):
        kth = kth,
    max_k = 0
    for k in kth:
        if k < 0:
            k += length
        if not (0 <= k < length):
            raise ValueError('kth(={}) out of bounds {}'.format(k, length))
        if max_k < k:
            max_k = k

    # For simplicity, max_k is round up to the power of 2. If max_k is
    # already the power of 2, it is round up to the next power of 2 because
    # we need to collect the first max(kth)+1 elements.
    max_k = max(32, 1 << max_k.bit_length())

    # The parameter t is the length of the list that stores elements to be
    # selected for each thread. We divide the array into sz subarrays.
    # These parameters are determined from the measurement on TITAN X.
    t = 4
    sz = 512
    while sz > 0 and length // sz < max_k + 32 * t:
        sz //= 2
    sz *= self.size // length

    # If the array size is small or k is large, we simply sort the array.
    if length < 32 or sz <= 32 or max_k >= 1024:
        # kth is ignored.
        data.sort(axis=-1)
    else:
        shape = data.shape
        data = data.ravel()

        # For each subarray, we collect first k elements to the head.
        kern, merge_kern = _partition_kernel(self.dtype)
        block_size = 32
        grid_size = sz
        kern(grid=(grid_size,), block=(block_size,), args=(
            data, max_k, self.size, t, sz))

        # Merge heads of subarrays.
        s = 1
        while s < sz // (self.size // length):
            block_size = 32
            grid_size = sz // s // 2
            merge_kern(grid=(grid_size,), block=(block_size,), args=(
                data, max_k, self.size, sz, s))
            s *= 2

        data = data.reshape(shape)

    if axis != ndim - 1:
        data = _manipulation.rollaxis(data, -1, axis)
        elementwise_copy(data, self)


cdef ndarray _ndarray_argpartition(self, kth, axis):
    """Returns the indices that would partially sort an array.

    Args:
        kth (int or sequence of ints): Element index to partition by. If
            supplied with a sequence of k-th it will partition all elements
            indexed by k-th of them into their sorted position at once.
        axis (int or None): Axis along which to sort. Default is -1, which
            means sort along the last axis. If None is supplied, the array
            is flattened before sorting.

    Returns:
        cupy.ndarray: Array of the same type and shape as ``a``.

    .. seealso::
        :func:`cupy.argpartition` for full documentation,
        :meth:`numpy.ndarray.argpartition`

    """
    cdef int _axis, ndim
    cdef Py_ssize_t k, length
    cdef ndarray data
    if axis is None:
        data = self.ravel()
        _axis = -1
    else:
        data = self
        _axis = axis

    ndim = data._shape.size()
    _axis = internal._normalize_axis_index(_axis, ndim)

    length = data._shape[_axis]
    if isinstance(kth, int):
        kth = kth,
    for k in kth:
        if k < 0:
            k += length
        if not (0 <= k < length):
            raise ValueError('kth(={}) out of bounds {}'.format(k, length))

    # TODO(takgi) For its implementation reason, cupy.ndarray.argsort
    # currently performs full argsort with Thrust's efficient radix sort
    # algorithm.

    # kth is ignored.
    return data.argsort(_axis)


@_util.memoize(for_each_device=True)
def _partition_kernel(dtype):
    name = 'partition_kernel'
    merge_kernel = 'partition_merge_kernel'
    dtype = _get_typename(dtype)
    source = string.Template('''
    template<typename T>
    __device__ void bitonic_sort_step(CArray<T, 1, true> a,
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
            CArray<T, 1, true> a, ptrdiff_t x, ptrdiff_t y, int i) {
        for (ptrdiff_t s = 2; s <= y - x; s *= 2) {
            for (ptrdiff_t w = s / 2; w >= 1; w /= 2) {
                bitonic_sort_step< T >(a, x, y, i, s, w);
            }
        }
    }

    // Merge first k elements and the next 32 times t elements.
    template<typename T>
    __device__ void merge(
            CArray<T, 1, true> a,
            int k, int i, ptrdiff_t x, ptrdiff_t z, int u) {
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
            bitonic_sort_step< T >(a, x, k + x, i, k, w);
        }
    }

    extern "C" {
    // In this function, 32 threads handle one subarray. This number equals to
    // the warp size. The first k elements are always sorted and the next 32
    // times t elements stored values that have possibilities to be selected.
    __global__ void ${name}(
            CArray<${dtype}, 1, true> a,
            int k, ptrdiff_t n, int t, ptrdiff_t sz) {

        // This thread handles a[z:m].
        ptrdiff_t i = static_cast<ptrdiff_t>(blockIdx.x) * blockDim.x
            + threadIdx.x;
        ptrdiff_t z = i / 32 * n / sz;
        ptrdiff_t m = (i / 32 + 1) * n / sz;
        int id = i % 32;
        int x = 0;

        bitonic_sort< ${dtype} >(a, z, k + z, id);
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
                bitonic_sort< ${dtype} >(a, k + z, 32 * t + k + z, id);
                merge< ${dtype} >(a, k, id, z, k + z, min(k, 32 * t));
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
        bitonic_sort< ${dtype} >(a, k + z, 32 * t + k + z, id);
        merge< ${dtype} >(a, k, id, z, k + z, min(k, 32 * t));
    }

    __global__ void ${merge_kernel}(
            CArray<${dtype}, 1, true> a, int k, ptrdiff_t n, int sz, int s) {
        ptrdiff_t i = static_cast<ptrdiff_t>(blockIdx.x) * blockDim.x
            + threadIdx.x;
        ptrdiff_t z = i / 32 * 2 * s * n / sz;
        ptrdiff_t m = (i / 32 * 2 + 1) * s * n / sz;
        int id = i % 32;
        merge< ${dtype} >(a, k, id, z, m, k);
    }
    }
    ''').substitute(name=name, merge_kernel=merge_kernel, dtype=dtype)
    module = compile_with_cache(source)
    return module.get_function(name), module.get_function(merge_kernel)
