import string

import numpy

import cupy
from cupy._core._reduction import create_reduction_func
from cupy._core._kernel import create_ufunc
from cupy._core._scalar import get_typename
from cupy._core._ufuncs import elementwise_copy
from cupy._core cimport internal
from cupy import _util

from cupy_backends.cuda.api cimport runtime
from cupy._core cimport _accelerator
from cupy._core._dtype cimport get_dtype
from cupy._core cimport _kernel
from cupy._core.core cimport _ndarray_init
from cupy._core.core cimport compile_with_cache
from cupy._core.core cimport ndarray
from cupy.cuda cimport memory

from cupy.cuda import cub

try:
    import cupy_backends.cuda.libs.cutensor as cuda_cutensor
    from cupy import cutensor
except ImportError:
    cuda_cutensor = None
    cutensor = None


# ndarray members


cdef ndarray _ndarray_conj(ndarray self):
    if self.dtype.kind == 'c':
        return _conjugate(self)
    else:
        return self


cdef ndarray _ndarray_real_getter(ndarray self):
    if self.dtype.kind == 'c':
        view = ndarray(
            shape=self._shape, dtype=get_dtype(self.dtype.char.lower()),
            memptr=self.data, strides=self._strides)
        view.base = self.base if self.base is not None else self
        return view
    return self


cdef ndarray _ndarray_real_setter(ndarray self, value):
    if self.dtype.kind == 'c':
        _real_setter(value, self)
    else:
        elementwise_copy(value, self)


cdef ndarray _ndarray_imag_getter(ndarray self):
    cdef memory.MemoryPointer memptr
    if self.dtype.kind == 'c':
        dtype = get_dtype(self.dtype.char.lower())
        memptr = self.data
        # Make the memory pointer point to the first imaginary element.
        # Note that even if the array doesn't have a valid memory (e.g. 0-size
        # array), the resulting array should be a view of the original array,
        # aligning with NumPy behavior.
        if memptr.ptr != 0:
            memptr = memptr + self.dtype.itemsize // 2
        view = ndarray(
            shape=self._shape, dtype=dtype, memptr=memptr,
            strides=self._strides)
        view.base = self.base if self.base is not None else self
        return view
    new_array = ndarray(self.shape, dtype=self.dtype)
    new_array.fill(0)
    return new_array


cdef ndarray _ndarray_imag_setter(ndarray self, value):
    if self.dtype.kind == 'c':
        _imag_setter(value, self)
    else:
        raise TypeError('cupy.ndarray does not have imaginary part to set')


cdef ndarray _ndarray_prod(ndarray self, axis, dtype, out, keepdims):
    for accelerator in _accelerator._routine_accelerators:
        result = None
        if accelerator == _accelerator.ACCELERATOR_CUB:
            # result will be None if the reduction is not compatible with CUB
            result = cub.cub_reduction(
                self, cub.CUPY_CUB_PROD, axis, dtype, out, keepdims)
        if accelerator == _accelerator.ACCELERATOR_CUTENSOR:
            result = cutensor._try_reduction_routine(
                self, axis, dtype, out, keepdims, cuda_cutensor.OP_MUL, 1, 0)
        if result is not None:
            return result
    if dtype is None:
        return _prod_auto_dtype(self, axis, dtype, out, keepdims)
    else:
        return _prod_keep_dtype(self, axis, dtype, out, keepdims)


cdef ndarray _ndarray_sum(ndarray self, axis, dtype, out, keepdims):
    for accelerator in _accelerator._routine_accelerators:
        result = None
        if accelerator == _accelerator.ACCELERATOR_CUB:
            # result will be None if the reduction is not compatible with CUB
            result = cub.cub_reduction(
                self, cub.CUPY_CUB_SUM, axis, dtype, out, keepdims)
        if accelerator == _accelerator.ACCELERATOR_CUTENSOR:
            result = cutensor._try_reduction_routine(
                self, axis, dtype, out, keepdims, cuda_cutensor.OP_ADD, 1, 0)
        if result is not None:
            return result

    if dtype is None:
        return _sum_auto_dtype(self, axis, dtype, out, keepdims)
    else:
        return _sum_keep_dtype(self, axis, dtype, out, keepdims)


cdef ndarray _ndarray_cumsum(ndarray self, axis, dtype, out):
    return cupy.cumsum(self, axis, dtype, out)


cdef ndarray _ndarray_cumprod(ndarray self, axis, dtype, out):
    return cupy.cumprod(self, axis, dtype, out)


cdef ndarray _ndarray_clip(ndarray self, a_min, a_max, out):
    if a_min is None and a_max is None:
        raise ValueError('array_clip: must set either max or min')
    kind = self.dtype.kind
    if a_min is None:
        if kind == 'f':
            a_min = self.dtype.type('-inf')
        elif kind in 'iu':
            a_min = numpy.iinfo(self.dtype.type).min
    if a_max is None:
        if kind == 'f':
            a_max = self.dtype.type('inf')
        elif kind in 'iu':
            a_max = numpy.iinfo(self.dtype.type).max
    return _clip(self, a_min, a_max, out=out)


# private/internal

_op_char = {scan_op.SCAN_SUM: '+', scan_op.SCAN_PROD: '*'}
_identity = {scan_op.SCAN_SUM: 0, scan_op.SCAN_PROD: 1}
_preamble = '' if not runtime._is_hip_environment else r'''
    // ignore mask
    #define __shfl_xor_sync(m, x, y, z) __shfl_xor(x, y, z)

    // It is guaranteed to be safe on AMD's hardware, see
    // https://rocmdocs.amd.com/en/latest/Programming_Guides/HIP-GUIDE.html#warp-cross-lane-functions  # NOQA
    #define __syncwarp() {}
    '''


@cupy._util.memoize(for_each_device=True)
def _cupy_bsum_shfl(op, chunk_size, warp_size=32):
    """Returns a kernel that computes the sum/prod of each thread-block.

    Args:
        op (int): Operation type. SCAN_SUM or SCAN_PROD.
        chunk_size (int): Number of array elements processed by a single
            thread-block.
        warp_size (int); Warp size.

    Returns:
        cupy.ElementwiseKernel

    Example:
        a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        _cupy_bsum(op=SCAN_SUM, chunk_size=4)(a, b, ...)
        b == [10, 26, 19]

    Note:
        This uses warp shuffle functions to exchange data in a warp.
        See the link below for details about warp shuffle functions.
        https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-shuffle-functions
    """
    block_size = chunk_size // 2  # each thread handles two elements
    in_params = 'raw T a'
    out_params = 'raw O b'
    loop_prep = string.Template("""
        __shared__ O smem[${block_size} / ${warp_size}];
        const int n_warp = ${block_size} / ${warp_size};
        const int warp_id = threadIdx.x / ${warp_size};
        const int lane_id = threadIdx.x % ${warp_size};
    """).substitute(block_size=block_size, warp_size=warp_size)
    loop_body = string.Template("""
        O x = ${identity};
        if (2*i < a.size()) x = a[2*i];
        if (2*i + 1 < a.size()) x ${op}= a[2*i + 1];
        for (int j = 1; j < ${warp_size}; j *= 2) {
            x ${op}= __shfl_xor_sync(0xffffffff, x, j, ${warp_size});
        }
        if (lane_id == 0) smem[warp_id] = x;
        __syncthreads();
        if (warp_id == 0) {
            x = ${identity};
            if (lane_id < n_warp) x = smem[lane_id];
            for (int j = 1; j < n_warp; j *= 2) {
                x ${op}= __shfl_xor_sync(0xffffffff, x, j, ${warp_size});
            }
            int block_id = i / ${block_size};
            if (lane_id == 0) b[block_id] = x;
        }
    """).substitute(block_size=block_size, warp_size=warp_size,
                    op=_op_char[op], identity=_identity[op])
    return cupy.ElementwiseKernel(in_params, out_params, loop_body,
                                  'cupy_bsum_shfl', loop_prep=loop_prep,
                                  preamble=_preamble)


@cupy._util.memoize(for_each_device=True)
def _cupy_bsum_smem(op, chunk_size, warp_size=32):
    """Returns a kernel that computes the sum/prod of each thread-block.

    Args:
        op (int): Operation type. SCAN_SUM or SCAN_PROD.
        chunk_size (int): Number of array elements processed by a single
            thread-block.
        warp_size (int); Warp size.

    Returns:
        cupy.ElementwiseKernel

    Example:
        a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        _cupy_bsum(op=SCAN_SUM, chunk_size=4)(a, b, ...)
        b == [10, 26, 19]

    Note:
        This uses shared memory to exchange data in a warp.
    """
    block_size = chunk_size // 2  # each thread handles two elements
    in_params = 'raw T a'
    out_params = 'raw O b'
    loop_prep = string.Template("""
        __shared__ O smem1[${block_size}];
        __shared__ O smem2[${warp_size}];
        const int n_warp = ${block_size} / ${warp_size};
        const int warp_id = threadIdx.x / ${warp_size};
        const int lane_id = threadIdx.x % ${warp_size};
    """).substitute(block_size=block_size, warp_size=warp_size)
    loop_body = string.Template("""
        O x = ${identity};
        if (2*i < a.size()) x = a[2*i];
        if (2*i + 1 < a.size()) x ${op}= a[2*i + 1];
        for (int j = 1; j < ${warp_size}; j *= 2) {
            smem1[threadIdx.x] = x;          __syncwarp();
            x ${op}= smem1[threadIdx.x ^ j]; __syncwarp();
        }
        if (lane_id == 0) smem2[warp_id] = x;
        __syncthreads();
        if (warp_id == 0) {
            x = ${identity};
            if (lane_id < n_warp) x = smem2[lane_id];
            for (int j = 1; j < n_warp; j *= 2) {
                smem2[lane_id] = x;          __syncwarp();
                x ${op}= smem2[lane_id ^ j]; __syncwarp();
            }
            int block_id = i / ${block_size};
            if (lane_id == 0) b[block_id] = x;
        }
    """).substitute(block_size=block_size, warp_size=warp_size,
                    op=_op_char[op], identity=_identity[op])
    return cupy.ElementwiseKernel(in_params, out_params, loop_body,
                                  'cupy_bsum_smem', loop_prep=loop_prep,
                                  preamble=_preamble)


@cupy._util.memoize(for_each_device=True)
def _cupy_scan_naive(op, chunk_size, warp_size=32):
    """Returns a kernel to compute an inclusive scan.

    It first performs an inclusive scan in each thread-block and then add the
    scan results for the sum/prod of the chunks.

    Args:
        op (int): Operation type. SCAN_SUM or SCAN_PROD.
        chunk_size (int): Number of array elements processed by a single
            thread-block.
        warp_size (int); Warp size.

    Returns:
        cupy.ElementwiseKernel

    Example:
        b = [10, 36, 55]
        a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        _cupy_scan(op=SCAN_SUM, chunk_size=4)(b, a, out, ...)
        out == [1, 3, 6, 10, 15, 21, 28, 36, 45, 55]

    Note:
        This uses a kind of method called "Naive Parallel Scan" for inclusive
        scan in each thread-block. See below for details about it.
        https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
    """
    in_params = 'raw O b'
    out_params = 'raw T a, raw O out'
    loop_prep = string.Template("""
        __shared__ O smem1[${block_size}];
        __shared__ O smem2[${warp_size}];
        const int n_warp = ${block_size} / ${warp_size};
        const int warp_id = threadIdx.x / ${warp_size};
        const int lane_id = threadIdx.x % ${warp_size};
    """).substitute(block_size=chunk_size, warp_size=warp_size)
    loop_body = string.Template("""
        O x = ${identity};
        if (i < a.size()) x = a[i];
        for (int j = 1; j < ${warp_size}; j *= 2) {
            smem1[threadIdx.x] = x;  __syncwarp();
            if (lane_id - j >= 0) x ${op}= smem1[threadIdx.x - j];
            __syncwarp();
        }
        if (lane_id == ${warp_size} - 1) smem2[warp_id] = x;
        __syncthreads();
        if (warp_id == 0) {
            O y = ${identity};
            if (lane_id < n_warp) y = smem2[lane_id];
            for (int j = 1; j < n_warp; j *= 2) {
                smem2[lane_id] = y;  __syncwarp();
                if (lane_id - j >= 0) y ${op}= smem2[lane_id - j];
                __syncwarp();
            }
            smem2[lane_id] = y;
        }
        __syncthreads();
        if (warp_id > 0) x ${op}= smem2[warp_id - 1];
        int block_id = i / ${block_size};
        if (block_id > 0) x ${op}= b[block_id - 1];
        if (i < a.size()) out[i] = x;
    """).substitute(block_size=chunk_size, warp_size=warp_size,
                    op=_op_char[op], identity=_identity[op])
    return cupy.ElementwiseKernel(in_params, out_params, loop_body,
                                  'cupy_scan_naive', loop_prep=loop_prep,
                                  preamble=_preamble)


@cupy._util.memoize(for_each_device=True)
def _cupy_scan_btree(op, chunk_size, warp_size=32):
    """Returns a kernel to compute an inclusive scan.

    It first performs an inclusive scan in each thread-block and then add the
    scan results for the sum/prod of the chunks.

    Args:
        op (int): Operation type. SCAN_SUM or SCAN_PROD.
        chunk_size (int): Number of array elements processed by a single
            thread-block.
        warp_size (int); Warp size.

    Returns:
        cupy.ElementwiseKernel

    Example:
        b = [10, 36, 55]
        a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        _cupy_scan(op=SCAN_SUM, chunk_size=4)(b, a, out, ...)
        out == [1, 3, 6, 10, 15, 21, 28, 36, 45, 55]

    Note:
        This uses a kind of method called "Work-Efficient Parallel Scan" for
        inclusive scan in each thread-block. See below link for details about
        "Work-Efficient Parallel Scan".
        https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
    """
    in_params = 'raw O b'
    out_params = 'raw T a, raw O out'
    loop_prep = string.Template("""
        __shared__ O smem0[${block_size} + 1];
        O *smem1 = smem0 + 1;
        __shared__ O smem2[${warp_size}];
        const int n_warp = ${block_size} / ${warp_size};
        const int warp_id = threadIdx.x / ${warp_size};
        const int lane_id = threadIdx.x % ${warp_size};
        if (threadIdx.x == 0) smem0[0] = ${identity};
    """).substitute(block_size=chunk_size, warp_size=warp_size,
                    identity=_identity[op])
    loop_body = string.Template("""
        O x = ${identity};
        if (i < a.size()) x = a[i];
        for (int j = 1; j < ${warp_size}; j *= 2) {
            smem1[threadIdx.x] = x;  __syncwarp();
            if (lane_id % (2*j) == (2*j)-1) {
                x ${op}= smem1[threadIdx.x - j];
            }
            __syncwarp();
        }
        smem1[threadIdx.x] = x;
        __syncthreads();
        if (warp_id == 0) {
            O y = ${identity};
            if (lane_id < n_warp) {
                y = smem0[${warp_size} * (lane_id + 1)];
            }
            for (int j = 1; j < n_warp; j *= 2) {
                smem2[lane_id] = y;  __syncwarp();
                if (lane_id % (2*j) == (2*j)-1) {
                    y ${op}= smem2[lane_id - j];
                }
                __syncwarp();
            }
            for (int j = n_warp / 4; j > 0; j /= 2) {
                smem2[lane_id] = y; __syncwarp();
                if ((lane_id % (2*j) == j-1) && (lane_id >= 2*j)) {
                    y ${op}= smem2[lane_id - j];
                }
                __syncwarp();
            }
            if (lane_id < n_warp) {
                smem0[${warp_size} * (lane_id + 1)] = y;
            }
        }
        __syncthreads();
        x = smem0[threadIdx.x];
        for (int j = ${warp_size} / 2; j > 0; j /= 2) {
            if (lane_id % (2*j) == j) {
                x ${op}= smem0[threadIdx.x - j];
            }
            __syncwarp();
            smem0[threadIdx.x] = x;  __syncwarp();
        }
        __syncthreads();
        x = smem1[threadIdx.x];
        int block_id = i / ${block_size};
        if (block_id > 0) x ${op}= b[block_id - 1];
        if (i < a.size()) out[i] = x;
    """).substitute(block_size=chunk_size, warp_size=warp_size,
                    op=_op_char[op], identity=_identity[op])
    return cupy.ElementwiseKernel(in_params, out_params, loop_body,
                                  'cupy_scan_btree', loop_prep=loop_prep,
                                  preamble=_preamble)


cdef ndarray scan(ndarray a, op, dtype=None, ndarray out=None,
                  incomplete=False, chunk_size=512):
    """Return the prefix sum(scan) of the elements.

    Args:
        a (cupy.ndarray): input array.
        out (cupy.ndarray): Alternative output array in which to place
            the result. The same size and same type as the input array(a).

    Returns:
        cupy.ndarray: A new array holding the result is returned.

    """
    if a._shape.size() != 1:
        raise TypeError('Input array should be 1D array.')

    if out is None:
        if dtype is None:
            dtype = a.dtype
        if not incomplete:
            out = _ndarray_init(a._shape, dtype)
    else:
        if a.size != out.size:
            raise ValueError('Provided out is the wrong size')
        dtype = out.dtype
    dtype = numpy.dtype(dtype)

    warp_size = 64 if runtime._is_hip_environment else 32
    if runtime._is_hip_environment:
        if dtype.char in 'iIfdlq':
            # On HIP, __shfl* supports int, unsigned int, float, double,
            # long, and long long. The documentation is too outdated and
            # unreliable; refer to the header at
            # $ROCM_HOME/include/hip/hcc_detail/device_functions.h
            bsum_kernel = _cupy_bsum_shfl(op, chunk_size, warp_size)
        else:
            bsum_kernel = _cupy_bsum_smem(op, chunk_size, warp_size)
    else:
        if dtype.char in 'iIlLqQfd':
            bsum_kernel = _cupy_bsum_shfl(op, chunk_size, warp_size)
        else:
            bsum_kernel = _cupy_bsum_smem(op, chunk_size, warp_size)
    if dtype.char in 'fdFD':
        scan_kernel = _cupy_scan_btree(op, chunk_size, warp_size)
    else:
        scan_kernel = _cupy_scan_naive(op, chunk_size, warp_size)
    b_size = (a.size + chunk_size - 1) // chunk_size
    b = cupy.empty((b_size,), dtype=dtype)
    size = b.size * chunk_size

    if a.size > chunk_size:
        bsum_kernel(a, b, size=size // 2, block_size=chunk_size // 2)
        scan(b, op, dtype=dtype, out=b)
        if incomplete:
            return b
        scan_kernel(b, a, out, size=size, block_size=chunk_size)
    else:
        scan_kernel(b, a, out, size=size, block_size=chunk_size)

    return out


@_util.memoize(for_each_device=True)
def _inclusive_batch_scan_kernel(
        dtype, block_size, op, src_c_cont, out_c_cont):
    """return Prefix Sum(Scan) cuda kernel
    for a 2d array over axis 1
    used for scanning over different axes

    e.g
    if blocksize > len(src[0])
    src [[1, 2, 3, 4],
         [5, 6, 7, 8]]
    dst [[1, 3, 6, 10],
         [5, 11, 18, 26]]

    if blocksize < len(src[0])
    block_size: 2
    # TODO show partialness
    src [[1, 2, 3, 4],
         [5, 6, 7, 8]]
    dst [[1, 3, 3, 7],
         [5, 11, 7, 15]]

    Args:
        dtype: src, dst array type
        block_size: block_size

    Returns:
         cupy.cuda.Function: cuda function
    """
    op_char = {scan_op.SCAN_SUM: '+', scan_op.SCAN_PROD: '*'}
    identity = {scan_op.SCAN_SUM: 0, scan_op.SCAN_PROD: 1}
    name = 'inclusive_batch_scan_kernel'
    dtype = get_typename(dtype)
    source = string.Template("""
    extern "C" __global__ void ${name}(
        const CArray<${dtype}, 2, ${src_c_cont}> src,
        CArray<${dtype}, 2, ${out_c_cont}> dst, int batch_size){
        long long n = src.size();

        extern __shared__ ${dtype} temp[];

        unsigned int thid = threadIdx.x;
        unsigned int block = blockIdx.x * blockDim.x;

        unsigned int pad_batch_size = batch_size;
        bool must_copy = true;

        if (batch_size & (batch_size -1)) {
            pad_batch_size = 1 << (32 - __clz(batch_size));
            must_copy = (thid & (pad_batch_size-1)) < batch_size;
        }
        if (pad_batch_size > ${block_size}) {
            int blocks_per_batch = (batch_size - 1) / ${block_size} + 1;
            pad_batch_size = ${block_size} * blocks_per_batch;

            // Must copy enables for all blocks but the last one in the batch
            bool last_block = (blockIdx.x + 1) % blocks_per_batch == 0;
            int remaining_batch = batch_size % ${block_size};
            if (remaining_batch == 0) {
                remaining_batch = ${block_size};
            }
            must_copy = !last_block || (thid < (remaining_batch));
        }

        int pad_per_batch = pad_batch_size-batch_size;
        int n_batches_block = ${block_size} / pad_batch_size;

        unsigned int idx0 = thid + block;

        int batch_id = idx0 / pad_batch_size;
        idx0 = idx0 - pad_per_batch * batch_id;

        int row = idx0 / batch_size;
        int col = idx0 % batch_size;
        const ptrdiff_t idx0_idx[] = {row, col};

        if(idx0 < n){
            temp[thid] = (must_copy) ? src[idx0_idx] : (${dtype}) ${identity};
            __syncthreads();
            if (!n_batches_block) {
                n_batches_block = 1;
                pad_batch_size = ${block_size};
            }
            for (int j = 0; j < n_batches_block; j++) {
                int offset = j * pad_batch_size;
                for (int i = 1; i <= pad_batch_size; i <<= 1) {
                    int index = ((threadIdx.x + 1) * 2 * i - 1);
                    int index_block = offset + index;
                    if (index < (pad_batch_size)){
                        temp[index_block] ${op}= temp[index_block - i];
                    }
                    __syncthreads();
                }
                for(int i = pad_batch_size >> 1; i > 0; i >>= 1){
                    int index = ((threadIdx.x + 1) * 2 * i - 1);
                    int index_block = offset + index;
                    if((index + i) < (pad_batch_size)){
                        temp[index_block + i] ${op}= temp[index_block];
                    }
                    __syncthreads();
                }
            }
            if(must_copy){
                dst[idx0_idx] = temp[thid];
            }
        }
    }
    """).substitute(name=name, dtype=dtype, block_size=block_size,
                    op=op_char[op], identity=identity[op],
                    src_c_cont=src_c_cont, out_c_cont=out_c_cont)
    module = compile_with_cache(source)
    return module.get_function(name)


@_util.memoize(for_each_device=True)
def _add_scan_batch_blocked_sum_kernel(dtype, op, block_size, c_cont):
    name = 'add_scan_blocked_sum_kernel'
    dtype = get_typename(dtype)
    ops = {scan_op.SCAN_SUM: '+', scan_op.SCAN_PROD: '*'}
    source = string.Template("""
    extern "C" __global__ void ${name}(CArray<${dtype}, 2, ${c_cont}> src_dst,
        int batch_size){
        long long n = src_dst.size();

        unsigned int thid = threadIdx.x;
        unsigned int block = blockIdx.x * ${block_size};

        unsigned int idx0 = thid + block;

        // Respect padding
        unsigned int row = idx0 / batch_size;
        unsigned int col = idx0 % batch_size;
        int my_block = ${block_size} * (col / ${block_size});
        const ptrdiff_t dst_idx[] = {row, col};
        const ptrdiff_t src_idx[] = {row, my_block - 1};

        // Avoid for the first block of every row
        // This can be tweaked with kernel launch settings
        bool first = col < ${block_size};
        bool is_block = (col % (${block_size})) == ${block_size} - 1;
        if(idx0 < n && !first && !is_block){
            src_dst[dst_idx] ${op}= src_dst[src_idx];
        }
    }
    """).substitute(name=name, dtype=dtype, op=ops[op], block_size=block_size,
                    c_cont=c_cont)
    module = compile_with_cache(source)
    return module.get_function(name)


cdef ndarray _batch_scan_op(ndarray a, scan_op op, ndarray out):
    batch_size = a.shape[1]
    # TODO(ecastill) replace this with "_reduction._block_size" once it is
    # properly exposed
    block_size = 512
    # Since we need to pad each batch we spawn more threads as some
    # of them will be idle
    # Calc the total number of blocks
    padded_bs = 1 << ((batch_size - 1).bit_length())
    if padded_bs > block_size:
        blocks_per_batch = (batch_size - 1) // block_size + 1
        padded_bs = block_size * blocks_per_batch
    padded_size = a.size // batch_size * padded_bs

    cdef int src_cont = int(a.flags.c_contiguous)
    cdef int out_cont = int(out.flags.c_contiguous)
    kern_scan = _inclusive_batch_scan_kernel(a.dtype, block_size, op,
                                             src_cont, out_cont)
    kern_scan(grid=((padded_size - 1) // (block_size) + 1,),
              block=(block_size,),
              args=(a, out, batch_size),
              shared_mem=a.itemsize * block_size)
    if batch_size > block_size:
        blocked_sum = out[:, block_size-1::block_size]
        _batch_scan_op(blocked_sum, op, blocked_sum)
        kern_add = _add_scan_batch_blocked_sum_kernel(
            out.dtype, op, block_size, out_cont)
        kern_add(
            grid=((out.size - 1) // (block_size) + 1,),
            block=(block_size,),
            args=(out, batch_size))
    return out


cdef _proc_as_batch(ndarray x, int axis, scan_op op):
    if x.shape[axis] == 0:
        return cupy.empty_like(x)
    t = cupy.rollaxis(x, axis, x.ndim)
    s = t.shape
    r = t.reshape(-1, x.shape[axis])
    _batch_scan_op(r, op, r)
    return cupy.rollaxis(r.reshape(s), x.ndim-1, axis)


cpdef scan_core(ndarray a, axis, scan_op op, dtype=None, ndarray out=None):
    if out is None:
        if dtype is None:
            kind = a.dtype.kind
            if kind == 'b':
                dtype = numpy.dtype('l')
            elif kind == 'i' and a.dtype.itemsize < numpy.dtype('l').itemsize:
                dtype = numpy.dtype('l')
            elif kind == 'u' and a.dtype.itemsize < numpy.dtype('L').itemsize:
                dtype = numpy.dtype('L')
            else:
                dtype = a.dtype
        result = None
    else:
        if (out.flags.c_contiguous or out.flags.f_contiguous):
            result = out
            result[...] = a
        else:
            result = a.astype(out.dtype, order='C')

    if axis is None:
        for accelerator in _accelerator._routine_accelerators:
            if accelerator == _accelerator.ACCELERATOR_CUB:
                if result is None:
                    result = a.astype(dtype, order='C').ravel()
                # result will be None if the scan is not compatible with CUB
                if op == scan_op.SCAN_SUM:
                    cub_op = cub.CUPY_CUB_CUMSUM
                else:
                    cub_op = cub.CUPY_CUB_CUMPROD
                res = cub.cub_scan(result, cub_op)
                if res is not None:
                    break
        else:
            if result is None:
                result = scan(a.ravel(), op, dtype=dtype)
            else:
                scan(result, op, dtype=dtype, out=result)
    else:
        if result is None:
            result = a.astype(dtype, order='C')
        axis = internal._normalize_axis_index(axis, a.ndim)
        result = _proc_as_batch(result, axis, op)
    # This is for when the original out param was not contiguous
    if out is not None and out.data != result.data:
        out[...] = result.reshape(out.shape)
    else:
        out = result
    return out


# Only for test
def _scan_for_test(a, out=None):
    return scan(a, scan_op.SCAN_SUM, dtype=None, out=out)


cpdef ndarray _nansum(ndarray a, axis, dtype, out, keepdims):
    if cupy.iscomplexobj(a):
        return _nansum_complex_dtype(a, axis, dtype, out, keepdims)
    elif dtype is None:
        return _nansum_auto_dtype(a, axis, dtype, out, keepdims)
    else:
        return _nansum_keep_dtype(a, axis, dtype, out, keepdims)


cpdef ndarray _nanprod(ndarray a, axis, dtype, out, keepdims):
    if cupy.iscomplexobj(a):
        return _nanprod_complex_dtype(a, axis, dtype, out, keepdims)
    elif dtype is None:
        return _nanprod_auto_dtype(a, axis, dtype, out, keepdims)
    else:
        return _nanprod_keep_dtype(a, axis, dtype, out, keepdims)


_sum_auto_dtype = create_reduction_func(
    'cupy_sum',
    ('?->l', 'b->l', 'B->L', 'h->l', 'H->L', 'i->l', 'I->L', 'l->l', 'L->L',
     'q->q', 'Q->Q',
     ('e->e', (None, None, None, 'float')),
     'f->f', 'd->d', 'F->F', 'D->D'),
    ('in0', 'a + b', 'out0 = type_out0_raw(a)', None), 0)


_sum_keep_dtype = create_reduction_func(
    'cupy_sum_with_dtype',
    ('?->?', 'b->b', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I', 'l->l', 'L->L',
     'q->q', 'Q->Q',
     ('e->e', (None, None, None, 'float')),
     'f->f', 'd->d', 'F->F', 'D->D'),
    ('in0', 'a + b', 'out0 = type_out0_raw(a)', None), 0)


_nansum_auto_dtype = create_reduction_func(
    'cupy_nansum',
    ('?->l', 'b->l', 'B->L', 'h->l', 'H->L', 'i->l', 'I->L', 'l->l', 'L->L',
     'q->q', 'Q->Q',
     ('e->e', (None, None, None, 'float')),
     'f->f', 'd->d', 'F->F', 'D->D'),
    ('(in0 == in0) ? in0 : type_in0_raw(0)',
     'a + b', 'out0 = type_out0_raw(a)', None), 0)


_nansum_keep_dtype = create_reduction_func(
    'cupy_nansum_with_dtype',
    ('?->?', 'b->b', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I', 'l->l', 'L->L',
     'q->q', 'Q->Q',
     ('e->e', (None, None, None, 'float')),
     'f->f', 'd->d', 'F->F', 'D->D'),
    ('(in0 == in0) ? in0 : type_in0_raw(0)',
     'a + b', 'out0 = type_out0_raw(a)', None), 0)


_nansum_complex_dtype = create_reduction_func(
    'cupy_nansum_complex_dtype',
    ('F->F', 'D->D'),
    ('''
    type_in0_raw((in0.real() == in0.real()) ? in0.real() : 0,
                 (in0.imag() == in0.imag()) ? in0.imag() : 0)
    ''',
     'a + b', 'out0 = type_out0_raw(a)', None), 0)


_prod_auto_dtype = create_reduction_func(
    'cupy_prod',
    ('?->l', 'b->l', 'B->L', 'h->l', 'H->L', 'i->l', 'I->L', 'l->l', 'L->L',
     'q->q', 'Q->Q',
     ('e->e', (None, None, None, 'float')),
     'f->f', 'd->d', 'F->F', 'D->D'),
    ('in0', 'a * b', 'out0 = type_out0_raw(a)', None), 1)


_prod_keep_dtype = create_reduction_func(
    'cupy_prod_with_dtype',
    ('?->?', 'b->b', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I', 'l->l', 'L->L',
     'q->q', 'Q->Q',
     ('e->e', (None, None, None, 'float')),
     'f->f', 'd->d', 'F->F', 'D->D'),
    ('in0', 'a * b', 'out0 = type_out0_raw(a)', None), 1)


_nanprod_auto_dtype = create_reduction_func(
    'cupy_nanprod',
    ('?->l', 'b->l', 'B->L', 'h->l', 'H->L', 'i->l', 'I->L', 'l->l', 'L->L',
     'q->q', 'Q->Q',
     ('e->e', (None, None, None, 'float')),
     'f->f', 'd->d', 'F->F', 'D->D'),
    ('(in0 == in0) ? in0 : type_in0_raw(1)',
     'a * b', 'out0 = type_out0_raw(a)', None), 1)


_nanprod_keep_dtype = create_reduction_func(
    'cupy_nanprod_with_dtype',
    ('?->?', 'b->b', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I', 'l->l', 'L->L',
     'q->q', 'Q->Q',
     ('e->e', (None, None, None, 'float')),
     'f->f', 'd->d', 'F->F', 'D->D'),
    ('(in0 == in0) ? in0 : type_in0_raw(1)',
     'a * b', 'out0 = type_out0_raw(a)', None), 1)


_nanprod_complex_dtype = create_reduction_func(
    'cupy_nanprod_complex_dtype',
    ('F->F', 'D->D'),
    ('''
    type_in0_raw((in0.real() == in0.real()) ? in0.real() : 1,
                 (in0.imag() == in0.imag()) ? in0.imag() : 1)
    ''',
     'a * b', 'out0 = type_out0_raw(a)', None), 1)

cdef create_arithmetic(name, op, boolop, doc):
    # boolop is either
    #  - str (the operator for bool-bool inputs) or
    #  - callable (a function to raise an error for bool-bool inputs).
    if isinstance(boolop, str):
        boolop = 'out0 = in0 %s in1' % boolop

    return create_ufunc(
        'cupy_' + name,
        (('??->?', boolop),
         'bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l',
         'LL->L', 'qq->q', 'QQ->Q', 'ee->e', 'ff->f', 'dd->d', 'FF->F',
         'DD->D'),
        'out0 = in0 %s in1' % op,
        doc=doc)


_add = create_arithmetic(
    'add', '+', '|',
    '''Adds two arrays elementwise.

    .. seealso:: :data:`numpy.add`

    ''')


_conjugate = create_ufunc(
    'cupy_conjugate',
    ('b->b', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I', 'l->l', 'L->L', 'q->q',
     'Q->Q', 'e->e', 'f->f', 'd->d',
     ('F->F', 'out0 = conj(in0)'),
     ('D->D', 'out0 = conj(in0)')),
    'out0 = in0',
    doc='''Returns the complex conjugate, element-wise.

    .. seealso:: :data:`numpy.conjugate`

    ''')


_angle = create_ufunc(
    'cupy_angle',
    ('?->d', 'e->e', 'f->f', 'd->d',
     ('F->f', 'out0 = arg(in0)'),
     ('D->d', 'out0 = arg(in0)')),
    'out0 = in0 >= 0 ? 0 : M_PI',
    doc='''Returns the angle of the complex argument.

    .. seealso:: :func:`numpy.angle`

    ''')


_real = create_ufunc(
    'cupy_real',
    ('?->?', 'b->b', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I', 'l->l', 'L->L',
     'q->q', 'Q->Q', 'e->e', 'f->f', 'd->d',
     ('F->f', 'out0 = in0.real()'),
     ('D->d', 'out0 = in0.real()')),
    'out0 = in0',
    doc='''Returns the real part of the elements of the array.

    .. seealso:: :func:`numpy.real`

    ''')

_real_setter = create_ufunc(
    'cupy_real_setter',
    ('f->F', 'd->D'),
    'out0.real(in0)',
    doc='''Sets the real part of the elements of the array.
    ''')


_imag = create_ufunc(
    'cupy_imag',
    ('?->?', 'b->b', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I', 'l->l', 'L->L',
     'q->q', 'Q->Q', 'e->e', 'f->f', 'd->d',
     ('F->f', 'out0 = in0.imag()'),
     ('D->d', 'out0 = in0.imag()')),
    'out0 = 0',
    doc='''Returns the imaginary part of the elements of the array.

    .. seealso:: :func:`numpy.imag`

    ''')


_imag_setter = create_ufunc(
    'cupy_imag_setter',
    ('f->F', 'd->D'),
    'out0.imag(in0)',
    doc='''Sets the imaginary part of the elements of the array.
    ''')


def _negative_boolean_error():
    raise TypeError(
        'The cupy boolean negative, the `-` operator, is not supported, '
        'use the `~` operator or the logical_not function instead.')


_negative = create_ufunc(
    'cupy_negative',
    (('?->?', _negative_boolean_error),
     'b->b', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I', 'l->l', 'L->L',
     'q->q', 'Q->Q', 'e->e', 'f->f', 'd->d', 'F->F', 'D->D'),
    'out0 = -in0',
    doc='''Takes numerical negative elementwise.

    .. seealso:: :data:`numpy.negative`

    ''')


_multiply = create_arithmetic(
    'multiply', '*', '&',
    '''Multiplies two arrays elementwise.

    .. seealso:: :data:`numpy.multiply`

    ''')


# `integral_power` should return somewhat appropriate values for negative
# integral powers (for which NumPy would raise errors). Hence the branches in
# the beginning. This behavior is not officially documented and could change.
cdef _power_preamble = '''
template <typename T>
inline __device__ T integral_power(T in0, T in1) {
    if (in1 < 0) {
        if (in0 == -1) {return (in1 & 1) ? -1 : 1;}
        else {return (in0 == 1) ? 1 : 0;}
    }
    T out0 = 1;
    while (in1 > 0) {
        if (in1 & 1) {
            out0 *= in0;
        }
        in0 *= in0;
        in1 >>= 1;
    }
    return out0;
}

template <typename T>
inline __device__ T complex_power(T in0, T in1) {
    return in1 == T(0) ? T(1): pow(in0, in1);
}
'''

_power = create_ufunc(
    'cupy_power',
    ('??->b', 'bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l',
     'LL->L', 'qq->q', 'QQ->Q',
     ('ee->e', 'out0 = powf(in0, in1)'),
     ('ff->f', 'out0 = powf(in0, in1)'),
     ('dd->d', 'out0 = pow(in0, in1)'),
     ('FF->F', 'out0 = complex_power(in0, in1)'),
     ('DD->D', 'out0 = complex_power(in0, in1)')),
    'out0 = integral_power(in0, in1)',
    preamble=_power_preamble,
    doc='''Computes ``x1 ** x2`` elementwise.

    .. seealso:: :data:`numpy.power`

    ''')


def _subtract_boolean_error():
    raise TypeError(
        'cupy boolean subtract, the `-` operator, is deprecated, use the '
        'bitwise_xor, the `^` operator, or the logical_xor function instead.')


_subtract = create_arithmetic(
    'subtract', '-', _subtract_boolean_error,
    '''Subtracts arguments elementwise.

    .. seealso:: :data:`numpy.subtract`

    ''')


_true_divide = create_ufunc(
    'cupy_true_divide',
    ('bb->d', 'BB->d', 'hh->d', 'HH->d', 'ii->d', 'II->d', 'll->d', 'LL->d',
     'qq->d', 'QQ->d', 'ee->e', 'ff->f', 'dd->d', 'FF->F', 'DD->D'),
    'out0 = (out0_type)in0 / (out0_type)in1',
    doc='''Elementwise true division (i.e. division as floating values).

    .. seealso:: :data:`numpy.true_divide`

    ''',
    out_ops=('ee->e', 'ff->f', 'dd->d', 'FF->F', 'DD->D'),
)


_divide = _true_divide


_floor_divide = create_ufunc(
    'cupy_floor_divide',
    ('bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l', 'LL->L',
     'qq->q', 'QQ->Q', 'ee->e', 'ff->f', 'dd->d'),
    'out0 = _floor_divide(in0, in1)',
    doc='''Elementwise floor division (i.e. integer quotient).

    .. seealso:: :data:`numpy.floor_divide`

    ''')


_remainder = create_ufunc(
    'cupy_remainder',
    ('bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l', 'LL->L',
     'qq->q', 'QQ->Q',
     ('ee->e', 'out0 = in0 - _floor_divide(in0, in1) * in1'),
     ('ff->f', 'out0 = in0 - _floor_divide(in0, in1) * in1'),
     ('dd->d', 'out0 = in0 - _floor_divide(in0, in1) * in1')),
    'out0 = (in0 - _floor_divide(in0, in1) * in1) * (in1 != 0)',
    doc='''Computes the remainder of Python division elementwise.

    .. seealso:: :data:`numpy.remainder`

    ''')


_absolute = create_ufunc(
    'cupy_absolute',
    (('?->?', 'out0 = in0'),
     'b->b', ('B->B', 'out0 = in0'), 'h->h', ('H->H', 'out0 = in0'),
     'i->i', ('I->I', 'out0 = in0'), 'l->l', ('L->L', 'out0 = in0'),
     'q->q', ('Q->Q', 'out0 = in0'),
     ('e->e', 'out0 = fabsf(in0)'),
     ('f->f', 'out0 = fabsf(in0)'),
     ('d->d', 'out0 = fabs(in0)'),
     ('F->f', 'out0 = abs(in0)'),
     ('D->d', 'out0 = abs(in0)')),
    'out0 = in0 > 0 ? in0 : -in0',
    doc='''Elementwise absolute value function.

    .. seealso:: :data:`numpy.absolute`

    ''')


_sqrt = create_ufunc(
    'cupy_sqrt',
    ('e->e', 'f->f', 'd->d', 'F->F', 'D->D'),
    'out0 = sqrt(in0)',
    doc='''Elementwise square root function.

    .. seealso:: :data:`numpy.sqrt`

    ''')


_clip = create_ufunc(
    'cupy_clip',
    ('???->?', 'bbb->b', 'BBB->B', 'hhh->h', 'HHH->H', 'iii->i', 'III->I',
     'lll->l', 'LLL->L', 'qqq->q', 'QQQ->Q', 'eee->e', 'fff->f', 'ddd->d'),
    'out0 = in0 < in1 ? in1 : (in0 > in2 ? in2 : in0)')


# Variables to expose to Python
# (cythonized data cannot be exposed to Python, even with cpdef.)


add = _add
conjugate = _conjugate
angle = _angle
real = _real
imag = _imag
negative = _negative
multiply = _multiply
divide = _divide
power = _power
subtract = _subtract
true_divide = _true_divide
floor_divide = _floor_divide
remainder = _remainder
absolute = _absolute
sqrt = _sqrt

sum_auto_dtype = _sum_auto_dtype  # used from cupy/math/sumprod.py
nansum_auto_dtype = _nansum_auto_dtype  # used from cupy/math/sumprod.py
prod_auto_dtype = _prod_auto_dtype  # used from cupy/math/sumprod.py
nanprod_auto_dtype = _nanprod_auto_dtype  # used from cupy/math/sumprod.py
clip = _clip  # used from cupy/math/misc.py
