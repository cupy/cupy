import math
import os
import warnings

import cython
import numpy

import cupy
from cupy._core._kernel import ElementwiseKernel
from cupy._core._reduction import ReductionKernel
from cupy._core._ufuncs import elementwise_copy


from libc.stdint cimport intptr_t

from cupy._core cimport _accelerator
from cupy._core._carray cimport shape_t
from cupy._core._dtype cimport to_cuda_dtype
from cupy._core._scalar cimport get_typename
from cupy._core.core cimport _internal_ascontiguousarray
from cupy._core.core cimport _ndarray_init
from cupy._core.core cimport ascontiguousarray
from cupy._core.core cimport ndarray
from cupy._core cimport _routines_manipulation as _manipulation
from cupy._core cimport _routines_math as _math
from cupy.cuda cimport device
from cupy_backends.cuda.api cimport runtime
from cupy_backends.cuda.libs cimport cublas


cdef extern from '../../cupy_backends/cupy_complex.h':
    ctypedef struct cuComplex 'cuComplex':
        float x, y

    ctypedef struct cuDoubleComplex 'cuDoubleComplex':
        double x, y


cdef int _cuda_runtime_version = -1


cdef list compute_types = [COMPUTE_TYPE_TBD,  # float16
                           COMPUTE_TYPE_TBD,  # float32
                           COMPUTE_TYPE_TBD]  # float64
cdef dict compute_type_str = {
    0: 'COMPUTE_TYPE_TBD',
    1: 'COMPUTE_TYPE_DEFAULT',
    2: 'COMPUTE_TYPE_PEDANTIC',
    3: 'COMPUTE_TYPE_FP16',
    4: 'COMPUTE_TYPE_FP32',
    5: 'COMPUTE_TYPE_FP64',
    6: 'COMPUTE_TYPE_BF16',
    7: 'COMPUTE_TYPE_TF32',
}


cpdef int to_compute_type_index(dtype) except -1:
    cdef str dtype_char = numpy.dtype(dtype).char
    if dtype_char == 'e':
        return 0
    elif dtype_char in 'fF':
        return 1
    elif dtype_char in 'dD':
        return 2
    else:
        raise TypeError('dtype is not supported: {}'.format(dtype))


cpdef set_compute_type(dtype, compute_type):
    global compute_types
    if compute_type in (COMPUTE_TYPE_TBD, COMPUTE_TYPE_DEFAULT,
                        COMPUTE_TYPE_PEDANTIC, COMPUTE_TYPE_FP16,
                        COMPUTE_TYPE_FP32, COMPUTE_TYPE_FP64):
        compute_types[to_compute_type_index(dtype)] = compute_type
    elif compute_type in (COMPUTE_TYPE_BF16, COMPUTE_TYPE_TF32):
        if int(device.get_compute_capability()) >= 80:
            compute_types[to_compute_type_index(dtype)] = compute_type
        else:
            warnings.warn('COMPUTE_TYPE_BF16 and COMPUTE_TYPE_TF32 are only '
                          'available on GPUs with compute capability 8.0 or '
                          'higher. COMPUTE_TYPE_DEFAULT will be used instead.')
            compute_types[to_compute_type_index(dtype)] = COMPUTE_TYPE_DEFAULT
    else:
        raise ValueError('Unknown compute type: {}'.format(compute_type))


cpdef compute_type_to_str(compute_type):
    if compute_type in compute_type_str:
        return compute_type_str[compute_type]
    else:
        return compute_type


@cupy._util.memoize(for_each_device=True)
def _tensordot_core_int_kernel(config, dtype):
    # This code is based in the GEMM implementation from MAGMA
    # (http://icl.cs.utk.edu/magma/)
    code = '''
#define fetch(arr, col, m, n, bound) arr[min(n*col + m, bound)]

template<typename T>
__global__ void _tensordot_core_int_kernel(
        int M, int N, int K,
        const T* A,
        const T* B,
        T * C)
{
    int idx = threadIdx.x;
    int idy = threadIdx.y;

    int idt = DIM_X * idy + idx;

    int idxA = idt % DIM_XA;
    int idyA = idt / DIM_XA;

    int idxB = idt % DIM_XB;
    int idyB = idt / DIM_XB;

    int blx = blockIdx.x;
    int bly = blockIdx.y;

    __shared__ T sA[BLK_K][BLK_M + 1];
    __shared__ T sB[BLK_N][BLK_K + 1];

    // registers for the innermost loop
    T rC[THR_N][THR_M];
    T rA[THR_M];
    T rB[THR_N];

    T ra[BLK_K / DIM_YA][BLK_M / DIM_XA];
    T rb[BLK_N / DIM_YB][BLK_K / DIM_XB];

    const T* offs_dA = A + blx * BLK_M       + idyA * M + idxA;
    int boundA = (M * (K - 1) + M) - (blx * BLK_M + idyA * M + idxA) - 1;
    const T* offs_dB = B + bly * BLK_N * K + idyB * K + idxB;
    int boundB = (K * (N - 1) + K) - (bly * BLK_N * K + idyB * K + idxB) - 1;

    int m, n, k, kk;

    #pragma unroll
    for (n = 0; n < THR_N; n++) {
        #pragma unroll
        for (m = 0 ; m < THR_M; m++) {
            rC[n][m] = 0;
        }
    }

    // blockwise transpose to transpose load
    #pragma unroll
    for (n = 0; n < BLK_K; n += DIM_YA) {
        #pragma unroll
        for (m = 0; m < BLK_M; m += DIM_XA) {
            sA[n + idyA][m + idxA] = fetch(offs_dA, M, m, n, boundA);
        }
    }
    // blockwise transpose to transpose load
    #pragma unroll
    for (n = 0; n < BLK_N; n += DIM_YB) {
        #pragma unroll
        for (m = 0; m < BLK_K; m += DIM_XB) {
            sB[n + idyB][m + idxB] = fetch(offs_dB, K, m, n, boundB);
        }
    }
    __syncthreads();

    for (kk = 0; kk < K - BLK_K; kk += BLK_K)
    {
        offs_dA += BLK_K * M;
        boundA -= BLK_K * M;
        offs_dB += BLK_K;
        boundB -= BLK_K;

        #pragma unroll
        for (n = 0; n < BLK_K / DIM_YA; n++) {
            #pragma unroll
            for (m = 0; m < BLK_M / DIM_XA; m++) {
                ra[n][m] = fetch(offs_dA, M, m * DIM_XA, n * DIM_YA, boundA);
            }
        }

        #pragma unroll
        for (n = 0; n < BLK_N / DIM_YB; n++) {
            #pragma unroll
            for (m = 0; m < BLK_K / DIM_XB; m++) {
                rb[n][m] = fetch(offs_dB, K, m * DIM_XB, n * DIM_YB, boundB);
            }
        }

        // multiply
        #pragma unroll
        for (k = 0; k < BLK_K; k++)
        {
            #pragma unroll
            for (m = 0; m < THR_M; m++) {
                rA[m] = sA[k][m * DIM_X + idx];
            }

            #pragma unroll
            for (n = 0; n < THR_N; n++) {
                rB[n] = sB[n * DIM_Y + idy][k];
            }

            // HIP is strange...
            #ifdef __HIP_DEVICE_COMPILE__
            __syncthreads();
            #endif

            #pragma unroll
            for (n = 0; n < THR_N; n++) {
                #pragma unroll
                for (m = 0; m < THR_M; m++) {
                    rC[n][m] += rA[m] * rB[n];
                }
            }
        }
        __syncthreads();

        // store A regs->smem
        #pragma unroll
        for (n = 0; n < BLK_K / DIM_YA; n++)
        {
            #pragma unroll
            for (m = 0; m < BLK_M / DIM_XA; m++)
            {
                sA[n * DIM_YA + idyA][m * DIM_XA + idxA] = ra[n][m];
            }
        }

        #pragma unroll
        for (n = 0; n < BLK_N / DIM_YB; n++)
        {
            #pragma unroll
            for (m = 0; m < BLK_K / DIM_XB; m++)
            {
                sB[n * DIM_YB + idyB][m * DIM_XB + idxB] = rb[n][m];
            }
        }
        __syncthreads();
    }

    // Multiply last full (BLK_K) or partial block of columns of A and
    // rows of B.
    // It's okay that m,n exceed matrix bounds as all work is in registers
    // or shared memory, and out-of-bounds rC[n][m] will not be saved later.

    kk = K - kk;
    #pragma unroll
    for (k = 0; k < kk; k++)
    {
        #pragma unroll
        for (m = 0; m < THR_M; m++) {
            rA[m] = sA[k][m * DIM_X + idx];
        }

        #pragma unroll
        for (n = 0; n < THR_N; n++) {
            rB[n] = sB[n * DIM_Y + idy][k];
        }

        // HIP is strange...
        #ifdef __HIP_DEVICE_COMPILE__
        __syncthreads();
        #endif

        #pragma unroll
        for (n = 0; n < THR_N; n++) {
            #pragma unroll
            for (m = 0; m < THR_M; m++) {
                rC[n][m] += rA[m] * rB[n];
            }
        }
    }

    #pragma unroll
    for (n = 0; n < THR_N; n++) {
        int coord_dCn = bly * BLK_N + n * DIM_Y + idy;
        #pragma unroll
        for (m = 0; m < THR_M; m++) {
            int coord_dCm = blx * BLK_M + m * DIM_X + idx;
            if (coord_dCm < M && coord_dCn < N) {
                C[coord_dCn * M + coord_dCm] = rC[n][m];
            }
        }
    }
}
'''
    for k, v in config:
        code = '#define ' + k + ' ' + str(v) + '\n' + code
    name_expressions = ['_tensordot_core_int_kernel<bool>',
                        '_tensordot_core_int_kernel<signed char>',
                        '_tensordot_core_int_kernel<unsigned char>',
                        '_tensordot_core_int_kernel<short>',
                        '_tensordot_core_int_kernel<unsigned short>',
                        '_tensordot_core_int_kernel<int>',
                        '_tensordot_core_int_kernel<unsigned int>',
                        '_tensordot_core_int_kernel<long>',
                        '_tensordot_core_int_kernel<unsigned long>',
                        '_tensordot_core_int_kernel<long long>',
                        '_tensordot_core_int_kernel<unsigned long long>']
    mod = cupy.RawModule(code=code, options=('--std=c++11',),
                         name_expressions=name_expressions)
    ker = mod.get_function(
        '_tensordot_core_int_kernel<'+get_typename(dtype)+'>')
    return ker


cdef ndarray _integral_tensordot_core(
        ndarray a, ndarray b, ndarray out, Py_ssize_t m, Py_ssize_t n,
        Py_ssize_t k, str dtype, const shape_t& ret_shape):

    # TODO(leofang): autotune the tuning parameters here? See the discussion
    # in this thread: https://groups.google.com/a/icl.utk.edu/g/magma-user/c/igc66uduTfI  # NOQA
    dim_x=16
    dim_y=16
    blk_m=64
    blk_n=64
    blk_k=4
    dim_xa=64
    dim_ya=4
    dim_xb=4
    dim_yb=64
    config = (('DIM_X', dim_x), ('DIM_Y', dim_y),
              ('BLK_M', blk_m), ('BLK_N', blk_n), ('BLK_K', blk_k),
              ('DIM_XA', dim_xa), ('DIM_YA', dim_ya),
              ('DIM_XB', dim_xb), ('DIM_YB', dim_yb),
              ('THR_M', blk_m // dim_x), ('THR_N', blk_n // dim_y))
    kern = _tensordot_core_int_kernel(config, dtype)
    args = (m, n, k, a, b, out)
    grid = (int(math.ceil(m / blk_m)), int(math.ceil(n / blk_n)), 1)
    block = (dim_x, dim_y, 1)
    kern(grid, block, args=args)
    return out


cdef _tensordot_core_mul_sum = ReductionKernel(
    'S x, T y', 'U out',
    'static_cast<U>(x) * static_cast<U>(y)',
    'a + b', 'out = a', '0', '_tensordot_core_mul_sum')


cpdef get_compute_type(dtype):
    global compute_types
    cdef int index = to_compute_type_index(dtype)
    if compute_types[index] == COMPUTE_TYPE_TBD:
        compute_type = COMPUTE_TYPE_DEFAULT
        dtype_char = numpy.dtype(dtype).char
        if dtype_char in 'fF' and int(os.getenv('CUPY_TF32', '0')) > 0:
            compute_type = COMPUTE_TYPE_TF32
        set_compute_type(dtype, compute_type)
    return compute_types[index]


@cython.profile(False)
cpdef inline tuple _mat_to_cublas_contiguous(ndarray a, Py_ssize_t trans):
    assert a.ndim == 2
    if a._f_contiguous:
        # builtin max function is not used for Cython 0.23
        lda = a._strides[1] // a.itemsize
        if lda < a._shape[0]:
            lda = a._shape[0]
        return a, trans, lda
    if not a._c_contiguous:
        a = a.copy()
    return a, 1 - trans, a._strides[0] // a.itemsize


cpdef ndarray dot(ndarray a, ndarray b, ndarray out=None):
    cdef Py_ssize_t a_ndim, b_ndim, a_axis, b_axis, n, m, k
    cdef bint input_a_is_vec, input_b_is_vec
    cdef shape_t ret_shape, shape

    a_ndim = a._shape.size()
    b_ndim = b._shape.size()

    if out is not None and numpy.result_type(a.dtype, b.dtype) != out.dtype:
        raise ValueError('Not supported dtype combination.')

    if a_ndim == 0 or b_ndim == 0:
        return _math._multiply(a, b, out=out)

    input_a_is_vec = a_ndim == 1
    input_b_is_vec = b_ndim == 1
    if input_a_is_vec:
        shape.clear()
        shape.push_back(1)
        shape.push_back(a.size)
        a = _manipulation._reshape(a, shape)
        a_ndim = 2
    if input_b_is_vec:
        shape.clear()
        shape.push_back(b.size)
        shape.push_back(1)
        b = _manipulation._reshape(b, shape)
        b_ndim = 2

    a_axis = a_ndim - 1
    b_axis = b_ndim - 2

    if a._shape[a_axis] != b._shape[b_axis]:
        raise ValueError('Axis dimension mismatch')

    if a_axis:
        a = _manipulation.rollaxis(a, a_axis, 0)
    if b_axis:
        b = _manipulation.rollaxis(b, b_axis, 0)

    k = a._shape[0]
    if k != 0:
        m = b.size // k
        n = a.size // k
    else:
        # When k==0, the function must return a matrix filled with zero
        # like NumPy.
        m = 0
        n = 0

    if not input_a_is_vec:
        ret_shape.insert(ret_shape.end(), a._shape.begin() + 1, a._shape.end())
    if not input_b_is_vec:
        ret_shape.insert(ret_shape.end(), b._shape.begin() + 1, b._shape.end())
    if out is not None:
        if k != 0 and out.size != n * m:
            raise ValueError('Output array has an invalid size')
        if not out._c_contiguous:
            raise ValueError('Output array must be C-contiguous')

    return tensordot_core(a, b, out, n, m, k, ret_shape)


cpdef ndarray tensordot_core(
        ndarray a, ndarray b, ndarray out, Py_ssize_t n, Py_ssize_t m,
        Py_ssize_t k, const shape_t& ret_shape):
    cdef shape_t shape
    cdef Py_ssize_t inca, incb, transa, transb, lda, ldb
    cdef Py_ssize_t mode
    cdef intptr_t handle
    cdef bint use_sgemmEx = True
    cdef str dtype = a.dtype.char
    cdef int compute_capability = int(device.get_compute_capability())
    if dtype != b.dtype.char:
        dtype = numpy.promote_types(dtype, b.dtype).char
    if not a.size or not b.size:
        if out is None:
            out = _ndarray_init(ret_shape, dtype)
        out.fill(0)
        return out

    if out is None:
        out = _ndarray_init(ret_shape, dtype)
    else:
        if out.dtype != dtype:
            out = _ndarray_init(ret_shape, dtype)
    cdef int ace
    if m == 1 and n == 1:
        for ace in _accelerator._routine_accelerators:
            ret = _ndarray_init(ret_shape, dtype)
            # fast path using CUB or cuTENSOR
            if ace in (_accelerator.ACCELERATOR_CUB,
                       _accelerator.ACCELERATOR_CUTENSOR):
                ret = (a.ravel() * b.ravel()).sum(
                    out=_manipulation._reshape(ret, ()))
                elementwise_copy(ret, out)
                break
        else:
            _tensordot_core_mul_sum(
                a.ravel(), b.ravel(),
                out=_manipulation._reshape(out, ()))
        return out

    a = a.astype(dtype, order='K', casting=None, subok=None, copy=False)
    b = b.astype(dtype, order='K', casting=None, subok=None, copy=False)
    # It copies the operands if needed
    if a._shape.size() != 2 or a._shape[0] != k or a._shape[1] != n:
        shape.clear()
        shape.push_back(k)
        shape.push_back(n)
        a = _manipulation._reshape(a, shape)
    if b._shape.size() != 2 or b._shape[0] != k or b._shape[1] != m:
        shape.clear()
        shape.push_back(k)
        shape.push_back(m)
        b = _manipulation._reshape(b, shape)
    c = out
    if c._shape.size() != 2 or c._shape[0] != n or c._shape[1] != m:
        c = c.view()
        c.shape = (n, m)

    # Be careful that cuBLAS uses the FORTRAN-order matrix representation.
    # Matrix-Matrix product A^T * B
    # c is C-contiguous while cuBLAS assumes F-contiguous inputs, so we
    # compute C^T = B^T * A here.
    a, transa, lda = _mat_to_cublas_contiguous(a, 0)
    b, transb, ldb = _mat_to_cublas_contiguous(b, 1)

    if dtype not in 'efdFD':
        if transa:
            a = a.T
            a = _internal_ascontiguousarray(a)
        if transb:
            b = _internal_ascontiguousarray(b)
        return _integral_tensordot_core(b, a, out, m, n, k, dtype, ret_shape)

    global _cuda_runtime_version
    if _cuda_runtime_version < 0:
        _cuda_runtime_version = runtime.runtimeGetVersion()

    if _cuda_runtime_version >= 11000 and compute_capability >= 50:
        tensordot_core_v11(transb, transa, m, n, k, b, ldb, a, lda, c, m)
        return out

    handle = device.get_cublas_handle()
    if dtype == 'e':
        coef_dtype = 'f'
    else:
        coef_dtype = dtype
    one = numpy.array(1.0, dtype=coef_dtype)
    zero = numpy.array(0.0, dtype=coef_dtype)
    if runtime._is_hip_environment and dtype == 'e':
        # On HIP, SgemmEx does not work for half precision
        dtype = 'f'
        a = a.astype(dtype, order='K', casting=None, subok=None, copy=True)
        b = b.astype(dtype, order='K', casting=None, subok=None, copy=True)
        c = _ndarray_init(ret_shape, dtype)
        use_sgemmEx = False
        warnings.warn('On ROCm/HIP, there is no specialized API to handle '
                      'half precision floating numbers, so the computation '
                      'will be done by casting to single precision')
    if dtype == 'e':
        use_tensor_core = (_cuda_runtime_version >= 9000 and
                           compute_capability >= 70)
        if use_tensor_core:
            cublas.setMathMode(handle, cublas.CUBLAS_TENSOR_OP_MATH)
            cublas.gemmEx(
                handle, <int>transb, <int> transa, <int>m, <int>n, <int>k,
                one.ctypes.data, b.data.ptr, runtime.CUDA_R_16F, <int>ldb,
                a.data.ptr, runtime.CUDA_R_16F, <int>lda, zero.ctypes.data,
                c.data.ptr, runtime.CUDA_R_16F, <int>m, runtime.CUDA_R_32F,
                cublas.CUBLAS_GEMM_DEFAULT_TENSOR_OP)
            cublas.setMathMode(handle, cublas.CUBLAS_DEFAULT_MATH)
        else:
            cublas.sgemmEx(
                handle, <int>transb, <int> transa, <int>m, <int>n, <int>k,
                one.ctypes.data, b.data.ptr, runtime.CUDA_R_16F, <int>ldb,
                a.data.ptr, runtime.CUDA_R_16F, <int>lda, zero.ctypes.data,
                c.data.ptr, runtime.CUDA_R_16F, <int>m)
    elif dtype == 'f':
        cublas.sgemmEx(
            handle, <int>transb, <int> transa, <int>m, <int>n, <int>k,
            one.ctypes.data, b.data.ptr, runtime.CUDA_R_32F, <int>ldb,
            a.data.ptr, runtime.CUDA_R_32F, <int>lda, zero.ctypes.data,
            c.data.ptr, runtime.CUDA_R_32F, <int>m)
    elif dtype == 'd':
        cublas.dgemm(
            handle, <int>transb, <int>transa, <int>m, <int>n, <int>k,
            one.ctypes.data, b.data.ptr, <int>ldb, a.data.ptr, <int>lda,
            zero.ctypes.data, c.data.ptr, <int>m)
    elif dtype == 'F':
        cublas.cgemm(
            handle, <int>transb, <int>transa, <int>m, <int>n, <int>k,
            one.ctypes.data, b.data.ptr, <int>ldb, a.data.ptr, <int>lda,
            zero.ctypes.data, c.data.ptr, <int>m)
    elif dtype == 'D':
        cublas.zgemm(
            handle, <int>transb, <int>transa, <int>m, <int>n, <int>k,
            one.ctypes.data, b.data.ptr, <int>ldb, a.data.ptr, <int>lda,
            zero.ctypes.data, c.data.ptr, <int>m)
    else:
        raise ValueError('Invalid dtype: %s' % str(dtype))
    if not use_sgemmEx:
        out[...] = c
    return out


cpdef ndarray tensordot_core_v11(
        Py_ssize_t transa, Py_ssize_t transb, Py_ssize_t m, Py_ssize_t n,
        Py_ssize_t k, ndarray a, Py_ssize_t lda, ndarray b, Py_ssize_t ldb,
        ndarray c, Py_ssize_t ldc):
    cdef float one_f, zero_f
    cdef double one_d, zero_d
    cdef cuComplex one_F, zero_F
    cdef cuDoubleComplex one_D, zero_D
    cdef size_t one_ptr, zero_ptr

    cdef int compute_capability = int(device.get_compute_capability())
    cdef int compute_type = get_compute_type(c.dtype)
    cdef int cublas_compute_type = -1
    if c.dtype.char in 'efF':
        if compute_type == COMPUTE_TYPE_PEDANTIC:
            cublas_compute_type = cublas.CUBLAS_COMPUTE_32F_PEDANTIC
        elif compute_type == COMPUTE_TYPE_TF32 and c.dtype.char in 'fF':
            cublas_compute_type = cublas.CUBLAS_COMPUTE_32F_FAST_TF32
        else:
            cublas_compute_type = cublas.CUBLAS_COMPUTE_32F
    elif c.dtype.char in 'dD':
        if compute_type == COMPUTE_TYPE_PEDANTIC:
            cublas_compute_type = cublas.CUBLAS_COMPUTE_64F_PEDANTIC
        else:
            cublas_compute_type = cublas.CUBLAS_COMPUTE_64F
    else:
        raise ValueError('Invalid dtype: {}'.format(c.dtype))

    cdef int algo = cublas.CUBLAS_GEMM_DEFAULT
    if ((compute_capability >= 80) or
            (compute_capability >= 70 and c.dtype == 'e')):
        algo = cublas.CUBLAS_GEMM_DEFAULT_TENSOR_OP

    if cublas_compute_type in (cublas.CUBLAS_COMPUTE_32F,
                               cublas.CUBLAS_COMPUTE_32F_PEDANTIC,
                               cublas.CUBLAS_COMPUTE_32F_FAST_TF32):
        if c.dtype.char in 'efd':
            one_f = 1
            zero_f = 0
            one_ptr = <size_t>&one_f
            zero_ptr = <size_t>&zero_f
        else:
            one_F = cuComplex(1, 0)
            zero_F = cuComplex(0, 0)
            one_ptr = <size_t>&one_F
            zero_ptr = <size_t>&zero_F
    elif cublas_compute_type in (cublas.CUBLAS_COMPUTE_64F,
                                 cublas.CUBLAS_COMPUTE_64F_PEDANTIC):
        if c.dtype.char in 'efd':
            one_d = 1
            zero_d = 0
            one_ptr = <size_t>&one_d
            zero_ptr = <size_t>&zero_d
        else:
            one_D = cuDoubleComplex(1, 0)
            zero_D = cuDoubleComplex(0, 0)
            one_ptr = <size_t>&one_D
            zero_ptr = <size_t>&zero_D
    else:
        raise ValueError('Invalid cublas compute type: {}'
                         .format(cublas_compute_type))

    cdef int a_cuda_dtype = to_cuda_dtype(a.dtype, is_half_allowed=True)
    cdef int b_cuda_dtype = to_cuda_dtype(b.dtype, is_half_allowed=True)
    cdef int c_cuda_dtype = to_cuda_dtype(c.dtype, is_half_allowed=True)
    cdef intptr_t handle = device.get_cublas_handle()
    cublas.gemmEx(
        handle, <int>transa, <int>transb, <int>m, <int>n, <int>k, one_ptr,
        a.data.ptr, a_cuda_dtype, <int>lda, b.data.ptr, b_cuda_dtype, <int>ldb,
        zero_ptr, c.data.ptr, c_cuda_dtype, <int>ldc, cublas_compute_type,
        algo)


cdef Py_ssize_t _get_stride_for_strided_batched_gemm(ndarray a) except? 0:
    cdef int ndim = a._shape.size()
    assert ndim > 2
    return a._strides[ndim - 3] // <Py_ssize_t>a.itemsize


cdef _mat_ptrs_kernel = ElementwiseKernel(
    'T base, T stride', 'T out',
    'out = base + _ind.get()[_ind.ndim - 1] * stride', 'mat_ptrs',
    reduce_dims=False)


cpdef ndarray _mat_ptrs(ndarray a):
    """Creates an array of pointers to matrices
    Args:
        a: A batch of matrices on GPU.
           shape: (A, B, C) -> A ptrs to mat o size (B, C)
           shape: (A_1, ..., A_N, B, C) -> A_1*...*A_N ptrs to mat of
                  size (B, C)
    Returns:
        GPU array of pointers to matrices.
    """
    cdef int ndim = a._shape.size()
    assert ndim > 2
    cdef Py_ssize_t sh_, st_
    cdef ndarray idx
    idx = _mat_ptrs_kernel(
        a.data.ptr, a._strides[0],
        ndarray((a._shape[0],), dtype=numpy.uintp))

    for i in range(1, ndim - 2):
        idx = _mat_ptrs_kernel(
            idx[:, None], a._strides[i],
            ndarray((idx.size, a._shape[i]), dtype=numpy.uintp))
        idx = idx.ravel()
    return idx


cpdef ndarray _matmul(ndarray a, ndarray b, ndarray out=None):
    """ Returns the matrix product of two arrays and is the implementation of
    the `@` operator introduced in Python 3.5 following PEP465.

    The main difference against cupy.dot are the handling of arrays with more
    than 2 dimensions. For more information see :func:`numpy.matmul`.

    .. note::
        The out array as input is currently not supported.

    Args:
        a (cupy.ndarray): The left argument.
        b (cupy.ndarray): The right argument.
        out (cupy.ndarray): Output array.

    Returns:
        cupy.ndarray: Output array.

    .. seealso:: :func:`numpy.matmul`

    """

    if out is not None:
        raise NotImplementedError('The out array as input is currently not '
                                  'supported')

    cdef Py_ssize_t i, n, m, ka, kb, a_sh, b_sh, c_sh
    cdef Py_ssize_t batchCount, a_part_outshape, b_part_outshape
    cdef int orig_a_ndim, orig_b_ndim, a_ndim, b_ndim, ndim
    cdef ndarray ap, bp, outp, out_view
    cdef bint use_broadcast

    orig_a_ndim = a._shape.size()
    orig_b_ndim = b._shape.size()
    if orig_a_ndim == 0 or orig_b_ndim == 0:
        raise ValueError('Scalar operands are not allowed, use \'*\' instead')

    ndim = max(orig_a_ndim, orig_b_ndim)
    if ndim <= 2:
        return dot(a, b, out)

    orig_a = a
    orig_b = b
    a_part_outshape = b_part_outshape = 0
    if orig_a_ndim == 1:
        a = _manipulation._reshape(a, (1, a.size))
    else:
        a = a.view()
        a_part_outshape = a._shape[orig_a_ndim - 2]
    if orig_b_ndim == 1:
        b = _manipulation._reshape(b, (b.size, 1))
        ldout = 1
    else:
        b = b.view()
        b_part_outshape = ldout = b._shape[orig_b_ndim - 1]

    # expand dims
    a_ndim = a._shape.size()
    b_ndim = b._shape.size()
    if a_ndim < ndim:
        # TODO(niboshi): Confirm update_x_contiguity flags
        a._set_shape_and_strides(
            (1,) * (ndim - a_ndim) + a.shape,
            (0,) * (ndim - a_ndim) + a.strides,
            True, True)
    if b_ndim < ndim:
        # TODO(niboshi): Confirm update_x_contiguity flags
        b._set_shape_and_strides(
            (1,) * (ndim - b_ndim) + b.shape,
            (0,) * (ndim - b_ndim) + b.strides,
            True, True)

    ret_dtype = numpy.promote_types(a.dtype, b.dtype)
    dtype = numpy.promote_types(ret_dtype, 'f')

    a = ascontiguousarray(a, dtype)
    b = ascontiguousarray(b, dtype)

    # broadcast
    batchCount = 1  # batchCount = numpy.prod(out_shape[:-2])
    out_shape = []
    use_broadcast = False
    for i in range(0, ndim - 2):
        a_sh = a._shape[i]
        b_sh = b._shape[i]
        if a_sh != b_sh and a_sh != 1 and b_sh != 1:
            raise ValueError(
                'operands could not be broadcast together with '
                'remapped shapes')

        if a_sh == 0 or b_sh == 0:
            c_sh = 0
        else:
            c_sh = max(a_sh, b_sh)
        batchCount *= c_sh
        out_shape.append(c_sh)
        if a_sh == 1 and c_sh > 1:
            a._strides[i] = 0
            a._shape[i] = c_sh
            a._c_contiguous = a._f_contiguous = False
            use_broadcast = True

        if b_sh == 1 and c_sh > 1:
            b._strides[i] = 0
            b._shape[i] = c_sh
            b._c_contiguous = b._f_contiguous = False
            use_broadcast = True

    if orig_a_ndim != 1:
        out_shape.append(a_part_outshape)
    if orig_b_ndim != 1:
        out_shape.append(b_part_outshape)

    # (A B)^T = B^T A^T
    a, b = b, a

    ka = a._shape[ndim - 2]
    lda = n = a._shape[ndim - 1]
    m = b._shape[ndim - 2]
    ldb = kb = b._shape[ndim - 1]

    if ka != kb:
        raise ValueError(
            'shapes ({}) and ({}) not aligned'.format(
                ','.join([str(_) for _ in orig_a.shape]),
                ','.join([str(_) for _ in orig_b.shape])))

    if a.size == 0 or b.size == 0:
        return cupy.zeros(out_shape, ret_dtype)

    out = ndarray(out_shape, dtype=dtype)

    if orig_a_ndim == 1 or orig_b_ndim == 1:
        out_view = out.view()
        if orig_b_ndim == 1:
            out_view._shape.push_back(1)
            out_view._strides.push_back(0)
        if orig_a_ndim == 1:
            out_view._shape.insert(out_view._shape.end() - 1, 1)
            out_view._strides.insert(out_view._strides.end() - 1, 0)
        assert out_view._c_contiguous
        out_view._update_f_contiguity()
    else:
        out_view = out

    global _cuda_runtime_version
    if _cuda_runtime_version < 0:
        _cuda_runtime_version = runtime.runtimeGetVersion()

    cdef intptr_t handle = device.get_cublas_handle()

    one = numpy.array(1, dtype=dtype)
    zero = numpy.array(0, dtype=dtype)
    # TODO(anaruse) use cublasGemmStridedBatchedEx() when cuda version >= 9.1
    if not use_broadcast:
        strideA = _get_stride_for_strided_batched_gemm(a)
        strideB = _get_stride_for_strided_batched_gemm(b)
        strideC = _get_stride_for_strided_batched_gemm(out_view)
        if dtype == numpy.float32:
            cublas.sgemmStridedBatched(
                handle,
                0,  # transa
                0,  # transb
                n, m, ka, one.ctypes.data,
                a.data.ptr, lda, strideA,
                b.data.ptr, ldb, strideB,
                zero.ctypes.data, out_view.data.ptr, ldout, strideC,
                batchCount)
        elif dtype == numpy.float64:
            cublas.dgemmStridedBatched(
                handle,
                0,  # transa
                0,  # transb
                n, m, ka, one.ctypes.data,
                a.data.ptr, lda, strideA,
                b.data.ptr, ldb, strideB,
                zero.ctypes.data, out_view.data.ptr, ldout, strideC,
                batchCount)
        elif dtype == numpy.complex64:
            cublas.cgemmStridedBatched(
                handle,
                0,  # transa
                0,  # transb
                n, m, ka, one.ctypes.data,
                a.data.ptr, lda, strideA,
                b.data.ptr, ldb, strideB,
                zero.ctypes.data, out_view.data.ptr, ldout, strideC,
                batchCount)
        elif dtype == numpy.complex128:
            cublas.zgemmStridedBatched(
                handle,
                0,  # transa
                0,  # transb
                n, m, ka, one.ctypes.data,
                a.data.ptr, lda, strideA,
                b.data.ptr, ldb, strideB,
                zero.ctypes.data, out_view.data.ptr, ldout, strideC,
                batchCount)
        else:
            raise TypeError(dtype, a.dtype, b.dtype)
    else:
        ap = _mat_ptrs(a)
        bp = _mat_ptrs(b)
        outp = _mat_ptrs(out_view)
        if dtype == numpy.float32:
            cublas.sgemmBatched(
                handle,
                0,  # transa
                0,  # transb
                n, m, ka, one.ctypes.data,
                ap.data.ptr, lda,
                bp.data.ptr, ldb,
                zero.ctypes.data, outp.data.ptr, ldout, batchCount)
        elif dtype == numpy.float64:
            cublas.dgemmBatched(
                handle,
                0,  # transa
                0,  # transb
                n, m, ka, one.ctypes.data,
                ap.data.ptr, lda,
                bp.data.ptr, ldb,
                zero.ctypes.data, outp.data.ptr, ldout, batchCount)
        elif dtype == numpy.complex64:
            cublas.cgemmBatched(
                handle,
                0,  # transa
                0,  # transb
                n, m, ka, one.ctypes.data,
                ap.data.ptr, lda,
                bp.data.ptr, ldb,
                zero.ctypes.data, outp.data.ptr, ldout, batchCount)
        elif dtype == numpy.complex128:
            cublas.zgemmBatched(
                handle,
                0,  # transa
                0,  # transb
                n, m, ka, one.ctypes.data,
                ap.data.ptr, lda,
                bp.data.ptr, ldb,
                zero.ctypes.data, outp.data.ptr, ldout, batchCount)
        else:
            raise TypeError(dtype, a.dtype, b.dtype)

    if dtype == ret_dtype:
        return out
    else:
        ret = ndarray(out_shape, ret_dtype)
        elementwise_copy(out, ret)
        return ret
