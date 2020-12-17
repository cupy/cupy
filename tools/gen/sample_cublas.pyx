# This code was automatically generated. Do not modify it directly.

cimport cython  # NOQA

from cupy_backends.cuda.api cimport driver
from cupy_backends.cuda.api cimport runtime
from cupy_backends.cuda cimport stream as stream_module


cdef extern from '../cupy_cuComplex.h':
    ctypedef struct cuComplex 'cuComplex':
        float x, y

    ctypedef struct cuDoubleComplex 'cuDoubleComplex':
        double x, y


cdef extern from '../cupy_cublas.h' nogil:

    # cuBLAS Helper Function
    Status cublasCreate_v2(Handle* handle)
    Status cublasDestroy_v2(Handle handle)
    Status cublasGetVersion_v2(Handle handle, int* version)
    Status cublasGetPointerMode_v2(Handle handle, PointerMode* mode)
    Status cublasSetPointerMode_v2(Handle handle, PointerMode mode)
    Status cublasSetStream_v2(Handle handle, driver.Stream streamId)
    Status cublasGetStream_v2(Handle handle, driver.Stream* streamId)
    Status cublasSetMathMode(Handle handle, Math mode)
    Status cublasGetMathMode(Handle handle, Math* mode)

    # cuBLAS Level-1 Function
    Status cublasIsamax_v2(Handle handle, int n, const float* x, int incx, int* result)
    Status cublasIdamax_v2(Handle handle, int n, const double* x, int incx, int* result)
    Status cublasIcamax_v2(Handle handle, int n, const cuComplex* x, int incx, int* result)
    Status cublasIzamax_v2(Handle handle, int n, const cuDoubleComplex* x, int incx, int* result)
    Status cublasIsamin_v2(Handle handle, int n, const float* x, int incx, int* result)
    Status cublasIdamin_v2(Handle handle, int n, const double* x, int incx, int* result)
    Status cublasIcamin_v2(Handle handle, int n, const cuComplex* x, int incx, int* result)
    Status cublasIzamin_v2(Handle handle, int n, const cuDoubleComplex* x, int incx, int* result)
    Status cublasSasum_v2(Handle handle, int n, const float* x, int incx, float* result)
    Status cublasDasum_v2(Handle handle, int n, const double* x, int incx, double* result)
    Status cublasScasum_v2(Handle handle, int n, const cuComplex* x, int incx, float* result)
    Status cublasDzasum_v2(Handle handle, int n, const cuDoubleComplex* x, int incx, double* result)
    Status cublasSaxpy_v2(Handle handle, int n, const float* alpha, const float* x, int incx, float* y, int incy)
    Status cublasDaxpy_v2(Handle handle, int n, const double* alpha, const double* x, int incx, double* y, int incy)
    Status cublasCaxpy_v2(Handle handle, int n, const cuComplex* alpha, const cuComplex* x, int incx, cuComplex* y, int incy)
    Status cublasZaxpy_v2(Handle handle, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy)
    Status cublasSdot_v2(Handle handle, int n, const float* x, int incx, const float* y, int incy, float* result)
    Status cublasDdot_v2(Handle handle, int n, const double* x, int incx, const double* y, int incy, double* result)
    Status cublasCdotu_v2(Handle handle, int n, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* result)
    Status cublasCdotc_v2(Handle handle, int n, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* result)
    Status cublasZdotu_v2(Handle handle, int n, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* result)
    Status cublasZdotc_v2(Handle handle, int n, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* result)
    Status cublasSnrm2_v2(Handle handle, int n, const float* x, int incx, float* result)
    Status cublasDnrm2_v2(Handle handle, int n, const double* x, int incx, double* result)
    Status cublasScnrm2_v2(Handle handle, int n, const cuComplex* x, int incx, float* result)
    Status cublasDznrm2_v2(Handle handle, int n, const cuDoubleComplex* x, int incx, double* result)
    Status cublasSscal_v2(Handle handle, int n, const float* alpha, float* x, int incx)
    Status cublasDscal_v2(Handle handle, int n, const double* alpha, double* x, int incx)
    Status cublasCscal_v2(Handle handle, int n, const cuComplex* alpha, cuComplex* x, int incx)
    Status cublasCsscal_v2(Handle handle, int n, const float* alpha, cuComplex* x, int incx)
    Status cublasZscal_v2(Handle handle, int n, const cuDoubleComplex* alpha, cuDoubleComplex* x, int incx)
    Status cublasZdscal_v2(Handle handle, int n, const double* alpha, cuDoubleComplex* x, int incx)

    # cuBLAS Level-2 Function
    Status cublasSgemv_v2(Handle handle, Operation trans, int m, int n, const float* alpha, const float* A, int lda, const float* x, int incx, const float* beta, float* y, int incy)
    Status cublasDgemv_v2(Handle handle, Operation trans, int m, int n, const double* alpha, const double* A, int lda, const double* x, int incx, const double* beta, double* y, int incy)
    Status cublasCgemv_v2(Handle handle, Operation trans, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* x, int incx, const cuComplex* beta, cuComplex* y, int incy)
    Status cublasZgemv_v2(Handle handle, Operation trans, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int incy)
    Status cublasSger_v2(Handle handle, int m, int n, const float* alpha, const float* x, int incx, const float* y, int incy, float* A, int lda)
    Status cublasDger_v2(Handle handle, int m, int n, const double* alpha, const double* x, int incx, const double* y, int incy, double* A, int lda)
    Status cublasCgeru_v2(Handle handle, int m, int n, const cuComplex* alpha, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* A, int lda)
    Status cublasCgerc_v2(Handle handle, int m, int n, const cuComplex* alpha, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* A, int lda)
    Status cublasZgeru_v2(Handle handle, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* A, int lda)
    Status cublasZgerc_v2(Handle handle, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* A, int lda)

    # cuBLAS Level-3 Function
    Status cublasSgemm_v2(Handle handle, Operation transa, Operation transb, int m, int n, int k, const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc)
    Status cublasDgemm_v2(Handle handle, Operation transa, Operation transb, int m, int n, int k, const double* alpha, const double* A, int lda, const double* B, int ldb, const double* beta, double* C, int ldc)
    Status cublasCgemm_v2(Handle handle, Operation transa, Operation transb, int m, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const cuComplex* beta, cuComplex* C, int ldc)
    Status cublasZgemm_v2(Handle handle, Operation transa, Operation transb, int m, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc)
    Status cublasSgemmBatched(Handle handle, Operation transa, Operation transb, int m, int n, int k, const float* alpha, const float** Aarray, int lda, const float** Barray, int ldb, const float* beta, float** Carray, int ldc, int batchCount)
    Status cublasDgemmBatched(Handle handle, Operation transa, Operation transb, int m, int n, int k, const double* alpha, const double** Aarray, int lda, const double** Barray, int ldb, const double* beta, double** Carray, int ldc, int batchCount)
    Status cublasCgemmBatched(Handle handle, Operation transa, Operation transb, int m, int n, int k, const cuComplex* alpha, const cuComplex** Aarray, int lda, const cuComplex** Barray, int ldb, const cuComplex* beta, cuComplex** Carray, int ldc, int batchCount)
    Status cublasZgemmBatched(Handle handle, Operation transa, Operation transb, int m, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex** Aarray, int lda, const cuDoubleComplex** Barray, int ldb, const cuDoubleComplex* beta, cuDoubleComplex** Carray, int ldc, int batchCount)
    Status cublasSgemmStridedBatched(Handle handle, Operation transa, Operation transb, int m, int n, int k, const float* alpha, const float* A, int lda, long long int strideA, const float* B, int ldb, long long int strideB, const float* beta, float* C, int ldc, long long int strideC, int batchCount)
    Status cublasDgemmStridedBatched(Handle handle, Operation transa, Operation transb, int m, int n, int k, const double* alpha, const double* A, int lda, long long int strideA, const double* B, int ldb, long long int strideB, const double* beta, double* C, int ldc, long long int strideC, int batchCount)
    Status cublasCgemmStridedBatched(Handle handle, Operation transa, Operation transb, int m, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, long long int strideA, const cuComplex* B, int ldb, long long int strideB, const cuComplex* beta, cuComplex* C, int ldc, long long int strideC, int batchCount)
    Status cublasZgemmStridedBatched(Handle handle, Operation transa, Operation transb, int m, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, long long int strideA, const cuDoubleComplex* B, int ldb, long long int strideB, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc, long long int strideC, int batchCount)
    Status cublasStrsm_v2(Handle handle, SideMode side, FillMode uplo, Operation trans, DiagType diag, int m, int n, const float* alpha, const float* A, int lda, float* B, int ldb)
    Status cublasDtrsm_v2(Handle handle, SideMode side, FillMode uplo, Operation trans, DiagType diag, int m, int n, const double* alpha, const double* A, int lda, double* B, int ldb)
    Status cublasCtrsm_v2(Handle handle, SideMode side, FillMode uplo, Operation trans, DiagType diag, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, cuComplex* B, int ldb)
    Status cublasZtrsm_v2(Handle handle, SideMode side, FillMode uplo, Operation trans, DiagType diag, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, cuDoubleComplex* B, int ldb)

    # cuBLAS BLAS-like Extension
    Status cublasSgeam(Handle handle, Operation transa, Operation transb, int m, int n, const float* alpha, const float* A, int lda, const float* beta, const float* B, int ldb, float* C, int ldc)
    Status cublasDgeam(Handle handle, Operation transa, Operation transb, int m, int n, const double* alpha, const double* A, int lda, const double* beta, const double* B, int ldb, double* C, int ldc)
    Status cublasCgeam(Handle handle, Operation transa, Operation transb, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* beta, const cuComplex* B, int ldb, cuComplex* C, int ldc)
    Status cublasZgeam(Handle handle, Operation transa, Operation transb, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* beta, const cuDoubleComplex* B, int ldb, cuDoubleComplex* C, int ldc)
    Status cublasSdgmm(Handle handle, SideMode mode, int m, int n, const float* A, int lda, const float* x, int incx, float* C, int ldc)
    Status cublasDdgmm(Handle handle, SideMode mode, int m, int n, const double* A, int lda, const double* x, int incx, double* C, int ldc)
    Status cublasCdgmm(Handle handle, SideMode mode, int m, int n, const cuComplex* A, int lda, const cuComplex* x, int incx, cuComplex* C, int ldc)
    Status cublasZdgmm(Handle handle, SideMode mode, int m, int n, const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx, cuDoubleComplex* C, int ldc)
    Status cublasSgemmEx(Handle handle, Operation transa, Operation transb, int m, int n, int k, const float* alpha, const void* A, DataType Atype, int lda, const void* B, DataType Btype, int ldb, const float* beta, void* C, DataType Ctype, int ldc)
    Status cublasCgemmEx(Handle handle, Operation transa, Operation transb, int m, int n, int k, const cuComplex* alpha, const void* A, DataType Atype, int lda, const void* B, DataType Btype, int ldb, const cuComplex* beta, void* C, DataType Ctype, int ldc)
    Status cublasSgetrfBatched(Handle handle, int n, float** A, int lda, int* P, int* info, int batchSize)
    Status cublasDgetrfBatched(Handle handle, int n, double** A, int lda, int* P, int* info, int batchSize)
    Status cublasCgetrfBatched(Handle handle, int n, cuComplex** A, int lda, int* P, int* info, int batchSize)
    Status cublasZgetrfBatched(Handle handle, int n, cuDoubleComplex** A, int lda, int* P, int* info, int batchSize)
    Status cublasSgetrsBatched(Handle handle, Operation trans, int n, int nrhs, const float** Aarray, int lda, const int* devIpiv, float** Barray, int ldb, int* info, int batchSize)
    Status cublasDgetrsBatched(Handle handle, Operation trans, int n, int nrhs, const double** Aarray, int lda, const int* devIpiv, double** Barray, int ldb, int* info, int batchSize)
    Status cublasCgetrsBatched(Handle handle, Operation trans, int n, int nrhs, const cuComplex** Aarray, int lda, const int* devIpiv, cuComplex** Barray, int ldb, int* info, int batchSize)
    Status cublasZgetrsBatched(Handle handle, Operation trans, int n, int nrhs, const cuDoubleComplex** Aarray, int lda, const int* devIpiv, cuDoubleComplex** Barray, int ldb, int* info, int batchSize)
    Status cublasSgetriBatched(Handle handle, int n, const float** A, int lda, const int* P, float** C, int ldc, int* info, int batchSize)
    Status cublasDgetriBatched(Handle handle, int n, const double** A, int lda, const int* P, double** C, int ldc, int* info, int batchSize)
    Status cublasCgetriBatched(Handle handle, int n, const cuComplex** A, int lda, const int* P, cuComplex** C, int ldc, int* info, int batchSize)
    Status cublasZgetriBatched(Handle handle, int n, const cuDoubleComplex** A, int lda, const int* P, cuDoubleComplex** C, int ldc, int* info, int batchSize)
    Status cublasStpttr(Handle handle, FillMode uplo, int n, const float* AP, float* A, int lda)
    Status cublasDtpttr(Handle handle, FillMode uplo, int n, const double* AP, double* A, int lda)
    Status cublasCtpttr(Handle handle, FillMode uplo, int n, const cuComplex* AP, cuComplex* A, int lda)
    Status cublasZtpttr(Handle handle, FillMode uplo, int n, const cuDoubleComplex* AP, cuDoubleComplex* A, int lda)
    Status cublasStrttp(Handle handle, FillMode uplo, int n, const float* A, int lda, float* AP)
    Status cublasDtrttp(Handle handle, FillMode uplo, int n, const double* A, int lda, double* AP)
    Status cublasCtrttp(Handle handle, FillMode uplo, int n, const cuComplex* A, int lda, cuComplex* AP)
    Status cublasZtrttp(Handle handle, FillMode uplo, int n, const cuDoubleComplex* A, int lda, cuDoubleComplex* AP)

    # Define `cublasGemmEx` by hands for a backward compatibility reason.
    int cublasGemmEx(
        Handle handle, Operation transa, Operation transb,
        int m, int n, int k,
        const void *alpha,
        const void *A, runtime.DataType Atype, int lda,
        const void *B, runtime.DataType Btype, int ldb,
        const void *beta,
        void *C, runtime.DataType Ctype, int ldc,
        runtime.DataType computetype, GemmAlgo algo)
    int cublasGemmEx_v11(
        Handle handle, Operation transa, Operation transb,
        int m, int n, int k,
        const void *alpha,
        const void *A, runtime.DataType Atype, int lda,
        const void *B, runtime.DataType Btype, int ldb,
        const void *beta,
        void *C, runtime.DataType Ctype, int ldc,
        ComputeType computetype, GemmAlgo algo)


###############################################################################
# Util
###############################################################################

cdef cuComplex get_cu_complex(float complex a):
    cdef cuComplex ret
    ret.x = a.real
    ret.y = a.imag
    return ret


cdef cuDoubleComplex get_cu_double_complex(double complex a):
    cdef cuDoubleComplex ret
    ret.x = a.real
    ret.y = a.imag
    return ret


###############################################################################
# Error handling
###############################################################################

cdef dict STATUS = {
    0: 'CUBLAS_STATUS_SUCCESS',
    1: 'CUBLAS_STATUS_NOT_INITIALIZED',
    3: 'CUBLAS_STATUS_ALLOC_FAILED',
    7: 'CUBLAS_STATUS_INVALID_VALUE',
    8: 'CUBLAS_STATUS_ARCH_MISMATCH',
    11: 'CUBLAS_STATUS_MAPPING_ERROR',
    13: 'CUBLAS_STATUS_EXECUTION_FAILED',
    14: 'CUBLAS_STATUS_INTERNAL_ERROR',
    15: 'CUBLAS_STATUS_NOT_SUPPORTED',
    16: 'CUBLAS_STATUS_LICENSE_ERROR',
}


cdef dict HIP_STATUS = {
    0: 'HIPBLAS_STATUS_SUCCESS',
    1: 'HIPBLAS_STATUS_NOT_INITIALIZED',
    2: 'HIPBLAS_STATUS_ALLOC_FAILED',
    3: 'HIPBLAS_STATUS_INVALID_VALUE',
    4: 'HIPBLAS_STATUS_MAPPING_ERROR',
    5: 'HIPBLAS_STATUS_EXECUTION_FAILED',
    6: 'HIPBLAS_STATUS_INTERNAL_ERROR',
    7: 'HIPBLAS_STATUS_NOT_SUPPORTED',
    8: 'HIPBLAS_STATUS_ARCH_MISMATCH',
    9: 'HIPBLAS_STATUS_HANDLE_IS_NULLPTR',
}


class CUBLASError(RuntimeError):

    def __init__(self, status):
        self.status = status
        cdef str err
        if runtime._is_hip_environment:
            err = HIP_STATUS[status]
        else:
            err = STATUS[status]
        super(CUBLASError, self).__init__(err)

    def __reduce__(self):
        return (type(self), (self.status,))


@cython.profile(False)
cpdef inline check_status(int status):
    if status != 0:
        raise CUBLASError(status)


########################################
# Auxiliary structures




########################################
# cuBLAS Helper Function

cpdef intptr_t create() except? 0:
    cdef Handle handle
    status = cublasCreate_v2(&handle)
    check_status(status)
    return <intptr_t>handle

cpdef destroy(intptr_t handle):
    status = cublasDestroy_v2(<Handle>handle)
    check_status(status)

cpdef int getVersion(intptr_t handle) except? -1:
    cdef int version
    status = cublasGetVersion_v2(<Handle>handle, &version)
    check_status(status)
    return version

cpdef int getPointerMode(intptr_t handle) except? 0:
    cdef PointerMode mode
    status = cublasGetPointerMode_v2(<Handle>handle, &mode)
    check_status(status)
    return <int>mode

cpdef setPointerMode(intptr_t handle, int mode):
    status = cublasSetPointerMode_v2(<Handle>handle, <PointerMode>mode)
    check_status(status)

cpdef setStream(intptr_t handle, size_t streamId):
    status = cublasSetStream_v2(<Handle>handle, <driver.Stream>streamId)
    check_status(status)

cpdef size_t getStream(intptr_t handle) except? 0:
    cdef driver.Stream streamId
    status = cublasGetStream_v2(<Handle>handle, &streamId)
    check_status(status)
    return <size_t>streamId

cpdef setMathMode(intptr_t handle, int mode):
    status = cublasSetMathMode(<Handle>handle, <Math>mode)
    check_status(status)

cpdef int getMathMode(intptr_t handle) except? -1:
    cdef Math mode
    status = cublasGetMathMode(<Handle>handle, &mode)
    check_status(status)
    return <int>mode


########################################
# cuBLAS Level-1 Function

cpdef isamax(intptr_t handle, int n, intptr_t x, int incx, intptr_t result):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasIsamax_v2(<Handle>handle, n, <const float*>x, incx, <int*>result)
    check_status(status)

cpdef idamax(intptr_t handle, int n, intptr_t x, int incx, intptr_t result):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasIdamax_v2(<Handle>handle, n, <const double*>x, incx, <int*>result)
    check_status(status)

cpdef icamax(intptr_t handle, int n, intptr_t x, int incx, intptr_t result):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasIcamax_v2(<Handle>handle, n, <const cuComplex*>x, incx, <int*>result)
    check_status(status)

cpdef izamax(intptr_t handle, int n, intptr_t x, int incx, intptr_t result):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasIzamax_v2(<Handle>handle, n, <const cuDoubleComplex*>x, incx, <int*>result)
    check_status(status)

cpdef isamin(intptr_t handle, int n, intptr_t x, int incx, intptr_t result):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasIsamin_v2(<Handle>handle, n, <const float*>x, incx, <int*>result)
    check_status(status)

cpdef idamin(intptr_t handle, int n, intptr_t x, int incx, intptr_t result):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasIdamin_v2(<Handle>handle, n, <const double*>x, incx, <int*>result)
    check_status(status)

cpdef icamin(intptr_t handle, int n, intptr_t x, int incx, intptr_t result):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasIcamin_v2(<Handle>handle, n, <const cuComplex*>x, incx, <int*>result)
    check_status(status)

cpdef izamin(intptr_t handle, int n, intptr_t x, int incx, intptr_t result):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasIzamin_v2(<Handle>handle, n, <const cuDoubleComplex*>x, incx, <int*>result)
    check_status(status)

cpdef sasum(intptr_t handle, int n, intptr_t x, int incx, intptr_t result):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasSasum_v2(<Handle>handle, n, <const float*>x, incx, <float*>result)
    check_status(status)

cpdef dasum(intptr_t handle, int n, intptr_t x, int incx, intptr_t result):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasDasum_v2(<Handle>handle, n, <const double*>x, incx, <double*>result)
    check_status(status)

cpdef scasum(intptr_t handle, int n, intptr_t x, int incx, intptr_t result):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasScasum_v2(<Handle>handle, n, <const cuComplex*>x, incx, <float*>result)
    check_status(status)

cpdef dzasum(intptr_t handle, int n, intptr_t x, int incx, intptr_t result):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasDzasum_v2(<Handle>handle, n, <const cuDoubleComplex*>x, incx, <double*>result)
    check_status(status)

cpdef saxpy(intptr_t handle, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasSaxpy_v2(<Handle>handle, n, <const float*>alpha, <const float*>x, incx, <float*>y, incy)
    check_status(status)

cpdef daxpy(intptr_t handle, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasDaxpy_v2(<Handle>handle, n, <const double*>alpha, <const double*>x, incx, <double*>y, incy)
    check_status(status)

cpdef caxpy(intptr_t handle, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasCaxpy_v2(<Handle>handle, n, <const cuComplex*>alpha, <const cuComplex*>x, incx, <cuComplex*>y, incy)
    check_status(status)

cpdef zaxpy(intptr_t handle, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasZaxpy_v2(<Handle>handle, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>x, incx, <cuDoubleComplex*>y, incy)
    check_status(status)

cpdef sdot(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy, intptr_t result):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasSdot_v2(<Handle>handle, n, <const float*>x, incx, <const float*>y, incy, <float*>result)
    check_status(status)

cpdef ddot(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy, intptr_t result):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasDdot_v2(<Handle>handle, n, <const double*>x, incx, <const double*>y, incy, <double*>result)
    check_status(status)

cpdef cdotu(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy, intptr_t result):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasCdotu_v2(<Handle>handle, n, <const cuComplex*>x, incx, <const cuComplex*>y, incy, <cuComplex*>result)
    check_status(status)

cpdef cdotc(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy, intptr_t result):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasCdotc_v2(<Handle>handle, n, <const cuComplex*>x, incx, <const cuComplex*>y, incy, <cuComplex*>result)
    check_status(status)

cpdef zdotu(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy, intptr_t result):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasZdotu_v2(<Handle>handle, n, <const cuDoubleComplex*>x, incx, <const cuDoubleComplex*>y, incy, <cuDoubleComplex*>result)
    check_status(status)

cpdef zdotc(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy, intptr_t result):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasZdotc_v2(<Handle>handle, n, <const cuDoubleComplex*>x, incx, <const cuDoubleComplex*>y, incy, <cuDoubleComplex*>result)
    check_status(status)

cpdef snrm2(intptr_t handle, int n, intptr_t x, int incx, intptr_t result):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasSnrm2_v2(<Handle>handle, n, <const float*>x, incx, <float*>result)
    check_status(status)

cpdef dnrm2(intptr_t handle, int n, intptr_t x, int incx, intptr_t result):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasDnrm2_v2(<Handle>handle, n, <const double*>x, incx, <double*>result)
    check_status(status)

cpdef scnrm2(intptr_t handle, int n, intptr_t x, int incx, intptr_t result):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasScnrm2_v2(<Handle>handle, n, <const cuComplex*>x, incx, <float*>result)
    check_status(status)

cpdef dznrm2(intptr_t handle, int n, intptr_t x, int incx, intptr_t result):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasDznrm2_v2(<Handle>handle, n, <const cuDoubleComplex*>x, incx, <double*>result)
    check_status(status)

cpdef sscal(intptr_t handle, int n, intptr_t alpha, intptr_t x, int incx):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasSscal_v2(<Handle>handle, n, <const float*>alpha, <float*>x, incx)
    check_status(status)

cpdef dscal(intptr_t handle, int n, intptr_t alpha, intptr_t x, int incx):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasDscal_v2(<Handle>handle, n, <const double*>alpha, <double*>x, incx)
    check_status(status)

cpdef cscal(intptr_t handle, int n, intptr_t alpha, intptr_t x, int incx):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasCscal_v2(<Handle>handle, n, <const cuComplex*>alpha, <cuComplex*>x, incx)
    check_status(status)

cpdef csscal(intptr_t handle, int n, intptr_t alpha, intptr_t x, int incx):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasCsscal_v2(<Handle>handle, n, <const float*>alpha, <cuComplex*>x, incx)
    check_status(status)

cpdef zscal(intptr_t handle, int n, intptr_t alpha, intptr_t x, int incx):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasZscal_v2(<Handle>handle, n, <const cuDoubleComplex*>alpha, <cuDoubleComplex*>x, incx)
    check_status(status)

cpdef zdscal(intptr_t handle, int n, intptr_t alpha, intptr_t x, int incx):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasZdscal_v2(<Handle>handle, n, <const double*>alpha, <cuDoubleComplex*>x, incx)
    check_status(status)


########################################
# cuBLAS Level-2 Function

cpdef sgemv(intptr_t handle, int trans, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasSgemv_v2(<Handle>handle, <Operation>trans, m, n, <const float*>alpha, <const float*>A, lda, <const float*>x, incx, <const float*>beta, <float*>y, incy)
    check_status(status)

cpdef dgemv(intptr_t handle, int trans, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasDgemv_v2(<Handle>handle, <Operation>trans, m, n, <const double*>alpha, <const double*>A, lda, <const double*>x, incx, <const double*>beta, <double*>y, incy)
    check_status(status)

cpdef cgemv(intptr_t handle, int trans, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasCgemv_v2(<Handle>handle, <Operation>trans, m, n, <const cuComplex*>alpha, <const cuComplex*>A, lda, <const cuComplex*>x, incx, <const cuComplex*>beta, <cuComplex*>y, incy)
    check_status(status)

cpdef zgemv(intptr_t handle, int trans, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasZgemv_v2(<Handle>handle, <Operation>trans, m, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>A, lda, <const cuDoubleComplex*>x, incx, <const cuDoubleComplex*>beta, <cuDoubleComplex*>y, incy)
    check_status(status)

cpdef sger(intptr_t handle, int m, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t A, int lda):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasSger_v2(<Handle>handle, m, n, <const float*>alpha, <const float*>x, incx, <const float*>y, incy, <float*>A, lda)
    check_status(status)

cpdef dger(intptr_t handle, int m, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t A, int lda):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasDger_v2(<Handle>handle, m, n, <const double*>alpha, <const double*>x, incx, <const double*>y, incy, <double*>A, lda)
    check_status(status)

cpdef cgeru(intptr_t handle, int m, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t A, int lda):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasCgeru_v2(<Handle>handle, m, n, <const cuComplex*>alpha, <const cuComplex*>x, incx, <const cuComplex*>y, incy, <cuComplex*>A, lda)
    check_status(status)

cpdef cgerc(intptr_t handle, int m, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t A, int lda):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasCgerc_v2(<Handle>handle, m, n, <const cuComplex*>alpha, <const cuComplex*>x, incx, <const cuComplex*>y, incy, <cuComplex*>A, lda)
    check_status(status)

cpdef zgeru(intptr_t handle, int m, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t A, int lda):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasZgeru_v2(<Handle>handle, m, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>x, incx, <const cuDoubleComplex*>y, incy, <cuDoubleComplex*>A, lda)
    check_status(status)

cpdef zgerc(intptr_t handle, int m, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t A, int lda):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasZgerc_v2(<Handle>handle, m, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>x, incx, <const cuDoubleComplex*>y, incy, <cuDoubleComplex*>A, lda)
    check_status(status)


########################################
# cuBLAS Level-3 Function

cpdef sgemm(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasSgemm_v2(<Handle>handle, <Operation>transa, <Operation>transb, m, n, k, <const float*>alpha, <const float*>A, lda, <const float*>B, ldb, <const float*>beta, <float*>C, ldc)
    check_status(status)

cpdef dgemm(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasDgemm_v2(<Handle>handle, <Operation>transa, <Operation>transb, m, n, k, <const double*>alpha, <const double*>A, lda, <const double*>B, ldb, <const double*>beta, <double*>C, ldc)
    check_status(status)

cpdef cgemm(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasCgemm_v2(<Handle>handle, <Operation>transa, <Operation>transb, m, n, k, <const cuComplex*>alpha, <const cuComplex*>A, lda, <const cuComplex*>B, ldb, <const cuComplex*>beta, <cuComplex*>C, ldc)
    check_status(status)

cpdef zgemm(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasZgemm_v2(<Handle>handle, <Operation>transa, <Operation>transb, m, n, k, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>A, lda, <const cuDoubleComplex*>B, ldb, <const cuDoubleComplex*>beta, <cuDoubleComplex*>C, ldc)
    check_status(status)

cpdef sgemmBatched(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t Aarray, int lda, intptr_t Barray, int ldb, intptr_t beta, intptr_t Carray, int ldc, int batchCount):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasSgemmBatched(<Handle>handle, <Operation>transa, <Operation>transb, m, n, k, <const float*>alpha, <const float**>Aarray, lda, <const float**>Barray, ldb, <const float*>beta, <float**>Carray, ldc, batchCount)
    check_status(status)

cpdef dgemmBatched(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t Aarray, int lda, intptr_t Barray, int ldb, intptr_t beta, intptr_t Carray, int ldc, int batchCount):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasDgemmBatched(<Handle>handle, <Operation>transa, <Operation>transb, m, n, k, <const double*>alpha, <const double**>Aarray, lda, <const double**>Barray, ldb, <const double*>beta, <double**>Carray, ldc, batchCount)
    check_status(status)

cpdef cgemmBatched(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t Aarray, int lda, intptr_t Barray, int ldb, intptr_t beta, intptr_t Carray, int ldc, int batchCount):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasCgemmBatched(<Handle>handle, <Operation>transa, <Operation>transb, m, n, k, <const cuComplex*>alpha, <const cuComplex**>Aarray, lda, <const cuComplex**>Barray, ldb, <const cuComplex*>beta, <cuComplex**>Carray, ldc, batchCount)
    check_status(status)

cpdef zgemmBatched(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t Aarray, int lda, intptr_t Barray, int ldb, intptr_t beta, intptr_t Carray, int ldc, int batchCount):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasZgemmBatched(<Handle>handle, <Operation>transa, <Operation>transb, m, n, k, <const cuDoubleComplex*>alpha, <const cuDoubleComplex**>Aarray, lda, <const cuDoubleComplex**>Barray, ldb, <const cuDoubleComplex*>beta, <cuDoubleComplex**>Carray, ldc, batchCount)
    check_status(status)

cpdef sgemmStridedBatched(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t A, int lda, long long int strideA, intptr_t B, int ldb, long long int strideB, intptr_t beta, intptr_t C, int ldc, long long int strideC, int batchCount):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasSgemmStridedBatched(<Handle>handle, <Operation>transa, <Operation>transb, m, n, k, <const float*>alpha, <const float*>A, lda, strideA, <const float*>B, ldb, strideB, <const float*>beta, <float*>C, ldc, strideC, batchCount)
    check_status(status)

cpdef dgemmStridedBatched(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t A, int lda, long long int strideA, intptr_t B, int ldb, long long int strideB, intptr_t beta, intptr_t C, int ldc, long long int strideC, int batchCount):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasDgemmStridedBatched(<Handle>handle, <Operation>transa, <Operation>transb, m, n, k, <const double*>alpha, <const double*>A, lda, strideA, <const double*>B, ldb, strideB, <const double*>beta, <double*>C, ldc, strideC, batchCount)
    check_status(status)

cpdef cgemmStridedBatched(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t A, int lda, long long int strideA, intptr_t B, int ldb, long long int strideB, intptr_t beta, intptr_t C, int ldc, long long int strideC, int batchCount):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasCgemmStridedBatched(<Handle>handle, <Operation>transa, <Operation>transb, m, n, k, <const cuComplex*>alpha, <const cuComplex*>A, lda, strideA, <const cuComplex*>B, ldb, strideB, <const cuComplex*>beta, <cuComplex*>C, ldc, strideC, batchCount)
    check_status(status)

cpdef zgemmStridedBatched(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t A, int lda, long long int strideA, intptr_t B, int ldb, long long int strideB, intptr_t beta, intptr_t C, int ldc, long long int strideC, int batchCount):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasZgemmStridedBatched(<Handle>handle, <Operation>transa, <Operation>transb, m, n, k, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>A, lda, strideA, <const cuDoubleComplex*>B, ldb, strideB, <const cuDoubleComplex*>beta, <cuDoubleComplex*>C, ldc, strideC, batchCount)
    check_status(status)

cpdef strsm(intptr_t handle, int side, int uplo, int trans, int diag, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasStrsm_v2(<Handle>handle, <SideMode>side, <FillMode>uplo, <Operation>trans, <DiagType>diag, m, n, <const float*>alpha, <const float*>A, lda, <float*>B, ldb)
    check_status(status)

cpdef dtrsm(intptr_t handle, int side, int uplo, int trans, int diag, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasDtrsm_v2(<Handle>handle, <SideMode>side, <FillMode>uplo, <Operation>trans, <DiagType>diag, m, n, <const double*>alpha, <const double*>A, lda, <double*>B, ldb)
    check_status(status)

cpdef ctrsm(intptr_t handle, int side, int uplo, int trans, int diag, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasCtrsm_v2(<Handle>handle, <SideMode>side, <FillMode>uplo, <Operation>trans, <DiagType>diag, m, n, <const cuComplex*>alpha, <const cuComplex*>A, lda, <cuComplex*>B, ldb)
    check_status(status)

cpdef ztrsm(intptr_t handle, int side, int uplo, int trans, int diag, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasZtrsm_v2(<Handle>handle, <SideMode>side, <FillMode>uplo, <Operation>trans, <DiagType>diag, m, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>A, lda, <cuDoubleComplex*>B, ldb)
    check_status(status)


########################################
# cuBLAS BLAS-like Extension

cpdef sgeam(intptr_t handle, int transa, int transb, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t beta, intptr_t B, int ldb, intptr_t C, int ldc):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasSgeam(<Handle>handle, <Operation>transa, <Operation>transb, m, n, <const float*>alpha, <const float*>A, lda, <const float*>beta, <const float*>B, ldb, <float*>C, ldc)
    check_status(status)

cpdef dgeam(intptr_t handle, int transa, int transb, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t beta, intptr_t B, int ldb, intptr_t C, int ldc):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasDgeam(<Handle>handle, <Operation>transa, <Operation>transb, m, n, <const double*>alpha, <const double*>A, lda, <const double*>beta, <const double*>B, ldb, <double*>C, ldc)
    check_status(status)

cpdef cgeam(intptr_t handle, int transa, int transb, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t beta, intptr_t B, int ldb, intptr_t C, int ldc):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasCgeam(<Handle>handle, <Operation>transa, <Operation>transb, m, n, <const cuComplex*>alpha, <const cuComplex*>A, lda, <const cuComplex*>beta, <const cuComplex*>B, ldb, <cuComplex*>C, ldc)
    check_status(status)

cpdef zgeam(intptr_t handle, int transa, int transb, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t beta, intptr_t B, int ldb, intptr_t C, int ldc):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasZgeam(<Handle>handle, <Operation>transa, <Operation>transb, m, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>A, lda, <const cuDoubleComplex*>beta, <const cuDoubleComplex*>B, ldb, <cuDoubleComplex*>C, ldc)
    check_status(status)

cpdef sdgmm(intptr_t handle, int mode, int m, int n, intptr_t A, int lda, intptr_t x, int incx, intptr_t C, int ldc):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasSdgmm(<Handle>handle, <SideMode>mode, m, n, <const float*>A, lda, <const float*>x, incx, <float*>C, ldc)
    check_status(status)

cpdef ddgmm(intptr_t handle, int mode, int m, int n, intptr_t A, int lda, intptr_t x, int incx, intptr_t C, int ldc):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasDdgmm(<Handle>handle, <SideMode>mode, m, n, <const double*>A, lda, <const double*>x, incx, <double*>C, ldc)
    check_status(status)

cpdef cdgmm(intptr_t handle, int mode, int m, int n, intptr_t A, int lda, intptr_t x, int incx, intptr_t C, int ldc):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasCdgmm(<Handle>handle, <SideMode>mode, m, n, <const cuComplex*>A, lda, <const cuComplex*>x, incx, <cuComplex*>C, ldc)
    check_status(status)

cpdef zdgmm(intptr_t handle, int mode, int m, int n, intptr_t A, int lda, intptr_t x, int incx, intptr_t C, int ldc):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasZdgmm(<Handle>handle, <SideMode>mode, m, n, <const cuDoubleComplex*>A, lda, <const cuDoubleComplex*>x, incx, <cuDoubleComplex*>C, ldc)
    check_status(status)

cpdef sgemmEx(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t A, size_t Atype, int lda, intptr_t B, size_t Btype, int ldb, intptr_t beta, intptr_t C, size_t Ctype, int ldc):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasSgemmEx(<Handle>handle, <Operation>transa, <Operation>transb, m, n, k, <const float*>alpha, <const void*>A, <DataType>Atype, lda, <const void*>B, <DataType>Btype, ldb, <const float*>beta, <void*>C, <DataType>Ctype, ldc)
    check_status(status)

cpdef cgemmEx(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t A, size_t Atype, int lda, intptr_t B, size_t Btype, int ldb, intptr_t beta, intptr_t C, size_t Ctype, int ldc):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasCgemmEx(<Handle>handle, <Operation>transa, <Operation>transb, m, n, k, <const cuComplex*>alpha, <const void*>A, <DataType>Atype, lda, <const void*>B, <DataType>Btype, ldb, <const cuComplex*>beta, <void*>C, <DataType>Ctype, ldc)
    check_status(status)

cpdef sgetrfBatched(intptr_t handle, int n, intptr_t A, int lda, intptr_t P, intptr_t info, int batchSize):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasSgetrfBatched(<Handle>handle, n, <float**>A, lda, <int*>P, <int*>info, batchSize)
    check_status(status)

cpdef dgetrfBatched(intptr_t handle, int n, intptr_t A, int lda, intptr_t P, intptr_t info, int batchSize):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasDgetrfBatched(<Handle>handle, n, <double**>A, lda, <int*>P, <int*>info, batchSize)
    check_status(status)

cpdef cgetrfBatched(intptr_t handle, int n, intptr_t A, int lda, intptr_t P, intptr_t info, int batchSize):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasCgetrfBatched(<Handle>handle, n, <cuComplex**>A, lda, <int*>P, <int*>info, batchSize)
    check_status(status)

cpdef zgetrfBatched(intptr_t handle, int n, intptr_t A, int lda, intptr_t P, intptr_t info, int batchSize):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasZgetrfBatched(<Handle>handle, n, <cuDoubleComplex**>A, lda, <int*>P, <int*>info, batchSize)
    check_status(status)

cpdef sgetrsBatched(intptr_t handle, int trans, int n, int nrhs, intptr_t Aarray, int lda, intptr_t devIpiv, intptr_t Barray, int ldb, intptr_t info, int batchSize):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasSgetrsBatched(<Handle>handle, <Operation>trans, n, nrhs, <const float**>Aarray, lda, <const int*>devIpiv, <float**>Barray, ldb, <int*>info, batchSize)
    check_status(status)

cpdef dgetrsBatched(intptr_t handle, int trans, int n, int nrhs, intptr_t Aarray, int lda, intptr_t devIpiv, intptr_t Barray, int ldb, intptr_t info, int batchSize):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasDgetrsBatched(<Handle>handle, <Operation>trans, n, nrhs, <const double**>Aarray, lda, <const int*>devIpiv, <double**>Barray, ldb, <int*>info, batchSize)
    check_status(status)

cpdef cgetrsBatched(intptr_t handle, int trans, int n, int nrhs, intptr_t Aarray, int lda, intptr_t devIpiv, intptr_t Barray, int ldb, intptr_t info, int batchSize):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasCgetrsBatched(<Handle>handle, <Operation>trans, n, nrhs, <const cuComplex**>Aarray, lda, <const int*>devIpiv, <cuComplex**>Barray, ldb, <int*>info, batchSize)
    check_status(status)

cpdef zgetrsBatched(intptr_t handle, int trans, int n, int nrhs, intptr_t Aarray, int lda, intptr_t devIpiv, intptr_t Barray, int ldb, intptr_t info, int batchSize):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasZgetrsBatched(<Handle>handle, <Operation>trans, n, nrhs, <const cuDoubleComplex**>Aarray, lda, <const int*>devIpiv, <cuDoubleComplex**>Barray, ldb, <int*>info, batchSize)
    check_status(status)

cpdef sgetriBatched(intptr_t handle, int n, intptr_t A, int lda, intptr_t P, intptr_t C, int ldc, intptr_t info, int batchSize):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasSgetriBatched(<Handle>handle, n, <const float**>A, lda, <const int*>P, <float**>C, ldc, <int*>info, batchSize)
    check_status(status)

cpdef dgetriBatched(intptr_t handle, int n, intptr_t A, int lda, intptr_t P, intptr_t C, int ldc, intptr_t info, int batchSize):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasDgetriBatched(<Handle>handle, n, <const double**>A, lda, <const int*>P, <double**>C, ldc, <int*>info, batchSize)
    check_status(status)

cpdef cgetriBatched(intptr_t handle, int n, intptr_t A, int lda, intptr_t P, intptr_t C, int ldc, intptr_t info, int batchSize):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasCgetriBatched(<Handle>handle, n, <const cuComplex**>A, lda, <const int*>P, <cuComplex**>C, ldc, <int*>info, batchSize)
    check_status(status)

cpdef zgetriBatched(intptr_t handle, int n, intptr_t A, int lda, intptr_t P, intptr_t C, int ldc, intptr_t info, int batchSize):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasZgetriBatched(<Handle>handle, n, <const cuDoubleComplex**>A, lda, <const int*>P, <cuDoubleComplex**>C, ldc, <int*>info, batchSize)
    check_status(status)

cpdef stpttr(intptr_t handle, int uplo, int n, intptr_t AP, intptr_t A, int lda):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasStpttr(<Handle>handle, <FillMode>uplo, n, <const float*>AP, <float*>A, lda)
    check_status(status)

cpdef dtpttr(intptr_t handle, int uplo, int n, intptr_t AP, intptr_t A, int lda):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasDtpttr(<Handle>handle, <FillMode>uplo, n, <const double*>AP, <double*>A, lda)
    check_status(status)

cpdef ctpttr(intptr_t handle, int uplo, int n, intptr_t AP, intptr_t A, int lda):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasCtpttr(<Handle>handle, <FillMode>uplo, n, <const cuComplex*>AP, <cuComplex*>A, lda)
    check_status(status)

cpdef ztpttr(intptr_t handle, int uplo, int n, intptr_t AP, intptr_t A, int lda):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasZtpttr(<Handle>handle, <FillMode>uplo, n, <const cuDoubleComplex*>AP, <cuDoubleComplex*>A, lda)
    check_status(status)

cpdef strttp(intptr_t handle, int uplo, int n, intptr_t A, int lda, intptr_t AP):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasStrttp(<Handle>handle, <FillMode>uplo, n, <const float*>A, lda, <float*>AP)
    check_status(status)

cpdef dtrttp(intptr_t handle, int uplo, int n, intptr_t A, int lda, intptr_t AP):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasDtrttp(<Handle>handle, <FillMode>uplo, n, <const double*>A, lda, <double*>AP)
    check_status(status)

cpdef ctrttp(intptr_t handle, int uplo, int n, intptr_t A, int lda, intptr_t AP):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasCtrttp(<Handle>handle, <FillMode>uplo, n, <const cuComplex*>A, lda, <cuComplex*>AP)
    check_status(status)

cpdef ztrttp(intptr_t handle, int uplo, int n, intptr_t A, int lda, intptr_t AP):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasZtrttp(<Handle>handle, <FillMode>uplo, n, <const cuDoubleComplex*>A, lda, <cuDoubleComplex*>AP)
    check_status(status)

# Define `gemmEx` by hands for a backward compatibility reason.	
cpdef gemmEx(
        intptr_t handle, int transa, int transb, int m, int n, int k,
        size_t alpha, size_t A, int Atype, int lda, size_t B,
        int Btype, int ldb, size_t beta, size_t C, int Ctype,
        int ldc, int computeType, int algo):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        if computeType >= CUBLAS_COMPUTE_16F:
            status = cublasGemmEx_v11(
                <Handle>handle, <Operation>transa, <Operation>transb, m, n, k,
                <const void*>alpha,
                <const void*>A, <runtime.DataType>Atype, lda,
                <const void*>B, <runtime.DataType>Btype, ldb,
                <const void*>beta,
                <void*>C, <runtime.DataType>Ctype, ldc,
                <ComputeType>computeType, <GemmAlgo>algo)
        else:
            status = cublasGemmEx(
                <Handle>handle, <Operation>transa, <Operation>transb, m, n, k,
                <const void*>alpha,
                <const void*>A, <runtime.DataType>Atype, lda,
                <const void*>B, <runtime.DataType>Btype, ldb,
                <const void*>beta,
                <void*>C, <runtime.DataType>Ctype, ldc,
                <runtime.DataType>computeType, <GemmAlgo>algo)
    check_status(status)

