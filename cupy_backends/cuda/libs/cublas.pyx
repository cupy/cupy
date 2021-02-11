# distutils: language = c++

"""Thin wrapper of CUBLAS."""

cimport cython  # NOQA

from cupy_backends.cuda.api cimport driver
from cupy_backends.cuda.api cimport runtime
from cupy_backends.cuda cimport stream as stream_module

###############################################################################
# Extern
###############################################################################

cdef extern from '../../cupy_complex.h':
    ctypedef struct cuComplex 'cuComplex':
        float x, y

    ctypedef struct cuDoubleComplex 'cuDoubleComplex':
        double x, y

cdef extern from '../../cupy_blas.h' nogil:
    # Context
    int cublasCreate(Handle* handle)
    int cublasDestroy(Handle handle)
    int cublasGetVersion(Handle handle, int* version)
    int cublasGetPointerMode(Handle handle, PointerMode* mode)
    int cublasSetPointerMode(Handle handle, PointerMode mode)

    # Stream
    int cublasSetStream(Handle handle, driver.Stream streamId)
    int cublasGetStream(Handle handle, driver.Stream* streamId)

    # Math Mode
    int cublasSetMathMode(Handle handle, Math mode)
    int cublasGetMathMode(Handle handle, Math* mode)

    # BLAS Level 1
    int cublasIsamax(Handle handle, int n, float* x, int incx,
                     int* result)
    int cublasIdamax(Handle handle, int n, double* x, int incx,
                     int* result)
    int cublasIcamax(Handle handle, int n, cuComplex* x, int incx,
                     int* result)
    int cublasIzamax(Handle handle, int n, cuDoubleComplex* x, int incx,
                     int* result)
    int cublasIsamin(Handle handle, int n, float* x, int incx,
                     int* result)
    int cublasIdamin(Handle handle, int n, double* x, int incx,
                     int* result)
    int cublasIcamin(Handle handle, int n, cuComplex* x, int incx,
                     int* result)
    int cublasIzamin(Handle handle, int n, cuDoubleComplex* x, int incx,
                     int* result)
    int cublasSasum(Handle handle, int n, float* x, int incx,
                    float* result)
    int cublasDasum(Handle handle, int n, double* x, int incx,
                    double* result)
    int cublasScasum(Handle handle, int n, cuComplex* x, int incx,
                     float* result)
    int cublasDzasum(Handle handle, int n, cuDoubleComplex* x, int incx,
                     double* result)
    int cublasSaxpy(Handle handle, int n, float* alpha, float* x,
                    int incx, float* y, int incy)
    int cublasDaxpy(Handle handle, int n, double* alpha, double* x,
                    int incx, double* y, int incy)
    int cublasCaxpy(Handle handle, int n, cuComplex* alpha, cuComplex* x,
                    int incx, cuComplex* y, int incy)
    int cublasZaxpy(Handle handle, int n, cuDoubleComplex* alpha,
                    cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy)
    int cublasSdot(Handle handle, int n, float* x, int incx,
                   float* y, int incy, float* result)
    int cublasDdot(Handle handle, int n, double* x, int incx,
                   double* y, int incy, double* result)
    int cublasCdotu(Handle handle, int n, cuComplex* x, int incx,
                    cuComplex* y, int incy, cuComplex* result)
    int cublasCdotc(Handle handle, int n, cuComplex* x, int incx,
                    cuComplex* y, int incy, cuComplex* result)
    int cublasZdotu(Handle handle, int n, cuDoubleComplex* x, int incx,
                    cuDoubleComplex* y, int incy,
                    cuDoubleComplex* result)
    int cublasZdotc(Handle handle, int n, cuDoubleComplex* x, int incx,
                    cuDoubleComplex* y, int incy,
                    cuDoubleComplex* result)
    int cublasSnrm2(Handle handle, int n, float* x, int incx, float* result)
    int cublasDnrm2(Handle handle, int n, double* x, int incx, double* result)
    int cublasScnrm2(Handle handle, int n, cuComplex* x, int incx,
                     float* result)
    int cublasDznrm2(Handle handle, int n, cuDoubleComplex* x, int incx,
                     double* result)
    int cublasSscal(Handle handle, int n, float* alpha, float* x, int incx)
    int cublasDscal(Handle handle, int n, double* alpha, double* x, int incx)
    int cublasCscal(Handle handle, int n, cuComplex* alpha,
                    cuComplex* x, int incx)
    int cublasCsscal(Handle handle, int n, float* alpha,
                     cuComplex* x, int incx)
    int cublasZscal(Handle handle, int n, cuDoubleComplex* alpha,
                    cuDoubleComplex* x, int incx)
    int cublasZdscal(Handle handle, int n, double* alpha,
                     cuDoubleComplex* x, int incx)

    # BLAS Level 2
    int cublasSgemv(
        Handle handle, Operation trans, int m, int n, float* alpha,
        float* A, int lda, float* x, int incx, float* beta,
        float* y, int incy)
    int cublasDgemv(
        Handle handle, Operation trans, int m, int n, double* alpha,
        double* A, int lda, double* x, int incx, double* beta,
        double* y, int incy)
    int cublasCgemv(
        Handle handle, Operation trans, int m, int n, cuComplex* alpha,
        cuComplex* A, int lda, cuComplex* x, int incx, cuComplex* beta,
        cuComplex* y, int incy)
    int cublasZgemv(
        Handle handle, Operation trans, int m, int n, cuDoubleComplex* alpha,
        cuDoubleComplex* A, int lda, cuDoubleComplex* x, int incx,
        cuDoubleComplex* beta, cuDoubleComplex* y, int incy)
    int cublasSger(
        Handle handle, int m, int n, float* alpha, float* x, int incx,
        float* y, int incy, float* A, int lda)
    int cublasDger(
        Handle handle, int m, int n, double* alpha, double* x,
        int incx, double* y, int incy, double* A, int lda)
    int cublasCgeru(
        Handle handle, int m, int n, cuComplex* alpha, cuComplex* x,
        int incx, cuComplex* y, int incy, cuComplex* A, int lda)
    int cublasCgerc(
        Handle handle, int m, int n, cuComplex* alpha, cuComplex* x,
        int incx, cuComplex* y, int incy, cuComplex* A, int lda)
    int cublasZgeru(
        Handle handle, int m, int n, cuDoubleComplex* alpha,
        cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy,
        cuDoubleComplex* A, int lda)
    int cublasZgerc(
        Handle handle, int m, int n, cuDoubleComplex* alpha,
        cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy,
        cuDoubleComplex* A, int lda)

    # BLAS Level 3
    int cublasSgemm(
        Handle handle, Operation transa, Operation transb, int m,
        int n, int k, float* alpha, float* A, int lda, float* B,
        int ldb, float* beta, float* C, int ldc)
    int cublasDgemm(
        Handle handle, Operation transa, Operation transb, int m,
        int n, int k, double* alpha, double* A, int lda, double* B,
        int ldb, double* beta, double* C, int ldc)
    int cublasCgemm(
        Handle handle, Operation transa, Operation transb, int m,
        int n, int k, cuComplex* alpha, cuComplex* A, int lda,
        cuComplex* B, int ldb, cuComplex* beta, cuComplex* C,
        int ldc)
    int cublasZgemm(
        Handle handle, Operation transa, Operation transb, int m,
        int n, int k, cuDoubleComplex* alpha, cuDoubleComplex* A, int lda,
        cuDoubleComplex* B, int ldb, cuDoubleComplex* beta,
        cuDoubleComplex* C, int ldc)
    int cublasSgemmBatched(
        Handle handle, Operation transa, Operation transb, int m,
        int n, int k, const float* alpha, const float** Aarray,
        int lda, const float** Barray, int ldb, const float* beta,
        float** Carray, int ldc, int batchCount)
    int cublasDgemmBatched(
        Handle handle, Operation transa, Operation transb, int m,
        int n, int k, const double* alpha, const double** Aarray,
        int lda, const double** Barray, int ldb, const double* beta,
        double** Carray, int ldc, int batchCount)
    int cublasCgemmBatched(
        Handle handle, Operation transa, Operation transb, int m,
        int n, int k, const cuComplex* alpha, const cuComplex** Aarray,
        int lda, const cuComplex** Barray, int ldb, const cuComplex* beta,
        cuComplex** Carray, int ldc, int batchCount)
    int cublasZgemmBatched(
        Handle handle, Operation transa, Operation transb, int m,
        int n, int k, const cuDoubleComplex* alpha,
        const cuDoubleComplex** Aarray, int lda,
        const cuDoubleComplex** Barray, int ldb,
        const cuDoubleComplex* beta, cuDoubleComplex** Carray, int ldc,
        int batchCount)
    int cublasSgemmStridedBatched(
        Handle handle, Operation transa, Operation transb,
        int m, int n, int k, const float* alpha,
        const float* A, int lda, long long strideA,
        const float* B, int ldb, long long strideB,
        const float* beta,
        float* C, int ldc, long long strideC, int batchCount)
    int cublasDgemmStridedBatched(
        Handle handle, Operation transa, Operation transb,
        int m, int n, int k, const double* alpha,
        const double* A, int lda, long long strideA,
        const double* B, int ldb, long long strideB,
        const double* beta,
        double* C, int ldc, long long strideC, int batchCount)
    int cublasCgemmStridedBatched(
        Handle handle, Operation transa, Operation transb,
        int m, int n, int k, const cuComplex* alpha,
        const cuComplex* A, int lda, long long strideA,
        const cuComplex* B, int ldb, long long strideB,
        const cuComplex* beta,
        cuComplex* C, int ldc, long long strideC, int batchCount)
    int cublasZgemmStridedBatched(
        Handle handle, Operation transa, Operation transb,
        int m, int n, int k, const cuDoubleComplex* alpha,
        const cuDoubleComplex* A, int lda, long long strideA,
        const cuDoubleComplex* B, int ldb, long long strideB,
        const cuDoubleComplex* beta,
        cuDoubleComplex* C, int ldc, long long strideC, int batchCount)
    int cublasStrsm(
        Handle handle, SideMode size, FillMode uplo, Operation trans,
        DiagType diag, int m, int n, const float* alpha,
        const float* A, int lda, float* B, int ldb)
    int cublasDtrsm(
        Handle handle, SideMode size, FillMode uplo, Operation trans,
        DiagType diag, int m, int n, const double* alpha,
        const double* A, int lda, double* B, int ldb)
    int cublasCtrsm(
        Handle handle, SideMode size, FillMode uplo, Operation trans,
        DiagType diag, int m, int n, const cuComplex* alpha,
        const cuComplex* A, int lda, cuComplex* B, int ldb)
    int cublasZtrsm(
        Handle handle, SideMode size, FillMode uplo, Operation trans,
        DiagType diag, int m, int n, const cuDoubleComplex* alpha,
        const cuDoubleComplex* A, int lda, cuDoubleComplex* B, int ldb)

    # BLAS extension
    int cublasSgeam(
        Handle handle, Operation transa, Operation transb, int m, int n,
        const float* alpha, const float* A, int lda,
        const float* beta, const float* B, int ldb,
        float* C, int ldc)
    int cublasDgeam(
        Handle handle, Operation transa, Operation transb, int m, int n,
        const double* alpha, const double* A, int lda,
        const double* beta, const double* B, int ldb,
        double* C, int ldc)
    int cublasCgeam(
        Handle handle, Operation transa, Operation transb, int m, int n,
        const cuComplex* alpha, const cuComplex* A, int lda,
        const cuComplex* beta, const cuComplex* B, int ldb,
        cuComplex* C, int ldc)
    int cublasZgeam(
        Handle handle, Operation transa, Operation transb, int m, int n,
        const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda,
        const cuDoubleComplex* beta, const cuDoubleComplex* B, int ldb,
        cuDoubleComplex* C, int ldc)
    int cublasSdgmm(
        Handle handle, SideMode mode, int m, int n, const float* A, int lda,
        const float* x, int incx, float* C, int ldc)
    int cublasDdgmm(
        Handle handle, SideMode mode, int m, int n, const double* A, int lda,
        const double* x, int incx, double* C, int ldc)
    int cublasCdgmm(
        Handle handle, SideMode mode, int m, int n, const cuComplex* A,
        int lda, const cuComplex* x, int incx, cuComplex* C, int ldc)
    int cublasZdgmm(
        Handle handle, SideMode mode, int m, int n, const cuDoubleComplex* A,
        int lda, const cuDoubleComplex* x, int incx, cuDoubleComplex* C,
        int ldc)
    int cublasSgemmEx(
        Handle handle, Operation transa,
        Operation transb, int m, int n, int k,
        const float *alpha, const void *A, runtime.DataType Atype,
        int lda, const void *B, runtime.DataType Btype, int ldb,
        const float *beta, void *C, runtime.DataType Ctype, int ldc)
    int cublasSgetrfBatched(
        Handle handle, int n, float **Aarray, int lda,
        int *PivotArray, int *infoArray, int batchSize)
    int cublasDgetrfBatched(
        Handle handle, int n, double **Aarray, int lda,
        int *PivotArray, int *infoArray, int batchSize)
    int cublasCgetrfBatched(
        Handle handle, int n, cuComplex **Aarray, int lda,
        int *PivotArray, int *infoArray, int batchSize)
    int cublasZgetrfBatched(
        Handle handle, int n, cuDoubleComplex **Aarray, int lda,
        int *PivotArray, int *infoArray, int batchSize)

    int cublasSgetrsBatched(
        Handle handle, Operation trans, int n, int nrhs,
        const float **Aarray, int lda, const int *devIpiv,
        float **Barray, int ldb, int *info, int batchSize)
    int cublasDgetrsBatched(
        Handle handle, Operation trans, int n, int nrhs,
        const double **Aarray, int lda, const int *devIpiv,
        double **Barray, int ldb, int *info, int batchSize)
    int cublasCgetrsBatched(
        Handle handle, Operation trans, int n, int nrhs,
        const cuComplex **Aarray, int lda, const int *devIpiv,
        cuComplex **Barray, int ldb, int *info, int batchSize)
    int cublasZgetrsBatched(
        Handle handle, Operation trans, int n, int nrhs,
        const cuDoubleComplex **Aarray, int lda, const int *devIpiv,
        cuDoubleComplex **Barray, int ldb, int *info, int batchSize)

    int cublasSgetriBatched(
        Handle handle, int n, const float **Aarray, int lda,
        int *PivotArray, float *Carray[], int ldc, int *infoArray,
        int batchSize)
    int cublasDgetriBatched(
        Handle handle, int n, const double **Aarray, int lda,
        int *PivotArray, double *Carray[], int ldc, int *infoArray,
        int batchSize)
    int cublasCgetriBatched(
        Handle handle, int n, const cuComplex **Aarray, int lda,
        int *PivotArray, cuComplex *Carray[], int ldc, int *infoArray,
        int batchSize)
    int cublasZgetriBatched(
        Handle handle, int n, const cuDoubleComplex **Aarray, int lda,
        int *PivotArray, cuDoubleComplex *Carray[], int ldc, int *infoArray,
        int batchSize)
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
    int cublasStpttr(
        Handle handle, FillMode uplo, int n, const float *AP, float *A,
        int lda)
    int cublasDtpttr(
        Handle handle, FillMode uplo, int n, const double *AP, double *A,
        int lda)
    int cublasStrttp(
        Handle handle, FillMode uplo, int n, const float *A, int lda,
        float *AP)
    int cublasDtrttp(
        Handle handle, FillMode uplo, int n, const double *A, int lda,
        double *AP)


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


###############################################################################
# Context
###############################################################################

cpdef intptr_t create() except? 0:
    cdef Handle handle
    with nogil:
        status = cublasCreate(&handle)
    check_status(status)
    return <intptr_t>handle


cpdef destroy(intptr_t handle):
    with nogil:
        status = cublasDestroy(<Handle>handle)
    check_status(status)


cpdef int getVersion(intptr_t handle) except? -1:
    cdef int version
    with nogil:
        status = cublasGetVersion(<Handle>handle, &version)
    check_status(status)
    return version


cpdef int getPointerMode(intptr_t handle) except? -1:
    cdef PointerMode mode
    with nogil:
        status = cublasGetPointerMode(<Handle>handle, &mode)
    check_status(status)
    return mode


cpdef setPointerMode(intptr_t handle, int mode):
    with nogil:
        status = cublasSetPointerMode(<Handle>handle, <PointerMode>mode)
    check_status(status)


###############################################################################
# Stream
###############################################################################

cpdef setStream(intptr_t handle, size_t stream):
    with nogil:
        status = cublasSetStream(<Handle>handle, <driver.Stream>stream)
    check_status(status)


cpdef size_t getStream(intptr_t handle) except? 0:
    cdef driver.Stream stream
    with nogil:
        status = cublasGetStream(<Handle>handle, &stream)
    check_status(status)
    return <size_t>stream


cdef _setStream(intptr_t handle):
    """Set current stream"""
    setStream(handle, stream_module.get_current_stream_ptr())

###############################################################################
# Math Mode
###############################################################################

cpdef setMathMode(intptr_t handle, int mode):
    with nogil:
        status = cublasSetMathMode(<Handle>handle, <Math>mode)
    check_status(status)


cpdef int getMathMode(intptr_t handle) except? -1:
    cdef Math mode
    with nogil:
        status = cublasGetMathMode(<Handle>handle, &mode)
    check_status(status)
    return <int>mode


###############################################################################
# BLAS Level 1
###############################################################################

cpdef isamax(intptr_t handle, int n, size_t x, int incx, size_t result):
    _setStream(handle)
    with nogil:
        status = cublasIsamax(
            <Handle>handle, n, <float*>x, incx, <int*>result)
    check_status(status)

cpdef idamax(intptr_t handle, int n, size_t x, int incx, size_t result):
    _setStream(handle)
    with nogil:
        status = cublasIdamax(
            <Handle>handle, n, <double*>x, incx, <int*>result)
    check_status(status)

cpdef icamax(intptr_t handle, int n, size_t x, int incx, size_t result):
    _setStream(handle)
    with nogil:
        status = cublasIcamax(
            <Handle>handle, n, <cuComplex*>x, incx, <int*>result)
    check_status(status)

cpdef izamax(intptr_t handle, int n, size_t x, int incx, size_t result):
    _setStream(handle)
    with nogil:
        status = cublasIzamax(
            <Handle>handle, n, <cuDoubleComplex*>x, incx, <int*>result)
    check_status(status)


cpdef isamin(intptr_t handle, int n, size_t x, int incx, size_t result):
    _setStream(handle)
    with nogil:
        status = cublasIsamin(
            <Handle>handle, n, <float*>x, incx, <int*>result)
    check_status(status)

cpdef idamin(intptr_t handle, int n, size_t x, int incx, size_t result):
    _setStream(handle)
    with nogil:
        status = cublasIdamin(
            <Handle>handle, n, <double*>x, incx, <int*>result)
    check_status(status)

cpdef icamin(intptr_t handle, int n, size_t x, int incx, size_t result):
    _setStream(handle)
    with nogil:
        status = cublasIcamin(
            <Handle>handle, n, <cuComplex*>x, incx, <int*>result)
    check_status(status)

cpdef izamin(intptr_t handle, int n, size_t x, int incx, size_t result):
    _setStream(handle)
    with nogil:
        status = cublasIzamin(
            <Handle>handle, n, <cuDoubleComplex*>x, incx, <int*>result)
    check_status(status)


cpdef sasum(intptr_t handle, int n, size_t x, int incx, size_t result):
    _setStream(handle)
    with nogil:
        status = cublasSasum(
            <Handle>handle, n, <float*>x, incx, <float*>result)
    check_status(status)

cpdef dasum(intptr_t handle, int n, size_t x, int incx, size_t result):
    _setStream(handle)
    with nogil:
        status = cublasDasum(
            <Handle>handle, n, <double*>x, incx, <double*>result)
    check_status(status)

cpdef scasum(intptr_t handle, int n, size_t x, int incx, size_t result):
    _setStream(handle)
    with nogil:
        status = cublasScasum(
            <Handle>handle, n, <cuComplex*>x, incx, <float*>result)
    check_status(status)

cpdef dzasum(intptr_t handle, int n, size_t x, int incx, size_t result):
    _setStream(handle)
    with nogil:
        status = cublasDzasum(
            <Handle>handle, n, <cuDoubleComplex*>x, incx, <double*>result)
    check_status(status)


cpdef saxpy(intptr_t handle, int n, size_t alpha, size_t x, int incx, size_t y,
            int incy):
    _setStream(handle)
    with nogil:
        status = cublasSaxpy(
            <Handle>handle, n, <float*>alpha, <float*>x, incx, <float*>y, incy)
    check_status(status)

cpdef daxpy(intptr_t handle, int n, size_t alpha, size_t x, int incx, size_t y,
            int incy):
    _setStream(handle)
    with nogil:
        status = cublasDaxpy(
            <Handle>handle, n, <double*>alpha, <double*>x, incx, <double*>y,
            incy)
    check_status(status)

cpdef caxpy(intptr_t handle, int n, size_t alpha, size_t x, int incx, size_t y,
            int incy):
    _setStream(handle)
    with nogil:
        status = cublasCaxpy(
            <Handle>handle, n, <cuComplex*>alpha, <cuComplex*>x, incx,
            <cuComplex*>y, incy)
    check_status(status)

cpdef zaxpy(intptr_t handle, int n, size_t alpha, size_t x, int incx, size_t y,
            int incy):
    _setStream(handle)
    with nogil:
        status = cublasZaxpy(
            <Handle>handle, n, <cuDoubleComplex*>alpha, <cuDoubleComplex*>x,
            incx, <cuDoubleComplex*>y, incy)
    check_status(status)


cpdef sdot(intptr_t handle, int n, size_t x, int incx, size_t y, int incy,
           size_t result):
    _setStream(handle)
    with nogil:
        status = cublasSdot(
            <Handle>handle, n, <float*>x, incx, <float*>y, incy,
            <float*>result)
    check_status(status)

cpdef ddot(intptr_t handle, int n, size_t x, int incx, size_t y, int incy,
           size_t result):
    _setStream(handle)
    with nogil:
        status = cublasDdot(
            <Handle>handle, n, <double*>x, incx, <double*>y, incy,
            <double*>result)
    check_status(status)

cpdef cdotu(intptr_t handle, int n, size_t x, int incx, size_t y, int incy,
            size_t result):
    _setStream(handle)
    with nogil:
        status = cublasCdotu(
            <Handle>handle, n, <cuComplex*>x, incx, <cuComplex*>y, incy,
            <cuComplex*>result)
    check_status(status)

cpdef cdotc(intptr_t handle, int n, size_t x, int incx, size_t y, int incy,
            size_t result):
    _setStream(handle)
    with nogil:
        status = cublasCdotc(
            <Handle>handle, n, <cuComplex*>x, incx, <cuComplex*>y, incy,
            <cuComplex*>result)
    check_status(status)

cpdef zdotu(intptr_t handle, int n, size_t x, int incx, size_t y, int incy,
            size_t result):
    _setStream(handle)
    with nogil:
        status = cublasZdotu(
            <Handle>handle, n, <cuDoubleComplex*>x, incx,
            <cuDoubleComplex*>y, incy, <cuDoubleComplex*>result)
    check_status(status)

cpdef zdotc(intptr_t handle, int n, size_t x, int incx, size_t y, int incy,
            size_t result):
    with nogil:
        status = cublasZdotc(
            <Handle>handle, n, <cuDoubleComplex*>x, incx,
            <cuDoubleComplex*>y, incy, <cuDoubleComplex*>result)
    check_status(status)


cpdef snrm2(intptr_t handle, int n, size_t x, int incx, size_t result):
    _setStream(handle)
    with nogil:
        status = cublasSnrm2(<Handle>handle, n, <float*>x, incx,
                             <float*>result)
    check_status(status)

cpdef dnrm2(intptr_t handle, int n, size_t x, int incx, size_t result):
    _setStream(handle)
    with nogil:
        status = cublasDnrm2(<Handle>handle, n, <double*>x, incx,
                             <double*>result)
    check_status(status)

cpdef scnrm2(intptr_t handle, int n, size_t x, int incx, size_t result):
    _setStream(handle)
    with nogil:
        status = cublasScnrm2(<Handle>handle, n, <cuComplex*>x, incx,
                              <float*>result)
    check_status(status)

cpdef dznrm2(intptr_t handle, int n, size_t x, int incx, size_t result):
    _setStream(handle)
    with nogil:
        status = cublasDznrm2(<Handle>handle, n, <cuDoubleComplex*>x, incx,
                              <double*>result)
    check_status(status)


cpdef sscal(intptr_t handle, int n, size_t alpha, size_t x, int incx):
    _setStream(handle)
    with nogil:
        status = cublasSscal(<Handle>handle, n, <float*>alpha,
                             <float*>x, incx)
    check_status(status)

cpdef dscal(intptr_t handle, int n, size_t alpha, size_t x, int incx):
    _setStream(handle)
    with nogil:
        status = cublasDscal(<Handle>handle, n, <double*>alpha,
                             <double*>x, incx)
    check_status(status)

cpdef cscal(intptr_t handle, int n, size_t alpha, size_t x, int incx):
    _setStream(handle)
    with nogil:
        status = cublasCscal(<Handle>handle, n, <cuComplex*>alpha,
                             <cuComplex*>x, incx)
    check_status(status)

cpdef csscal(intptr_t handle, int n, size_t alpha, size_t x, int incx):
    _setStream(handle)
    with nogil:
        status = cublasCsscal(<Handle>handle, n, <float*>alpha,
                              <cuComplex*>x, incx)
    check_status(status)

cpdef zscal(intptr_t handle, int n, size_t alpha, size_t x, int incx):
    _setStream(handle)
    with nogil:
        status = cublasZscal(<Handle>handle, n, <cuDoubleComplex*>alpha,
                             <cuDoubleComplex*>x, incx)
    check_status(status)

cpdef zdscal(intptr_t handle, int n, size_t alpha, size_t x, int incx):
    _setStream(handle)
    with nogil:
        status = cublasZdscal(<Handle>handle, n, <double*>alpha,
                              <cuDoubleComplex*>x, incx)
    check_status(status)


###############################################################################
# BLAS Level 2
###############################################################################

cpdef sgemv(intptr_t handle, int trans, int m, int n, size_t alpha, size_t A,
            int lda, size_t x, int incx, size_t beta, size_t y, int incy):
    _setStream(handle)
    with nogil:
        status = cublasSgemv(
            <Handle>handle, <Operation>trans, m, n, <float*>alpha,
            <float*>A, lda, <float*>x, incx, <float*>beta, <float*>y, incy)
    check_status(status)


cpdef dgemv(intptr_t handle, int trans, int m, int n, size_t alpha, size_t A,
            int lda, size_t x, int incx, size_t beta, size_t y, int incy):
    _setStream(handle)
    with nogil:
        status = cublasDgemv(
            <Handle>handle, <Operation>trans, m, n, <double*>alpha,
            <double*>A, lda, <double*>x, incx, <double*>beta, <double*>y, incy)
    check_status(status)


cpdef cgemv(intptr_t handle, int trans, int m, int n, size_t alpha, size_t A,
            int lda, size_t x, int incx, size_t beta, size_t y, int incy):
    _setStream(handle)
    with nogil:
        status = cublasCgemv(
            <Handle>handle, <Operation>trans, m, n, <cuComplex*>alpha,
            <cuComplex*>A, lda, <cuComplex*>x, incx, <cuComplex*>beta,
            <cuComplex*>y, incy)
    check_status(status)


cpdef zgemv(intptr_t handle, int trans, int m, int n, size_t alpha, size_t A,
            int lda, size_t x, int incx, size_t beta, size_t y, int incy):
    _setStream(handle)
    with nogil:
        status = cublasZgemv(
            <Handle>handle, <Operation>trans, m, n, <cuDoubleComplex*>alpha,
            <cuDoubleComplex*>A, lda, <cuDoubleComplex*>x, incx,
            <cuDoubleComplex*>beta, <cuDoubleComplex*>y, incy)
    check_status(status)


cpdef sger(intptr_t handle, int m, int n, size_t alpha, size_t x, int incx,
           size_t y, int incy, size_t A, int lda):
    _setStream(handle)
    with nogil:
        status = cublasSger(
            <Handle>handle, m, n, <float*>alpha, <float*>x, incx, <float*>y,
            incy, <float*>A, lda)
    check_status(status)


cpdef dger(intptr_t handle, int m, int n, size_t alpha, size_t x, int incx,
           size_t y, int incy, size_t A, int lda):
    _setStream(handle)
    with nogil:
        status = cublasDger(
            <Handle>handle, m, n, <double*>alpha, <double*>x, incx, <double*>y,
            incy, <double*>A, lda)
    check_status(status)


cpdef cgeru(intptr_t handle, int m, int n, size_t alpha, size_t x, int incx,
            size_t y, int incy, size_t A, int lda):
    _setStream(handle)
    with nogil:
        status = cublasCgeru(
            <Handle>handle, m, n, <cuComplex*>alpha, <cuComplex*>x, incx,
            <cuComplex*>y, incy, <cuComplex*>A, lda)
    check_status(status)


cpdef cgerc(intptr_t handle, int m, int n, size_t alpha, size_t x, int incx,
            size_t y, int incy, size_t A, int lda):
    _setStream(handle)
    with nogil:
        status = cublasCgerc(
            <Handle>handle, m, n, <cuComplex*>alpha, <cuComplex*>x, incx,
            <cuComplex*>y, incy, <cuComplex*>A, lda)
    check_status(status)


cpdef zgeru(intptr_t handle, int m, int n, size_t alpha, size_t x, int incx,
            size_t y, int incy, size_t A, int lda):
    _setStream(handle)
    with nogil:
        status = cublasZgeru(
            <Handle>handle, m, n, <cuDoubleComplex*>alpha,
            <cuDoubleComplex*>x, incx, <cuDoubleComplex*>y, incy,
            <cuDoubleComplex*>A, lda)
    check_status(status)


cpdef zgerc(intptr_t handle, int m, int n, size_t alpha, size_t x, int incx,
            size_t y, int incy, size_t A, int lda):
    _setStream(handle)
    with nogil:
        status = cublasZgerc(
            <Handle>handle, m, n, <cuDoubleComplex*>alpha,
            <cuDoubleComplex*>x, incx, <cuDoubleComplex*>y, incy,
            <cuDoubleComplex*>A, lda)
    check_status(status)


###############################################################################
# BLAS Level 3
###############################################################################

cpdef sgemm(intptr_t handle, int transa, int transb,
            int m, int n, int k, size_t alpha, size_t A, int lda,
            size_t B, int ldb, size_t beta, size_t C, int ldc):
    _setStream(handle)
    with nogil:
        status = cublasSgemm(
            <Handle>handle, <Operation>transa, <Operation>transb, m, n, k,
            <float*>alpha, <float*>A, lda, <float*>B, ldb, <float*>beta,
            <float*>C, ldc)
    check_status(status)


cpdef dgemm(intptr_t handle, int transa, int transb,
            int m, int n, int k, size_t alpha, size_t A, int lda,
            size_t B, int ldb, size_t beta, size_t C, int ldc):
    _setStream(handle)
    with nogil:
        status = cublasDgemm(
            <Handle>handle, <Operation>transa, <Operation>transb, m, n, k,
            <double*>alpha, <double*>A, lda, <double*>B, ldb, <double*>beta,
            <double*>C, ldc)
    check_status(status)


cpdef cgemm(intptr_t handle, int transa, int transb,
            int m, int n, int k, size_t alpha, size_t A, int lda,
            size_t B, int ldb, size_t beta, size_t C, int ldc):
    _setStream(handle)
    with nogil:
        status = cublasCgemm(
            <Handle>handle, <Operation>transa, <Operation>transb, m, n, k,
            <cuComplex*>alpha, <cuComplex*>A, lda, <cuComplex*>B, ldb,
            <cuComplex*>beta, <cuComplex*>C, ldc)
    check_status(status)


cpdef zgemm(intptr_t handle, int transa, int transb,
            int m, int n, int k, size_t alpha, size_t A, int lda,
            size_t B, int ldb, size_t beta, size_t C, int ldc):
    _setStream(handle)
    with nogil:
        status = cublasZgemm(
            <Handle>handle, <Operation>transa, <Operation>transb, m, n, k,
            <cuDoubleComplex*>alpha, <cuDoubleComplex*>A, lda,
            <cuDoubleComplex*>B, ldb, <cuDoubleComplex*>beta,
            <cuDoubleComplex*>C, ldc)
    check_status(status)


cpdef sgemmBatched(
        intptr_t handle, int transa, int transb, int m, int n, int k,
        size_t alpha, size_t Aarray, int lda, size_t Barray, int ldb,
        size_t beta, size_t Carray, int ldc, int batchCount):
    _setStream(handle)
    with nogil:
        status = cublasSgemmBatched(
            <Handle>handle, <Operation>transa, <Operation>transb, m, n, k,
            <float*>alpha, <const float**>Aarray, lda, <const float**>Barray,
            ldb, <float*>beta, <float**>Carray, ldc, batchCount)
    check_status(status)


cpdef dgemmBatched(
        intptr_t handle, int transa, int transb, int m, int n, int k,
        size_t alpha, size_t Aarray, int lda, size_t Barray, int ldb,
        size_t beta, size_t Carray, int ldc, int batchCount):
    _setStream(handle)
    with nogil:
        status = cublasDgemmBatched(
            <Handle>handle, <Operation>transa, <Operation>transb, m, n, k,
            <double*>alpha, <const double**>Aarray, lda,
            <const double**>Barray, ldb, <double*>beta,
            <double**>Carray, ldc, batchCount)
    check_status(status)


cpdef cgemmBatched(
        intptr_t handle, int transa, int transb, int m, int n, int k,
        size_t alpha, size_t Aarray, int lda, size_t Barray, int ldb,
        size_t beta, size_t Carray, int ldc, int batchCount):
    _setStream(handle)
    with nogil:
        status = cublasCgemmBatched(
            <Handle>handle, <Operation>transa, <Operation>transb, m, n, k,
            <cuComplex*>alpha, <const cuComplex**>Aarray, lda,
            <const cuComplex**>Barray, ldb, <cuComplex*>beta,
            <cuComplex**>Carray, ldc, batchCount)
    check_status(status)


cpdef zgemmBatched(
        intptr_t handle, int transa, int transb, int m, int n, int k,
        size_t alpha, size_t Aarray, int lda, size_t Barray, int ldb,
        size_t beta, size_t Carray, int ldc, int batchCount):
    _setStream(handle)
    with nogil:
        status = cublasZgemmBatched(
            <Handle>handle, <Operation>transa, <Operation>transb, m, n, k,
            <cuDoubleComplex*>alpha, <const cuDoubleComplex**>Aarray, lda,
            <const cuDoubleComplex**>Barray, ldb, <cuDoubleComplex*>beta,
            <cuDoubleComplex**>Carray, ldc, batchCount)


cpdef sgemmStridedBatched(
        intptr_t handle, int transa, int transb, int m, int n, int k,
        size_t alpha, size_t A, int lda, long long strideA, size_t B, int ldb,
        long long strideB, size_t beta, size_t C, int ldc, long long strideC,
        int batchCount):
    _setStream(handle)
    with nogil:
        status = cublasSgemmStridedBatched(
            <Handle>handle, <Operation>transa, <Operation>transb, m, n, k,
            <const float*>alpha,
            <const float*>A, lda, <long long>strideA,
            <const float*>B, ldb, <long long>strideB,
            <const float*>beta,
            <float*>C, ldc, <long long>strideC,
            batchCount)
    check_status(status)


cpdef dgemmStridedBatched(
        intptr_t handle, int transa, int transb, int m, int n, int k,
        size_t alpha, size_t A, int lda, long long strideA, size_t B, int ldb,
        long long strideB, size_t beta, size_t C, int ldc, long long strideC,
        int batchCount):
    _setStream(handle)
    with nogil:
        status = cublasDgemmStridedBatched(
            <Handle>handle, <Operation>transa, <Operation>transb, m, n, k,
            <const double*>alpha,
            <const double*>A, lda, <long long>strideA,
            <const double*>B, ldb, <long long>strideB,
            <const double*>beta,
            <double*>C, ldc, <long long>strideC,
            batchCount)
    check_status(status)


cpdef cgemmStridedBatched(
        intptr_t handle, int transa, int transb, int m, int n, int k,
        size_t alpha, size_t A, int lda, long long strideA, size_t B, int ldb,
        long long strideB, size_t beta, size_t C, int ldc, long long strideC,
        int batchCount):
    _setStream(handle)
    with nogil:
        status = cublasCgemmStridedBatched(
            <Handle>handle, <Operation>transa, <Operation>transb, m, n, k,
            <const cuComplex*>alpha,
            <const cuComplex*>A, lda, <long long>strideA,
            <const cuComplex*>B, ldb, <long long>strideB,
            <const cuComplex*>beta,
            <cuComplex*>C, ldc, <long long>strideC,
            batchCount)
    check_status(status)


cpdef zgemmStridedBatched(
        intptr_t handle, int transa, int transb, int m, int n, int k,
        size_t alpha, size_t A, int lda, long long strideA, size_t B, int ldb,
        long long strideB, size_t beta, size_t C, int ldc, long long strideC,
        int batchCount):
    _setStream(handle)
    with nogil:
        status = cublasZgemmStridedBatched(
            <Handle>handle, <Operation>transa, <Operation>transb, m, n, k,
            <const cuDoubleComplex*>alpha,
            <const cuDoubleComplex*>A, lda, <long long>strideA,
            <const cuDoubleComplex*>B, ldb, <long long>strideB,
            <const cuDoubleComplex*>beta,
            <cuDoubleComplex*>C, ldc, <long long>strideC,
            batchCount)
    check_status(status)


cpdef strsm(
        intptr_t handle, int side, int uplo, int trans, int diag,
        int m, int n, size_t alpha, size_t Aarray, int lda,
        size_t Barray, int ldb):
    _setStream(handle)
    with nogil:
        status = cublasStrsm(
            <Handle>handle, <SideMode>side, <FillMode>uplo, <Operation>trans,
            <DiagType>diag, m, n, <const float*>alpha, <const float*>Aarray,
            lda, <float*>Barray, ldb)
    check_status(status)


cpdef dtrsm(
        intptr_t handle, int side, int uplo, int trans, int diag,
        int m, int n, size_t alpha, size_t Aarray, int lda,
        size_t Barray, int ldb):
    _setStream(handle)
    with nogil:
        status = cublasDtrsm(
            <Handle>handle, <SideMode>side, <FillMode>uplo, <Operation>trans,
            <DiagType>diag, m, n, <const double*>alpha, <const double*>Aarray,
            lda, <double*>Barray, ldb)
    check_status(status)

cpdef ctrsm(
        intptr_t handle, int side, int uplo, int trans, int diag,
        int m, int n, size_t alpha, size_t Aarray, int lda,
        size_t Barray, int ldb):
    _setStream(handle)
    with nogil:
        status = cublasCtrsm(
            <Handle>handle, <SideMode>side, <FillMode>uplo, <Operation>trans,
            <DiagType>diag, m, n, <const cuComplex*>alpha,
            <const cuComplex*>Aarray, lda, <cuComplex*>Barray, ldb)
    check_status(status)


cpdef ztrsm(
        intptr_t handle, int side, int uplo, int trans, int diag,
        int m, int n, size_t alpha, size_t Aarray, int lda,
        size_t Barray, int ldb):
    _setStream(handle)
    with nogil:
        status = cublasZtrsm(
            <Handle>handle, <SideMode>side, <FillMode>uplo, <Operation>trans,
            <DiagType>diag, m, n, <const cuDoubleComplex*>alpha,
            <const cuDoubleComplex*>Aarray, lda, <cuDoubleComplex*>Barray, ldb)
    check_status(status)

###############################################################################
# BLAS extension
###############################################################################

cpdef sgeam(intptr_t handle, int transa, int transb, int m, int n,
            size_t alpha, size_t A, int lda, size_t beta, size_t B, int ldb,
            size_t C, int ldc):
    _setStream(handle)
    with nogil:
        status = cublasSgeam(
            <Handle>handle, <Operation>transa, <Operation>transb, m, n,
            <const float*>alpha, <const float*>A, lda, <const float*>beta,
            <const float*>B, ldb, <float*>C, ldc)
    check_status(status)

cpdef dgeam(intptr_t handle, int transa, int transb, int m, int n,
            size_t alpha, size_t A, int lda, size_t beta, size_t B, int ldb,
            size_t C, int ldc):
    _setStream(handle)
    with nogil:
        status = cublasDgeam(
            <Handle>handle, <Operation>transa, <Operation>transb, m, n,
            <const double*>alpha, <const double*>A, lda, <const double*>beta,
            <const double*>B, ldb, <double*>C, ldc)
    check_status(status)

cpdef cgeam(intptr_t handle, int transa, int transb, int m, int n,
            size_t alpha, size_t A, int lda, size_t beta, size_t B, int ldb,
            size_t C, int ldc):
    _setStream(handle)
    with nogil:
        status = cublasCgeam(
            <Handle>handle, <Operation>transa, <Operation>transb, m, n,
            <const cuComplex*>alpha, <const cuComplex*>A, lda,
            <const cuComplex*>beta, <const cuComplex*>B, ldb,
            <cuComplex*>C, ldc)
    check_status(status)

cpdef zgeam(intptr_t handle, int transa, int transb, int m, int n,
            size_t alpha, size_t A, int lda, size_t beta, size_t B, int ldb,
            size_t C, int ldc):
    _setStream(handle)
    with nogil:
        status = cublasZgeam(
            <Handle>handle, <Operation>transa, <Operation>transb, m, n,
            <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>A, lda,
            <const cuDoubleComplex*>beta, <const cuDoubleComplex*>B, ldb,
            <cuDoubleComplex*>C, ldc)
    check_status(status)


cpdef sdgmm(intptr_t handle, int mode, int m, int n, size_t A, int lda,
            size_t x, int incx, size_t C, int ldc):
    _setStream(handle)
    with nogil:
        status = cublasSdgmm(
            <Handle>handle, <SideMode>mode, m, n, <const float*>A, lda,
            <const float*>x, incx, <float*>C, ldc)
    check_status(status)

cpdef ddgmm(intptr_t handle, int mode, int m, int n, size_t A, int lda,
            size_t x, int incx, size_t C, int ldc):
    _setStream(handle)
    with nogil:
        status = cublasDdgmm(
            <Handle>handle, <SideMode>mode, m, n, <const double*>A, lda,
            <const double*>x, incx, <double*>C, ldc)
    check_status(status)

cpdef cdgmm(intptr_t handle, int mode, int m, int n, size_t A, int lda,
            size_t x, int incx, size_t C, int ldc):
    _setStream(handle)
    with nogil:
        status = cublasCdgmm(
            <Handle>handle, <SideMode>mode, m, n, <const cuComplex*>A, lda,
            <const cuComplex*>x, incx, <cuComplex*>C, ldc)
    check_status(status)

cpdef zdgmm(intptr_t handle, int mode, int m, int n, size_t A, int lda,
            size_t x, int incx, size_t C, int ldc):
    _setStream(handle)
    with nogil:
        status = cublasZdgmm(
            <Handle>handle, <SideMode>mode, m, n, <const cuDoubleComplex*>A,
            lda, <const cuDoubleComplex*>x, incx, <cuDoubleComplex*>C, ldc)
    check_status(status)


cpdef sgemmEx(
        intptr_t handle, int transa, int transb, int m, int n, int k,
        size_t alpha, size_t A, int Atype, int lda, size_t B,
        int Btype, int ldb, size_t beta, size_t C, int Ctype,
        int ldc):
    _setStream(handle)
    with nogil:
        status = cublasSgemmEx(
            <Handle>handle, <Operation>transa, <Operation>transb, m, n, k,
            <const float*>alpha, <const void*>A, <runtime.DataType>Atype, lda,
            <const void*>B, <runtime.DataType>Btype, ldb, <const float*>beta,
            <void*>C, <runtime.DataType>Ctype, ldc)
    check_status(status)


cpdef sgetrfBatched(intptr_t handle, int n, size_t Aarray, int lda,
                    size_t PivotArray, size_t infoArray, int batchSize):
    _setStream(handle)
    with nogil:
        status = cublasSgetrfBatched(
            <Handle>handle, n, <float**>Aarray, lda,
            <int*>PivotArray, <int*>infoArray, batchSize)
    check_status(status)


cpdef dgetrfBatched(intptr_t handle, int n, size_t Aarray, int lda,
                    size_t PivotArray, size_t infoArray, int batchSize):
    _setStream(handle)
    with nogil:
        status = cublasDgetrfBatched(
            <Handle>handle, n, <double**>Aarray, lda,
            <int*>PivotArray, <int*>infoArray, batchSize)
    check_status(status)


cpdef cgetrfBatched(intptr_t handle, int n, size_t Aarray, int lda,
                    size_t PivotArray, size_t infoArray, int batchSize):
    _setStream(handle)
    with nogil:
        status = cublasCgetrfBatched(
            <Handle>handle, n, <cuComplex**>Aarray, lda,
            <int*>PivotArray, <int*>infoArray, batchSize)
    check_status(status)


cpdef zgetrfBatched(intptr_t handle, int n, size_t Aarray, int lda,
                    size_t PivotArray, size_t infoArray, int batchSize):
    _setStream(handle)
    with nogil:
        status = cublasZgetrfBatched(
            <Handle>handle, n, <cuDoubleComplex**>Aarray, lda,
            <int*>PivotArray, <int*>infoArray, batchSize)
    check_status(status)


cpdef int sgetrsBatched(intptr_t handle, int trans, int n, int nrhs,
                        size_t Aarray, int lda, size_t devIpiv,
                        size_t Barray, int ldb, size_t info, int batchSize):
    _setStream(handle)
    with nogil:
        status = cublasSgetrsBatched(
            <Handle>handle, <Operation>trans, n, nrhs,
            <const float**>Aarray, lda, <const int*>devIpiv,
            <float**>Barray, ldb, <int*>info, batchSize)
    check_status(status)

cpdef int dgetrsBatched(intptr_t handle, int trans, int n, int nrhs,
                        size_t Aarray, int lda, size_t devIpiv,
                        size_t Barray, int ldb, size_t info, int batchSize):
    _setStream(handle)
    with nogil:
        status = cublasDgetrsBatched(
            <Handle>handle, <Operation>trans, n, nrhs,
            <const double**>Aarray, lda, <const int*>devIpiv,
            <double**>Barray, ldb, <int*>info, batchSize)
    check_status(status)

cpdef int cgetrsBatched(intptr_t handle, int trans, int n, int nrhs,
                        size_t Aarray, int lda, size_t devIpiv,
                        size_t Barray, int ldb, size_t info, int batchSize):
    _setStream(handle)
    with nogil:
        status = cublasCgetrsBatched(
            <Handle>handle, <Operation>trans, n, nrhs,
            <const cuComplex**>Aarray, lda, <const int*>devIpiv,
            <cuComplex**>Barray, ldb, <int*>info, batchSize)
    check_status(status)

cpdef int zgetrsBatched(intptr_t handle, int trans, int n, int nrhs,
                        size_t Aarray, int lda, size_t devIpiv,
                        size_t Barray, int ldb, size_t info, int batchSize):
    _setStream(handle)
    with nogil:
        status = cublasZgetrsBatched(
            <Handle>handle, <Operation>trans, n, nrhs,
            <const cuDoubleComplex**>Aarray, lda, <const int*>devIpiv,
            <cuDoubleComplex**>Barray, ldb, <int*>info, batchSize)
    check_status(status)


cpdef sgetriBatched(
        intptr_t handle, int n, size_t Aarray, int lda, size_t PivotArray,
        size_t Carray, int ldc, size_t infoArray, int batchSize):
    _setStream(handle)
    with nogil:
        status = cublasSgetriBatched(
            <Handle>handle, n, <const float**>Aarray, lda, <int*>PivotArray,
            <float**>Carray, ldc, <int*>infoArray, batchSize)
    check_status(status)


cpdef dgetriBatched(
        intptr_t handle, int n, size_t Aarray, int lda, size_t PivotArray,
        size_t Carray, int ldc, size_t infoArray, int batchSize):
    _setStream(handle)
    with nogil:
        status = cublasDgetriBatched(
            <Handle>handle, n, <const double**>Aarray, lda, <int*>PivotArray,
            <double**>Carray, ldc, <int*>infoArray, batchSize)
    check_status(status)


cpdef cgetriBatched(
        intptr_t handle, int n, size_t Aarray, int lda, size_t PivotArray,
        size_t Carray, int ldc, size_t infoArray, int batchSize):
    _setStream(handle)
    with nogil:
        status = cublasCgetriBatched(
            <Handle>handle, n, <const cuComplex**>Aarray, lda,
            <int*>PivotArray,
            <cuComplex**>Carray, ldc, <int*>infoArray, batchSize)
    check_status(status)


cpdef zgetriBatched(
        intptr_t handle, int n, size_t Aarray, int lda, size_t PivotArray,
        size_t Carray, int ldc, size_t infoArray, int batchSize):
    _setStream(handle)
    with nogil:
        status = cublasZgetriBatched(
            <Handle>handle, n, <const cuDoubleComplex**>Aarray, lda,
            <int*>PivotArray,
            <cuDoubleComplex**>Carray, ldc, <int*>infoArray, batchSize)
    check_status(status)


cpdef gemmEx(
        intptr_t handle, int transa, int transb, int m, int n, int k,
        size_t alpha, size_t A, int Atype, int lda, size_t B,
        int Btype, int ldb, size_t beta, size_t C, int Ctype,
        int ldc, int computeType, int algo):
    _setStream(handle)
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


cpdef stpttr(intptr_t handle, int uplo, int n, size_t AP, size_t A, int lda):
    _setStream(handle)
    with nogil:
        status = cublasStpttr(<Handle>handle, <FillMode>uplo, n,
                              <const float*>AP, <float*>A, lda)
    check_status(status)


cpdef dtpttr(intptr_t handle, int uplo, int n, size_t AP, size_t A, int lda):
    _setStream(handle)
    with nogil:
        status = cublasDtpttr(<Handle>handle, <FillMode>uplo, n,
                              <const double*>AP, <double*>A, lda)
    check_status(status)


cpdef strttp(intptr_t handle, int uplo, int n, size_t A, int lda, size_t AP):
    _setStream(handle)
    with nogil:
        status = cublasStrttp(<Handle>handle, <FillMode>uplo, n,
                              <const float*>A, lda, <float*>AP)
    check_status(status)


cpdef dtrttp(intptr_t handle, int uplo, int n, size_t A, int lda, size_t AP):
    _setStream(handle)
    with nogil:
        status = cublasDtrttp(<Handle>handle, <FillMode>uplo, n,
                              <const double*>A, lda, <double*>AP)
    check_status(status)
