# distutils: language = c++

"""Thin wrapper of CUBLAS."""
cimport cython

from cupy.cuda cimport driver
from cupy.cuda cimport runtime


###############################################################################
# Extern
###############################################################################

cdef extern from 'cupy_cuda.h':
    # Context
    int cublasCreate(Handle* handle) nogil
    int cublasDestroy(Handle handle) nogil
    int cublasGetVersion(Handle handle, int* version) nogil
    int cublasGetPointerMode(Handle handle, PointerMode* mode) nogil
    int cublasSetPointerMode(Handle handle, PointerMode mode) nogil

    # Stream
    int cublasSetStream(Handle handle, driver.Stream streamId) nogil
    int cublasGetStream(Handle handle, driver.Stream* streamId) nogil

    # BLAS Level 1
    int cublasIsamax(Handle handle, int n, float* x, int incx,
                     int* result) nogil
    int cublasIsamin(Handle handle, int n, float* x, int incx,
                     int* result) nogil
    int cublasSasum(Handle handle, int n, float* x, int incx,
                    float* result) nogil
    int cublasSaxpy(Handle handle, int n, float* alpha, float* x,
                    int incx, float* y, int incy) nogil
    int cublasDaxpy(Handle handle, int n, double* alpha, double* x,
                    int incx, double* y, int incy) nogil
    int cublasSdot(Handle handle, int n, float* x, int incx,
                   float* y, int incy, float* result) nogil
    int cublasDdot(Handle handle, int n, double* x, int incx,
                   double* y, int incy, double* result) nogil
    int cublasSnrm2(Handle handle, int n, float* x, int incx,
                    float* result) nogil
    int cublasSscal(Handle handle, int n, float* alpha, float* x,
                    int incx) nogil

    # BLAS Level 2
    int cublasSgemv(
        Handle handle, Operation trans, int m, int n, float* alpha,
        float* A, int lda, float* x, int incx, float* beta,
        float* y, int incy) nogil
    int cublasDgemv(
        Handle handle, Operation trans, int m, int n, double* alpha,
        double* A, int lda, double* x, int incx, double* beta,
        double* y, int incy) nogil
    int cublasSger(
        Handle handle, int m, int n, float* alpha, float* x, int incx,
        float* y, int incy, float* A, int lda) nogil
    int cublasDger(
        Handle handle, int m, int n, double* alpha, double* x,
        int incx, double* y, int incy, double* A, int lda) nogil

    # BLAS Level 3
    int cublasSgemm(
        Handle handle, Operation transa, Operation transb, int m,
        int n, int k, float* alpha, float* A, int lda, float* B,
        int ldb, float* beta, float* C, int ldc) nogil
    int cublasDgemm(
        Handle handle, Operation transa, Operation transb, int m,
        int n, int k, double* alpha, double* A, int lda, double* B,
        int ldb, double* beta, double* C, int ldc) nogil
    int cublasSgemmBatched(
        Handle handle, Operation transa, Operation transb, int m,
        int n, int k, const float* alpha, const float** Aarray,
        int lda, const float** Barray, int ldb, const float* beta,
        float** Carray, int ldc, int batchCount) nogil
    int cublasDgemmBatched(
        Handle handle, Operation transa, Operation transb, int m,
        int n, int k, const double* alpha, const double** Aarray,
        int lda, const double** Barray, int ldb, const double* beta,
        double** Carray, int ldc, int batchCount) nogil

    # BLAS extension
    int cublasSgeam(
        Handle handle, Operation transa, Operation transb, int m, int n,
        const float* alpha, const float* A, int lda,
        const float* beta, const float* B, int ldb,
        float* C, int ldc) nogil
    int cublasDgeam(
        Handle handle, Operation transa, Operation transb, int m, int n,
        const double* alpha, const double* A, int lda,
        const double* beta, const double* B, int ldb,
        double* C, int ldc) nogil
    int cublasSdgmm(
        Handle handle, SideMode mode, int m, int n, float* A, int lda,
        float* x, int incx, float* C, int ldc) nogil
    int cublasSgemmEx(
        Handle handle, Operation transa,
        Operation transb, int m, int n, int k,
        const float *alpha, const void *A, runtime.DataType Atype,
        int lda, const void *B, runtime.DataType Btype, int ldb,
        const float *beta, void *C, runtime.DataType Ctype, int ldc) nogil
    int cublasSgetrfBatched(
        Handle handle, int n, float **Aarray, int lda,
        int *PivotArray, int *infoArray, int batchSize) nogil
    int cublasSgetriBatched(
        Handle handle, int n, const float **Aarray, int lda,
        int *PivotArray, float *Carray[], int ldc, int *infoArray,
        int batchSize) nogil


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


class CUBLASError(RuntimeError):

    def __init__(self, status):
        self.status = status
        super(CUBLASError, self).__init__(STATUS[status])


@cython.profile(False)
cpdef inline check_status(int status):
    if status != 0:
        raise CUBLASError(status)


###############################################################################
# Context
###############################################################################

cpdef size_t create() except *:
    cdef Handle handle
    with nogil:
        status = cublasCreate(&handle)
    check_status(status)
    return <size_t>handle


cpdef void destroy(size_t handle) except *:
    with nogil:
        status = cublasDestroy(<Handle>handle)
    check_status(status)


cpdef int getVersion(size_t handle) except *:
    cdef int version
    with nogil:
        status = cublasGetVersion(<Handle>handle, &version)
    check_status(status)
    return version


cpdef int getPointerMode(size_t handle) except *:
    cdef PointerMode mode
    with nogil:
        status = cublasGetPointerMode(<Handle>handle, &mode)
    check_status(status)
    return mode


cpdef setPointerMode(size_t handle, int mode):
    with nogil:
        status = cublasSetPointerMode(<Handle>handle, <PointerMode>mode)
    check_status(status)


###############################################################################
# Stream
###############################################################################

cpdef setStream(size_t handle, size_t stream):
    with nogil:
        status = cublasSetStream(<Handle>handle, <driver.Stream>stream)
    check_status(status)


cpdef size_t getStream(size_t handle) except *:
    cdef driver.Stream stream
    with nogil:
        status = cublasGetStream(<Handle>handle, &stream)
    check_status(status)
    return <size_t>stream


###############################################################################
# BLAS Level 1
###############################################################################

cpdef int isamax(size_t handle, int n, size_t x, int incx) except *:
    cdef int result
    with nogil:
        status = cublasIsamax(
            <Handle>handle, n, <float*>x, incx, &result)
    check_status(status)
    return result


cpdef int isamin(size_t handle, int n, size_t x, int incx) except *:
    cdef int result
    with nogil:
        status = cublasIsamin(
            <Handle>handle, n, <float*>x, incx, &result)
    check_status(status)
    return result


cpdef float sasum(size_t handle, int n, size_t x, int incx) except *:
    cdef float result
    with nogil:
        status = cublasSasum(
            <Handle>handle, n, <float*>x, incx, &result)
    check_status(status)
    return result


cpdef saxpy(size_t handle, int n, float alpha, size_t x, int incx, size_t y,
            int incy):
    with nogil:
        status = cublasSaxpy(
            <Handle>handle, n, &alpha, <float*>x, incx, <float*>y, incy)
    check_status(status)


cpdef daxpy(size_t handle, int n, double alpha, size_t x, int incx, size_t y,
            int incy):
    with nogil:
        status = cublasDaxpy(
            <Handle>handle, n, &alpha, <double*>x, incx, <double*>y, incy)
    check_status(status)


cpdef sdot(size_t handle, int n, size_t x, int incx, size_t y, int incy,
           size_t result):
    with nogil:
        status = cublasSdot(
            <Handle>handle, n, <float*>x, incx, <float*>y, incy,
            <float*>result)
    check_status(status)


cpdef ddot(size_t handle, int n, size_t x, int incx, size_t y, int incy,
           size_t result):
    with nogil:
        status = cublasDdot(
            <Handle>handle, n, <double*>x, incx, <double*>y, incy,
            <double*>result)
    check_status(status)


cpdef float snrm2(size_t handle, int n, size_t x, int incx) except *:
    cdef float result
    with nogil:
        status = cublasSnrm2(<Handle>handle, n, <float*>x, incx, &result)
    check_status(status)
    return result


cpdef sscal(size_t handle, int n, float alpha, size_t x, int incx):
    with nogil:
        status = cublasSscal(<Handle>handle, n, &alpha, <float*>x, incx)
    check_status(status)


###############################################################################
# BLAS Level 2
###############################################################################

cpdef sgemv(size_t handle, int trans, int m, int n, float alpha, size_t A,
            int lda, size_t x, int incx, float beta, size_t y, int incy):
    with nogil:
        status = cublasSgemv(
            <Handle>handle, <Operation>trans, m, n, &alpha,
            <float*>A, lda, <float*>x, incx, &beta, <float*>y, incy)
    check_status(status)


cpdef dgemv(size_t handle, int trans, int m, int n, double alpha, size_t A,
            int lda, size_t x, int incx, double beta, size_t y, int incy):
    with nogil:
        status = cublasDgemv(
            <Handle>handle, <Operation>trans, m, n, &alpha,
            <double*>A, lda, <double*>x, incx, &beta, <double*>y, incy)
    check_status(status)


cpdef sger(size_t handle, int m, int n, float alpha, size_t x, int incx,
           size_t y, int incy, size_t A, int lda):
    with nogil:
        status = cublasSger(
            <Handle>handle, m, n, &alpha, <float*>x, incx, <float*>y, incy,
            <float*>A, lda)
    check_status(status)


cpdef dger(size_t handle, int m, int n, double alpha, size_t x, int incx,
           size_t y, int incy, size_t A, int lda):
    with nogil:
        status = cublasDger(
            <Handle>handle, m, n, &alpha, <double*>x, incx, <double*>y, incy,
            <double*>A, lda)
    check_status(status)


###############################################################################
# BLAS Level 3
###############################################################################

cpdef sgemm(size_t handle, int transa, int transb,
            int m, int n, int k, float alpha, size_t A, int lda,
            size_t B, int ldb, float beta, size_t C, int ldc):
    with nogil:
        status = cublasSgemm(
            <Handle>handle, <Operation>transa, <Operation>transb, m, n, k,
            &alpha, <float*>A, lda, <float*>B, ldb, &beta, <float*>C, ldc)
    check_status(status)


cpdef dgemm(size_t handle, int transa, int transb,
            int m, int n, int k, double alpha, size_t A, int lda,
            size_t B, int ldb, double beta, size_t C, int ldc):
    with nogil:
        status = cublasDgemm(
            <Handle>handle, <Operation>transa, <Operation>transb, m, n, k,
            &alpha, <double*>A, lda, <double*>B, ldb, &beta, <double*>C, ldc)
    check_status(status)


cpdef sgemmBatched(
        size_t handle, int transa, int transb, int m, int n, int k,
        float alpha, size_t Aarray, int lda, size_t Barray, int ldb,
        float beta, size_t Carray, int ldc, int batchCount):
    with nogil:
        status = cublasSgemmBatched(
            <Handle>handle, <Operation>transa, <Operation>transb, m, n, k,
            &alpha, <const float**>Aarray, lda, <const float**>Barray, ldb,
            &beta, <float**>Carray, ldc, batchCount)
    check_status(status)


cpdef dgemmBatched(
        size_t handle, int transa, int transb, int m, int n, int k,
        double alpha, size_t Aarray, int lda, size_t Barray, int ldb,
        double beta, size_t Carray, int ldc, int batchCount):
    with nogil:
        status = cublasDgemmBatched(
            <Handle>handle, <Operation>transa, <Operation>transb, m, n, k,
            &alpha, <const double**>Aarray, lda, <const double**>Barray, ldb,
            &beta, <double**>Carray, ldc, batchCount)
    check_status(status)

###############################################################################
# BLAS extension
###############################################################################

cpdef sgeam(size_t handle, int transa, int transb, int m, int n,
            float alpha, size_t A, int lda, float beta, size_t B, int ldb,
            size_t C, int ldc):
    with nogil:
        status = cublasSgeam(
            <Handle>handle, <Operation>transa, <Operation>transb, m, n,
            &alpha, <const float*>A, lda, &beta, <const float*>B, ldb,
            <float*>C, ldc)
    check_status(status)


cpdef dgeam(size_t handle, int transa, int transb, int m, int n,
            double alpha, size_t A, int lda, double beta, size_t B, int ldb,
            size_t C, int ldc):
    with nogil:
        status = cublasDgeam(
            <Handle>handle, <Operation>transa, <Operation>transb, m, n,
            &alpha, <const double*>A, lda, &beta, <const double*>B, ldb,
            <double*>C, ldc)
    check_status(status)


cpdef sdgmm(size_t handle, int mode, int m, int n, size_t A, int lda,
            size_t x, int incx, size_t C, int ldc):
    with nogil:
        status = cublasSdgmm(
            <Handle>handle, <SideMode>mode, m, n, <float*>A, lda, <float*>x,
            incx, <float*>C, ldc)
    check_status(status)


cpdef sgemmEx(
        size_t handle, int transa, int transb, int m, int n, int k,
        float alpha, size_t A, int Atype, int lda, size_t B,
        int Btype, int ldb, float beta, size_t C, int Ctype,
        int ldc):
    with nogil:
        status = cublasSgemmEx(
            <Handle>handle, <Operation>transa, <Operation>transb, m, n, k,
            &alpha, <const void*>A, <runtime.DataType>Atype, lda,
            <const void*>B, <runtime.DataType>Btype, ldb, &beta, <void*>C,
            <runtime.DataType>Ctype, ldc)
    check_status(status)


cpdef sgetrfBatched(size_t handle, int n, size_t Aarray, int lda,
                    size_t PivotArray, size_t infoArray, int batchSize):
    with nogil:
        status = cublasSgetrfBatched(
            <Handle>handle, n, <float**>Aarray, lda, <int*>PivotArray,
            <int*>infoArray, batchSize)
    check_status(status)


cpdef sgetriBatched(
        size_t handle, int n, size_t Aarray, int lda, size_t PivotArray,
        size_t Carray, int ldc, size_t infoArray, int batchSize):
    with nogil:
        status = cublasSgetriBatched(
            <Handle>handle, n, <const float**>Aarray, lda, <int*>PivotArray,
            <float**>Carray, ldc, <int*>infoArray, batchSize)
    check_status(status)
