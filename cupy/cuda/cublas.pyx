"""Thin wrapper of CUBLAS."""
cimport cython

from cupy.cuda cimport driver
from cupy.cuda cimport runtime


###############################################################################
# Extern
###############################################################################

cdef extern from "cuComplex.h":
    ctypedef void* cuComplexPtr 'cuComplex*'
    ctypedef void* cuDoubleComplexPtr 'cuDoubleComplex*'
    ctypedef const void* constCuComplexPtr 'const cuComplex*'
    ctypedef const void* constCuDoubleComplexPtr 'const cuDoubleComplex*'
    ctypedef void** cuComplexPtrPtr 'cuComplex**'
    ctypedef void** cuDoubleComplexPtrPtr 'cuDoubleComplex**'
    ctypedef const void** constCuComplexPtrPtr 'const cuComplex**'
    ctypedef const void** constCuDoubleComplexPtrPtr 'const cuDoubleComplex**'

cdef extern from 'cupy_cuda.h':
    # Context
    int cublasCreate(Handle* handle)
    int cublasDestroy(Handle handle)
    int cublasGetVersion(Handle handle, int* version)
    int cublasGetPointerMode(Handle handle, PointerMode* mode)
    int cublasSetPointerMode(Handle handle, PointerMode mode)

    # Stream
    int cublasSetStream(Handle handle, driver.Stream streamId)
    int cublasGetStream(Handle handle, driver.Stream* streamId)

    # BLAS Level 1
    int cublasIsamax(Handle handle, int n, float* x, int incx, int* result)
    int cublasIsamin(Handle handle, int n, float* x, int incx, int* result)
    int cublasSasum(Handle handle, int n, float* x, int incx,
                    float* result)
    int cublasSaxpy(Handle handle, int n, float* alpha, float* x,
                    int incx, float* y, int incy)
    int cublasDaxpy(Handle handle, int n, double* alpha, double* x,
                    int incx, double* y, int incy)
    int cublasSdot(Handle handle, int n, float* x, int incx,
                   float* y, int incy, float* result)
    int cublasDdot(Handle handle, int n, double* x, int incx,
                   double* y, int incy, double* result)
    int cublasCdotu(Handle handle, int n, cuComplexPtr x, int incx,
                    cuComplexPtr y, int incy, cuComplexPtr result)
    int cublasCdotc(Handle handle, int n, cuComplexPtr x, int incx,
                    cuComplexPtr y, int incy, cuComplexPtr result)
    int cublasZdotu(Handle handle, int n, cuDoubleComplexPtr x, int incx,
                    cuDoubleComplexPtr y, int incy, cuDoubleComplexPtr result)
    int cublasZdotc(Handle handle, int n, cuDoubleComplexPtr x, int incx,
                    cuDoubleComplexPtr y, int incy, cuDoubleComplexPtr result)
    int cublasSnrm2(Handle handle, int n, float* x, int incx,
                    float* result)
    int cublasSscal(Handle handle, int n, float* alpha, float* x, int incx)

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
        Handle handle, Operation trans, int m, int n, cuComplexPtr alpha,
        cuComplexPtr A, int lda, cuComplexPtr x, int incx, cuComplexPtr beta,
        cuComplexPtr y, int incy)
    int cublasZgemv(
        Handle handle, Operation trans, int m, int n, cuDoubleComplexPtr alpha,
        cuDoubleComplexPtr A, int lda, cuDoubleComplexPtr x, int incx,
        cuDoubleComplexPtr beta, cuDoubleComplexPtr y, int incy)
    int cublasSger(
        Handle handle, int m, int n, float* alpha, float* x, int incx,
        float* y, int incy, float* A, int lda)
    int cublasDger(
        Handle handle, int m, int n, double* alpha, double* x,
        int incx, double* y, int incy, double* A, int lda)
    int cublasCgeru(
        Handle handle, int m, int n, cuComplexPtr alpha, cuComplexPtr x,
        int incx, cuComplexPtr y, int incy, cuComplexPtr A, int lda)
    int cublasCgerc(
        Handle handle, int m, int n, cuComplexPtr alpha, cuComplexPtr x,
        int incx, cuComplexPtr y, int incy, cuComplexPtr A, int lda)
    int cublasZgeru(
        Handle handle, int m, int n, cuDoubleComplexPtr alpha,
        cuDoubleComplexPtr x, int incx, cuDoubleComplexPtr y, int incy,
        cuDoubleComplexPtr A, int lda)
    int cublasZgerc(
        Handle handle, int m, int n, cuDoubleComplexPtr alpha,
        cuDoubleComplexPtr x, int incx, cuDoubleComplexPtr y, int incy,
        cuDoubleComplexPtr A, int lda)

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
        int n, int k, cuComplexPtr alpha, cuComplexPtr A, int lda,
        cuComplexPtr B, int ldb, cuComplexPtr beta, cuComplexPtr C, int ldc)
    int cublasZgemm(
        Handle handle, Operation transa, Operation transb, int m,
        int n, int k, cuDoubleComplexPtr alpha, cuDoubleComplexPtr A, int lda,
        cuDoubleComplexPtr B, int ldb, cuDoubleComplexPtr beta,
        cuDoubleComplexPtr C, int ldc)
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
        int n, int k, constCuComplexPtr alpha, constCuComplexPtrPtr Aarray,
        int lda, constCuComplexPtrPtr Barray, int ldb, constCuComplexPtr beta,
        cuComplexPtrPtr Carray, int ldc, int batchCount)
    int cublasZgemmBatched(
        Handle handle, Operation transa, Operation transb, int m,
        int n, int k, constCuDoubleComplexPtr alpha,
        constCuDoubleComplexPtrPtr Aarray, int lda,
        constCuDoubleComplexPtrPtr Barray, int ldb,
        constCuDoubleComplexPtr beta, cuDoubleComplexPtrPtr Carray, int ldc,
        int batchCount)

    # BLAS extension
    int cublasSdgmm(
        Handle handle, SideMode mode, int m, int n, float* A, int lda,
        float* x, int incx, float* C, int ldc)
    int cublasSgemmEx(
        Handle handle, Operation transa,
        Operation transb, int m, int n, int k,
        const float *alpha, const void *A, runtime.DataType Atype,
        int lda, const void *B, runtime.DataType Btype, int ldb,
        const float *beta, void *C, runtime.DataType Ctype, int ldc)
    int cublasSgetrfBatched(
        Handle handle, int n, float **Aarray, int lda,
        int *PivotArray, int *infoArray, int batchSize)
    int cublasSgetriBatched(
        Handle handle, int n, const float **Aarray, int lda,
        int *PivotArray, float *Carray[], int ldc, int *infoArray,
        int batchSize)


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
    status = cublasCreate(&handle)
    check_status(status)
    return <size_t>handle


cpdef void destroy(size_t handle) except *:
    status = cublasDestroy(<Handle>handle)
    check_status(status)


cpdef int getVersion(size_t handle) except *:
    cdef int version
    status = cublasGetVersion(<Handle>handle, &version)
    check_status(status)
    return version


cpdef int getPointerMode(size_t handle) except *:
    cdef PointerMode mode
    status = cublasGetPointerMode(<Handle>handle, &mode)
    check_status(status)
    return mode


cpdef setPointerMode(size_t handle, int mode):
    status = cublasSetPointerMode(<Handle>handle, <PointerMode>mode)
    check_status(status)


###############################################################################
# Stream
###############################################################################

cpdef setStream(size_t handle, size_t stream):
    status = cublasSetStream(<Handle>handle, <driver.Stream>stream)
    check_status(status)


cpdef size_t getStream(size_t handle) except *:
    cdef driver.Stream stream
    status = cublasGetStream(<Handle>handle, &stream)
    check_status(status)
    return <size_t>stream


###############################################################################
# BLAS Level 1
###############################################################################

cpdef int isamax(size_t handle, int n, size_t x, int incx) except *:
    cdef int result
    status = cublasIsamax(
        <Handle>handle, n, <float*>x, incx, &result)
    check_status(status)
    return result


cpdef int isamin(size_t handle, int n, size_t x, int incx) except *:
    cdef int result
    status = cublasIsamin(
        <Handle>handle, n, <float*>x, incx, &result)
    check_status(status)
    return result


cpdef float sasum(size_t handle, int n, size_t x, int incx) except *:
    cdef float result
    status = cublasSasum(
        <Handle>handle, n, <float*>x, incx, &result)
    check_status(status)
    return result


cpdef saxpy(size_t handle, int n, float alpha, size_t x, int incx, size_t y,
            int incy):
    status = cublasSaxpy(
        <Handle>handle, n, &alpha, <float*>x, incx, <float*>y, incy)
    check_status(status)


cpdef daxpy(size_t handle, int n, double alpha, size_t x, int incx, size_t y,
            int incy):
    status = cublasDaxpy(
        <Handle>handle, n, &alpha, <double*>x, incx, <double*>y, incy)
    check_status(status)


cpdef sdot(size_t handle, int n, size_t x, int incx, size_t y, int incy,
           size_t result):
    status = cublasSdot(
        <Handle>handle, n, <float*>x, incx, <float*>y, incy, <float*>result)
    check_status(status)


cpdef ddot(size_t handle, int n, size_t x, int incx, size_t y, int incy,
           size_t result):
    status = cublasDdot(
        <Handle>handle, n, <double*>x, incx, <double*>y, incy, <double*>result)
    check_status(status)


cpdef cdotu(size_t handle, int n, size_t x, int incx, size_t y, int incy,
            size_t result):
    status = cublasCdotu(
        <Handle>handle, n, <cuComplexPtr>x, incx, <cuComplexPtr>y, incy,
        <cuComplexPtr>result)
    check_status(status)


cpdef cdotc(size_t handle, int n, size_t x, int incx, size_t y, int incy,
            size_t result):
    status = cublasCdotc(
        <Handle>handle, n, <cuComplexPtr>x, incx, <cuComplexPtr>y, incy,
        <cuComplexPtr>result)
    check_status(status)


cpdef zdotu(size_t handle, int n, size_t x, int incx, size_t y, int incy,
            size_t result):
    status = cublasZdotu(
        <Handle>handle, n, <cuDoubleComplexPtr>x, incx, <cuDoubleComplexPtr>y,
        incy, <cuDoubleComplexPtr>result)
    check_status(status)


cpdef zdotc(size_t handle, int n, size_t x, int incx, size_t y, int incy,
            size_t result):
    status = cublasZdotc(
        <Handle>handle, n, <cuDoubleComplexPtr>x, incx, <cuDoubleComplexPtr>y,
        incy, <cuDoubleComplexPtr>result)
    check_status(status)


cpdef float snrm2(size_t handle, int n, size_t x, int incx) except *:
    cdef float result
    status = cublasSnrm2(<Handle>handle, n, <float*>x, incx, &result)
    check_status(status)
    return result


cpdef sscal(size_t handle, int n, float alpha, size_t x, int incx):
    status = cublasSscal(<Handle>handle, n, &alpha, <float*>x, incx)
    check_status(status)


###############################################################################
# BLAS Level 2
###############################################################################

cpdef sgemv(size_t handle, int trans, int m, int n, float alpha, size_t A,
            int lda, size_t x, int incx, float beta, size_t y, int incy):
    status = cublasSgemv(
        <Handle>handle, <Operation>trans, m, n, &alpha,
        <float*>A, lda, <float*>x, incx, &beta, <float*>y, incy)
    check_status(status)


cpdef dgemv(size_t handle, int trans, int m, int n, double alpha, size_t A,
            int lda, size_t x, int incx, double beta, size_t y, int incy):
    status = cublasDgemv(
        <Handle>handle, <Operation>trans, m, n, &alpha,
        <double*>A, lda, <double*>x, incx, &beta, <double*>y, incy)
    check_status(status)


cpdef cgemv(size_t handle, int trans, int m, int n, float complex alpha,
            size_t A, int lda, size_t x, int incx, float complex beta,
            size_t y, int incy):
    status = cublasCgemv(
        <Handle>handle, <Operation>trans, m, n, <cuComplexPtr>&alpha,
        <cuComplexPtr>A, lda, <cuComplexPtr>x, incx, <cuComplexPtr>&beta,
        <cuComplexPtr>y, incy)
    check_status(status)


cpdef zgemv(size_t handle, int trans, int m, int n, double complex alpha,
            size_t A, int lda, size_t x, int incx, double complex beta,
            size_t y, int incy):
    status = cublasZgemv(
        <Handle>handle, <Operation>trans, m, n, <cuDoubleComplexPtr>&alpha,
        <cuDoubleComplexPtr>A, lda, <cuDoubleComplexPtr>x, incx,
        <cuDoubleComplexPtr>&beta, <cuDoubleComplexPtr>y, incy)
    check_status(status)


cpdef sger(size_t handle, int m, int n, float alpha, size_t x, int incx,
           size_t y, int incy, size_t A, int lda):
    status = cublasSger(
        <Handle>handle, m, n, &alpha, <float*>x, incx, <float*>y, incy,
        <float*>A, lda)
    check_status(status)


cpdef dger(size_t handle, int m, int n, double alpha, size_t x, int incx,
           size_t y, int incy, size_t A, int lda):
    status = cublasDger(
        <Handle>handle, m, n, &alpha, <double*>x, incx, <double*>y, incy,
        <double*>A, lda)
    check_status(status)


cpdef cgeru(size_t handle, int m, int n, float complex alpha, size_t x,
            int incx, size_t y, int incy, size_t A, int lda):
    status = cublasCgeru(
        <Handle>handle, m, n, <cuComplexPtr>&alpha, <cuComplexPtr>x, incx,
        <cuComplexPtr>y, incy, <cuComplexPtr>A, lda)
    check_status(status)


cpdef cgerc(size_t handle, int m, int n, float complex alpha, size_t x,
            int incx, size_t y, int incy, size_t A, int lda):
    status = cublasCgerc(
        <Handle>handle, m, n, <cuComplexPtr>&alpha, <cuComplexPtr>x, incx,
        <cuComplexPtr>y, incy, <cuComplexPtr>A, lda)
    check_status(status)


cpdef zgeru(size_t handle, int m, int n, double complex alpha, size_t x,
            int incx, size_t y, int incy, size_t A, int lda):
    status = cublasZgeru(
        <Handle>handle, m, n, <cuDoubleComplexPtr>&alpha,
        <cuDoubleComplexPtr>x, incx, <cuDoubleComplexPtr>y, incy,
        <cuDoubleComplexPtr>A, lda)
    check_status(status)


cpdef zgerc(size_t handle, int m, int n, double complex alpha, size_t x,
            int incx, size_t y, int incy, size_t A, int lda):
    status = cublasZgerc(
        <Handle>handle, m, n, <cuDoubleComplexPtr>&alpha,
        <cuDoubleComplexPtr>x, incx, <cuDoubleComplexPtr>y, incy,
        <cuDoubleComplexPtr>A, lda)
    check_status(status)


###############################################################################
# BLAS Level 3
###############################################################################

cpdef sgemm(size_t handle, int transa, int transb,
            int m, int n, int k, float alpha, size_t A, int lda,
            size_t B, int ldb, float beta, size_t C, int ldc):
    status = cublasSgemm(
        <Handle>handle, <Operation>transa, <Operation>transb, m, n, k,
        &alpha, <float*>A, lda, <float*>B, ldb, &beta, <float*>C, ldc)
    check_status(status)


cpdef dgemm(size_t handle, int transa, int transb,
            int m, int n, int k, double alpha, size_t A, int lda,
            size_t B, int ldb, double beta, size_t C, int ldc):
    status = cublasDgemm(
        <Handle>handle, <Operation>transa, <Operation>transb, m, n, k,
        &alpha, <double*>A, lda, <double*>B, ldb, &beta, <double*>C, ldc)
    check_status(status)


cpdef cgemm(size_t handle, int transa, int transb,
            int m, int n, int k, float complex alpha, size_t A, int lda,
            size_t B, int ldb, float complex beta, size_t C, int ldc):
    status = cublasCgemm(
        <Handle>handle, <Operation>transa, <Operation>transb, m, n, k,
        <cuComplexPtr>&alpha, <cuComplexPtr>A, lda, <cuComplexPtr>B, ldb,
        <cuComplexPtr>&beta, <cuComplexPtr>C, ldc)
    check_status(status)


cpdef zgemm(size_t handle, int transa, int transb,
            int m, int n, int k, double complex alpha, size_t A, int lda,
            size_t B, int ldb, double complex beta, size_t C, int ldc):
    status = cublasZgemm(
        <Handle>handle, <Operation>transa, <Operation>transb, m, n, k,
        <cuDoubleComplexPtr>&alpha, <cuDoubleComplexPtr>A, lda,
        <cuDoubleComplexPtr>B, ldb, <cuDoubleComplexPtr>&beta,
        <cuDoubleComplexPtr>C, ldc)
    check_status(status)


cpdef sgemmBatched(
        size_t handle, int transa, int transb, int m, int n, int k,
        float alpha, size_t Aarray, int lda, size_t Barray, int ldb,
        float beta, size_t Carray, int ldc, int batchCount):
    status = cublasSgemmBatched(
        <Handle>handle, <Operation>transa, <Operation>transb, m, n, k,
        &alpha, <const float**>Aarray, lda, <const float**>Barray, ldb, &beta,
        <float**>Carray, ldc, batchCount)
    check_status(status)


cpdef dgemmBatched(
        size_t handle, int transa, int transb, int m, int n, int k,
        double alpha, size_t Aarray, int lda, size_t Barray, int ldb,
        double beta, size_t Carray, int ldc, int batchCount):
    status = cublasDgemmBatched(
        <Handle>handle, <Operation>transa, <Operation>transb, m, n, k,
        &alpha, <const double**>Aarray, lda, <const double**>Barray, ldb,
        &beta, <double**>Carray, ldc, batchCount)
    check_status(status)


cpdef cgemmBatched(
        size_t handle, int transa, int transb, int m, int n, int k,
        float complex alpha, size_t Aarray, int lda, size_t Barray, int ldb,
        float complex beta, size_t Carray, int ldc, int batchCount):
    status = cublasCgemmBatched(
        <Handle>handle, <Operation>transa, <Operation>transb, m, n, k,
        <constCuComplexPtr>&alpha, <constCuComplexPtrPtr>Aarray, lda,
        <constCuComplexPtrPtr>Barray, ldb, <constCuComplexPtr>&beta,
        <cuComplexPtrPtr>Carray, ldc, batchCount)
    check_status(status)


cpdef zgemmBatched(
        size_t handle, int transa, int transb, int m, int n, int k,
        double complex alpha, size_t Aarray, int lda, size_t Barray, int ldb,
        double complex beta, size_t Carray, int ldc, int batchCount):
    status = cublasZgemmBatched(
        <Handle>handle, <Operation>transa, <Operation>transb, m, n, k,
        <constCuDoubleComplexPtr>&alpha, <constCuDoubleComplexPtrPtr>Aarray,
        lda, <constCuDoubleComplexPtrPtr>Barray, ldb,
        <constCuDoubleComplexPtr>&beta, <cuDoubleComplexPtrPtr>Carray, ldc,
        batchCount)
    check_status(status)


###############################################################################
# BLAS extension
###############################################################################

cpdef sdgmm(size_t handle, int mode, int m, int n, size_t A, int lda,
            size_t x, int incx, size_t C, int ldc):
    status = cublasSdgmm(
        <Handle>handle, <SideMode>mode, m, n, <float*>A, lda, <float*>x, incx,
        <float*>C, ldc)
    check_status(status)


cpdef sgemmEx(
        size_t handle, int transa, int transb, int m, int n, int k,
        float alpha, size_t A, int Atype, int lda, size_t B,
        int Btype, int ldb, float beta, size_t C, int Ctype,
        int ldc):
    status = cublasSgemmEx(
        <Handle>handle, <Operation>transa, <Operation>transb, m, n, k,
        &alpha, <const void*>A, <runtime.DataType>Atype, lda, <const void*>B,
        <runtime.DataType>Btype, ldb, &beta, <void*>C,
        <runtime.DataType>Ctype, ldc)
    check_status(status)


cpdef sgetrfBatched(size_t handle, int n, size_t Aarray, int lda,
                    size_t PivotArray, size_t infoArray, int batchSize):
    status = cublasSgetrfBatched(
        <Handle>handle, n, <float**>Aarray, lda, <int*>PivotArray,
        <int*>infoArray, batchSize)
    check_status(status)


cpdef sgetriBatched(
        size_t handle, int n, size_t Aarray, int lda, size_t PivotArray,
        size_t Carray, int ldc, size_t infoArray, int batchSize):
    status = cublasSgetriBatched(
        <Handle>handle, n, <const float**>Aarray, lda, <int*>PivotArray,
        <float**>Carray, ldc, <int*>infoArray, batchSize)
    check_status(status)
