"""Thin wrapper of CUBLAS."""


###############################################################################
# Types
###############################################################################

cdef extern from *:
    ctypedef void* cuComplexPtr 'cuComplex*'
    ctypedef void* cuDoubleComplexPtr 'cuDoubleComplex*'


cdef extern from *:
    ctypedef void* Handle 'cublasHandle_t'

    ctypedef int Operation 'cublasOperation_t'
    ctypedef int PointerMode 'cublasPointerMode_t'
    ctypedef int SideMode 'cublasSideMode_t'


###############################################################################
# Enum
###############################################################################

cpdef enum:
    CUBLAS_OP_N = 0
    CUBLAS_OP_T = 1
    CUBLAS_OP_C = 2

    CUBLAS_POINTER_MODE_HOST = 0
    CUBLAS_POINTER_MODE_DEVICE = 1

    CUBLAS_SIDE_LEFT = 0
    CUBLAS_SIDE_RIGHT = 1

    CUBLAS_FILL_MODE_LOWER = 0
    CUBLAS_FILL_MODE_UPPER = 1


###############################################################################
# Context
###############################################################################

cpdef size_t create() except *
cpdef void destroy(size_t handle) except *
cpdef int getVersion(size_t handle) except *
cpdef int getPointerMode(size_t handle) except *
cpdef setPointerMode(size_t handle, int mode)


###############################################################################
# Stream
###############################################################################

cpdef setStream(size_t handle, size_t stream)
cpdef size_t getStream(size_t handle) except *


###############################################################################
# BLAS Level 1
###############################################################################

cpdef int isamax(size_t handle, int n, size_t x, int incx) except *
cpdef int isamin(size_t handle, int n, size_t x, int incx) except *
cpdef float sasum(size_t handle, int n, size_t x, int incx) except *
cpdef saxpy(size_t handle, int n, float alpha, size_t x, int incx, size_t y,
            int incy)
cpdef daxpy(size_t handle, int n, double alpha, size_t x, int incx, size_t y,
            int incy)
cpdef sdot(size_t handle, int n, size_t x, int incx, size_t y, int incy,
           size_t result)
cpdef ddot(size_t handle, int n, size_t x, int incx, size_t y, int incy,
           size_t result)
cpdef cdotu(size_t handle, int n, size_t x, int incx, size_t y, int incy,
            size_t result)
cpdef cdotc(size_t handle, int n, size_t x, int incx, size_t y, int incy,
            size_t result)
cpdef zdotu(size_t handle, int n, size_t x, int incx, size_t y, int incy,
            size_t result)
cpdef zdotc(size_t handle, int n, size_t x, int incx, size_t y, int incy,
            size_t result)
cpdef float snrm2(size_t handle, int n, size_t x, int incx) except *
cpdef sscal(size_t handle, int n, float alpha, size_t x, int incx)


###############################################################################
# BLAS Level 2
###############################################################################

cpdef sgemv(size_t handle, int trans, int m, int n, float alpha, size_t A,
            int lda, size_t x, int incx, float beta, size_t y, int incy)
cpdef dgemv(size_t handle, int trans, int m, int n, double alpha, size_t A,
            int lda, size_t x, int incx, double beta, size_t y, int incy)
cpdef cgemv(size_t handle, int trans, int m, int n, float complex alpha,
            size_t A, int lda, size_t x, int incx, float complex beta,
            size_t y, int incy)
cpdef zgemv(size_t handle, int trans, int m, int n, double complex alpha,
            size_t A, int lda, size_t x, int incx, double complex beta,
            size_t y, int incy)
cpdef sger(size_t handle, int m, int n, float alpha, size_t x, int incx,
           size_t y, int incy, size_t A, int lda)
cpdef dger(size_t handle, int m, int n, double alpha, size_t x, int incx,
           size_t y, int incy, size_t A, int lda)
cpdef cgeru(size_t handle, int m, int n, float complex alpha, size_t x,
            int incx, size_t y, int incy, size_t A, int lda)
cpdef cgerc(size_t handle, int m, int n, float complex alpha, size_t x,
            int incx, size_t y, int incy, size_t A, int lda)
cpdef zgeru(size_t handle, int m, int n, double complex alpha, size_t x,
            int incx, size_t y, int incy, size_t A, int lda)
cpdef zgerc(size_t handle, int m, int n, double complex alpha, size_t x,
            int incx, size_t y, int incy, size_t A, int lda)


###############################################################################
# BLAS Level 3
###############################################################################

cpdef sgemm(size_t handle, int transa, int transb,
            int m, int n, int k, float alpha, size_t A, int lda,
            size_t B, int ldb, float beta, size_t C, int ldc)
cpdef dgemm(size_t handle, int transa, int transb,
            int m, int n, int k, double alpha, size_t A, int lda,
            size_t B, int ldb, double beta, size_t C, int ldc)
cpdef cgemm(size_t handle, int transa, int transb,
            int m, int n, int k, float complex alpha, size_t A, int lda,
            size_t B, int ldb, float complex beta, size_t C, int ldc)
cpdef zgemm(size_t handle, int transa, int transb,
            int m, int n, int k, double complex alpha, size_t A, int lda,
            size_t B, int ldb, double complex beta, size_t C, int ldc)
cpdef sgemmBatched(size_t handle, int transa, int transb,
                   int m, int n, int k, float alpha, size_t Aarray, int lda,
                   size_t Barray, int ldb, float beta, size_t Carray, int ldc,
                   int batchCount)
cpdef dgemmBatched(size_t handle, int transa, int transb,
                   int m, int n, int k, double alpha, size_t Aarray, int lda,
                   size_t Barray, int ldb, double beta, size_t Carray, int ldc,
                   int batchCount)
cpdef cgemmBatched(size_t handle, int transa, int transb,
                   int m, int n, int k, float complex alpha, size_t Aarray,
                   int lda, size_t Barray, int ldb, float complex beta,
                   size_t Carray, int ldc, int batchCount)
cpdef zgemmBatched(size_t handle, int transa, int transb,
                   int m, int n, int k, double complex alpha, size_t Aarray,
                   int lda, size_t Barray, int ldb, double complex beta,
                   size_t Carray, int ldc, int batchCount)

###############################################################################
# BLAS extension
###############################################################################

cpdef sgeam(size_t handle, int transa, int transb, int m, int n,
            float alpha, size_t A, int lda, float beta, size_t B, int ldb,
            size_t C, int ldc)
cpdef dgeam(size_t handle, int transa, int transb, int m, int n,
            double alpha, size_t A, int lda, double beta, size_t B, int ldb,
            size_t C, int ldc)
cpdef sdgmm(size_t handle, int mode, int m, int n, size_t A, int lda,
            size_t x, int incx, size_t C, int ldc)
cpdef sgemmEx(size_t handle, int transa, int transb, int m, int n, int k,
              float alpha, size_t A, int Atype, int lda, size_t B,
              int Btype, int ldb, float beta, size_t C, int Ctype,
              int ldc)
cpdef sgetrfBatched(size_t handle, int n, size_t Aarray, int lda,
                    size_t PivotArray, size_t infoArray, int batchSize)

cpdef sgetriBatched(size_t handle, int n, size_t Aarray, int lda,
                    size_t PivotArray, size_t Carray, int ldc,
                    size_t infoArray, int batchSize)
