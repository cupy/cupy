"""Thin wrapper of CUBLAS."""


###############################################################################
# Types
###############################################################################

cdef extern from *:
    ctypedef void* cuComplexPtr 'cuComplex*'
    ctypedef void* cuDoubleComplexPtr 'cuDoubleComplex*'


cdef extern from *:
    ctypedef void* Handle 'cublasHandle_t'

    ctypedef int DiagType 'cublasDiagType_t'
    ctypedef int FillMode 'cublasFillMode_t'
    ctypedef int Operation 'cublasOperation_t'
    ctypedef int PointerMode 'cublasPointerMode_t'
    ctypedef int SideMode 'cublasSideMode_t'
    ctypedef int GemmAlgo 'cublasGemmAlgo_t'
    ctypedef int Math 'cublasMath_t'


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

    CUBLAS_DIAG_NON_UNIT = 0
    CUBLAS_DIAG_UNIT = 1

    CUBLAS_GEMM_DEFAULT = -1
    CUBLAS_GEMM_DEFAULT_TENSOR_OP = 99

    # The following two are left for backward compatibility; renamed from
    # `DFALT` to `DEFAULT` in CUDA 9.1.
    CUBLAS_GEMM_DFALT = -1
    CUBLAS_GEMM_DFALT_TENSOR_OP = 99

    CUBLAS_DEFAULT_MATH = 0
    CUBLAS_TENSOR_OP_MATH = 1

###############################################################################
# Context
###############################################################################

cpdef size_t create() except? 0
cpdef destroy(size_t handle)
cpdef int getVersion(size_t handle) except? -1
cpdef int getPointerMode(size_t handle) except? -1
cpdef setPointerMode(size_t handle, int mode)


###############################################################################
# Stream
###############################################################################

cpdef setStream(size_t handle, size_t stream)
cpdef size_t getStream(size_t handle) except? 0


###############################################################################
# Math Mode
###############################################################################

cpdef setMathMode(size_t handle, int mode)
cpdef int getMathMode(size_t handle) except? -1


###############################################################################
# BLAS Level 1
###############################################################################

cpdef int isamax(size_t handle, int n, size_t x, int incx) except? 0
cpdef int isamin(size_t handle, int n, size_t x, int incx) except? 0
cpdef float sasum(size_t handle, int n, size_t x, int incx) except? 0
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
cpdef float snrm2(size_t handle, int n, size_t x, int incx) except? 0
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
cpdef sgemmStridedBatched(size_t handle, int transa, int transb,
                          int m, int n, int k, float alpha,
                          size_t A, int lda, long long strideA,
                          size_t B, int ldb, long long strideB,
                          float beta,
                          size_t C, int ldc, long long strideC,
                          int batchCount)
cpdef dgemmStridedBatched(size_t handle, int transa, int transb,
                          int m, int n, int k, double alpha,
                          size_t A, int lda, long long strideA,
                          size_t B, int ldb, long long strideB,
                          double beta,
                          size_t C, int ldc, long long strideC,
                          int batchCount)
cpdef cgemmStridedBatched(size_t handle, int transa, int transb,
                          int m, int n, int k, float complex alpha,
                          size_t A, int lda, long long strideA,
                          size_t B, int ldb, long long strideB,
                          float complex beta,
                          size_t C, int ldc, long long strideC,
                          int batchCount)
cpdef zgemmStridedBatched(size_t handle, int transa, int transb,
                          int m, int n, int k, double complex alpha,
                          size_t A, int lda, long long strideA,
                          size_t B, int ldb, long long strideB,
                          double complex beta,
                          size_t C, int ldc, long long strideC,
                          int batchCount)
cpdef strsm(size_t handle, int side, int uplo, int trans, int diag,
            int m, int n, float alpha, size_t Aarray, int lda,
            size_t Barray, int ldb)
cpdef dtrsm(size_t handle, int side, int uplo, int trans, int diag,
            int m, int n, double alpha, size_t Aarray, int lda,
            size_t Barray, int ldb)
cpdef ctrsm(size_t handle, int side, int uplo, int trans, int diag,
            int m, int n, float complex alpha, size_t Aarray, int lda,
            size_t Barray, int ldb)
cpdef ztrsm(size_t handle, int side, int uplo, int trans, int diag,
            int m, int n, double complex alpha, size_t Aarray, int lda,
            size_t Barray, int ldb)

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
cpdef dgetrfBatched(size_t handle, int n, size_t Aarray, int lda,
                    size_t PivotArray, size_t infoArray, int batchSize)
cpdef cgetrfBatched(size_t handle, int n, size_t Aarray, int lda,
                    size_t PivotArray, size_t infoArray, int batchSize)
cpdef zgetrfBatched(size_t handle, int n, size_t Aarray, int lda,
                    size_t PivotArray, size_t infoArray, int batchSize)

cpdef sgetriBatched(size_t handle, int n, size_t Aarray, int lda,
                    size_t PivotArray, size_t Carray, int ldc,
                    size_t infoArray, int batchSize)
cpdef dgetriBatched(size_t handle, int n, size_t Aarray, int lda,
                    size_t PivotArray, size_t Carray, int ldc,
                    size_t infoArray, int batchSize)
cpdef cgetriBatched(size_t handle, int n, size_t Aarray, int lda,
                    size_t PivotArray, size_t Carray, int ldc,
                    size_t infoArray, int batchSize)
cpdef zgetriBatched(size_t handle, int n, size_t Aarray, int lda,
                    size_t PivotArray, size_t Carray, int ldc,
                    size_t infoArray, int batchSize)
cpdef gemmEx(size_t handle, int transa, int transb, int m, int n, int k,
             size_t alpha, size_t A, int Atype, int lda, size_t B,
             int Btype, int ldb, size_t beta, size_t C, int Ctype,
             int ldc, int computeType, int algo)

cpdef stpttr(size_t handle, int uplo, int n, size_t AP, size_t A, int lda)
cpdef dtpttr(size_t handle, int uplo, int n, size_t AP, size_t A, int lda)

cpdef strttp(size_t handle, int uplo, int n, size_t A, int lda, size_t AP)
cpdef dtrttp(size_t handle, int uplo, int n, size_t A, int lda, size_t AP)
