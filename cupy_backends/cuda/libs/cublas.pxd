"""Thin wrapper of CUBLAS."""
from libc.stdint cimport intptr_t


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
    ctypedef int ComputeType 'cublasComputeType_t'


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

    # cublasComputeType_t added in CUDA 11.0
    CUBLAS_COMPUTE_16F = 64            # half - default
    CUBLAS_COMPUTE_16F_PEDANTIC = 65   # half - pedantic
    CUBLAS_COMPUTE_32F = 68            # float - default
    CUBLAS_COMPUTE_32F_PEDANTIC = 69   # float - pedantic
    CUBLAS_COMPUTE_32F_FAST_16F = 74   # float - fast, allows down-converting inputs to half or TF32  # NOQA
    CUBLAS_COMPUTE_32F_FAST_16BF = 75  # float - fast, allows down-converting inputs to bfloat16 or TF32  # NOQA
    CUBLAS_COMPUTE_32F_FAST_TF32 = 77  # float - fast, allows down-converting inputs to TF32  # NOQA
    CUBLAS_COMPUTE_64F = 70            # double - default
    CUBLAS_COMPUTE_64F_PEDANTIC = 71   # double - pedantic
    CUBLAS_COMPUTE_32I = 72            # signed 32-bit int - default
    CUBLAS_COMPUTE_32I_PEDANTIC = 73   # signed 32-bit int - pedantic

###############################################################################
# Context
###############################################################################

cpdef intptr_t create() except? 0
cpdef destroy(intptr_t handle)
cpdef int getVersion(intptr_t handle) except? -1
cpdef int getPointerMode(intptr_t handle) except? -1
cpdef setPointerMode(intptr_t handle, int mode)


###############################################################################
# Stream
###############################################################################

cpdef setStream(intptr_t handle, size_t stream)
cpdef size_t getStream(intptr_t handle) except? 0


###############################################################################
# Math Mode
###############################################################################

cpdef setMathMode(intptr_t handle, int mode)
cpdef int getMathMode(intptr_t handle) except? -1


###############################################################################
# BLAS Level 1
###############################################################################

cpdef isamax(intptr_t handle, int n, size_t x, int incx, size_t result)
cpdef idamax(intptr_t handle, int n, size_t x, int incx, size_t result)
cpdef icamax(intptr_t handle, int n, size_t x, int incx, size_t result)
cpdef izamax(intptr_t handle, int n, size_t x, int incx, size_t result)
cpdef isamin(intptr_t handle, int n, size_t x, int incx, size_t result)
cpdef idamin(intptr_t handle, int n, size_t x, int incx, size_t result)
cpdef icamin(intptr_t handle, int n, size_t x, int incx, size_t result)
cpdef izamin(intptr_t handle, int n, size_t x, int incx, size_t result)
cpdef sasum(intptr_t handle, int n, size_t x, int incx, size_t result)
cpdef dasum(intptr_t handle, int n, size_t x, int incx, size_t result)
cpdef scasum(intptr_t handle, int n, size_t x, int incx, size_t result)
cpdef dzasum(intptr_t handle, int n, size_t x, int incx, size_t result)
cpdef saxpy(intptr_t handle, int n, size_t alpha, size_t x, int incx, size_t y,
            int incy)
cpdef daxpy(intptr_t handle, int n, size_t alpha, size_t x, int incx, size_t y,
            int incy)
cpdef caxpy(intptr_t handle, int n, size_t alpha, size_t x, int incx, size_t y,
            int incy)
cpdef zaxpy(intptr_t handle, int n, size_t alpha, size_t x, int incx, size_t y,
            int incy)
cpdef sdot(intptr_t handle, int n, size_t x, int incx, size_t y, int incy,
           size_t result)
cpdef ddot(intptr_t handle, int n, size_t x, int incx, size_t y, int incy,
           size_t result)
cpdef cdotu(intptr_t handle, int n, size_t x, int incx, size_t y, int incy,
            size_t result)
cpdef cdotc(intptr_t handle, int n, size_t x, int incx, size_t y, int incy,
            size_t result)
cpdef zdotu(intptr_t handle, int n, size_t x, int incx, size_t y, int incy,
            size_t result)
cpdef zdotc(intptr_t handle, int n, size_t x, int incx, size_t y, int incy,
            size_t result)
cpdef snrm2(intptr_t handle, int n, size_t x, int incx, size_t result)
cpdef dnrm2(intptr_t handle, int n, size_t x, int incx, size_t result)
cpdef scnrm2(intptr_t handle, int n, size_t x, int incx, size_t result)
cpdef dznrm2(intptr_t handle, int n, size_t x, int incx, size_t result)
cpdef sscal(intptr_t handle, int n, size_t alpha, size_t x, int incx)
cpdef dscal(intptr_t handle, int n, size_t alpha, size_t x, int incx)
cpdef cscal(intptr_t handle, int n, size_t alpha, size_t x, int incx)
cpdef csscal(intptr_t handle, int n, size_t alpha, size_t x, int incx)
cpdef zscal(intptr_t handle, int n, size_t alpha, size_t x, int incx)
cpdef zdscal(intptr_t handle, int n, size_t alpha, size_t x, int incx)


###############################################################################
# BLAS Level 2
###############################################################################

cpdef sgemv(intptr_t handle, int trans, int m, int n, size_t alpha, size_t A,
            int lda, size_t x, int incx, size_t beta, size_t y, int incy)
cpdef dgemv(intptr_t handle, int trans, int m, int n, size_t alpha, size_t A,
            int lda, size_t x, int incx, size_t beta, size_t y, int incy)
cpdef cgemv(intptr_t handle, int trans, int m, int n, size_t alpha, size_t A,
            int lda, size_t x, int incx, size_t beta, size_t y, int incy)
cpdef zgemv(intptr_t handle, int trans, int m, int n, size_t alpha, size_t A,
            int lda, size_t x, int incx, size_t beta, size_t y, int incy)
cpdef sger(intptr_t handle, int m, int n, size_t alpha, size_t x, int incx,
           size_t y, int incy, size_t A, int lda)
cpdef dger(intptr_t handle, int m, int n, size_t alpha, size_t x, int incx,
           size_t y, int incy, size_t A, int lda)
cpdef cgeru(intptr_t handle, int m, int n, size_t alpha, size_t x, int incx,
            size_t y, int incy, size_t A, int lda)
cpdef cgerc(intptr_t handle, int m, int n, size_t alpha, size_t x, int incx,
            size_t y, int incy, size_t A, int lda)
cpdef zgeru(intptr_t handle, int m, int n, size_t alpha, size_t x, int incx,
            size_t y, int incy, size_t A, int lda)
cpdef zgerc(intptr_t handle, int m, int n, size_t alpha, size_t x, int incx,
            size_t y, int incy, size_t A, int lda)


###############################################################################
# BLAS Level 3
###############################################################################

cpdef sgemm(intptr_t handle, int transa, int transb,
            int m, int n, int k, size_t alpha, size_t A, int lda,
            size_t B, int ldb, size_t beta, size_t C, int ldc)
cpdef dgemm(intptr_t handle, int transa, int transb,
            int m, int n, int k, size_t alpha, size_t A, int lda,
            size_t B, int ldb, size_t beta, size_t C, int ldc)
cpdef cgemm(intptr_t handle, int transa, int transb,
            int m, int n, int k, size_t alpha, size_t A, int lda,
            size_t B, int ldb, size_t beta, size_t C, int ldc)
cpdef zgemm(intptr_t handle, int transa, int transb,
            int m, int n, int k, size_t alpha, size_t A, int lda,
            size_t B, int ldb, size_t beta, size_t C, int ldc)
cpdef sgemmBatched(intptr_t handle, int transa, int transb,
                   int m, int n, int k, size_t alpha, size_t Aarray, int lda,
                   size_t Barray, int ldb, size_t beta, size_t Carray, int ldc,
                   int batchCount)
cpdef dgemmBatched(intptr_t handle, int transa, int transb,
                   int m, int n, int k, size_t alpha, size_t Aarray, int lda,
                   size_t Barray, int ldb, size_t beta, size_t Carray, int ldc,
                   int batchCount)
cpdef cgemmBatched(intptr_t handle, int transa, int transb,
                   int m, int n, int k, size_t alpha, size_t Aarray, int lda,
                   size_t Barray, int ldb, size_t beta, size_t Carray, int ldc,
                   int batchCount)
cpdef zgemmBatched(intptr_t handle, int transa, int transb,
                   int m, int n, int k, size_t alpha, size_t Aarray, int lda,
                   size_t Barray, int ldb, size_t beta, size_t Carray, int ldc,
                   int batchCount)
cpdef sgemmStridedBatched(intptr_t handle, int transa, int transb, int m,
                          int n, int k, size_t alpha, size_t A, int lda,
                          long long strideA, size_t B, int ldb,
                          long long strideB, size_t beta, size_t C, int ldc,
                          long long strideC, int batchCount)
cpdef dgemmStridedBatched(intptr_t handle, int transa, int transb, int m,
                          int n, int k, size_t alpha, size_t A, int lda,
                          long long strideA, size_t B, int ldb,
                          long long strideB, size_t beta, size_t C, int ldc,
                          long long strideC, int batchCount)
cpdef cgemmStridedBatched(intptr_t handle, int transa, int transb, int m,
                          int n, int k, size_t alpha, size_t A, int lda,
                          long long strideA, size_t B, int ldb,
                          long long strideB, size_t beta, size_t C, int ldc,
                          long long strideC, int batchCount)
cpdef zgemmStridedBatched(intptr_t handle, int transa, int transb, int m,
                          int n, int k, size_t alpha, size_t A, int lda,
                          long long strideA, size_t B, int ldb,
                          long long strideB, size_t beta, size_t C, int ldc,
                          long long strideC, int batchCount)
cpdef strsm(intptr_t handle, int side, int uplo, int trans, int diag,
            int m, int n, size_t alpha, size_t Aarray, int lda,
            size_t Barray, int ldb)
cpdef dtrsm(intptr_t handle, int side, int uplo, int trans, int diag,
            int m, int n, size_t alpha, size_t Aarray, int lda,
            size_t Barray, int ldb)
cpdef ctrsm(intptr_t handle, int side, int uplo, int trans, int diag,
            int m, int n, size_t alpha, size_t Aarray, int lda,
            size_t Barray, int ldb)
cpdef ztrsm(intptr_t handle, int side, int uplo, int trans, int diag,
            int m, int n, size_t alpha, size_t Aarray, int lda,
            size_t Barray, int ldb)

###############################################################################
# BLAS extension
###############################################################################

cpdef sgeam(intptr_t handle, int transa, int transb, int m, int n,
            size_t alpha, size_t A, int lda, size_t beta, size_t B, int ldb,
            size_t C, int ldc)
cpdef dgeam(intptr_t handle, int transa, int transb, int m, int n,
            size_t alpha, size_t A, int lda, size_t beta, size_t B, int ldb,
            size_t C, int ldc)
cpdef cgeam(intptr_t handle, int transa, int transb, int m, int n,
            size_t alpha, size_t A, int lda, size_t beta, size_t B, int ldb,
            size_t C, int ldc)
cpdef zgeam(intptr_t handle, int transa, int transb, int m, int n,
            size_t alpha, size_t A, int lda, size_t beta, size_t B, int ldb,
            size_t C, int ldc)
cpdef sdgmm(intptr_t handle, int mode, int m, int n, size_t A, int lda,
            size_t x, int incx, size_t C, int ldc)
cpdef ddgmm(intptr_t handle, int mode, int m, int n, size_t A, int lda,
            size_t x, int incx, size_t C, int ldc)
cpdef cdgmm(intptr_t handle, int mode, int m, int n, size_t A, int lda,
            size_t x, int incx, size_t C, int ldc)
cpdef zdgmm(intptr_t handle, int mode, int m, int n, size_t A, int lda,
            size_t x, int incx, size_t C, int ldc)
cpdef sgemmEx(intptr_t handle, int transa, int transb, int m, int n, int k,
              size_t alpha, size_t A, int Atype, int lda, size_t B,
              int Btype, int ldb, size_t beta, size_t C, int Ctype,
              int ldc)
cpdef sgetrfBatched(intptr_t handle, int n, size_t Aarray, int lda,
                    size_t PivotArray, size_t infoArray, int batchSize)
cpdef dgetrfBatched(intptr_t handle, int n, size_t Aarray, int lda,
                    size_t PivotArray, size_t infoArray, int batchSize)
cpdef cgetrfBatched(intptr_t handle, int n, size_t Aarray, int lda,
                    size_t PivotArray, size_t infoArray, int batchSize)
cpdef zgetrfBatched(intptr_t handle, int n, size_t Aarray, int lda,
                    size_t PivotArray, size_t infoArray, int batchSize)

cpdef int sgetrsBatched(intptr_t handle, int trans, int n, int nrhs,
                        size_t Aarray, int lda, size_t devIpiv,
                        size_t Barray, int ldb, size_t info, int batchSize)
cpdef int dgetrsBatched(intptr_t handle, int trans, int n, int nrhs,
                        size_t Aarray, int lda, size_t devIpiv,
                        size_t Barray, int ldb, size_t info, int batchSize)
cpdef int cgetrsBatched(intptr_t handle, int trans, int n, int nrhs,
                        size_t Aarray, int lda, size_t devIpiv,
                        size_t Barray, int ldb, size_t info, int batchSize)
cpdef int zgetrsBatched(intptr_t handle, int trans, int n, int nrhs,
                        size_t Aarray, int lda, size_t devIpiv,
                        size_t Barray, int ldb, size_t info, int batchSize)

cpdef sgetriBatched(intptr_t handle, int n, size_t Aarray, int lda,
                    size_t PivotArray, size_t Carray, int ldc,
                    size_t infoArray, int batchSize)
cpdef dgetriBatched(intptr_t handle, int n, size_t Aarray, int lda,
                    size_t PivotArray, size_t Carray, int ldc,
                    size_t infoArray, int batchSize)
cpdef cgetriBatched(intptr_t handle, int n, size_t Aarray, int lda,
                    size_t PivotArray, size_t Carray, int ldc,
                    size_t infoArray, int batchSize)
cpdef zgetriBatched(intptr_t handle, int n, size_t Aarray, int lda,
                    size_t PivotArray, size_t Carray, int ldc,
                    size_t infoArray, int batchSize)
cpdef gemmEx(intptr_t handle, int transa, int transb, int m, int n, int k,
             size_t alpha, size_t A, int Atype, int lda, size_t B,
             int Btype, int ldb, size_t beta, size_t C, int Ctype,
             int ldc, int computeType, int algo)

cpdef stpttr(intptr_t handle, int uplo, int n, size_t AP, size_t A, int lda)
cpdef dtpttr(intptr_t handle, int uplo, int n, size_t AP, size_t A, int lda)

cpdef strttp(intptr_t handle, int uplo, int n, size_t A, int lda, size_t AP)
cpdef dtrttp(intptr_t handle, int uplo, int n, size_t A, int lda, size_t AP)
