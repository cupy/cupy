# This code was automatically generated. Do not modify it directly.

from libc.stdint cimport intptr_t, int64_t


########################################
# Opaque data structures

cdef extern from *:
    ctypedef int DataType 'cudaDataType'


cdef extern from *:
    ctypedef void* Handle 'cublasHandle_t'


########################################
# Enumerators

cdef extern from *:
    ctypedef int Status 'cublasStatus_t'
    ctypedef int FillMode 'cublasFillMode_t'
    ctypedef int DiagType 'cublasDiagType_t'
    ctypedef int SideMode 'cublasSideMode_t'
    ctypedef int Operation 'cublasOperation_t'
    ctypedef int PointerMode 'cublasPointerMode_t'
    ctypedef int AtomicsMode 'cublasAtomicsMode_t'
    ctypedef int GemmAlgo 'cublasGemmAlgo_t'
    ctypedef int Math 'cublasMath_t'
    ctypedef int ComputeType 'cublasComputeType_t'


cpdef enum:
    CUBLAS_STATUS_SUCCESS = 0
    CUBLAS_STATUS_NOT_INITIALIZED = 1
    CUBLAS_STATUS_ALLOC_FAILED = 3
    CUBLAS_STATUS_INVALID_VALUE = 7
    CUBLAS_STATUS_ARCH_MISMATCH = 8
    CUBLAS_STATUS_MAPPING_ERROR = 11
    CUBLAS_STATUS_EXECUTION_FAILED = 13
    CUBLAS_STATUS_INTERNAL_ERROR = 14
    CUBLAS_STATUS_NOT_SUPPORTED = 15
    CUBLAS_STATUS_LICENSE_ERROR = 16


cpdef enum:
    CUBLAS_FILL_MODE_LOWER = 0
    CUBLAS_FILL_MODE_UPPER = 1
    CUBLAS_FILL_MODE_FULL = 2


cpdef enum:
    CUBLAS_DIAG_NON_UNIT = 0
    CUBLAS_DIAG_UNIT = 1


cpdef enum:
    CUBLAS_SIDE_LEFT = 0
    CUBLAS_SIDE_RIGHT = 1


cpdef enum:
    CUBLAS_OP_N = 0
    CUBLAS_OP_T = 1
    CUBLAS_OP_C = 2
    CUBLAS_OP_HERMITAN = 2
    CUBLAS_OP_CONJG = 3


cpdef enum:
    CUBLAS_POINTER_MODE_HOST = 0
    CUBLAS_POINTER_MODE_DEVICE = 1


cpdef enum:
    CUBLAS_ATOMICS_NOT_ALLOWED = 0
    CUBLAS_ATOMICS_ALLOWED = 1


cpdef enum:
    CUBLAS_GEMM_DFALT = -1
    CUBLAS_GEMM_DEFAULT = -1
    CUBLAS_GEMM_ALGO0 = 0
    CUBLAS_GEMM_ALGO1 = 1
    CUBLAS_GEMM_ALGO2 = 2
    CUBLAS_GEMM_ALGO3 = 3
    CUBLAS_GEMM_ALGO4 = 4
    CUBLAS_GEMM_ALGO5 = 5
    CUBLAS_GEMM_ALGO6 = 6
    CUBLAS_GEMM_ALGO7 = 7
    CUBLAS_GEMM_ALGO8 = 8
    CUBLAS_GEMM_ALGO9 = 9
    CUBLAS_GEMM_ALGO10 = 10
    CUBLAS_GEMM_ALGO11 = 11
    CUBLAS_GEMM_ALGO12 = 12
    CUBLAS_GEMM_ALGO13 = 13
    CUBLAS_GEMM_ALGO14 = 14
    CUBLAS_GEMM_ALGO15 = 15
    CUBLAS_GEMM_ALGO16 = 16
    CUBLAS_GEMM_ALGO17 = 17
    CUBLAS_GEMM_ALGO18 = 18
    CUBLAS_GEMM_ALGO19 = 19
    CUBLAS_GEMM_ALGO20 = 20
    CUBLAS_GEMM_ALGO21 = 21
    CUBLAS_GEMM_ALGO22 = 22
    CUBLAS_GEMM_ALGO23 = 23
    CUBLAS_GEMM_DEFAULT_TENSOR_OP = 99
    CUBLAS_GEMM_DFALT_TENSOR_OP = 99
    CUBLAS_GEMM_ALGO0_TENSOR_OP = 100
    CUBLAS_GEMM_ALGO1_TENSOR_OP = 101
    CUBLAS_GEMM_ALGO2_TENSOR_OP = 102
    CUBLAS_GEMM_ALGO3_TENSOR_OP = 103
    CUBLAS_GEMM_ALGO4_TENSOR_OP = 104
    CUBLAS_GEMM_ALGO5_TENSOR_OP = 105
    CUBLAS_GEMM_ALGO6_TENSOR_OP = 106
    CUBLAS_GEMM_ALGO7_TENSOR_OP = 107
    CUBLAS_GEMM_ALGO8_TENSOR_OP = 108
    CUBLAS_GEMM_ALGO9_TENSOR_OP = 109
    CUBLAS_GEMM_ALGO10_TENSOR_OP = 110
    CUBLAS_GEMM_ALGO11_TENSOR_OP = 111
    CUBLAS_GEMM_ALGO12_TENSOR_OP = 112
    CUBLAS_GEMM_ALGO13_TENSOR_OP = 113
    CUBLAS_GEMM_ALGO14_TENSOR_OP = 114
    CUBLAS_GEMM_ALGO15_TENSOR_OP = 115


cpdef enum:
    CUBLAS_DEFAULT_MATH = 0
    CUBLAS_TENSOR_OP_MATH = 1
    CUBLAS_PEDANTIC_MATH = 2
    CUBLAS_TF32_TENSOR_OP_MATH = 3
    CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION = 16


cpdef enum:
    CUBLAS_COMPUTE_16F = 64
    CUBLAS_COMPUTE_16F_PEDANTIC = 65
    CUBLAS_COMPUTE_32F = 68
    CUBLAS_COMPUTE_32F_PEDANTIC = 69
    CUBLAS_COMPUTE_32F_FAST_16F = 74
    CUBLAS_COMPUTE_32F_FAST_16BF = 75
    CUBLAS_COMPUTE_32F_FAST_TF32 = 77
    CUBLAS_COMPUTE_64F = 70
    CUBLAS_COMPUTE_64F_PEDANTIC = 71
    CUBLAS_COMPUTE_32I = 72
    CUBLAS_COMPUTE_32I_PEDANTIC = 73


########################################
# Auxiliary structures




# TODO: should also expose functions?


########################################
# cuBLAS Helper Function

cpdef intptr_t create() except? 0

cpdef destroy(intptr_t handle)

cpdef int getVersion(intptr_t handle) except? -1

cpdef int getPointerMode(intptr_t handle) except? 0

cpdef setPointerMode(intptr_t handle, int mode)

cpdef setStream(intptr_t handle, size_t streamId)

cpdef size_t getStream(intptr_t handle) except? 0

cpdef setMathMode(intptr_t handle, int mode)

cpdef int getMathMode(intptr_t handle) except? -1


########################################
# cuBLAS Level-1 Function

cpdef isamax(intptr_t handle, int n, intptr_t x, int incx, intptr_t result)
cpdef idamax(intptr_t handle, int n, intptr_t x, int incx, intptr_t result)
cpdef icamax(intptr_t handle, int n, intptr_t x, int incx, intptr_t result)
cpdef izamax(intptr_t handle, int n, intptr_t x, int incx, intptr_t result)

cpdef isamin(intptr_t handle, int n, intptr_t x, int incx, intptr_t result)
cpdef idamin(intptr_t handle, int n, intptr_t x, int incx, intptr_t result)
cpdef icamin(intptr_t handle, int n, intptr_t x, int incx, intptr_t result)
cpdef izamin(intptr_t handle, int n, intptr_t x, int incx, intptr_t result)

cpdef sasum(intptr_t handle, int n, intptr_t x, int incx, intptr_t result)
cpdef dasum(intptr_t handle, int n, intptr_t x, int incx, intptr_t result)
cpdef scasum(intptr_t handle, int n, intptr_t x, int incx, intptr_t result)
cpdef dzasum(intptr_t handle, int n, intptr_t x, int incx, intptr_t result)

cpdef saxpy(intptr_t handle, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy)
cpdef daxpy(intptr_t handle, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy)
cpdef caxpy(intptr_t handle, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy)
cpdef zaxpy(intptr_t handle, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy)

cpdef sdot(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy, intptr_t result)
cpdef ddot(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy, intptr_t result)
cpdef cdotu(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy, intptr_t result)
cpdef cdotc(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy, intptr_t result)
cpdef zdotu(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy, intptr_t result)
cpdef zdotc(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy, intptr_t result)

cpdef snrm2(intptr_t handle, int n, intptr_t x, int incx, intptr_t result)
cpdef dnrm2(intptr_t handle, int n, intptr_t x, int incx, intptr_t result)
cpdef scnrm2(intptr_t handle, int n, intptr_t x, int incx, intptr_t result)
cpdef dznrm2(intptr_t handle, int n, intptr_t x, int incx, intptr_t result)

cpdef sscal(intptr_t handle, int n, intptr_t alpha, intptr_t x, int incx)
cpdef dscal(intptr_t handle, int n, intptr_t alpha, intptr_t x, int incx)
cpdef cscal(intptr_t handle, int n, intptr_t alpha, intptr_t x, int incx)
cpdef csscal(intptr_t handle, int n, intptr_t alpha, intptr_t x, int incx)
cpdef zscal(intptr_t handle, int n, intptr_t alpha, intptr_t x, int incx)
cpdef zdscal(intptr_t handle, int n, intptr_t alpha, intptr_t x, int incx)


########################################
# cuBLAS Level-2 Function

cpdef sgemv(intptr_t handle, int trans, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy)
cpdef dgemv(intptr_t handle, int trans, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy)
cpdef cgemv(intptr_t handle, int trans, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy)
cpdef zgemv(intptr_t handle, int trans, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy)

cpdef sger(intptr_t handle, int m, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t A, int lda)
cpdef dger(intptr_t handle, int m, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t A, int lda)
cpdef cgeru(intptr_t handle, int m, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t A, int lda)
cpdef cgerc(intptr_t handle, int m, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t A, int lda)
cpdef zgeru(intptr_t handle, int m, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t A, int lda)
cpdef zgerc(intptr_t handle, int m, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t A, int lda)


########################################
# cuBLAS Level-3 Function

cpdef sgemm(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc)
cpdef dgemm(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc)
cpdef cgemm(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc)
cpdef zgemm(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc)

cpdef sgemmBatched(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t Aarray, int lda, intptr_t Barray, int ldb, intptr_t beta, intptr_t Carray, int ldc, int batchCount)
cpdef dgemmBatched(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t Aarray, int lda, intptr_t Barray, int ldb, intptr_t beta, intptr_t Carray, int ldc, int batchCount)
cpdef cgemmBatched(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t Aarray, int lda, intptr_t Barray, int ldb, intptr_t beta, intptr_t Carray, int ldc, int batchCount)
cpdef zgemmBatched(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t Aarray, int lda, intptr_t Barray, int ldb, intptr_t beta, intptr_t Carray, int ldc, int batchCount)

cpdef sgemmStridedBatched(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t A, int lda, long long int strideA, intptr_t B, int ldb, long long int strideB, intptr_t beta, intptr_t C, int ldc, long long int strideC, int batchCount)
cpdef dgemmStridedBatched(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t A, int lda, long long int strideA, intptr_t B, int ldb, long long int strideB, intptr_t beta, intptr_t C, int ldc, long long int strideC, int batchCount)
cpdef cgemmStridedBatched(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t A, int lda, long long int strideA, intptr_t B, int ldb, long long int strideB, intptr_t beta, intptr_t C, int ldc, long long int strideC, int batchCount)
cpdef zgemmStridedBatched(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t A, int lda, long long int strideA, intptr_t B, int ldb, long long int strideB, intptr_t beta, intptr_t C, int ldc, long long int strideC, int batchCount)

cpdef strsm(intptr_t handle, int side, int uplo, int trans, int diag, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb)
cpdef dtrsm(intptr_t handle, int side, int uplo, int trans, int diag, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb)
cpdef ctrsm(intptr_t handle, int side, int uplo, int trans, int diag, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb)
cpdef ztrsm(intptr_t handle, int side, int uplo, int trans, int diag, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb)


########################################
# cuBLAS BLAS-like Extension

cpdef sgeam(intptr_t handle, int transa, int transb, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t beta, intptr_t B, int ldb, intptr_t C, int ldc)
cpdef dgeam(intptr_t handle, int transa, int transb, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t beta, intptr_t B, int ldb, intptr_t C, int ldc)
cpdef cgeam(intptr_t handle, int transa, int transb, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t beta, intptr_t B, int ldb, intptr_t C, int ldc)
cpdef zgeam(intptr_t handle, int transa, int transb, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t beta, intptr_t B, int ldb, intptr_t C, int ldc)

cpdef sdgmm(intptr_t handle, int mode, int m, int n, intptr_t A, int lda, intptr_t x, int incx, intptr_t C, int ldc)
cpdef ddgmm(intptr_t handle, int mode, int m, int n, intptr_t A, int lda, intptr_t x, int incx, intptr_t C, int ldc)
cpdef cdgmm(intptr_t handle, int mode, int m, int n, intptr_t A, int lda, intptr_t x, int incx, intptr_t C, int ldc)
cpdef zdgmm(intptr_t handle, int mode, int m, int n, intptr_t A, int lda, intptr_t x, int incx, intptr_t C, int ldc)

cpdef sgemmEx(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t A, size_t Atype, int lda, intptr_t B, size_t Btype, int ldb, intptr_t beta, intptr_t C, size_t Ctype, int ldc)
cpdef cgemmEx(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t A, size_t Atype, int lda, intptr_t B, size_t Btype, int ldb, intptr_t beta, intptr_t C, size_t Ctype, int ldc)

cpdef sgetrfBatched(intptr_t handle, int n, intptr_t A, int lda, intptr_t P, intptr_t info, int batchSize)
cpdef dgetrfBatched(intptr_t handle, int n, intptr_t A, int lda, intptr_t P, intptr_t info, int batchSize)
cpdef cgetrfBatched(intptr_t handle, int n, intptr_t A, int lda, intptr_t P, intptr_t info, int batchSize)
cpdef zgetrfBatched(intptr_t handle, int n, intptr_t A, int lda, intptr_t P, intptr_t info, int batchSize)

cpdef sgetrsBatched(intptr_t handle, int trans, int n, int nrhs, intptr_t Aarray, int lda, intptr_t devIpiv, intptr_t Barray, int ldb, intptr_t info, int batchSize)
cpdef dgetrsBatched(intptr_t handle, int trans, int n, int nrhs, intptr_t Aarray, int lda, intptr_t devIpiv, intptr_t Barray, int ldb, intptr_t info, int batchSize)
cpdef cgetrsBatched(intptr_t handle, int trans, int n, int nrhs, intptr_t Aarray, int lda, intptr_t devIpiv, intptr_t Barray, int ldb, intptr_t info, int batchSize)
cpdef zgetrsBatched(intptr_t handle, int trans, int n, int nrhs, intptr_t Aarray, int lda, intptr_t devIpiv, intptr_t Barray, int ldb, intptr_t info, int batchSize)

cpdef sgetriBatched(intptr_t handle, int n, intptr_t A, int lda, intptr_t P, intptr_t C, int ldc, intptr_t info, int batchSize)
cpdef dgetriBatched(intptr_t handle, int n, intptr_t A, int lda, intptr_t P, intptr_t C, int ldc, intptr_t info, int batchSize)
cpdef cgetriBatched(intptr_t handle, int n, intptr_t A, int lda, intptr_t P, intptr_t C, int ldc, intptr_t info, int batchSize)
cpdef zgetriBatched(intptr_t handle, int n, intptr_t A, int lda, intptr_t P, intptr_t C, int ldc, intptr_t info, int batchSize)

cpdef stpttr(intptr_t handle, int uplo, int n, intptr_t AP, intptr_t A, int lda)
cpdef dtpttr(intptr_t handle, int uplo, int n, intptr_t AP, intptr_t A, int lda)
cpdef ctpttr(intptr_t handle, int uplo, int n, intptr_t AP, intptr_t A, int lda)
cpdef ztpttr(intptr_t handle, int uplo, int n, intptr_t AP, intptr_t A, int lda)

cpdef strttp(intptr_t handle, int uplo, int n, intptr_t A, int lda, intptr_t AP)
cpdef dtrttp(intptr_t handle, int uplo, int n, intptr_t A, int lda, intptr_t AP)
cpdef ctrttp(intptr_t handle, int uplo, int n, intptr_t A, int lda, intptr_t AP)
cpdef ztrttp(intptr_t handle, int uplo, int n, intptr_t A, int lda, intptr_t AP)

# Define `gemmEx` by hands for a backward compatibility reason.
cpdef gemmEx(
        intptr_t handle, int transa, int transb, int m, int n, int k,
        size_t alpha, size_t A, int Atype, int lda, size_t B,
        int Btype, int ldb, size_t beta, size_t C, int Ctype,
        int ldc, int computeType, int algo)
