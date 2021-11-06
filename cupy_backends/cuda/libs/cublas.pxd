# This code was automatically generated. Do not modify it directly.

from libc.stdint cimport intptr_t


###############################################################################
# Types
###############################################################################

cdef extern from *:
    ctypedef void* Handle 'cublasHandle_t'

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


###############################################################################
# Enum
###############################################################################

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


###############################################################################
# Functions
###############################################################################

cpdef intptr_t create() except? 0
cpdef destroy(intptr_t handle)
cpdef int getVersion(intptr_t handle) except? -1
cpdef int getProperty(int type) except? -1
cpdef size_t getCudartVersion()
cpdef setStream(intptr_t handle, size_t streamId)
cpdef size_t getStream(intptr_t handle) except? 0
cpdef int getPointerMode(intptr_t handle) except? -1
cpdef setPointerMode(intptr_t handle, int mode)
cpdef int getAtomicsMode(intptr_t handle) except? -1
cpdef setAtomicsMode(intptr_t handle, int mode)
cpdef int getMathMode(intptr_t handle) except? -1
cpdef setMathMode(intptr_t handle, int mode)
cpdef setVector(int n, int elemSize, intptr_t x, int incx, intptr_t devicePtr, int incy)
cpdef getVector(int n, int elemSize, intptr_t x, int incx, intptr_t y, int incy)
cpdef setMatrix(int rows, int cols, int elemSize, intptr_t A, int lda, intptr_t B, int ldb)
cpdef getMatrix(int rows, int cols, int elemSize, intptr_t A, int lda, intptr_t B, int ldb)
cpdef setVectorAsync(int n, int elemSize, intptr_t hostPtr, int incx, intptr_t devicePtr, int incy)
cpdef getVectorAsync(int n, int elemSize, intptr_t devicePtr, int incx, intptr_t hostPtr, int incy)
cpdef setMatrixAsync(int rows, int cols, int elemSize, intptr_t A, int lda, intptr_t B, int ldb)
cpdef getMatrixAsync(int rows, int cols, int elemSize, intptr_t A, int lda, intptr_t B, int ldb)
cpdef nrm2Ex(intptr_t handle, int n, intptr_t x, size_t xType, int incx, intptr_t result, size_t resultType, size_t executionType)
cpdef snrm2(intptr_t handle, int n, intptr_t x, int incx, intptr_t result)
cpdef dnrm2(intptr_t handle, int n, intptr_t x, int incx, intptr_t result)
cpdef scnrm2(intptr_t handle, int n, intptr_t x, int incx, intptr_t result)
cpdef dznrm2(intptr_t handle, int n, intptr_t x, int incx, intptr_t result)
cpdef dotEx(intptr_t handle, int n, intptr_t x, size_t xType, int incx, intptr_t y, size_t yType, int incy, intptr_t result, size_t resultType, size_t executionType)
cpdef dotcEx(intptr_t handle, int n, intptr_t x, size_t xType, int incx, intptr_t y, size_t yType, int incy, intptr_t result, size_t resultType, size_t executionType)
cpdef sdot(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy, intptr_t result)
cpdef ddot(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy, intptr_t result)
cpdef cdotu(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy, intptr_t result)
cpdef cdotc(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy, intptr_t result)
cpdef zdotu(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy, intptr_t result)
cpdef zdotc(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy, intptr_t result)
cpdef scalEx(intptr_t handle, int n, intptr_t alpha, size_t alphaType, intptr_t x, size_t xType, int incx, size_t executionType)
cpdef sscal(intptr_t handle, int n, intptr_t alpha, intptr_t x, int incx)
cpdef dscal(intptr_t handle, int n, intptr_t alpha, intptr_t x, int incx)
cpdef cscal(intptr_t handle, int n, intptr_t alpha, intptr_t x, int incx)
cpdef csscal(intptr_t handle, int n, intptr_t alpha, intptr_t x, int incx)
cpdef zscal(intptr_t handle, int n, intptr_t alpha, intptr_t x, int incx)
cpdef zdscal(intptr_t handle, int n, intptr_t alpha, intptr_t x, int incx)
cpdef axpyEx(intptr_t handle, int n, intptr_t alpha, size_t alphaType, intptr_t x, size_t xType, int incx, intptr_t y, size_t yType, int incy, size_t executiontype)
cpdef saxpy(intptr_t handle, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy)
cpdef daxpy(intptr_t handle, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy)
cpdef caxpy(intptr_t handle, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy)
cpdef zaxpy(intptr_t handle, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy)
cpdef copyEx(intptr_t handle, int n, intptr_t x, size_t xType, int incx, intptr_t y, size_t yType, int incy)
cpdef scopy(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy)
cpdef dcopy(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy)
cpdef ccopy(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy)
cpdef zcopy(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy)
cpdef sswap(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy)
cpdef dswap(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy)
cpdef cswap(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy)
cpdef zswap(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy)
cpdef swapEx(intptr_t handle, int n, intptr_t x, size_t xType, int incx, intptr_t y, size_t yType, int incy)
cpdef isamax(intptr_t handle, int n, intptr_t x, int incx, intptr_t result)
cpdef idamax(intptr_t handle, int n, intptr_t x, int incx, intptr_t result)
cpdef icamax(intptr_t handle, int n, intptr_t x, int incx, intptr_t result)
cpdef izamax(intptr_t handle, int n, intptr_t x, int incx, intptr_t result)
cpdef iamaxEx(intptr_t handle, int n, intptr_t x, size_t xType, int incx, intptr_t result)
cpdef isamin(intptr_t handle, int n, intptr_t x, int incx, intptr_t result)
cpdef idamin(intptr_t handle, int n, intptr_t x, int incx, intptr_t result)
cpdef icamin(intptr_t handle, int n, intptr_t x, int incx, intptr_t result)
cpdef izamin(intptr_t handle, int n, intptr_t x, int incx, intptr_t result)
cpdef iaminEx(intptr_t handle, int n, intptr_t x, size_t xType, int incx, intptr_t result)
cpdef asumEx(intptr_t handle, int n, intptr_t x, size_t xType, int incx, intptr_t result, size_t resultType, size_t executiontype)
cpdef sasum(intptr_t handle, int n, intptr_t x, int incx, intptr_t result)
cpdef dasum(intptr_t handle, int n, intptr_t x, int incx, intptr_t result)
cpdef scasum(intptr_t handle, int n, intptr_t x, int incx, intptr_t result)
cpdef dzasum(intptr_t handle, int n, intptr_t x, int incx, intptr_t result)
cpdef srot(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy, intptr_t c, intptr_t s)
cpdef drot(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy, intptr_t c, intptr_t s)
cpdef crot(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy, intptr_t c, intptr_t s)
cpdef csrot(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy, intptr_t c, intptr_t s)
cpdef zrot(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy, intptr_t c, intptr_t s)
cpdef zdrot(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy, intptr_t c, intptr_t s)
cpdef rotEx(intptr_t handle, int n, intptr_t x, size_t xType, int incx, intptr_t y, size_t yType, int incy, intptr_t c, intptr_t s, size_t csType, size_t executiontype)
cpdef srotg(intptr_t handle, intptr_t a, intptr_t b, intptr_t c, intptr_t s)
cpdef drotg(intptr_t handle, intptr_t a, intptr_t b, intptr_t c, intptr_t s)
cpdef crotg(intptr_t handle, intptr_t a, intptr_t b, intptr_t c, intptr_t s)
cpdef zrotg(intptr_t handle, intptr_t a, intptr_t b, intptr_t c, intptr_t s)
cpdef rotgEx(intptr_t handle, intptr_t a, intptr_t b, size_t abType, intptr_t c, intptr_t s, size_t csType, size_t executiontype)
cpdef srotm(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy, intptr_t param)
cpdef drotm(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy, intptr_t param)
cpdef rotmEx(intptr_t handle, int n, intptr_t x, size_t xType, int incx, intptr_t y, size_t yType, int incy, intptr_t param, size_t paramType, size_t executiontype)
cpdef srotmg(intptr_t handle, intptr_t d1, intptr_t d2, intptr_t x1, intptr_t y1, intptr_t param)
cpdef drotmg(intptr_t handle, intptr_t d1, intptr_t d2, intptr_t x1, intptr_t y1, intptr_t param)
cpdef rotmgEx(intptr_t handle, intptr_t d1, size_t d1Type, intptr_t d2, size_t d2Type, intptr_t x1, size_t x1Type, intptr_t y1, size_t y1Type, intptr_t param, size_t paramType, size_t executiontype)
cpdef sgemv(intptr_t handle, int trans, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy)
cpdef dgemv(intptr_t handle, int trans, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy)
cpdef cgemv(intptr_t handle, int trans, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy)
cpdef zgemv(intptr_t handle, int trans, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy)
cpdef sgbmv(intptr_t handle, int trans, int m, int n, int kl, int ku, intptr_t alpha, intptr_t A, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy)
cpdef dgbmv(intptr_t handle, int trans, int m, int n, int kl, int ku, intptr_t alpha, intptr_t A, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy)
cpdef cgbmv(intptr_t handle, int trans, int m, int n, int kl, int ku, intptr_t alpha, intptr_t A, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy)
cpdef zgbmv(intptr_t handle, int trans, int m, int n, int kl, int ku, intptr_t alpha, intptr_t A, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy)
cpdef strmv(intptr_t handle, int uplo, int trans, int diag, int n, intptr_t A, int lda, intptr_t x, int incx)
cpdef dtrmv(intptr_t handle, int uplo, int trans, int diag, int n, intptr_t A, int lda, intptr_t x, int incx)
cpdef ctrmv(intptr_t handle, int uplo, int trans, int diag, int n, intptr_t A, int lda, intptr_t x, int incx)
cpdef ztrmv(intptr_t handle, int uplo, int trans, int diag, int n, intptr_t A, int lda, intptr_t x, int incx)
cpdef stbmv(intptr_t handle, int uplo, int trans, int diag, int n, int k, intptr_t A, int lda, intptr_t x, int incx)
cpdef dtbmv(intptr_t handle, int uplo, int trans, int diag, int n, int k, intptr_t A, int lda, intptr_t x, int incx)
cpdef ctbmv(intptr_t handle, int uplo, int trans, int diag, int n, int k, intptr_t A, int lda, intptr_t x, int incx)
cpdef ztbmv(intptr_t handle, int uplo, int trans, int diag, int n, int k, intptr_t A, int lda, intptr_t x, int incx)
cpdef stpmv(intptr_t handle, int uplo, int trans, int diag, int n, intptr_t AP, intptr_t x, int incx)
cpdef dtpmv(intptr_t handle, int uplo, int trans, int diag, int n, intptr_t AP, intptr_t x, int incx)
cpdef ctpmv(intptr_t handle, int uplo, int trans, int diag, int n, intptr_t AP, intptr_t x, int incx)
cpdef ztpmv(intptr_t handle, int uplo, int trans, int diag, int n, intptr_t AP, intptr_t x, int incx)
cpdef strsv(intptr_t handle, int uplo, int trans, int diag, int n, intptr_t A, int lda, intptr_t x, int incx)
cpdef dtrsv(intptr_t handle, int uplo, int trans, int diag, int n, intptr_t A, int lda, intptr_t x, int incx)
cpdef ctrsv(intptr_t handle, int uplo, int trans, int diag, int n, intptr_t A, int lda, intptr_t x, int incx)
cpdef ztrsv(intptr_t handle, int uplo, int trans, int diag, int n, intptr_t A, int lda, intptr_t x, int incx)
cpdef stpsv(intptr_t handle, int uplo, int trans, int diag, int n, intptr_t AP, intptr_t x, int incx)
cpdef dtpsv(intptr_t handle, int uplo, int trans, int diag, int n, intptr_t AP, intptr_t x, int incx)
cpdef ctpsv(intptr_t handle, int uplo, int trans, int diag, int n, intptr_t AP, intptr_t x, int incx)
cpdef ztpsv(intptr_t handle, int uplo, int trans, int diag, int n, intptr_t AP, intptr_t x, int incx)
cpdef stbsv(intptr_t handle, int uplo, int trans, int diag, int n, int k, intptr_t A, int lda, intptr_t x, int incx)
cpdef dtbsv(intptr_t handle, int uplo, int trans, int diag, int n, int k, intptr_t A, int lda, intptr_t x, int incx)
cpdef ctbsv(intptr_t handle, int uplo, int trans, int diag, int n, int k, intptr_t A, int lda, intptr_t x, int incx)
cpdef ztbsv(intptr_t handle, int uplo, int trans, int diag, int n, int k, intptr_t A, int lda, intptr_t x, int incx)
cpdef ssymv(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t A, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy)
cpdef dsymv(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t A, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy)
cpdef csymv(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t A, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy)
cpdef zsymv(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t A, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy)
cpdef chemv(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t A, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy)
cpdef zhemv(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t A, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy)
cpdef ssbmv(intptr_t handle, int uplo, int n, int k, intptr_t alpha, intptr_t A, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy)
cpdef dsbmv(intptr_t handle, int uplo, int n, int k, intptr_t alpha, intptr_t A, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy)
cpdef chbmv(intptr_t handle, int uplo, int n, int k, intptr_t alpha, intptr_t A, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy)
cpdef zhbmv(intptr_t handle, int uplo, int n, int k, intptr_t alpha, intptr_t A, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy)
cpdef sspmv(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t AP, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy)
cpdef dspmv(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t AP, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy)
cpdef chpmv(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t AP, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy)
cpdef zhpmv(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t AP, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy)
cpdef sger(intptr_t handle, int m, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t A, int lda)
cpdef dger(intptr_t handle, int m, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t A, int lda)
cpdef cgeru(intptr_t handle, int m, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t A, int lda)
cpdef cgerc(intptr_t handle, int m, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t A, int lda)
cpdef zgeru(intptr_t handle, int m, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t A, int lda)
cpdef zgerc(intptr_t handle, int m, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t A, int lda)
cpdef ssyr(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t A, int lda)
cpdef dsyr(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t A, int lda)
cpdef csyr(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t A, int lda)
cpdef zsyr(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t A, int lda)
cpdef cher(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t A, int lda)
cpdef zher(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t A, int lda)
cpdef sspr(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t AP)
cpdef dspr(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t AP)
cpdef chpr(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t AP)
cpdef zhpr(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t AP)
cpdef ssyr2(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t A, int lda)
cpdef dsyr2(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t A, int lda)
cpdef csyr2(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t A, int lda)
cpdef zsyr2(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t A, int lda)
cpdef cher2(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t A, int lda)
cpdef zher2(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t A, int lda)
cpdef sspr2(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t AP)
cpdef dspr2(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t AP)
cpdef chpr2(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t AP)
cpdef zhpr2(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t AP)
cpdef sgemm(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc)
cpdef dgemm(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc)
cpdef cgemm(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc)
cpdef cgemm3m(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc)
cpdef zgemm(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc)
cpdef zgemm3m(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc)
cpdef sgemmEx(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t A, size_t Atype, int lda, intptr_t B, size_t Btype, int ldb, intptr_t beta, intptr_t C, size_t Ctype, int ldc)
cpdef cgemmEx(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t A, size_t Atype, int lda, intptr_t B, size_t Btype, int ldb, intptr_t beta, intptr_t C, size_t Ctype, int ldc)
cpdef ssyrk(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t A, int lda, intptr_t beta, intptr_t C, int ldc)
cpdef dsyrk(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t A, int lda, intptr_t beta, intptr_t C, int ldc)
cpdef csyrk(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t A, int lda, intptr_t beta, intptr_t C, int ldc)
cpdef zsyrk(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t A, int lda, intptr_t beta, intptr_t C, int ldc)
cpdef csyrkEx(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t A, size_t Atype, int lda, intptr_t beta, intptr_t C, size_t Ctype, int ldc)
cpdef csyrk3mEx(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t A, size_t Atype, int lda, intptr_t beta, intptr_t C, size_t Ctype, int ldc)
cpdef cherk(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t A, int lda, intptr_t beta, intptr_t C, int ldc)
cpdef zherk(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t A, int lda, intptr_t beta, intptr_t C, int ldc)
cpdef cherkEx(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t A, size_t Atype, int lda, intptr_t beta, intptr_t C, size_t Ctype, int ldc)
cpdef cherk3mEx(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t A, size_t Atype, int lda, intptr_t beta, intptr_t C, size_t Ctype, int ldc)
cpdef ssyr2k(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc)
cpdef dsyr2k(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc)
cpdef csyr2k(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc)
cpdef zsyr2k(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc)
cpdef cher2k(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc)
cpdef zher2k(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc)
cpdef ssyrkx(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc)
cpdef dsyrkx(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc)
cpdef csyrkx(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc)
cpdef zsyrkx(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc)
cpdef cherkx(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc)
cpdef zherkx(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc)
cpdef ssymm(intptr_t handle, int side, int uplo, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc)
cpdef dsymm(intptr_t handle, int side, int uplo, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc)
cpdef csymm(intptr_t handle, int side, int uplo, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc)
cpdef zsymm(intptr_t handle, int side, int uplo, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc)
cpdef chemm(intptr_t handle, int side, int uplo, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc)
cpdef zhemm(intptr_t handle, int side, int uplo, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc)
cpdef strsm(intptr_t handle, int side, int uplo, int trans, int diag, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb)
cpdef dtrsm(intptr_t handle, int side, int uplo, int trans, int diag, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb)
cpdef ctrsm(intptr_t handle, int side, int uplo, int trans, int diag, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb)
cpdef ztrsm(intptr_t handle, int side, int uplo, int trans, int diag, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb)
cpdef strmm(intptr_t handle, int side, int uplo, int trans, int diag, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, intptr_t C, int ldc)
cpdef dtrmm(intptr_t handle, int side, int uplo, int trans, int diag, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, intptr_t C, int ldc)
cpdef ctrmm(intptr_t handle, int side, int uplo, int trans, int diag, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, intptr_t C, int ldc)
cpdef ztrmm(intptr_t handle, int side, int uplo, int trans, int diag, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, intptr_t C, int ldc)
cpdef sgemmBatched(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t Aarray, int lda, intptr_t Barray, int ldb, intptr_t beta, intptr_t Carray, int ldc, int batchCount)
cpdef dgemmBatched(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t Aarray, int lda, intptr_t Barray, int ldb, intptr_t beta, intptr_t Carray, int ldc, int batchCount)
cpdef cgemmBatched(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t Aarray, int lda, intptr_t Barray, int ldb, intptr_t beta, intptr_t Carray, int ldc, int batchCount)
cpdef zgemmBatched(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t Aarray, int lda, intptr_t Barray, int ldb, intptr_t beta, intptr_t Carray, int ldc, int batchCount)
cpdef sgemmStridedBatched(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t A, int lda, long long int strideA, intptr_t B, int ldb, long long int strideB, intptr_t beta, intptr_t C, int ldc, long long int strideC, int batchCount)
cpdef dgemmStridedBatched(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t A, int lda, long long int strideA, intptr_t B, int ldb, long long int strideB, intptr_t beta, intptr_t C, int ldc, long long int strideC, int batchCount)
cpdef cgemmStridedBatched(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t A, int lda, long long int strideA, intptr_t B, int ldb, long long int strideB, intptr_t beta, intptr_t C, int ldc, long long int strideC, int batchCount)
cpdef cgemm3mStridedBatched(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t A, int lda, long long int strideA, intptr_t B, int ldb, long long int strideB, intptr_t beta, intptr_t C, int ldc, long long int strideC, int batchCount)
cpdef zgemmStridedBatched(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t A, int lda, long long int strideA, intptr_t B, int ldb, long long int strideB, intptr_t beta, intptr_t C, int ldc, long long int strideC, int batchCount)
cpdef sgeam(intptr_t handle, int transa, int transb, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t beta, intptr_t B, int ldb, intptr_t C, int ldc)
cpdef dgeam(intptr_t handle, int transa, int transb, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t beta, intptr_t B, int ldb, intptr_t C, int ldc)
cpdef cgeam(intptr_t handle, int transa, int transb, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t beta, intptr_t B, int ldb, intptr_t C, int ldc)
cpdef zgeam(intptr_t handle, int transa, int transb, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t beta, intptr_t B, int ldb, intptr_t C, int ldc)
cpdef sgetrfBatched(intptr_t handle, int n, intptr_t A, int lda, intptr_t P, intptr_t info, int batchSize)
cpdef dgetrfBatched(intptr_t handle, int n, intptr_t A, int lda, intptr_t P, intptr_t info, int batchSize)
cpdef cgetrfBatched(intptr_t handle, int n, intptr_t A, int lda, intptr_t P, intptr_t info, int batchSize)
cpdef zgetrfBatched(intptr_t handle, int n, intptr_t A, int lda, intptr_t P, intptr_t info, int batchSize)
cpdef sgetriBatched(intptr_t handle, int n, intptr_t A, int lda, intptr_t P, intptr_t C, int ldc, intptr_t info, int batchSize)
cpdef dgetriBatched(intptr_t handle, int n, intptr_t A, int lda, intptr_t P, intptr_t C, int ldc, intptr_t info, int batchSize)
cpdef cgetriBatched(intptr_t handle, int n, intptr_t A, int lda, intptr_t P, intptr_t C, int ldc, intptr_t info, int batchSize)
cpdef zgetriBatched(intptr_t handle, int n, intptr_t A, int lda, intptr_t P, intptr_t C, int ldc, intptr_t info, int batchSize)
cpdef sgetrsBatched(intptr_t handle, int trans, int n, int nrhs, intptr_t Aarray, int lda, intptr_t devIpiv, intptr_t Barray, int ldb, intptr_t info, int batchSize)
cpdef dgetrsBatched(intptr_t handle, int trans, int n, int nrhs, intptr_t Aarray, int lda, intptr_t devIpiv, intptr_t Barray, int ldb, intptr_t info, int batchSize)
cpdef cgetrsBatched(intptr_t handle, int trans, int n, int nrhs, intptr_t Aarray, int lda, intptr_t devIpiv, intptr_t Barray, int ldb, intptr_t info, int batchSize)
cpdef zgetrsBatched(intptr_t handle, int trans, int n, int nrhs, intptr_t Aarray, int lda, intptr_t devIpiv, intptr_t Barray, int ldb, intptr_t info, int batchSize)
cpdef strsmBatched(intptr_t handle, int side, int uplo, int trans, int diag, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, int batchCount)
cpdef dtrsmBatched(intptr_t handle, int side, int uplo, int trans, int diag, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, int batchCount)
cpdef ctrsmBatched(intptr_t handle, int side, int uplo, int trans, int diag, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, int batchCount)
cpdef ztrsmBatched(intptr_t handle, int side, int uplo, int trans, int diag, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, int batchCount)
cpdef smatinvBatched(intptr_t handle, int n, intptr_t A, int lda, intptr_t Ainv, int lda_inv, intptr_t info, int batchSize)
cpdef dmatinvBatched(intptr_t handle, int n, intptr_t A, int lda, intptr_t Ainv, int lda_inv, intptr_t info, int batchSize)
cpdef cmatinvBatched(intptr_t handle, int n, intptr_t A, int lda, intptr_t Ainv, int lda_inv, intptr_t info, int batchSize)
cpdef zmatinvBatched(intptr_t handle, int n, intptr_t A, int lda, intptr_t Ainv, int lda_inv, intptr_t info, int batchSize)
cpdef sgeqrfBatched(intptr_t handle, int m, int n, intptr_t Aarray, int lda, intptr_t TauArray, intptr_t info, int batchSize)
cpdef dgeqrfBatched(intptr_t handle, int m, int n, intptr_t Aarray, int lda, intptr_t TauArray, intptr_t info, int batchSize)
cpdef cgeqrfBatched(intptr_t handle, int m, int n, intptr_t Aarray, int lda, intptr_t TauArray, intptr_t info, int batchSize)
cpdef zgeqrfBatched(intptr_t handle, int m, int n, intptr_t Aarray, int lda, intptr_t TauArray, intptr_t info, int batchSize)
cpdef sgelsBatched(intptr_t handle, int trans, int m, int n, int nrhs, intptr_t Aarray, int lda, intptr_t Carray, int ldc, intptr_t info, intptr_t devInfoArray, int batchSize)
cpdef dgelsBatched(intptr_t handle, int trans, int m, int n, int nrhs, intptr_t Aarray, int lda, intptr_t Carray, int ldc, intptr_t info, intptr_t devInfoArray, int batchSize)
cpdef cgelsBatched(intptr_t handle, int trans, int m, int n, int nrhs, intptr_t Aarray, int lda, intptr_t Carray, int ldc, intptr_t info, intptr_t devInfoArray, int batchSize)
cpdef zgelsBatched(intptr_t handle, int trans, int m, int n, int nrhs, intptr_t Aarray, int lda, intptr_t Carray, int ldc, intptr_t info, intptr_t devInfoArray, int batchSize)
cpdef sdgmm(intptr_t handle, int mode, int m, int n, intptr_t A, int lda, intptr_t x, int incx, intptr_t C, int ldc)
cpdef ddgmm(intptr_t handle, int mode, int m, int n, intptr_t A, int lda, intptr_t x, int incx, intptr_t C, int ldc)
cpdef cdgmm(intptr_t handle, int mode, int m, int n, intptr_t A, int lda, intptr_t x, int incx, intptr_t C, int ldc)
cpdef zdgmm(intptr_t handle, int mode, int m, int n, intptr_t A, int lda, intptr_t x, int incx, intptr_t C, int ldc)
cpdef stpttr(intptr_t handle, int uplo, int n, intptr_t AP, intptr_t A, int lda)
cpdef dtpttr(intptr_t handle, int uplo, int n, intptr_t AP, intptr_t A, int lda)
cpdef ctpttr(intptr_t handle, int uplo, int n, intptr_t AP, intptr_t A, int lda)
cpdef ztpttr(intptr_t handle, int uplo, int n, intptr_t AP, intptr_t A, int lda)
cpdef strttp(intptr_t handle, int uplo, int n, intptr_t A, int lda, intptr_t AP)
cpdef dtrttp(intptr_t handle, int uplo, int n, intptr_t A, int lda, intptr_t AP)
cpdef ctrttp(intptr_t handle, int uplo, int n, intptr_t A, int lda, intptr_t AP)
cpdef ztrttp(intptr_t handle, int uplo, int n, intptr_t A, int lda, intptr_t AP)
cpdef setWorkspace(intptr_t handle, intptr_t workspace, size_t workspaceSizeInBytes)

# Define by hand for backward compatibility
cpdef gemmEx(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t A, size_t Atype, int lda, intptr_t B, size_t Btype, int ldb, intptr_t beta, intptr_t C, size_t Ctype, int ldc, int computeType, int algo)
cpdef gemmBatchedEx(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t Aarray, size_t Atype, int lda, intptr_t Barray, size_t Btype, int ldb, intptr_t beta, intptr_t Carray, size_t Ctype, int ldc, int batchCount, int computeType, int algo)
cpdef gemmStridedBatchedEx(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t A, size_t Atype, int lda, long long int strideA, intptr_t B, size_t Btype, int ldb, long long int strideB, intptr_t beta, intptr_t C, size_t Ctype, int ldc, long long int strideC, int batchCount, int computeType, int algo)
