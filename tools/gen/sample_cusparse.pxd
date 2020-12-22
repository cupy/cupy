# This code was automatically generated. Do not modify it directly.

from libc.stdint cimport intptr_t, int64_t


########################################
# Opaque data structures

cdef extern from *:
    ctypedef int DataType 'cudaDataType'


cdef extern from *:
    ctypedef void* Handle 'cusparseHandle_t'
    ctypedef void* MatDescr 'cusparseMatDescr_t'
    ctypedef void* csrsv2Info_t 'csrsv2Info_t'
    ctypedef void* csrsm2Info_t 'csrsm2Info_t'
    ctypedef void* bsrsv2Info_t 'bsrsv2Info_t'
    ctypedef void* bsrsm2Info_t 'bsrsm2Info_t'
    ctypedef void* csric02Info_t 'csric02Info_t'
    ctypedef void* bsric02Info_t 'bsric02Info_t'
    ctypedef void* csrilu02Info_t 'csrilu02Info_t'
    ctypedef void* bsrilu02Info_t 'bsrilu02Info_t'
    ctypedef void* csrgemm2Info_t 'csrgemm2Info_t'
    ctypedef void* csru2csrInfo_t 'csru2csrInfo_t'
    ctypedef void* ColorInfo 'cusparseColorInfo_t'
    ctypedef void* pruneInfo_t 'pruneInfo_t'
    ctypedef void* SpVecDescr 'cusparseSpVecDescr_t'
    ctypedef void* DnVecDescr 'cusparseDnVecDescr_t'
    ctypedef void* SpMatDescr 'cusparseSpMatDescr_t'
    ctypedef void* DnMatDescr 'cusparseDnMatDescr_t'
    ctypedef void* SpGEMMDescr 'cusparseSpGEMMDescr_t'


########################################
# Enumerators

cdef extern from *:
    ctypedef int Status 'cusparseStatus_t'
    ctypedef int PointerMode 'cusparsePointerMode_t'
    ctypedef int Action 'cusparseAction_t'
    ctypedef int MatrixType 'cusparseMatrixType_t'
    ctypedef int FillMode 'cusparseFillMode_t'
    ctypedef int DiagType 'cusparseDiagType_t'
    ctypedef int IndexBase 'cusparseIndexBase_t'
    ctypedef int Operation 'cusparseOperation_t'
    ctypedef int Direction 'cusparseDirection_t'
    ctypedef int SolvePolicy 'cusparseSolvePolicy_t'
    ctypedef int SideMode 'cusparseSideMode_t'
    ctypedef int ColorAlg 'cusparseColorAlg_t'
    ctypedef int AlgMode 'cusparseAlgMode_t'
    ctypedef int Csr2CscAlg 'cusparseCsr2CscAlg_t'
    ctypedef int Format 'cusparseFormat_t'
    ctypedef int Order 'cusparseOrder_t'
    ctypedef int IndexType 'cusparseIndexType_t'
    ctypedef int SpMVAlg 'cusparseSpMVAlg_t'
    ctypedef int SpMMAlg 'cusparseSpMMAlg_t'
    ctypedef int SpGEMMAlg 'cusparseSpGEMMAlg_t'


cpdef enum:
    CUSPARSE_STATUS_SUCCESS = 0
    CUSPARSE_STATUS_NOT_INITIALIZED = 1
    CUSPARSE_STATUS_ALLOC_FAILED = 2
    CUSPARSE_STATUS_INVALID_VALUE = 3
    CUSPARSE_STATUS_ARCH_MISMATCH = 4
    CUSPARSE_STATUS_MAPPING_ERROR = 5
    CUSPARSE_STATUS_EXECUTION_FAILED = 6
    CUSPARSE_STATUS_INTERNAL_ERROR = 7
    CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED = 8
    CUSPARSE_STATUS_ZERO_PIVOT = 9
    CUSPARSE_STATUS_NOT_SUPPORTED = 10
    CUSPARSE_STATUS_INSUFFICIENT_RESOURCES = 11


cpdef enum:
    CUSPARSE_POINTER_MODE_HOST = 0
    CUSPARSE_POINTER_MODE_DEVICE = 1


cpdef enum:
    CUSPARSE_ACTION_SYMBOLIC = 0
    CUSPARSE_ACTION_NUMERIC = 1


cpdef enum:
    CUSPARSE_MATRIX_TYPE_GENERAL = 0
    CUSPARSE_MATRIX_TYPE_SYMMETRIC = 1
    CUSPARSE_MATRIX_TYPE_HERMITIAN = 2
    CUSPARSE_MATRIX_TYPE_TRIANGULAR = 3


cpdef enum:
    CUSPARSE_FILL_MODE_LOWER = 0
    CUSPARSE_FILL_MODE_UPPER = 1


cpdef enum:
    CUSPARSE_DIAG_TYPE_NON_UNIT = 0
    CUSPARSE_DIAG_TYPE_UNIT = 1


cpdef enum:
    CUSPARSE_INDEX_BASE_ZERO = 0
    CUSPARSE_INDEX_BASE_ONE = 1


cpdef enum:
    CUSPARSE_OPERATION_NON_TRANSPOSE = 0
    CUSPARSE_OPERATION_TRANSPOSE = 1
    CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE = 2


cpdef enum:
    CUSPARSE_DIRECTION_ROW = 0
    CUSPARSE_DIRECTION_COLUMN = 1


cpdef enum:
    CUSPARSE_SOLVE_POLICY_NO_LEVEL = 0
    CUSPARSE_SOLVE_POLICY_USE_LEVEL = 1


cpdef enum:
    CUSPARSE_SIDE_LEFT = 0
    CUSPARSE_SIDE_RIGHT = 1


cpdef enum:
    CUSPARSE_COLOR_ALG0 = 0
    CUSPARSE_COLOR_ALG1 = 1


cpdef enum:
    CUSPARSE_ALG_MERGE_PATH


cpdef enum:
    CUSPARSE_CSR2CSC_ALG1 = 1
    CUSPARSE_CSR2CSC_ALG2 = 2


cpdef enum:
    CUSPARSE_FORMAT_CSR = 1
    CUSPARSE_FORMAT_CSC = 2
    CUSPARSE_FORMAT_COO = 3
    CUSPARSE_FORMAT_COO_AOS = 4


cpdef enum:
    CUSPARSE_ORDER_COL = 1
    CUSPARSE_ORDER_ROW = 2


cpdef enum:
    CUSPARSE_INDEX_16U = 1
    CUSPARSE_INDEX_32I = 2
    CUSPARSE_INDEX_64I = 3


cpdef enum:
    CUSPARSE_MV_ALG_DEFAULT = 0
    CUSPARSE_COOMV_ALG = 1
    CUSPARSE_CSRMV_ALG1 = 2
    CUSPARSE_CSRMV_ALG2 = 3


cpdef enum:
    CUSPARSE_MM_ALG_DEFAULT = 0
    CUSPARSE_COOMM_ALG1 = 1
    CUSPARSE_COOMM_ALG2 = 2
    CUSPARSE_COOMM_ALG3 = 3
    CUSPARSE_CSRMM_ALG1 = 4
    CUSPARSE_SPMM_ALG_DEFAULT = 0
    CUSPARSE_SPMM_COO_ALG1 = 1
    CUSPARSE_SPMM_COO_ALG2 = 2
    CUSPARSE_SPMM_COO_ALG3 = 3
    CUSPARSE_SPMM_COO_ALG4 = 5
    CUSPARSE_SPMM_CSR_ALG1 = 4
    CUSPARSE_SPMM_CSR_ALG2 = 6


cpdef enum:
    CUSPARSE_SPGEMM_DEFAULT = 0


########################################
# Auxiliary structures

cdef class SpVecAttributes:
    cdef:
        public int64_t size
        public int64_t nnz
        public intptr_t indices
        public intptr_t values
        public IndexType idxType
        public IndexBase idxBase
        public DataType valueType


cdef class CooAttributes:
    cdef:
        public int64_t rows
        public int64_t cols
        public int64_t nnz
        public intptr_t cooRowInd
        public intptr_t cooColInd
        public intptr_t cooValues
        public IndexType idxType
        public IndexBase idxBase
        public DataType valueType


cdef class CooAoSAttributes:
    cdef:
        public int64_t rows
        public int64_t cols
        public int64_t nnz
        public intptr_t cooInd
        public intptr_t cooValues
        public IndexType idxType
        public IndexBase idxBase
        public DataType valueType


cdef class CsrAttributes:
    cdef:
        public int64_t rows
        public int64_t cols
        public int64_t nnz
        public intptr_t csrRowOffsets
        public intptr_t csrColInd
        public intptr_t csrValues
        public IndexType csrRowOffsetsType
        public IndexType csrColIndType
        public IndexBase idxBase
        public DataType valueType


cdef class DnVecAttributes:
    cdef:
        public int64_t size
        public intptr_t values
        public DataType valueType


cdef class DnMatAttributes:
    cdef:
        public int64_t rows
        public int64_t cols
        public int64_t ld
        public intptr_t values
        public DataType type
        public Order order


cdef class DnMatBatchAttributes:
    cdef:
        public int batchCount
        public int64_t batchStride


########################################
# cuSPARSE Management Function

cpdef intptr_t create() except? 0

cpdef destroy(intptr_t handle)

cpdef int getVersion(intptr_t handle) except -1

cpdef setPointerMode(intptr_t handle, int mode)

cpdef size_t getStream(intptr_t handle) except? 0

cpdef setStream(intptr_t handle, size_t streamId)


########################################
# cuSPARSE Helper Function

cpdef size_t createMatDescr() except? 0

cpdef destroyMatDescr(size_t descrA)

cpdef setMatDiagType(size_t descrA, int diagType)

cpdef setMatFillMode(size_t descrA, int fillMode)

cpdef setMatIndexBase(size_t descrA, int base)

cpdef setMatType(size_t descrA, int type)

cpdef size_t createCsrsv2Info() except? 0

cpdef destroyCsrsv2Info(size_t info)

cpdef size_t createCsrsm2Info() except? 0

cpdef destroyCsrsm2Info(size_t info)

cpdef size_t createCsric02Info() except? 0

cpdef destroyCsric02Info(size_t info)

cpdef size_t createCsrilu02Info() except? 0

cpdef destroyCsrilu02Info(size_t info)

cpdef size_t createBsric02Info() except? 0

cpdef destroyBsric02Info(size_t info)

cpdef size_t createBsrilu02Info() except? 0

cpdef destroyBsrilu02Info(size_t info)

cpdef size_t createCsrgemm2Info() except? 0

cpdef destroyCsrgemm2Info(size_t info)


########################################
# cuSPARSE Level 1 Function

cpdef sgthr(intptr_t handle, int nnz, intptr_t y, intptr_t xVal, intptr_t xInd, int idxBase)
cpdef dgthr(intptr_t handle, int nnz, intptr_t y, intptr_t xVal, intptr_t xInd, int idxBase)
cpdef cgthr(intptr_t handle, int nnz, intptr_t y, intptr_t xVal, intptr_t xInd, int idxBase)
cpdef zgthr(intptr_t handle, int nnz, intptr_t y, intptr_t xVal, intptr_t xInd, int idxBase)


########################################
# cuSPARSE Level 2 Function

# REMOVED
cpdef scsrmv(intptr_t handle, int transA, int m, int n, int nnz, intptr_t alpha, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t x, intptr_t beta, intptr_t y)
# REMOVED
cpdef dcsrmv(intptr_t handle, int transA, int m, int n, int nnz, intptr_t alpha, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t x, intptr_t beta, intptr_t y)
# REMOVED
cpdef ccsrmv(intptr_t handle, int transA, int m, int n, int nnz, intptr_t alpha, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t x, intptr_t beta, intptr_t y)
# REMOVED
cpdef zcsrmv(intptr_t handle, int transA, int m, int n, int nnz, intptr_t alpha, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t x, intptr_t beta, intptr_t y)

cpdef size_t csrmvEx_bufferSize(intptr_t handle, int alg, int transA, int m, int n, int nnz, intptr_t alpha, size_t alphatype, size_t descrA, intptr_t csrValA, size_t csrValAtype, intptr_t csrRowPtrA, intptr_t csrColIndA, intptr_t x, size_t xtype, intptr_t beta, size_t betatype, intptr_t y, size_t ytype, size_t executiontype) except? 0

cpdef csrmvEx(intptr_t handle, int alg, int transA, int m, int n, int nnz, intptr_t alpha, size_t alphatype, size_t descrA, intptr_t csrValA, size_t csrValAtype, intptr_t csrRowPtrA, intptr_t csrColIndA, intptr_t x, size_t xtype, intptr_t beta, size_t betatype, intptr_t y, size_t ytype, size_t executiontype, intptr_t buffer)

cpdef int scsrsv2_bufferSize(intptr_t handle, int transA, int m, int nnz, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info) except? 0
cpdef int dcsrsv2_bufferSize(intptr_t handle, int transA, int m, int nnz, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info) except? 0
cpdef int ccsrsv2_bufferSize(intptr_t handle, int transA, int m, int nnz, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info) except? 0
cpdef int zcsrsv2_bufferSize(intptr_t handle, int transA, int m, int nnz, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info) except? 0

cpdef scsrsv2_analysis(intptr_t handle, int transA, int m, int nnz, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info, int policy, intptr_t pBuffer)
cpdef dcsrsv2_analysis(intptr_t handle, int transA, int m, int nnz, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info, int policy, intptr_t pBuffer)
cpdef ccsrsv2_analysis(intptr_t handle, int transA, int m, int nnz, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info, int policy, intptr_t pBuffer)
cpdef zcsrsv2_analysis(intptr_t handle, int transA, int m, int nnz, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info, int policy, intptr_t pBuffer)

cpdef scsrsv2_solve(intptr_t handle, int transA, int m, int nnz, intptr_t alpha, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info, intptr_t f, intptr_t x, int policy, intptr_t pBuffer)
cpdef dcsrsv2_solve(intptr_t handle, int transA, int m, int nnz, intptr_t alpha, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info, intptr_t f, intptr_t x, int policy, intptr_t pBuffer)
cpdef ccsrsv2_solve(intptr_t handle, int transA, int m, int nnz, intptr_t alpha, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info, intptr_t f, intptr_t x, int policy, intptr_t pBuffer)
cpdef zcsrsv2_solve(intptr_t handle, int transA, int m, int nnz, intptr_t alpha, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info, intptr_t f, intptr_t x, int policy, intptr_t pBuffer)

cpdef xcsrsv2_zeroPivot(intptr_t handle, size_t info, intptr_t position)


########################################
# cuSPARSE Level 3 Function

# REMOVED
cpdef scsrmm(intptr_t handle, int transA, int m, int n, int k, int nnz, intptr_t alpha, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc)
# REMOVED
cpdef dcsrmm(intptr_t handle, int transA, int m, int n, int k, int nnz, intptr_t alpha, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc)
# REMOVED
cpdef ccsrmm(intptr_t handle, int transA, int m, int n, int k, int nnz, intptr_t alpha, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc)
# REMOVED
cpdef zcsrmm(intptr_t handle, int transA, int m, int n, int k, int nnz, intptr_t alpha, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc)

# REMOVED
cpdef scsrmm2(intptr_t handle, int transA, int transB, int m, int n, int k, int nnz, intptr_t alpha, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc)
# REMOVED
cpdef dcsrmm2(intptr_t handle, int transA, int transB, int m, int n, int k, int nnz, intptr_t alpha, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc)
# REMOVED
cpdef ccsrmm2(intptr_t handle, int transA, int transB, int m, int n, int k, int nnz, intptr_t alpha, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc)
# REMOVED
cpdef zcsrmm2(intptr_t handle, int transA, int transB, int m, int n, int k, int nnz, intptr_t alpha, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc)

cpdef size_t scsrsm2_bufferSizeExt(intptr_t handle, int algo, int transA, int transB, int m, int nrhs, int nnz, intptr_t alpha, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t B, int ldb, size_t info, int policy) except? 0
cpdef size_t dcsrsm2_bufferSizeExt(intptr_t handle, int algo, int transA, int transB, int m, int nrhs, int nnz, intptr_t alpha, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t B, int ldb, size_t info, int policy) except? 0
cpdef size_t ccsrsm2_bufferSizeExt(intptr_t handle, int algo, int transA, int transB, int m, int nrhs, int nnz, intptr_t alpha, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t B, int ldb, size_t info, int policy) except? 0
cpdef size_t zcsrsm2_bufferSizeExt(intptr_t handle, int algo, int transA, int transB, int m, int nrhs, int nnz, intptr_t alpha, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t B, int ldb, size_t info, int policy) except? 0

cpdef scsrsm2_analysis(intptr_t handle, int algo, int transA, int transB, int m, int nrhs, int nnz, intptr_t alpha, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t B, int ldb, size_t info, int policy, intptr_t pBuffer)
cpdef dcsrsm2_analysis(intptr_t handle, int algo, int transA, int transB, int m, int nrhs, int nnz, intptr_t alpha, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t B, int ldb, size_t info, int policy, intptr_t pBuffer)
cpdef ccsrsm2_analysis(intptr_t handle, int algo, int transA, int transB, int m, int nrhs, int nnz, intptr_t alpha, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t B, int ldb, size_t info, int policy, intptr_t pBuffer)
cpdef zcsrsm2_analysis(intptr_t handle, int algo, int transA, int transB, int m, int nrhs, int nnz, intptr_t alpha, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t B, int ldb, size_t info, int policy, intptr_t pBuffer)

cpdef scsrsm2_solve(intptr_t handle, int algo, int transA, int transB, int m, int nrhs, int nnz, intptr_t alpha, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t B, int ldb, size_t info, int policy, intptr_t pBuffer)
cpdef dcsrsm2_solve(intptr_t handle, int algo, int transA, int transB, int m, int nrhs, int nnz, intptr_t alpha, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t B, int ldb, size_t info, int policy, intptr_t pBuffer)
cpdef ccsrsm2_solve(intptr_t handle, int algo, int transA, int transB, int m, int nrhs, int nnz, intptr_t alpha, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t B, int ldb, size_t info, int policy, intptr_t pBuffer)
cpdef zcsrsm2_solve(intptr_t handle, int algo, int transA, int transB, int m, int nrhs, int nnz, intptr_t alpha, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t B, int ldb, size_t info, int policy, intptr_t pBuffer)

cpdef xcsrsm2_zeroPivot(intptr_t handle, size_t info, intptr_t position)


########################################
# cuSPARSE Extra Function

# REMOVED
cpdef xcsrgeamNnz(intptr_t handle, int m, int n, size_t descrA, int nnzA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t descrB, int nnzB, intptr_t csrSortedRowPtrB, intptr_t csrSortedColIndB, size_t descrC, intptr_t csrSortedRowPtrC, intptr_t nnzTotalDevHostPtr)

# REMOVED
cpdef scsrgeam(intptr_t handle, int m, int n, intptr_t alpha, size_t descrA, int nnzA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t beta, size_t descrB, int nnzB, intptr_t csrSortedValB, intptr_t csrSortedRowPtrB, intptr_t csrSortedColIndB, size_t descrC, intptr_t csrSortedValC, intptr_t csrSortedRowPtrC, intptr_t csrSortedColIndC)
# REMOVED
cpdef dcsrgeam(intptr_t handle, int m, int n, intptr_t alpha, size_t descrA, int nnzA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t beta, size_t descrB, int nnzB, intptr_t csrSortedValB, intptr_t csrSortedRowPtrB, intptr_t csrSortedColIndB, size_t descrC, intptr_t csrSortedValC, intptr_t csrSortedRowPtrC, intptr_t csrSortedColIndC)
# REMOVED
cpdef ccsrgeam(intptr_t handle, int m, int n, intptr_t alpha, size_t descrA, int nnzA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t beta, size_t descrB, int nnzB, intptr_t csrSortedValB, intptr_t csrSortedRowPtrB, intptr_t csrSortedColIndB, size_t descrC, intptr_t csrSortedValC, intptr_t csrSortedRowPtrC, intptr_t csrSortedColIndC)
# REMOVED
cpdef zcsrgeam(intptr_t handle, int m, int n, intptr_t alpha, size_t descrA, int nnzA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t beta, size_t descrB, int nnzB, intptr_t csrSortedValB, intptr_t csrSortedRowPtrB, intptr_t csrSortedColIndB, size_t descrC, intptr_t csrSortedValC, intptr_t csrSortedRowPtrC, intptr_t csrSortedColIndC)

cpdef size_t scsrgeam2_bufferSizeExt(intptr_t handle, int m, int n, intptr_t alpha, size_t descrA, int nnzA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t beta, size_t descrB, int nnzB, intptr_t csrSortedValB, intptr_t csrSortedRowPtrB, intptr_t csrSortedColIndB, size_t descrC, intptr_t csrSortedValC, intptr_t csrSortedRowPtrC, intptr_t csrSortedColIndC) except? 0
cpdef size_t dcsrgeam2_bufferSizeExt(intptr_t handle, int m, int n, intptr_t alpha, size_t descrA, int nnzA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t beta, size_t descrB, int nnzB, intptr_t csrSortedValB, intptr_t csrSortedRowPtrB, intptr_t csrSortedColIndB, size_t descrC, intptr_t csrSortedValC, intptr_t csrSortedRowPtrC, intptr_t csrSortedColIndC) except? 0
cpdef size_t ccsrgeam2_bufferSizeExt(intptr_t handle, int m, int n, intptr_t alpha, size_t descrA, int nnzA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t beta, size_t descrB, int nnzB, intptr_t csrSortedValB, intptr_t csrSortedRowPtrB, intptr_t csrSortedColIndB, size_t descrC, intptr_t csrSortedValC, intptr_t csrSortedRowPtrC, intptr_t csrSortedColIndC) except? 0
cpdef size_t zcsrgeam2_bufferSizeExt(intptr_t handle, int m, int n, intptr_t alpha, size_t descrA, int nnzA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t beta, size_t descrB, int nnzB, intptr_t csrSortedValB, intptr_t csrSortedRowPtrB, intptr_t csrSortedColIndB, size_t descrC, intptr_t csrSortedValC, intptr_t csrSortedRowPtrC, intptr_t csrSortedColIndC) except? 0

cpdef xcsrgeam2Nnz(intptr_t handle, int m, int n, size_t descrA, int nnzA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t descrB, int nnzB, intptr_t csrSortedRowPtrB, intptr_t csrSortedColIndB, size_t descrC, intptr_t csrSortedRowPtrC, intptr_t nnzTotalDevHostPtr, intptr_t workspace)

cpdef scsrgeam2(intptr_t handle, int m, int n, intptr_t alpha, size_t descrA, int nnzA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t beta, size_t descrB, int nnzB, intptr_t csrSortedValB, intptr_t csrSortedRowPtrB, intptr_t csrSortedColIndB, size_t descrC, intptr_t csrSortedValC, intptr_t csrSortedRowPtrC, intptr_t csrSortedColIndC, intptr_t pBuffer)
cpdef dcsrgeam2(intptr_t handle, int m, int n, intptr_t alpha, size_t descrA, int nnzA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t beta, size_t descrB, int nnzB, intptr_t csrSortedValB, intptr_t csrSortedRowPtrB, intptr_t csrSortedColIndB, size_t descrC, intptr_t csrSortedValC, intptr_t csrSortedRowPtrC, intptr_t csrSortedColIndC, intptr_t pBuffer)
cpdef ccsrgeam2(intptr_t handle, int m, int n, intptr_t alpha, size_t descrA, int nnzA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t beta, size_t descrB, int nnzB, intptr_t csrSortedValB, intptr_t csrSortedRowPtrB, intptr_t csrSortedColIndB, size_t descrC, intptr_t csrSortedValC, intptr_t csrSortedRowPtrC, intptr_t csrSortedColIndC, intptr_t pBuffer)
cpdef zcsrgeam2(intptr_t handle, int m, int n, intptr_t alpha, size_t descrA, int nnzA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t beta, size_t descrB, int nnzB, intptr_t csrSortedValB, intptr_t csrSortedRowPtrB, intptr_t csrSortedColIndB, size_t descrC, intptr_t csrSortedValC, intptr_t csrSortedRowPtrC, intptr_t csrSortedColIndC, intptr_t pBuffer)

# REMOVED
cpdef xcsrgemmNnz(intptr_t handle, int transA, int transB, int m, int n, int k, size_t descrA, const int nnzA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t descrB, const int nnzB, intptr_t csrSortedRowPtrB, intptr_t csrSortedColIndB, size_t descrC, intptr_t csrSortedRowPtrC, intptr_t nnzTotalDevHostPtr)

# REMOVED
cpdef scsrgemm(intptr_t handle, int transA, int transB, int m, int n, int k, size_t descrA, const int nnzA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t descrB, const int nnzB, intptr_t csrSortedValB, intptr_t csrSortedRowPtrB, intptr_t csrSortedColIndB, size_t descrC, intptr_t csrSortedValC, intptr_t csrSortedRowPtrC, intptr_t csrSortedColIndC)
# REMOVED
cpdef dcsrgemm(intptr_t handle, int transA, int transB, int m, int n, int k, size_t descrA, int nnzA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t descrB, int nnzB, intptr_t csrSortedValB, intptr_t csrSortedRowPtrB, intptr_t csrSortedColIndB, size_t descrC, intptr_t csrSortedValC, intptr_t csrSortedRowPtrC, intptr_t csrSortedColIndC)
# REMOVED
cpdef ccsrgemm(intptr_t handle, int transA, int transB, int m, int n, int k, size_t descrA, int nnzA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t descrB, int nnzB, intptr_t csrSortedValB, intptr_t csrSortedRowPtrB, intptr_t csrSortedColIndB, size_t descrC, intptr_t csrSortedValC, intptr_t csrSortedRowPtrC, intptr_t csrSortedColIndC)
# REMOVED
cpdef zcsrgemm(intptr_t handle, int transA, int transB, int m, int n, int k, size_t descrA, int nnzA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t descrB, int nnzB, intptr_t csrSortedValB, intptr_t csrSortedRowPtrB, intptr_t csrSortedColIndB, size_t descrC, intptr_t csrSortedValC, intptr_t csrSortedRowPtrC, intptr_t csrSortedColIndC)

cpdef size_t scsrgemm2_bufferSizeExt(intptr_t handle, int m, int n, int k, intptr_t alpha, size_t descrA, int nnzA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t descrB, int nnzB, intptr_t csrSortedRowPtrB, intptr_t csrSortedColIndB, intptr_t beta, size_t descrD, int nnzD, intptr_t csrSortedRowPtrD, intptr_t csrSortedColIndD, size_t info) except? 0
cpdef size_t dcsrgemm2_bufferSizeExt(intptr_t handle, int m, int n, int k, intptr_t alpha, size_t descrA, int nnzA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t descrB, int nnzB, intptr_t csrSortedRowPtrB, intptr_t csrSortedColIndB, intptr_t beta, size_t descrD, int nnzD, intptr_t csrSortedRowPtrD, intptr_t csrSortedColIndD, size_t info) except? 0
cpdef size_t ccsrgemm2_bufferSizeExt(intptr_t handle, int m, int n, int k, intptr_t alpha, size_t descrA, int nnzA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t descrB, int nnzB, intptr_t csrSortedRowPtrB, intptr_t csrSortedColIndB, intptr_t beta, size_t descrD, int nnzD, intptr_t csrSortedRowPtrD, intptr_t csrSortedColIndD, size_t info) except? 0
cpdef size_t zcsrgemm2_bufferSizeExt(intptr_t handle, int m, int n, int k, intptr_t alpha, size_t descrA, int nnzA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t descrB, int nnzB, intptr_t csrSortedRowPtrB, intptr_t csrSortedColIndB, intptr_t beta, size_t descrD, int nnzD, intptr_t csrSortedRowPtrD, intptr_t csrSortedColIndD, size_t info) except? 0

cpdef xcsrgemm2Nnz(intptr_t handle, int m, int n, int k, size_t descrA, int nnzA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t descrB, int nnzB, intptr_t csrSortedRowPtrB, intptr_t csrSortedColIndB, size_t descrD, int nnzD, intptr_t csrSortedRowPtrD, intptr_t csrSortedColIndD, size_t descrC, intptr_t csrSortedRowPtrC, intptr_t nnzTotalDevHostPtr, size_t info, intptr_t pBuffer)

cpdef scsrgemm2(intptr_t handle, int m, int n, int k, intptr_t alpha, size_t descrA, int nnzA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t descrB, int nnzB, intptr_t csrSortedValB, intptr_t csrSortedRowPtrB, intptr_t csrSortedColIndB, intptr_t beta, size_t descrD, int nnzD, intptr_t csrSortedValD, intptr_t csrSortedRowPtrD, intptr_t csrSortedColIndD, size_t descrC, intptr_t csrSortedValC, intptr_t csrSortedRowPtrC, intptr_t csrSortedColIndC, size_t info, intptr_t pBuffer)
cpdef dcsrgemm2(intptr_t handle, int m, int n, int k, intptr_t alpha, size_t descrA, int nnzA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t descrB, int nnzB, intptr_t csrSortedValB, intptr_t csrSortedRowPtrB, intptr_t csrSortedColIndB, intptr_t beta, size_t descrD, int nnzD, intptr_t csrSortedValD, intptr_t csrSortedRowPtrD, intptr_t csrSortedColIndD, size_t descrC, intptr_t csrSortedValC, intptr_t csrSortedRowPtrC, intptr_t csrSortedColIndC, size_t info, intptr_t pBuffer)
cpdef ccsrgemm2(intptr_t handle, int m, int n, int k, intptr_t alpha, size_t descrA, int nnzA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t descrB, int nnzB, intptr_t csrSortedValB, intptr_t csrSortedRowPtrB, intptr_t csrSortedColIndB, intptr_t beta, size_t descrD, int nnzD, intptr_t csrSortedValD, intptr_t csrSortedRowPtrD, intptr_t csrSortedColIndD, size_t descrC, intptr_t csrSortedValC, intptr_t csrSortedRowPtrC, intptr_t csrSortedColIndC, size_t info, intptr_t pBuffer)
cpdef zcsrgemm2(intptr_t handle, int m, int n, int k, intptr_t alpha, size_t descrA, int nnzA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t descrB, int nnzB, intptr_t csrSortedValB, intptr_t csrSortedRowPtrB, intptr_t csrSortedColIndB, intptr_t beta, size_t descrD, int nnzD, intptr_t csrSortedValD, intptr_t csrSortedRowPtrD, intptr_t csrSortedColIndD, size_t descrC, intptr_t csrSortedValC, intptr_t csrSortedRowPtrC, intptr_t csrSortedColIndC, size_t info, intptr_t pBuffer)


#######################################################################
# cuSPARSE Preconditioners - Incomplete Cholesky Factorization: level 0

cpdef int scsric02_bufferSize(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info) except? 0
cpdef int dcsric02_bufferSize(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info) except? 0
cpdef int ccsric02_bufferSize(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info) except? 0
cpdef int zcsric02_bufferSize(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info) except? 0

cpdef scsric02_analysis(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info, int policy, intptr_t pBuffer)
cpdef dcsric02_analysis(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info, int policy, intptr_t pBuffer)
cpdef ccsric02_analysis(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info, int policy, intptr_t pBuffer)
cpdef zcsric02_analysis(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info, int policy, intptr_t pBuffer)

cpdef scsric02(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrSortedValA_valM, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info, int policy, intptr_t pBuffer)
cpdef dcsric02(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrSortedValA_valM, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info, int policy, intptr_t pBuffer)
cpdef ccsric02(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrSortedValA_valM, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info, int policy, intptr_t pBuffer)
cpdef zcsric02(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrSortedValA_valM, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info, int policy, intptr_t pBuffer)

cpdef int xcsric02_zeroPivot(intptr_t handle, size_t info) except? 0

cpdef int sbsric02_bufferSize(intptr_t handle, int dirA, int mb, int nnzb, size_t descrA, intptr_t bsrSortedVal, intptr_t bsrSortedRowPtr, intptr_t bsrSortedColInd, int blockDim, size_t info) except? 0
cpdef int dbsric02_bufferSize(intptr_t handle, int dirA, int mb, int nnzb, size_t descrA, intptr_t bsrSortedVal, intptr_t bsrSortedRowPtr, intptr_t bsrSortedColInd, int blockDim, size_t info) except? 0
cpdef int cbsric02_bufferSize(intptr_t handle, int dirA, int mb, int nnzb, size_t descrA, intptr_t bsrSortedVal, intptr_t bsrSortedRowPtr, intptr_t bsrSortedColInd, int blockDim, size_t info) except? 0
cpdef int zbsric02_bufferSize(intptr_t handle, int dirA, int mb, int nnzb, size_t descrA, intptr_t bsrSortedVal, intptr_t bsrSortedRowPtr, intptr_t bsrSortedColInd, int blockDim, size_t info) except? 0

cpdef sbsric02_analysis(intptr_t handle, int dirA, int mb, int nnzb, size_t descrA, intptr_t bsrSortedVal, intptr_t bsrSortedRowPtr, intptr_t bsrSortedColInd, int blockDim, size_t info, int policy, intptr_t pInputBuffer)
cpdef dbsric02_analysis(intptr_t handle, int dirA, int mb, int nnzb, size_t descrA, intptr_t bsrSortedVal, intptr_t bsrSortedRowPtr, intptr_t bsrSortedColInd, int blockDim, size_t info, int policy, intptr_t pInputBuffer)
cpdef cbsric02_analysis(intptr_t handle, int dirA, int mb, int nnzb, size_t descrA, intptr_t bsrSortedVal, intptr_t bsrSortedRowPtr, intptr_t bsrSortedColInd, int blockDim, size_t info, int policy, intptr_t pInputBuffer)
cpdef zbsric02_analysis(intptr_t handle, int dirA, int mb, int nnzb, size_t descrA, intptr_t bsrSortedVal, intptr_t bsrSortedRowPtr, intptr_t bsrSortedColInd, int blockDim, size_t info, int policy, intptr_t pInputBuffer)

cpdef sbsric02(intptr_t handle, int dirA, int mb, int nnzb, size_t descrA, intptr_t bsrSortedVal, intptr_t bsrSortedRowPtr, intptr_t bsrSortedColInd, int blockDim, size_t info, int policy, intptr_t pBuffer)
cpdef dbsric02(intptr_t handle, int dirA, int mb, int nnzb, size_t descrA, intptr_t bsrSortedVal, intptr_t bsrSortedRowPtr, intptr_t bsrSortedColInd, int blockDim, size_t info, int policy, intptr_t pBuffer)
cpdef cbsric02(intptr_t handle, int dirA, int mb, int nnzb, size_t descrA, intptr_t bsrSortedVal, intptr_t bsrSortedRowPtr, intptr_t bsrSortedColInd, int blockDim, size_t info, int policy, intptr_t pBuffer)
cpdef zbsric02(intptr_t handle, int dirA, int mb, int nnzb, size_t descrA, intptr_t bsrSortedVal, intptr_t bsrSortedRowPtr, intptr_t bsrSortedColInd, int blockDim, size_t info, int policy, intptr_t pBuffer)

cpdef int xbsric02_zeroPivot(intptr_t handle, size_t info) except? 0


#################################################################
# cuSPARSE Preconditioners - Incomplete LU Factorization: level 0

cpdef scsrilu02_numericBoost(intptr_t handle, size_t info, int enable_boost, intptr_t tol, intptr_t boost_val)
cpdef dcsrilu02_numericBoost(intptr_t handle, size_t info, int enable_boost, intptr_t tol, intptr_t boost_val)
cpdef ccsrilu02_numericBoost(intptr_t handle, size_t info, int enable_boost, intptr_t tol, intptr_t boost_val)
cpdef zcsrilu02_numericBoost(intptr_t handle, size_t info, int enable_boost, intptr_t tol, intptr_t boost_val)

cpdef int scsrilu02_bufferSize(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info) except? 0
cpdef int dcsrilu02_bufferSize(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info) except? 0
cpdef int ccsrilu02_bufferSize(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info) except? 0
cpdef int zcsrilu02_bufferSize(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info) except? 0

cpdef scsrilu02_analysis(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info, int policy, intptr_t pBuffer)
cpdef dcsrilu02_analysis(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info, int policy, intptr_t pBuffer)
cpdef ccsrilu02_analysis(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info, int policy, intptr_t pBuffer)
cpdef zcsrilu02_analysis(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info, int policy, intptr_t pBuffer)

cpdef scsrilu02(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrSortedValA_valM, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info, int policy, intptr_t pBuffer)
cpdef dcsrilu02(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrSortedValA_valM, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info, int policy, intptr_t pBuffer)
cpdef ccsrilu02(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrSortedValA_valM, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info, int policy, intptr_t pBuffer)
cpdef zcsrilu02(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrSortedValA_valM, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info, int policy, intptr_t pBuffer)

cpdef xcsrilu02_zeroPivot(intptr_t handle, size_t info, intptr_t position)

cpdef sbsrilu02_numericBoost(intptr_t handle, size_t info, int enable_boost, intptr_t tol, intptr_t boost_val)
cpdef dbsrilu02_numericBoost(intptr_t handle, size_t info, int enable_boost, intptr_t tol, intptr_t boost_val)
cpdef cbsrilu02_numericBoost(intptr_t handle, size_t info, int enable_boost, intptr_t tol, intptr_t boost_val)
cpdef zbsrilu02_numericBoost(intptr_t handle, size_t info, int enable_boost, intptr_t tol, intptr_t boost_val)

cpdef int sbsrilu02_bufferSize(intptr_t handle, int dirA, int mb, int nnzb, size_t descrA, intptr_t bsrSortedVal, intptr_t bsrSortedRowPtr, intptr_t bsrSortedColInd, int blockDim, size_t info) except? 0
cpdef int dbsrilu02_bufferSize(intptr_t handle, int dirA, int mb, int nnzb, size_t descrA, intptr_t bsrSortedVal, intptr_t bsrSortedRowPtr, intptr_t bsrSortedColInd, int blockDim, size_t info) except? 0
cpdef int cbsrilu02_bufferSize(intptr_t handle, int dirA, int mb, int nnzb, size_t descrA, intptr_t bsrSortedVal, intptr_t bsrSortedRowPtr, intptr_t bsrSortedColInd, int blockDim, size_t info) except? 0
cpdef int zbsrilu02_bufferSize(intptr_t handle, int dirA, int mb, int nnzb, size_t descrA, intptr_t bsrSortedVal, intptr_t bsrSortedRowPtr, intptr_t bsrSortedColInd, int blockDim, size_t info) except? 0

cpdef sbsrilu02_analysis(intptr_t handle, int dirA, int mb, int nnzb, size_t descrA, intptr_t bsrSortedVal, intptr_t bsrSortedRowPtr, intptr_t bsrSortedColInd, int blockDim, size_t info, int policy, intptr_t pBuffer)
cpdef dbsrilu02_analysis(intptr_t handle, int dirA, int mb, int nnzb, size_t descrA, intptr_t bsrSortedVal, intptr_t bsrSortedRowPtr, intptr_t bsrSortedColInd, int blockDim, size_t info, int policy, intptr_t pBuffer)
cpdef cbsrilu02_analysis(intptr_t handle, int dirA, int mb, int nnzb, size_t descrA, intptr_t bsrSortedVal, intptr_t bsrSortedRowPtr, intptr_t bsrSortedColInd, int blockDim, size_t info, int policy, intptr_t pBuffer)
cpdef zbsrilu02_analysis(intptr_t handle, int dirA, int mb, int nnzb, size_t descrA, intptr_t bsrSortedVal, intptr_t bsrSortedRowPtr, intptr_t bsrSortedColInd, int blockDim, size_t info, int policy, intptr_t pBuffer)

cpdef sbsrilu02(intptr_t handle, int dirA, int mb, int nnzb, size_t descrA, intptr_t bsrSortedVal, intptr_t bsrSortedRowPtr, intptr_t bsrSortedColInd, int blockDim, size_t info, int policy, intptr_t pBuffer)
cpdef dbsrilu02(intptr_t handle, int dirA, int mb, int nnzb, size_t descrA, intptr_t bsrSortedVal, intptr_t bsrSortedRowPtr, intptr_t bsrSortedColInd, int blockDim, size_t info, int policy, intptr_t pBuffer)
cpdef cbsrilu02(intptr_t handle, int dirA, int mb, int nnzb, size_t descrA, intptr_t bsrSortedVal, intptr_t bsrSortedRowPtr, intptr_t bsrSortedColInd, int blockDim, size_t info, int policy, intptr_t pBuffer)
cpdef zbsrilu02(intptr_t handle, int dirA, int mb, int nnzb, size_t descrA, intptr_t bsrSortedVal, intptr_t bsrSortedRowPtr, intptr_t bsrSortedColInd, int blockDim, size_t info, int policy, intptr_t pBuffer)

cpdef xbsrilu02_zeroPivot(intptr_t handle, size_t info, intptr_t position)


##############################################
# cuSPARSE Preconditioners - Tridiagonal Solve

cpdef size_t sgtsv2_bufferSizeExt(intptr_t handle, int m, int n, intptr_t dl, intptr_t d, intptr_t du, intptr_t B, int ldb) except? 0
cpdef size_t dgtsv2_bufferSizeExt(intptr_t handle, int m, int n, intptr_t dl, intptr_t d, intptr_t du, intptr_t B, int ldb) except? 0
cpdef size_t cgtsv2_bufferSizeExt(intptr_t handle, int m, int n, intptr_t dl, intptr_t d, intptr_t du, intptr_t B, int ldb) except? 0
cpdef size_t zgtsv2_bufferSizeExt(intptr_t handle, int m, int n, intptr_t dl, intptr_t d, intptr_t du, intptr_t B, int ldb) except? 0

cpdef sgtsv2(intptr_t handle, int m, int n, intptr_t dl, intptr_t d, intptr_t du, intptr_t B, int ldb, intptr_t pBuffer)
cpdef dgtsv2(intptr_t handle, int m, int n, intptr_t dl, intptr_t d, intptr_t du, intptr_t B, int ldb, intptr_t pBuffer)
cpdef cgtsv2(intptr_t handle, int m, int n, intptr_t dl, intptr_t d, intptr_t du, intptr_t B, int ldb, intptr_t pBuffer)
cpdef zgtsv2(intptr_t handle, int m, int n, intptr_t dl, intptr_t d, intptr_t du, intptr_t B, int ldb, intptr_t pBuffer)

cpdef size_t sgtsv2_nopivot_bufferSizeExt(intptr_t handle, int m, int n, intptr_t dl, intptr_t d, intptr_t du, intptr_t B, int ldb) except? 0
cpdef size_t dgtsv2_nopivot_bufferSizeExt(intptr_t handle, int m, int n, intptr_t dl, intptr_t d, intptr_t du, intptr_t B, int ldb) except? 0
cpdef size_t cgtsv2_nopivot_bufferSizeExt(intptr_t handle, int m, int n, intptr_t dl, intptr_t d, intptr_t du, intptr_t B, int ldb) except? 0
cpdef size_t zgtsv2_nopivot_bufferSizeExt(intptr_t handle, int m, int n, intptr_t dl, intptr_t d, intptr_t du, intptr_t B, int ldb) except? 0

cpdef sgtsv2_nopivot(intptr_t handle, int m, int n, intptr_t dl, intptr_t d, intptr_t du, intptr_t B, int ldb, intptr_t pBuffer)
cpdef dgtsv2_nopivot(intptr_t handle, int m, int n, intptr_t dl, intptr_t d, intptr_t du, intptr_t B, int ldb, intptr_t pBuffer)
cpdef cgtsv2_nopivot(intptr_t handle, int m, int n, intptr_t dl, intptr_t d, intptr_t du, intptr_t B, int ldb, intptr_t pBuffer)
cpdef zgtsv2_nopivot(intptr_t handle, int m, int n, intptr_t dl, intptr_t d, intptr_t du, intptr_t B, int ldb, intptr_t pBuffer)


######################################################
# cuSPARSE Preconditioners - Batched Tridiagonal Solve

cpdef size_t sgtsv2StridedBatch_bufferSizeExt(intptr_t handle, int m, intptr_t dl, intptr_t d, intptr_t du, intptr_t x, int batchCount, int batchStride) except? 0
cpdef size_t dgtsv2StridedBatch_bufferSizeExt(intptr_t handle, int m, intptr_t dl, intptr_t d, intptr_t du, intptr_t x, int batchCount, int batchStride) except? 0
cpdef size_t cgtsv2StridedBatch_bufferSizeExt(intptr_t handle, int m, intptr_t dl, intptr_t d, intptr_t du, intptr_t x, int batchCount, int batchStride) except? 0
cpdef size_t zgtsv2StridedBatch_bufferSizeExt(intptr_t handle, int m, intptr_t dl, intptr_t d, intptr_t du, intptr_t x, int batchCount, int batchStride) except? 0

cpdef sgtsv2StridedBatch(intptr_t handle, int m, intptr_t dl, intptr_t d, intptr_t du, intptr_t x, int batchCount, int batchStride, intptr_t pBuffer)
cpdef dgtsv2StridedBatch(intptr_t handle, int m, intptr_t dl, intptr_t d, intptr_t du, intptr_t x, int batchCount, int batchStride, intptr_t pBuffer)
cpdef cgtsv2StridedBatch(intptr_t handle, int m, intptr_t dl, intptr_t d, intptr_t du, intptr_t x, int batchCount, int batchStride, intptr_t pBuffer)
cpdef zgtsv2StridedBatch(intptr_t handle, int m, intptr_t dl, intptr_t d, intptr_t du, intptr_t x, int batchCount, int batchStride, intptr_t pBuffer)

cpdef size_t sgtsvInterleavedBatch_bufferSizeExt(intptr_t handle, int algo, int m, intptr_t dl, intptr_t d, intptr_t du, intptr_t x, int batchCount) except? 0
cpdef size_t dgtsvInterleavedBatch_bufferSizeExt(intptr_t handle, int algo, int m, intptr_t dl, intptr_t d, intptr_t du, intptr_t x, int batchCount) except? 0
cpdef size_t cgtsvInterleavedBatch_bufferSizeExt(intptr_t handle, int algo, int m, intptr_t dl, intptr_t d, intptr_t du, intptr_t x, int batchCount) except? 0
cpdef size_t zgtsvInterleavedBatch_bufferSizeExt(intptr_t handle, int algo, int m, intptr_t dl, intptr_t d, intptr_t du, intptr_t x, int batchCount) except? 0

cpdef sgtsvInterleavedBatch(intptr_t handle, int algo, int m, intptr_t dl, intptr_t d, intptr_t du, intptr_t x, int batchCount, intptr_t pBuffer)
cpdef dgtsvInterleavedBatch(intptr_t handle, int algo, int m, intptr_t dl, intptr_t d, intptr_t du, intptr_t x, int batchCount, intptr_t pBuffer)
cpdef cgtsvInterleavedBatch(intptr_t handle, int algo, int m, intptr_t dl, intptr_t d, intptr_t du, intptr_t x, int batchCount, intptr_t pBuffer)
cpdef zgtsvInterleavedBatch(intptr_t handle, int algo, int m, intptr_t dl, intptr_t d, intptr_t du, intptr_t x, int batchCount, intptr_t pBuffer)


########################################################
# cuSPARSE Preconditioners - Batched Pentadiagonal Solve

cpdef size_t sgpsvInterleavedBatch_bufferSizeExt(intptr_t handle, int algo, int m, intptr_t ds, intptr_t dl, intptr_t d, intptr_t du, intptr_t dw, intptr_t x, int batchCount) except? 0
cpdef size_t dgpsvInterleavedBatch_bufferSizeExt(intptr_t handle, int algo, int m, intptr_t ds, intptr_t dl, intptr_t d, intptr_t du, intptr_t dw, intptr_t x, int batchCount) except? 0
cpdef size_t cgpsvInterleavedBatch_bufferSizeExt(intptr_t handle, int algo, int m, intptr_t ds, intptr_t dl, intptr_t d, intptr_t du, intptr_t dw, intptr_t x, int batchCount) except? 0
cpdef size_t zgpsvInterleavedBatch_bufferSizeExt(intptr_t handle, int algo, int m, intptr_t ds, intptr_t dl, intptr_t d, intptr_t du, intptr_t dw, intptr_t x, int batchCount) except? 0

cpdef sgpsvInterleavedBatch(intptr_t handle, int algo, int m, intptr_t ds, intptr_t dl, intptr_t d, intptr_t du, intptr_t dw, intptr_t x, int batchCount, intptr_t pBuffer)
cpdef dgpsvInterleavedBatch(intptr_t handle, int algo, int m, intptr_t ds, intptr_t dl, intptr_t d, intptr_t du, intptr_t dw, intptr_t x, int batchCount, intptr_t pBuffer)
cpdef cgpsvInterleavedBatch(intptr_t handle, int algo, int m, intptr_t ds, intptr_t dl, intptr_t d, intptr_t du, intptr_t dw, intptr_t x, int batchCount, intptr_t pBuffer)
cpdef zgpsvInterleavedBatch(intptr_t handle, int algo, int m, intptr_t ds, intptr_t dl, intptr_t d, intptr_t du, intptr_t dw, intptr_t x, int batchCount, intptr_t pBuffer)


########################################
# cuSPARSE Reorderings


########################################
# cuSPARSE Format Conversion

cpdef xcoo2csr(intptr_t handle, intptr_t cooRowInd, int nnz, int m, intptr_t csrSortedRowPtr, int idxBase)

cpdef scsc2dense(intptr_t handle, int m, int n, size_t descrA, intptr_t cscSortedValA, intptr_t cscSortedRowIndA, intptr_t cscSortedColPtrA, intptr_t A, int lda)
cpdef dcsc2dense(intptr_t handle, int m, int n, size_t descrA, intptr_t cscSortedValA, intptr_t cscSortedRowIndA, intptr_t cscSortedColPtrA, intptr_t A, int lda)
cpdef ccsc2dense(intptr_t handle, int m, int n, size_t descrA, intptr_t cscSortedValA, intptr_t cscSortedRowIndA, intptr_t cscSortedColPtrA, intptr_t A, int lda)
cpdef zcsc2dense(intptr_t handle, int m, int n, size_t descrA, intptr_t cscSortedValA, intptr_t cscSortedRowIndA, intptr_t cscSortedColPtrA, intptr_t A, int lda)

cpdef xcsr2coo(intptr_t handle, intptr_t csrSortedRowPtr, int nnz, int m, intptr_t cooRowInd, int idxBase)

# REMOVED
cpdef scsr2csc(intptr_t handle, int m, int n, int nnz, intptr_t csrSortedVal, intptr_t csrSortedRowPtr, intptr_t csrSortedColInd, intptr_t cscSortedVal, intptr_t cscSortedRowInd, intptr_t cscSortedColPtr, int copyValues, int idxBase)
# REMOVED
cpdef dcsr2csc(intptr_t handle, int m, int n, int nnz, intptr_t csrSortedVal, intptr_t csrSortedRowPtr, intptr_t csrSortedColInd, intptr_t cscSortedVal, intptr_t cscSortedRowInd, intptr_t cscSortedColPtr, int copyValues, int idxBase)
# REMOVED
cpdef ccsr2csc(intptr_t handle, int m, int n, int nnz, intptr_t csrSortedVal, intptr_t csrSortedRowPtr, intptr_t csrSortedColInd, intptr_t cscSortedVal, intptr_t cscSortedRowInd, intptr_t cscSortedColPtr, int copyValues, int idxBase)
# REMOVED
cpdef zcsr2csc(intptr_t handle, int m, int n, int nnz, intptr_t csrSortedVal, intptr_t csrSortedRowPtr, intptr_t csrSortedColInd, intptr_t cscSortedVal, intptr_t cscSortedRowInd, intptr_t cscSortedColPtr, int copyValues, int idxBase)

cpdef size_t csr2cscEx2_bufferSize(intptr_t handle, int m, int n, int nnz, intptr_t csrVal, intptr_t csrRowPtr, intptr_t csrColInd, intptr_t cscVal, intptr_t cscColPtr, intptr_t cscRowInd, size_t valType, int copyValues, int idxBase, int alg) except? 0

cpdef csr2cscEx2(intptr_t handle, int m, int n, int nnz, intptr_t csrVal, intptr_t csrRowPtr, intptr_t csrColInd, intptr_t cscVal, intptr_t cscColPtr, intptr_t cscRowInd, size_t valType, int copyValues, int idxBase, int alg, intptr_t buffer)

cpdef scsr2dense(intptr_t handle, int m, int n, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t A, int lda)
cpdef dcsr2dense(intptr_t handle, int m, int n, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t A, int lda)
cpdef ccsr2dense(intptr_t handle, int m, int n, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t A, int lda)
cpdef zcsr2dense(intptr_t handle, int m, int n, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t A, int lda)

cpdef int snnz_compress(intptr_t handle, int m, size_t descr, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t nnzPerRow, float tol) except? 0
cpdef int dnnz_compress(intptr_t handle, int m, size_t descr, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t nnzPerRow, double tol) except? 0
cpdef int cnnz_compress(intptr_t handle, int m, size_t descr, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t nnzPerRow, complex tol) except? 0
cpdef int znnz_compress(intptr_t handle, int m, size_t descr, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t nnzPerRow, double complex tol) except? 0

cpdef scsr2csr_compress(intptr_t handle, int m, int n, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedColIndA, intptr_t csrSortedRowPtrA, int nnzA, intptr_t nnzPerRow, intptr_t csrSortedValC, intptr_t csrSortedColIndC, intptr_t csrSortedRowPtrC, float tol)
cpdef dcsr2csr_compress(intptr_t handle, int m, int n, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedColIndA, intptr_t csrSortedRowPtrA, int nnzA, intptr_t nnzPerRow, intptr_t csrSortedValC, intptr_t csrSortedColIndC, intptr_t csrSortedRowPtrC, double tol)
cpdef ccsr2csr_compress(intptr_t handle, int m, int n, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedColIndA, intptr_t csrSortedRowPtrA, int nnzA, intptr_t nnzPerRow, intptr_t csrSortedValC, intptr_t csrSortedColIndC, intptr_t csrSortedRowPtrC, complex tol)
cpdef zcsr2csr_compress(intptr_t handle, int m, int n, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedColIndA, intptr_t csrSortedRowPtrA, int nnzA, intptr_t nnzPerRow, intptr_t csrSortedValC, intptr_t csrSortedColIndC, intptr_t csrSortedRowPtrC, double complex tol)

cpdef sdense2csc(intptr_t handle, int m, int n, size_t descrA, intptr_t A, int lda, intptr_t nnzPerCol, intptr_t cscSortedValA, intptr_t cscSortedRowIndA, intptr_t cscSortedColPtrA)
cpdef ddense2csc(intptr_t handle, int m, int n, size_t descrA, intptr_t A, int lda, intptr_t nnzPerCol, intptr_t cscSortedValA, intptr_t cscSortedRowIndA, intptr_t cscSortedColPtrA)
cpdef cdense2csc(intptr_t handle, int m, int n, size_t descrA, intptr_t A, int lda, intptr_t nnzPerCol, intptr_t cscSortedValA, intptr_t cscSortedRowIndA, intptr_t cscSortedColPtrA)
cpdef zdense2csc(intptr_t handle, int m, int n, size_t descrA, intptr_t A, int lda, intptr_t nnzPerCol, intptr_t cscSortedValA, intptr_t cscSortedRowIndA, intptr_t cscSortedColPtrA)

cpdef sdense2csr(intptr_t handle, int m, int n, size_t descrA, intptr_t A, int lda, intptr_t nnzPerRow, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA)
cpdef ddense2csr(intptr_t handle, int m, int n, size_t descrA, intptr_t A, int lda, intptr_t nnzPerRow, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA)
cpdef cdense2csr(intptr_t handle, int m, int n, size_t descrA, intptr_t A, int lda, intptr_t nnzPerRow, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA)
cpdef zdense2csr(intptr_t handle, int m, int n, size_t descrA, intptr_t A, int lda, intptr_t nnzPerRow, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA)

cpdef snnz(intptr_t handle, int dirA, int m, int n, size_t descrA, intptr_t A, int lda, intptr_t nnzPerRowCol, intptr_t nnzTotalDevHostPtr)
cpdef dnnz(intptr_t handle, int dirA, int m, int n, size_t descrA, intptr_t A, int lda, intptr_t nnzPerRowCol, intptr_t nnzTotalDevHostPtr)
cpdef cnnz(intptr_t handle, int dirA, int m, int n, size_t descrA, intptr_t A, int lda, intptr_t nnzPerRowCol, intptr_t nnzTotalDevHostPtr)
cpdef znnz(intptr_t handle, int dirA, int m, int n, size_t descrA, intptr_t A, int lda, intptr_t nnzPerRowCol, intptr_t nnzTotalDevHostPtr)

cpdef createIdentityPermutation(intptr_t handle, int n, intptr_t p)

cpdef size_t xcoosort_bufferSizeExt(intptr_t handle, int m, int n, int nnz, intptr_t cooRowsA, intptr_t cooColsA) except? 0

cpdef xcoosortByRow(intptr_t handle, int m, int n, int nnz, intptr_t cooRowsA, intptr_t cooColsA, intptr_t P, intptr_t pBuffer)

cpdef xcoosortByColumn(intptr_t handle, int m, int n, int nnz, intptr_t cooRowsA, intptr_t cooColsA, intptr_t P, intptr_t pBuffer)

cpdef size_t xcsrsort_bufferSizeExt(intptr_t handle, int m, int n, int nnz, intptr_t csrRowPtrA, intptr_t csrColIndA) except? 0

cpdef xcsrsort(intptr_t handle, int m, int n, int nnz, size_t descrA, intptr_t csrRowPtrA, intptr_t csrColIndA, intptr_t P, intptr_t pBuffer)

cpdef size_t xcscsort_bufferSizeExt(intptr_t handle, int m, int n, int nnz, intptr_t cscColPtrA, intptr_t cscRowIndA) except? 0

cpdef xcscsort(intptr_t handle, int m, int n, int nnz, size_t descrA, intptr_t cscColPtrA, intptr_t cscRowIndA, intptr_t P, intptr_t pBuffer)


###########################################
# cuSPARSE Generic API - Sparse Vector APIs

cpdef size_t createSpVec(int64_t size, int64_t nnz, intptr_t indices, intptr_t values, int idxType, int idxBase, size_t valueType) except? 0

cpdef destroySpVec(size_t spVecDescr)

cpdef SpVecAttributes spVecGet(size_t spVecDescr)

cpdef int spVecGetIndexBase(size_t spVecDescr) except? 0

cpdef intptr_t spVecGetValues(size_t spVecDescr) except? 0

cpdef spVecSetValues(size_t spVecDescr, intptr_t values)


###########################################
# cuSPARSE Generic API - Sparse Matrix APIs

cpdef size_t createCoo(int64_t rows, int64_t cols, int64_t nnz, intptr_t cooRowInd, intptr_t cooColInd, intptr_t cooValues, int cooIdxType, int idxBase, size_t valueType) except? 0

cpdef size_t createCooAoS(int64_t rows, int64_t cols, int64_t nnz, intptr_t cooInd, intptr_t cooValues, int cooIdxType, int idxBase, size_t valueType) except? 0

cpdef size_t createCsr(int64_t rows, int64_t cols, int64_t nnz, intptr_t csrRowOffsets, intptr_t csrColInd, intptr_t csrValues, int csrRowOffsetsType, int csrColIndType, int idxBase, size_t valueType) except? 0

cpdef destroySpMat(size_t spMatDescr)

cpdef CooAttributes cooGet(size_t spMatDescr)

cpdef CooAoSAttributes cooAoSGet(size_t spMatDescr)

cpdef CsrAttributes csrGet(size_t spMatDescr)

cpdef int spMatGetFormat(size_t spMatDescr) except? 0

cpdef int spMatGetIndexBase(size_t spMatDescr) except? 0

cpdef intptr_t spMatGetValues(size_t spMatDescr) except? 0

cpdef spMatSetValues(size_t spMatDescr, intptr_t values)

cpdef int spMatGetStridedBatch(size_t spMatDescr) except? 0

cpdef spMatSetStridedBatch(size_t spMatDescr, int batchCount)


##########################################
# cuSPARSE Generic API - Dense Vector APIs

cpdef size_t createDnVec(int64_t size, intptr_t values, size_t valueType) except? 0

cpdef destroyDnVec(size_t dnVecDescr)

cpdef DnVecAttributes dnVecGet(size_t dnVecDescr)

cpdef intptr_t dnVecGetValues(size_t dnVecDescr) except? 0

cpdef dnVecSetValues(size_t dnVecDescr, intptr_t values)


##########################################
# cuSPARSE Generic API - Dense Matrix APIs

cpdef size_t createDnMat(int64_t rows, int64_t cols, int64_t ld, intptr_t values, size_t valueType, int order) except? 0

cpdef destroyDnMat(size_t dnMatDescr)

cpdef DnMatAttributes dnMatGet(size_t dnMatDescr)

cpdef intptr_t dnMatGetValues(size_t dnMatDescr) except? 0

cpdef dnMatSetValues(size_t dnMatDescr, intptr_t values)

cpdef DnMatBatchAttributes dnMatGetStridedBatch(size_t dnMatDescr)

cpdef dnMatSetStridedBatch(size_t dnMatDescr, int batchCount, int64_t batchStride)


##############################################
# cuSPARSE Generic API - Generic API Functions

cpdef size_t spVV_bufferSize(intptr_t handle, int opX, size_t vecX, size_t vecY, intptr_t result, size_t computeType) except? 0

cpdef spVV(intptr_t handle, int opX, size_t vecX, size_t vecY, intptr_t result, size_t computeType, intptr_t externalBuffer)

cpdef size_t spMV_bufferSize(intptr_t handle, int opA, intptr_t alpha, size_t matA, size_t vecX, intptr_t beta, size_t vecY, size_t computeType, int alg) except? 0

cpdef spMV(intptr_t handle, int opA, intptr_t alpha, size_t matA, size_t vecX, intptr_t beta, size_t vecY, size_t computeType, int alg, intptr_t externalBuffer)

cpdef size_t spMM_bufferSize(intptr_t handle, int opA, int opB, intptr_t alpha, size_t matA, size_t matB, intptr_t beta, size_t matC, size_t computeType, int alg) except? 0

cpdef spMM(intptr_t handle, int opA, int opB, intptr_t alpha, size_t matA, size_t matB, intptr_t beta, size_t matC, size_t computeType, int alg, intptr_t externalBuffer)

cpdef size_t constrainedGeMM_bufferSize(intptr_t handle, int opA, int opB, intptr_t alpha, size_t matA, size_t matB, intptr_t beta, size_t matC, size_t computeType) except? 0

cpdef constrainedGeMM(intptr_t handle, int opA, int opB, intptr_t alpha, size_t matA, size_t matB, intptr_t beta, size_t matC, size_t computeType, intptr_t externalBuffer)
