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


# TODO: should also expose functions?

