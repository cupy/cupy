
from libc.stdint cimport intptr_t

cdef extern from *:
    ctypedef int IndexBase 'cusparseIndexBase_t'
    ctypedef int Status 'cusparseStatus_t'

    ctypedef void* Handle 'cusparseHandle_t'

    ctypedef void* MatDescr 'cusparseMatDescr_t'

    ctypedef int Direction 'cusparseDirection_t'

    ctypedef int MatrixType 'cusparseMatrixType_t'

    ctypedef int Operation 'cusparseOperation_t'

    ctypedef int PointerMode 'cusparsePointerMode_t'

    ctypedef int Action 'cusparseAction_t'
    ctypedef int AlgMode 'cusparseAlgMode_t'

    ctypedef void* cusparseHandle_t
    ctypedef void* cusparseMatDescr_t
    ctypedef void* csric02Info_t
    ctypedef void* bsric02Info_t
    ctypedef void* csrilu02Info_t
    ctypedef void* bsrilu02Info_t
    ctypedef void* csrgemm2Info_t

    ctypedef int cusparseStatus_t
    ctypedef int cusparseDirection_t
    ctypedef int cusparseSolvePolicy_t

cpdef enum:
    CUSPARSE_POINTER_MODE_HOST = 0
    CUSPARSE_POINTER_MODE_DEVICE = 1

    CUSPARSE_ACTION_SYMBOLIC = 0
    CUSPARSE_ACTION_NUMERIC = 1

    CUSPARSE_INDEX_BASE_ZERO = 0
    CUSPARSE_INDEX_BASE_ONE = 1

    CUSPARSE_MATRIX_TYPE_GENERAL = 0
    CUSPARSE_MATRIX_TYPE_SYMMETRIC = 1
    CUSPARSE_MATRIX_TYPE_HERMITIAN = 2
    CUSPARSE_MATRIX_TYPE_TRIANGULAR = 3

    CUSPARSE_OPERATION_NON_TRANSPOSE = 0
    CUSPARSE_OPERATION_TRANSPOSE = 1
    CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE = 2

    CUSPARSE_DIRECTION_ROW = 0
    CUSPARSE_DIRECTION_COLUMN = 1

    CUSPARSE_SOLVE_POLICY_NO_LEVEL = 0
    CUSPARSE_SOLVE_POLICY_USE_LEVEL = 1

    CUSPARSE_ALG_NAIVE = 0
    CUSPARSE_ALG_MERGE_PATH = 1

cpdef intptr_t create() except? 0
cpdef destroy(intptr_t handle)
