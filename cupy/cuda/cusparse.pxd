cdef extern from *:
    ctypedef int IndexBase 'cusparseIndexBase_t'
    ctypedef int Status 'cusparseStatus_t'

    ctypedef void* Handle 'cusparseHandle_t'

    ctypedef void* MatDescr 'cusparseMatDescr_t'

    ctypedef int MatrixType 'cusparseMatrixType_t'

    ctypedef int Operation 'cusparseOperation_t'

    ctypedef int PointerMode 'cusparsePointerMode_t'

    ctypedef int Action 'cusparseAction_t'


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


cpdef size_t create() except *
cpdef void destroy(size_t handle)
