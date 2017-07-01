cimport cython

cdef extern from "cupy_cusparse.h":

    # cuSPARSE Helper Function
    Status cusparseCreate(Handle *handle)
    Status cusparseCreateMatDescr(MatDescr descr)
    Status cusparseDestroy(Handle handle)
    Status cusparseDestroyMatDescr(MatDescr descr)
    Status cusparseSetMatIndexBase(MatDescr descr, IndexBase base)
    Status cusparseSetMatType(MatDescr descr, MatrixType type)
    Status cusparseSetPointerMode(Handle handle, PointerMode mode)

    # cuSPARSE Level1 Function
    Status cusparseSgthr(
        Handle handle, int nnz, const float *y, float *xVal, const int *xInd,
        IndexBase idxBase)

    Status cusparseDgthr(
        Handle handle, int nnz, const double *y, double *xVal, const int *xInd,
        IndexBase idxBase)

    # cuSPARSE Level2 Function
    Status cusparseScsrmv(
        Handle handle, Operation transA, int m, int n, int nnz,
        const float *alpha, MatDescr descrA, const float *csrSortedValA,
        const int *csrSortedRowPtrA, const int *csrSortedColIndA,
        const float *x, const float *beta, float *y)

    Status cusparseDcsrmv(
        Handle handle, Operation transA, int m, int n, int nnz,
        const double *alpha, MatDescr descrA, const double *csrSortedValA,
        const int *csrSortedRowPtrA, const int *csrSortedColIndA,
        const double *x, const double *beta, double *y)

    # cuSPARSE Level3 Function
    Status cusparseScsrmm(
        Handle handle, Operation transA, int m, int n, int k, int nnz,
        const float *alpha, const MatDescr descrA, const float *csrSortedValA,
        const int *csrSortedRowPtrA, const int *csrSortedColIndA,
        const float *B, int ldb, const float *beta, float *C, int ldc)

    Status cusparseDcsrmm(
        Handle handle, Operation transA, int m, int n, int k, int nnz,
        const double *alpha, const MatDescr descrA,
        const double *csrSortedValA,
        const int *csrSortedRowPtrA, const int *csrSortedColIndA,
        const double *B, int ldb, const double *beta, double *C, int ldc)

    Status cusparseScsrmm2(
        Handle handle, Operation transA, Operation transB, int m, int n, int k,
        int nnz, const float *alpha, const MatDescr descrA,
        const float *csrValA, const int *csrRowPtrA, const int *csrColIndA,
        const float *B, int ldb, const float *beta, float *C, int ldc)

    Status cusparseDcsrmm2(
        Handle handle, Operation transA, Operation transB, int m, int n, int k,
        int nnz, const double *alpha, const MatDescr descrA,
        const double *csrValA, const int *csrRowPtrA, const int *csrColIndA,
        const double *B, int ldb, const double *beta, double *C, int ldc)

    # cuSPARSE Extra Function
    Status cusparseXcsrgeamNnz(
        Handle handle, int m, int n, const MatDescr descrA, int nnzA,
        const int *csrRowPtrA, const int *csrColIndA, const MatDescr descrB,
        int nnzB, const int *csrRowPtrB, const int *csrColIndB,
        const MatDescr descrC, int *csrRowPtrC, int *nnzTotalDevHostPtr)

    Status cusparseScsrgeam(
        Handle handle, int m, int n, const float *alpha, const MatDescr descrA,
        int nnzA, const float *csrValA, const int *csrRowPtrA,
        const int *csrColIndA, const float *beta, const MatDescr descrB,
        int nnzB, const float *csrValB, const int *csrRowPtrB,
        const int *csrColIndB, const MatDescr descrC, float *csrValC,
        int *csrRowPtrC, int *csrColIndC)

    Status cusparseDcsrgeam(
        Handle handle, int m, int n, const double *alpha,
        const MatDescr descrA,
        int nnzA, const double *csrValA, const int *csrRowPtrA,
        const int *csrColIndA, const double *beta, const MatDescr descrB,
        int nnzB, const double *csrValB, const int *csrRowPtrB,
        const int *csrColIndB, const MatDescr descrC, double *csrValC,
        int *csrRowPtrC, int *csrColIndC)

    Status cusparseXcsrgemmNnz(
        Handle handle, Operation transA, Operation transB, int m, int n, int k,
        const MatDescr descrA, const int nnzA, const int *csrRowPtrA,
        const int *csrColIndA, const MatDescr descrB, const int nnzB,
        const int *csrRowPtrB, const int *csrColIndB,
        const MatDescr descrC, int *csrRowPtrC, int *nnzTotalDevHostPtr)

    Status cusparseScsrgemm(
        Handle handle, Operation transA, Operation transB, int m, int n, int k,
        const MatDescr descrA, const int nnzA, const float *csrValA,
        const int *csrRowPtrA, const int *csrColIndA, const MatDescr descrB,
        const int nnzB, const float *csrValB, const int *csrRowPtrB,
        const int *csrColIndB, const MatDescr descrC, float *csrValC,
        const int *csrRowPtrC, int *csrColIndC)

    Status cusparseDcsrgemm(
        Handle handle, Operation transA, Operation transB, int m, int n, int k,
        const MatDescr descrA, const int nnzA, const double *csrValA,
        const int *csrRowPtrA, const int *csrColIndA, const MatDescr descrB,
        const int nnzB, const double *csrValB, const int *csrRowPtrB,
        const int *csrColIndB, const MatDescr descrC, double *csrValC,
        const int *csrRowPtrC, int *csrColIndC)

    # cuSPARSE Format Convrsion
    Status cusparseXcoo2csr(
        Handle handle, const int *cooRowInd, int nnz, int m, int *csrRowPtr,
        IndexBase idxBase)

    Status cusparseXcsr2coo(
        Handle handle, const int *csrRowPtr, int nnz, int m, int *cooRowInd,
        IndexBase idxBase)

    Status cusparseScsr2csc(
        Handle handle, int m, int n, int nnz, const float *csrVal,
        const int *csrRowPtr, const int *csrColInd, float *cscVal,
        int *cscRowInd, int *cscColPtr, Action copyValues, IndexBase idxBase)

    Status cusparseDcsr2csc(
        Handle handle, int m, int n, int nnz, const double *csrVal,
        const int *csrRowPtr, const int *csrColInd, double *cscVal,
        int *cscRowInd, int *cscColPtr, Action copyValues, IndexBase idxBase)

    Status cusparseScsr2dense(
        Handle handle, int m, int n, const MatDescr descrA,
        const float *csrSortedValA, const int *csrSortedRowPtrA,
        const int *csrSortedColIndA, float *A, int lda)

    Status cusparseDcsr2dense(
        Handle handle, int m, int n, const MatDescr descrA,
        const double *csrSortedValA, const int *csrSortedRowPtrA,
        const int *csrSortedColIndA, double *A, int lda)

    Status cusparseCreateIdentityPermutation(
        Handle handle, int n, int *p)

    Status cusparseXcoosort_bufferSizeExt(
        Handle handle, int m, int n, int nnz, const int *cooRows,
        const int *cooCols, size_t *pBufferSizeInBytes)

    Status cusparseXcoosortByRow(
        Handle handle, int m, int n, int nnz, int *cooRows, int *cooCols,
        int *P, void *pBuffer)

    Status cusparseXcsrsort_bufferSizeExt(
        Handle handle, int m, int n, int nnz, const int *csrRowPtr,
        const int *csrColInd, size_t *pBufferSizeInBytes)

    Status cusparseXcsrsort(
        Handle handle, int m, int n, int nnz, const MatDescr descrA,
        const int *csrRowPtr, int *csrColInd, int *P, void *pBuffer)

    Status cusparseXcscsort_bufferSizeExt(
        Handle handle, int m, int n, int nnz, const int *cscColPtr,
        const int *cscRowInd, size_t *pBufferSizeInBytes)

    Status cusparseXcscsort(
        Handle handle, int m, int n, int nnz, const MatDescr descrA,
        const int *cscColPtr, int *cscRowInd, int *P, void *pBuffer)


cdef dict STATUS = {
    0: 'CUSPARSE_STATUS_SUCCESS',
    1: 'CUSPARSE_STATUS_NOT_INITIALIZED',
    2: 'CUSPARSE_STATUS_ALLOC_FAILED',
    3: 'CUSPARSE_STATUS_INVALID_VALUE',
    4: 'CUSPARSE_STATUS_ARCH_MISMATCH',
    5: 'CUSPARSE_STATUS_MAPPING_ERROR',
    6: 'CUSPARSE_STATUS_EXECUTION_FAILED',
    7: 'CUSPARSE_STATUS_INTERNAL_ERROR',
    8: 'CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED',
    9: 'CUSPARSE_STATUS_ZERO_PIVOT',
}


class CuSparseError(RuntimeError):

    def __init__(self, int status):
        self.status = status
        super(CuSparseError, self).__init__('%s' % (STATUS[status]))


@cython.profile(False)
cpdef inline check_status(int status):
    if status != 0:
        raise CuSparseError(status)


########################################
# cuSPARSE Helper Function

cpdef size_t create() except *:
    cdef Handle handle
    status = cusparseCreate(& handle)
    check_status(status)
    return <size_t >handle


cpdef createMatDescr():
    cdef MatDescr desc
    status = cusparseCreateMatDescr(& desc)
    check_status(status)
    return <size_t>desc


cpdef destroy(size_t handle):
    status = cusparseDestroy(<Handle >handle)
    check_status(status)


cpdef destroyMatDescr(size_t descr):
    status = cusparseDestroyMatDescr(<MatDescr>descr)
    check_status(status)


cpdef setMatIndexBase(size_t descr, base):
    status = cusparseSetMatIndexBase(<MatDescr>descr, base)
    check_status(status)


cpdef setMatType(size_t descr, typ):
    status = cusparseSetMatType(<MatDescr>descr, typ)
    check_status(status)


cpdef setPointerMode(size_t handle, int mode):
    status = cusparseSetPointerMode(<Handle>handle, <PointerMode>mode)
    check_status(status)


########################################
# cuSPARSE Level1 Function

cpdef sgthr(size_t handle, int nnz, size_t y, size_t xVal, size_t xInd,
            int idxBase):
    status = cusparseSgthr(
        <Handle>handle, nnz, <const float *>y, <float *>xVal,
        <const int *>xInd, <IndexBase>idxBase)
    check_status(status)

cpdef dgthr(size_t handle, int nnz, size_t y, size_t xVal, size_t xInd,
            int idxBase):
    status = cusparseDgthr(
        <Handle>handle, nnz, <const double *>y, <double *>xVal,
        <const int *>xInd, <IndexBase>idxBase)
    check_status(status)


########################################
# cuSPARSE Level2 Function

cpdef scsrmv(
        size_t handle, int transA, int m, int n, int nnz,
        size_t alpha, size_t descrA, size_t csrSortedValA,
        size_t csrSortedRowPtrA, size_t csrSortedColIndA,
        size_t x, size_t beta, size_t y):
    status = cusparseScsrmv(
        <Handle>handle, <Operation>transA, m, n, nnz,
        <const float *>alpha, <MatDescr>descrA, <const float *>csrSortedValA,
        <const int *>csrSortedRowPtrA, <const int *>csrSortedColIndA,
        <const float *>x, <const float *>beta, <float *>y)
    check_status(status)

cpdef dcsrmv(
        size_t handle, int transA, int m, int n, int nnz,
        size_t alpha, size_t descrA, size_t csrSortedValA,
        size_t csrSortedRowPtrA, size_t csrSortedColIndA,
        size_t x, size_t beta, size_t y):
    status = cusparseDcsrmv(
        <Handle>handle, <Operation>transA, m, n, nnz,
        <const double *>alpha, <MatDescr>descrA, <const double *>csrSortedValA,
        <const int *>csrSortedRowPtrA, <const int *>csrSortedColIndA,
        <const double *>x, <const double *>beta, <double *>y)
    check_status(status)


########################################
# cuSPARSE Level3 Function

cpdef scsrmm(
        size_t handle, int transA, int m, int n, int k, int nnz,
        size_t alpha, size_t descrA, size_t csrSortedValA,
        size_t csrSortedRowPtrA, size_t csrSortedColIndA,
        size_t B, int ldb, size_t beta, size_t C, int ldc):
    status = cusparseScsrmm(
        <Handle>handle, <Operation>transA, m, n, k, nnz,
        <const float *>alpha, <MatDescr>descrA, <const float *>csrSortedValA,
        <const int *>csrSortedRowPtrA, <const int *>csrSortedColIndA,
        <const float *>B, ldb, <const float *>beta, <float *>C, ldc)
    check_status(status)

cpdef dcsrmm(
        size_t handle, int transA, int m, int n, int k, int nnz,
        size_t alpha, size_t descrA, size_t csrSortedValA,
        size_t csrSortedRowPtrA, size_t csrSortedColIndA,
        size_t B, int ldb, size_t beta, size_t C, int ldc):
    status = cusparseDcsrmm(
        <Handle>handle, <Operation>transA, m, n, k, nnz,
        <const double *>alpha, <MatDescr>descrA, <const double *>csrSortedValA,
        <const int *>csrSortedRowPtrA, <const int *>csrSortedColIndA,
        <const double *>B, ldb, <const double *>beta, <double *>C, ldc)
    check_status(status)

cpdef scsrmm2(
        size_t handle, int transA, int transB, int m, int n, int k, int nnz,
        size_t alpha, size_t descrA, size_t csrValA,
        size_t csrRowPtrA, size_t csrColIndA,
        size_t B, int ldb, size_t beta, size_t C, int ldc):
    status = cusparseScsrmm2(
        <Handle>handle, <Operation>transA, <Operation>transB, m, n, k, nnz,
        <const float *>alpha, <MatDescr>descrA, <const float *>csrValA,
        <const int *>csrRowPtrA, <const int *>csrColIndA,
        <const float *>B, ldb, <const float *>beta, <float *>C, ldc)
    check_status(status)

cpdef dcsrmm2(
        size_t handle, int transA, int transB, int m, int n, int k, int nnz,
        size_t alpha, size_t descrA, size_t csrValA,
        size_t csrRowPtrA, size_t csrColIndA,
        size_t B, int ldb, size_t beta, size_t C, int ldc):
    status = cusparseDcsrmm2(
        <Handle>handle, <Operation>transA, <Operation>transB, m, n, k, nnz,
        <const double *>alpha, <MatDescr>descrA, <const double *>csrValA,
        <const int *>csrRowPtrA, <const int *>csrColIndA,
        <const double *>B, ldb, <const double *>beta, <double *>C, ldc)
    check_status(status)


########################################
# cuSPARSE Extra Function

cpdef xcsrgeamNnz(
        size_t handle, int m, int n, size_t descrA, int nnzA,
        size_t csrRowPtrA, size_t csrColIndA, size_t descrB,
        int nnzB, size_t csrRowPtrB, size_t csrColIndB,
        size_t descrC, size_t csrRowPtrC, size_t nnzTotalDevHostPtr):
    status = cusparseXcsrgeamNnz(
        <Handle>handle, m, n, <const MatDescr>descrA, nnzA,
        <const int *>csrRowPtrA, <const int *>csrColIndA,
        <const MatDescr>descrB, nnzB, <const int *>csrRowPtrB,
        <const int *>csrColIndB, <const MatDescr>descrC, <int *>csrRowPtrC,
        <int *>nnzTotalDevHostPtr)
    check_status(status)

cpdef scsrgeam(
        size_t handle, int m, int n, size_t alpha, size_t descrA,
        int nnzA, size_t csrValA, size_t csrRowPtrA,
        size_t csrColIndA, size_t beta, size_t descrB,
        int nnzB, size_t csrValB, size_t csrRowPtrB,
        size_t csrColIndB, size_t descrC, size_t csrValC,
        size_t csrRowPtrC, size_t csrColIndC):
    status = cusparseScsrgeam(
        <Handle>handle, m, n, <const float *>alpha,
        <const MatDescr>descrA, nnzA, <const float *>csrValA,
        <const int *>csrRowPtrA, <const int *>csrColIndA, <const float *>beta,
        <const MatDescr>descrB, nnzB, <const float *>csrValB,
        <const int *>csrRowPtrB, <const int *>csrColIndB,
        <const MatDescr>descrC, <float *>csrValC, <int *>csrRowPtrC,
        <int *>csrColIndC)
    check_status(status)


cpdef dcsrgeam(
        size_t handle, int m, int n, size_t alpha, size_t descrA,
        int nnzA, size_t csrValA, size_t csrRowPtrA,
        size_t csrColIndA, size_t beta, size_t descrB,
        int nnzB, size_t csrValB, size_t csrRowPtrB,
        size_t csrColIndB, size_t descrC, size_t csrValC,
        size_t csrRowPtrC, size_t csrColIndC):
    status = cusparseDcsrgeam(
        <Handle>handle, m, n, <const double *>alpha,
        <const MatDescr>descrA, nnzA, <const double *>csrValA,
        <const int *>csrRowPtrA, <const int *>csrColIndA, <const double *>beta,
        <const MatDescr>descrB, nnzB, <const double *>csrValB,
        <const int *>csrRowPtrB, <const int *>csrColIndB,
        <const MatDescr>descrC, <double *>csrValC, <int *>csrRowPtrC,
        <int *>csrColIndC)
    check_status(status)


cpdef xcsrgemmNnz(
        size_t handle, int transA, int transB, int m, int n, int k,
        size_t descrA, int nnzA, size_t csrRowPtrA,
        size_t csrColIndA, size_t descrB, int nnzB,
        size_t csrRowPtrB, size_t csrColIndB,
        size_t descrC, size_t csrRowPtrC, size_t nnzTotalDevHostPtr):
    status = cusparseXcsrgemmNnz(
        <Handle>handle, <Operation>transA, <Operation>transB, m, n, k,
        <const MatDescr>descrA, nnzA, <const int *>csrRowPtrA,
        <const int *>csrColIndA, <const MatDescr>descrB, nnzB,
        <const int *>csrRowPtrB, <const int *>csrColIndB,
        <const MatDescr>descrC, <int *>csrRowPtrC, <int *>nnzTotalDevHostPtr)
    check_status(status)


cpdef scsrgemm(
        size_t handle, int transA, int transB, int m, int n, int k,
        size_t descrA, const int nnzA, size_t csrValA,
        size_t csrRowPtrA, size_t csrColIndA, size_t descrB,
        const int nnzB, size_t csrValB, size_t csrRowPtrB,
        size_t csrColIndB, size_t descrC, size_t csrValC,
        size_t csrRowPtrC, size_t csrColIndC):
    status = cusparseScsrgemm(
        <Handle>handle, <Operation>transA, <Operation>transB, m, n, k,
        <const MatDescr>descrA, nnzA, <const float *>csrValA,
        <const int *>csrRowPtrA, <const int *>csrColIndA,
        <const MatDescr>descrB, nnzB, <const float *>csrValB,
        <const int *>csrRowPtrB, <const int *>csrColIndB,
        <const MatDescr>descrC, <float *>csrValC,
        <const int *>csrRowPtrC, <int *>csrColIndC)
    check_status(status)


cpdef dcsrgemm(
        size_t handle, int transA, int transB, int m, int n, int k,
        size_t descrA, const int nnzA, size_t csrValA,
        size_t csrRowPtrA, size_t csrColIndA, size_t descrB,
        const int nnzB, size_t csrValB, size_t csrRowPtrB,
        size_t csrColIndB, size_t descrC, size_t csrValC,
        size_t csrRowPtrC, size_t csrColIndC):
    status = cusparseDcsrgemm(
        <Handle>handle, <Operation>transA, <Operation>transB, m, n, k,
        <const MatDescr>descrA, nnzA, <const double *>csrValA,
        <const int *>csrRowPtrA, <const int *>csrColIndA,
        <const MatDescr>descrB, nnzB, <const double *>csrValB,
        <const int *>csrRowPtrB, <const int *>csrColIndB,
        <const MatDescr>descrC, <double *>csrValC,
        <const int *>csrRowPtrC, <int *>csrColIndC)
    check_status(status)


########################################
# cuSPARSE Format Convrsion

cpdef xcoo2csr(
        size_t handle, size_t cooRowInd, int nnz, int m, size_t csrRowPtr,
        int idxBase):
    status = cusparseXcoo2csr(
        <Handle>handle, <const int *>cooRowInd, nnz, m, <int *>csrRowPtr,
        <IndexBase>idxBase)
    check_status(status)


cpdef xcsr2coo(
        size_t handle, size_t csrRowPtr, int nnz, int m, size_t cooRowInd,
        int idxBase):
    status = cusparseXcsr2coo(
        <Handle>handle, <const int *>csrRowPtr, nnz, m, <int *>cooRowInd,
        <IndexBase>idxBase)
    check_status(status)


cpdef scsr2csc(
        size_t handle, int m, int n, int nnz, size_t csrVal,
        size_t csrRowPtr, size_t csrColInd, size_t cscVal,
        size_t cscRowInd, size_t cscColPtr, int copyValues, int idxBase):
    status = cusparseScsr2csc(
        <Handle>handle, m, n, nnz, <const float *>csrVal,
        <const int *>csrRowPtr, <const int *>csrColInd, <float *>cscVal,
        <int *>cscRowInd, <int *>cscColPtr, <Action>copyValues,
        <IndexBase>idxBase)
    check_status(status)


cpdef dcsr2csc(
        size_t handle, int m, int n, int nnz, size_t csrVal,
        size_t csrRowPtr, size_t csrColInd, size_t cscVal,
        size_t cscRowInd, size_t cscColPtr, int copyValues, int idxBase):
    status = cusparseDcsr2csc(
        <Handle>handle, m, n, nnz, <const double *>csrVal,
        <const int *>csrRowPtr, <const int *>csrColInd, <double *>cscVal,
        <int *>cscRowInd, <int *>cscColPtr, <Action>copyValues,
        <IndexBase>idxBase)
    check_status(status)


cpdef scsr2dense(
        size_t handle, int m, int n, size_t descrA,
        size_t csrSortedValA, size_t csrSortedRowPtrA,
        size_t csrSortedColIndA, size_t A, int lda):
    status = cusparseScsr2dense(
        <Handle>handle, m, n, <MatDescr>descrA,
        <const float *>csrSortedValA, <const int *>csrSortedRowPtrA,
        <const int *>csrSortedColIndA, <float *>A, lda)
    check_status(status)


cpdef dcsr2dense(
        size_t handle, int m, int n, size_t descrA,
        size_t csrSortedValA, size_t csrSortedRowPtrA,
        size_t csrSortedColIndA, size_t A, int lda):
    status = cusparseDcsr2dense(
        <Handle>handle, m, n, <MatDescr>descrA,
        <const double *>csrSortedValA, <const int *>csrSortedRowPtrA,
        <const int *>csrSortedColIndA, <double *>A, lda)
    check_status(status)


cpdef createIdentityPermutation(
        size_t handle, int n, size_t p):
    status = cusparseCreateIdentityPermutation(
        <Handle>handle, n, <int *>p)
    check_status(status)


cpdef size_t xcoosort_bufferSizeExt(
        size_t handle, int m, int n, int nnz, size_t cooRows,
        size_t cooCols):
    cdef size_t bufferSizeInBytes
    status = cusparseXcoosort_bufferSizeExt(
        <Handle>handle, m, n, nnz, <const int *>cooRows,
        <const int *>cooCols, &bufferSizeInBytes)
    check_status(status)
    return bufferSizeInBytes


cpdef xcoosortByRow(
        size_t handle, int m, int n, int nnz, size_t cooRows, size_t cooCols,
        size_t P, size_t pBuffer):
    status = cusparseXcoosortByRow(
        <Handle>handle, m, n, nnz, <int *>cooRows, <int *>cooCols,
        <int *>P, <void *>pBuffer)
    check_status(status)


cpdef size_t xcsrsort_bufferSizeExt(
        size_t handle, int m, int n, int nnz, size_t csrRowPtr,
        size_t csrColInd):
    cdef size_t bufferSizeInBytes
    status = cusparseXcsrsort_bufferSizeExt(
        <Handle>handle, m, n, nnz, <const int *>csrRowPtr,
        <const int *>csrColInd, &bufferSizeInBytes)
    check_status(status)
    return bufferSizeInBytes


cpdef xcsrsort(
        size_t handle, int m, int n, int nnz, size_t descrA,
        size_t csrRowPtr, size_t csrColInd, size_t P, size_t pBuffer):
    status = cusparseXcsrsort(
        <Handle>handle, m, n, nnz, <const MatDescr>descrA,
        <const int *>csrRowPtr, <int *>csrColInd, <int *>P, <void *>pBuffer)
    check_status(status)


cpdef size_t xcscsort_bufferSizeExt(
        size_t handle, int m, int n, int nnz, size_t cscColPtr,
        size_t cscRowInd):
    cdef size_t bufferSizeInBytes
    status = cusparseXcscsort_bufferSizeExt(
        <Handle>handle, m, n, nnz, <const int *>cscColPtr,
        <const int *>cscRowInd, &bufferSizeInBytes)
    check_status(status)
    return bufferSizeInBytes


cpdef xcscsort(
        size_t handle, int m, int n, int nnz, size_t descrA,
        size_t cscColPtr, size_t cscRowInd, size_t P, size_t pBuffer):
    status = cusparseXcscsort(
        <Handle>handle, m, n, nnz, <const MatDescr>descrA,
        <const int *>cscColPtr, <int *>cscRowInd, <int *>P, <void *>pBuffer)
    check_status(status)
