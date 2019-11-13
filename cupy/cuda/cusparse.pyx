cimport cython  # NOQA

from cupy.cuda cimport driver
from cupy.cuda cimport stream as stream_module
from cupy.cuda.runtime cimport DataType

cdef extern from 'cupy_cuComplex.h':
    ctypedef struct cuComplex 'cuComplex':
        float x, y

    ctypedef struct cuDoubleComplex 'cuDoubleComplex':
        double x, y

cdef extern from 'cupy_cusparse.h' nogil:

    # cuSPARSE Helper Function
    Status cusparseCreate(Handle *handle)
    Status cusparseCreateMatDescr(MatDescr descr)
    Status cusparseDestroy(Handle handle)
    Status cusparseDestroyMatDescr(MatDescr descr)
    Status cusparseSetMatIndexBase(MatDescr descr, IndexBase base)
    Status cusparseSetMatType(MatDescr descr, MatrixType type)
    Status cusparseSetPointerMode(Handle handle, PointerMode mode)

    # Stream
    Status cusparseSetStream(Handle handle, driver.Stream streamId)
    Status cusparseGetStream(Handle handle, driver.Stream* streamId)

    # cuSPARSE Level1 Function
    Status cusparseSgthr(
        Handle handle, int nnz, const float *y, float *xVal, const int *xInd,
        IndexBase idxBase)

    Status cusparseDgthr(
        Handle handle, int nnz, const double *y, double *xVal, const int *xInd,
        IndexBase idxBase)

    Status cusparseCgthr(
        Handle handle, int nnz, const cuComplex *y, cuComplex *xVal,
        const int *xInd,
        IndexBase idxBase)

    Status cusparseZgthr(
        Handle handle, int nnz, const cuDoubleComplex *y,
        cuDoubleComplex *xVal, const int *xInd,
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

    Status cusparseCcsrmv(
        Handle handle, Operation transA, int m, int n, int nnz,
        const cuComplex *alpha, MatDescr descrA,
        const cuComplex *csrSortedValA,
        const int *csrSortedRowPtrA, const int *csrSortedColIndA,
        const cuComplex *x, const cuComplex *beta, cuComplex *y)

    Status cusparseZcsrmv(
        Handle handle, Operation transA, int m, int n, int nnz,
        const cuDoubleComplex *alpha, MatDescr descrA,
        const cuDoubleComplex *csrSortedValA,
        const int *csrSortedRowPtrA, const int *csrSortedColIndA,
        const cuDoubleComplex *x, const cuDoubleComplex *beta,
        cuDoubleComplex *y)

    Status cusparseCsrmvEx_bufferSize(
        Handle handle, AlgMode alg, Operation transA, int m, int n,
        int nnz, const void *alpha, DataType alphatype,
        MatDescr descrA, const void *csrValA, DataType csrValAtype,
        const int *csrRowPtrA, const int *csrColIndA,
        const void *x, DataType xtype, const void *beta,
        DataType betatype, void *y, DataType ytype,
        DataType executiontype, size_t *bufferSizeInBytes)

    Status cusparseCsrmvEx(
        Handle handle, AlgMode alg, Operation transA, int m, int n,
        int nnz, const void *alpha, DataType alphatype,
        MatDescr descrA, const void *csrValA, DataType csrValAtype,
        const int *csrRowPtrA, const int *csrColIndA,
        const void *x, DataType xtype, const void *beta,
        DataType betatype, void *y, DataType ytype,
        DataType executiontype, void* buffer)

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

    Status cusparseCcsrmm(
        Handle handle, Operation transA, int m, int n, int k, int nnz,
        const cuComplex *alpha, const MatDescr descrA,
        const cuComplex *csrSortedValA,
        const int *csrSortedRowPtrA, const int *csrSortedColIndA,
        const cuComplex *B, int ldb, const cuComplex *beta,
        cuComplex *C, int ldc)

    Status cusparseZcsrmm(
        Handle handle, Operation transA, int m, int n, int k, int nnz,
        const cuDoubleComplex *alpha, const MatDescr descrA,
        const cuDoubleComplex *csrSortedValA,
        const int *csrSortedRowPtrA, const int *csrSortedColIndA,
        const cuDoubleComplex *B, int ldb,
        const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc)

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

    Status cusparseCcsrmm2(
        Handle handle, Operation transA, Operation transB, int m, int n, int k,
        int nnz, const cuComplex *alpha, const MatDescr descrA,
        const cuComplex *csrValA, const int *csrRowPtrA, const int *csrColIndA,
        const cuComplex *B, int ldb, const cuComplex *beta,
        cuComplex *C, int ldc)

    Status cusparseZcsrmm2(
        Handle handle, Operation transA, Operation transB, int m, int n, int k,
        int nnz, const cuDoubleComplex *alpha, const MatDescr descrA,
        const cuDoubleComplex *csrValA,
        const int *csrRowPtrA, const int *csrColIndA,
        const cuDoubleComplex *B, int ldb, const cuDoubleComplex *beta,
        cuDoubleComplex *C, int ldc)

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

    Status cusparseCcsrgeam(
        Handle handle, int m, int n, const cuComplex *alpha,
        const MatDescr descrA,
        int nnzA, const cuComplex *csrValA, const int *csrRowPtrA,
        const int *csrColIndA, const cuComplex *beta, const MatDescr descrB,
        int nnzB, const cuComplex *csrValB, const int *csrRowPtrB,
        const int *csrColIndB, const MatDescr descrC, cuComplex *csrValC,
        int *csrRowPtrC, int *csrColIndC)

    Status cusparseZcsrgeam(
        Handle handle, int m, int n, const cuDoubleComplex *alpha,
        const MatDescr descrA,
        int nnzA, const cuDoubleComplex *csrValA, const int *csrRowPtrA,
        const int *csrColIndA, const cuDoubleComplex *beta,
        const MatDescr descrB,
        int nnzB, const cuDoubleComplex *csrValB, const int *csrRowPtrB,
        const int *csrColIndB, const MatDescr descrC,
        cuDoubleComplex *csrValC, int *csrRowPtrC, int *csrColIndC)

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

    Status cusparseCcsrgemm(
        Handle handle, Operation transA, Operation transB, int m, int n, int k,
        const MatDescr descrA, const int nnzA, const cuComplex *csrValA,
        const int *csrRowPtrA, const int *csrColIndA, const MatDescr descrB,
        const int nnzB, const cuComplex *csrValB, const int *csrRowPtrB,
        const int *csrColIndB, const MatDescr descrC, cuComplex *csrValC,
        const int *csrRowPtrC, int *csrColIndC)

    Status cusparseZcsrgemm(
        Handle handle, Operation transA, Operation transB, int m, int n, int k,
        const MatDescr descrA, const int nnzA, const cuDoubleComplex *csrValA,
        const int *csrRowPtrA, const int *csrColIndA, const MatDescr descrB,
        const int nnzB, const cuDoubleComplex *csrValB, const int *csrRowPtrB,
        const int *csrColIndB, const MatDescr descrC, cuDoubleComplex *csrValC,
        const int *csrRowPtrC, int *csrColIndC)

    # cuSPARSE Format Convrsion
    Status cusparseXcoo2csr(
        Handle handle, const int *cooRowInd, int nnz, int m, int *csrRowPtr,
        IndexBase idxBase)

    Status cusparseScsc2dense(
        Handle handle, int m, int n, const MatDescr descrA,
        const float *cscSortedValA, const int *cscSortedRowIndA,
        const int *cscSortedColPtrA, float *A, int lda)

    Status cusparseDcsc2dense(
        Handle handle, int m, int n, const MatDescr descrA,
        const double *cscSortedValA, const int *cscSortedRowIndA,
        const int *cscSortedColPtrA, double *A, int lda)

    Status cusparseCcsc2dense(
        Handle handle, int m, int n, const MatDescr descrA,
        const cuComplex *cscSortedValA, const int *cscSortedRowIndA,
        const int *cscSortedColPtrA, cuComplex *A, int lda)

    Status cusparseZcsc2dense(
        Handle handle, int m, int n, const MatDescr descrA,
        const cuDoubleComplex *cscSortedValA, const int *cscSortedRowIndA,
        const int *cscSortedColPtrA, cuDoubleComplex *A, int lda)

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

    Status cusparseCcsr2csc(
        Handle handle, int m, int n, int nnz, const cuComplex *csrVal,
        const int *csrRowPtr, const int *csrColInd, cuComplex *cscVal,
        int *cscRowInd, int *cscColPtr, Action copyValues, IndexBase idxBase)

    Status cusparseZcsr2csc(
        Handle handle, int m, int n, int nnz, const cuDoubleComplex *csrVal,
        const int *csrRowPtr, const int *csrColInd, cuDoubleComplex *cscVal,
        int *cscRowInd, int *cscColPtr, Action copyValues, IndexBase idxBase)

    Status cusparseScsr2dense(
        Handle handle, int m, int n, const MatDescr descrA,
        const float *csrSortedValA, const int *csrSortedRowPtrA,
        const int *csrSortedColIndA, float *A, int lda)

    Status cusparseDcsr2dense(
        Handle handle, int m, int n, const MatDescr descrA,
        const double *csrSortedValA, const int *csrSortedRowPtrA,
        const int *csrSortedColIndA, double *A, int lda)

    Status cusparseCcsr2dense(
        Handle handle, int m, int n, const MatDescr descrA,
        const cuComplex *csrSortedValA, const int *csrSortedRowPtrA,
        const int *csrSortedColIndA, cuComplex *A, int lda)

    Status cusparseZcsr2dense(
        Handle handle, int m, int n, const MatDescr descrA,
        const cuDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA,
        const int *csrSortedColIndA, cuDoubleComplex *A, int lda)

    Status cusparseSnnz_compress(
        Handle handle, int m, const MatDescr descr,
        const float *values, const int *rowPtr, int *nnzPerRow,
        int *nnzTotal, float tol)

    Status cusparseDnnz_compress(
        Handle handle, int m, const MatDescr descr,
        const double *values, const int *rowPtr, int *nnzPerRow,
        int *nnzTotal, double tol)

    Status cusparseCnnz_compress(
        Handle handle, int m, const MatDescr descr,
        const cuComplex *values, const int *rowPtr, int *nnzPerRow,
        int *nnzTotal, cuComplex tol)

    Status cusparseZnnz_compress(
        Handle handle, int m, const MatDescr descr,
        const cuDoubleComplex *values, const int *rowPtr, int *nnzPerRow,
        int *nnzTotal, cuDoubleComplex tol)

    Status cusparseScsr2csr_compress(
        Handle handle, int m, int n, const MatDescr descrA,
        const float *inVal, const int *inColInd, const int *inRowPtr,
        int inNnz, int *nnzPerRow, float *outVal, int *outColInd,
        int *outRowPtr, float tol)

    Status cusparseDcsr2csr_compress(
        Handle handle, int m, int n, const MatDescr descrA,
        const double *inVal, const int *inColInd, const int *inRowPtr,
        int inNnz, int *nnzPerRow, double *outVal, int *outColInd,
        int *outRowPtr, double tol)

    Status cusparseCcsr2csr_compress(
        Handle handle, int m, int n, const MatDescr descrA,
        const cuComplex *inVal, const int *inColInd, const int *inRowPtr,
        int inNnz, int *nnzPerRow, cuComplex *outVal, int *outColInd,
        int *outRowPtr, cuComplex tol)

    Status cusparseZcsr2csr_compress(
        Handle handle, int m, int n, const MatDescr descrA,
        const cuDoubleComplex *inVal, const int *inColInd, const int *inRowPtr,
        int inNnz, int *nnzPerRow, cuDoubleComplex *outVal, int *outColInd,
        int *outRowPtr, cuDoubleComplex tol)

    Status cusparseSdense2csc(
        Handle handle, int m, int n, const MatDescr descrA, const float *A,
        int lda, const int *nnzPerCol, float *cscValA, int *cscRowIndA,
        int *cscColPtrA)

    Status cusparseDdense2csc(
        Handle handle, int m, int n, const MatDescr descrA, const double *A,
        int lda, const int *nnzPerCol, double *cscValA, int *cscRowIndA,
        int *cscColPtrA)

    Status cusparseCdense2csc(
        Handle handle, int m, int n, const MatDescr descrA, const cuComplex *A,
        int lda, const int *nnzPerCol, cuComplex *cscValA, int *cscRowIndA,
        int *cscColPtrA)

    Status cusparseZdense2csc(
        Handle handle, int m, int n, const MatDescr descrA,
        const cuDoubleComplex *A,
        int lda, const int *nnzPerCol, cuDoubleComplex *cscValA,
        int *cscRowIndA, int *cscColPtrA)

    Status cusparseSdense2csr(
        Handle handle, int m, int n, const MatDescr descrA,
        const float *A, int lda, const int *nnzPerRow, float *csrValA,
        int *csrRowPtrA, int *csrColIndA)

    Status cusparseDdense2csr(
        Handle handle, int m, int n, const MatDescr descrA,
        const double *A, int lda, const int *nnzPerRow, double *csrValA,
        int *csrRowPtrA, int *csrColIndA)

    Status cusparseCdense2csr(
        Handle handle, int m, int n, const MatDescr descrA,
        const cuComplex *A, int lda, const int *nnzPerRow, cuComplex *csrValA,
        int *csrRowPtrA, int *csrColIndA)

    Status cusparseZdense2csr(
        Handle handle, int m, int n, const MatDescr descrA,
        const cuDoubleComplex *A, int lda, const int *nnzPerRow,
        cuDoubleComplex *csrValA,
        int *csrRowPtrA, int *csrColIndA)

    Status cusparseSnnz(
        Handle handle, Direction dirA, int m, int n, const MatDescr descrA,
        const float *A, int lda, int *nnzPerRowColumn, int *nnzTotalDevHostPtr)

    Status cusparseDnnz(
        Handle handle, Direction dirA, int m, int n, const MatDescr descrA,
        const double *A, int lda, int *nnzPerRowColumn,
        int *nnzTotalDevHostPtr)

    Status cusparseCnnz(
        Handle handle, Direction dirA, int m, int n, const MatDescr descrA,
        const cuComplex *A, int lda, int *nnzPerRowColumn,
        int *nnzTotalDevHostPtr)

    Status cusparseZnnz(
        Handle handle, Direction dirA, int m, int n, const MatDescr descrA,
        const cuDoubleComplex *A, int lda, int *nnzPerRowColumn,
        int *nnzTotalDevHostPtr)

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

    # cuSparse PRECONDITIONERS
    cusparseStatus_t cusparseCreateCsrilu02Info(csrilu02Info_t *info)
    cusparseStatus_t cusparseDestroyCsrilu02Info(csrilu02Info_t info)
    cusparseStatus_t cusparseCreateBsrilu02Info(bsrilu02Info_t *info)
    cusparseStatus_t cusparseDestroyBsrilu02Info(bsrilu02Info_t info)
    cusparseStatus_t cusparseCreateCsric02Info(csric02Info_t *info)
    cusparseStatus_t cusparseDestroyCsric02Info(csric02Info_t info)
    cusparseStatus_t cusparseCreateBsric02Info(bsric02Info_t *info)
    cusparseStatus_t cusparseDestroyBsric02Info(bsric02Info_t info)
    cusparseStatus_t cusparseScsrilu02_numericBoost(
        cusparseHandle_t handle, csrilu02Info_t info, int enable_boost,
        double *tol, float *boost_val)
    cusparseStatus_t cusparseDcsrilu02_numericBoost(
        cusparseHandle_t handle, csrilu02Info_t info, int enable_boost,
        double *tol, double *boost_val)
    cusparseStatus_t cusparseCcsrilu02_numericBoost(
        cusparseHandle_t handle, csrilu02Info_t info, int enable_boost,
        double *tol, cuComplex *boost_val)
    cusparseStatus_t cusparseZcsrilu02_numericBoost(
        cusparseHandle_t handle, csrilu02Info_t info, int enable_boost,
        double *tol, cuDoubleComplex *boost_val)
    cusparseStatus_t cusparseXcsrilu02_zeroPivot(
        cusparseHandle_t handle, csrilu02Info_t info, int *position)
    cusparseStatus_t cusparseScsrilu02_bufferSize(
        cusparseHandle_t handle, int m, int nnz,
        const cusparseMatDescr_t descrA, float *csrSortedValA,
        const int *csrSortedRowPtrA, const int *csrSortedColIndA,
        csrilu02Info_t info, int *pBufferSizeInBytes)
    cusparseStatus_t cusparseDcsrilu02_bufferSize(
        cusparseHandle_t handle, int m, int nnz,
        const cusparseMatDescr_t descrA, double *csrSortedValA,
        const int *csrSortedRowPtrA, const int *csrSortedColIndA,
        csrilu02Info_t info, int *pBufferSizeInBytes)
    cusparseStatus_t cusparseCcsrilu02_bufferSize(
        cusparseHandle_t handle, int m, int nnz,
        const cusparseMatDescr_t descrA, cuComplex *csrSortedValA,
        const int *csrSortedRowPtrA, const int *csrSortedColIndA,
        csrilu02Info_t info, int *pBufferSizeInBytes)
    cusparseStatus_t cusparseZcsrilu02_bufferSize(
        cusparseHandle_t handle, int m, int nnz,
        const cusparseMatDescr_t descrA, cuDoubleComplex *csrSortedValA,
        const int *csrSortedRowPtrA, const int *csrSortedColIndA,
        csrilu02Info_t info, int *pBufferSizeInBytes)
    cusparseStatus_t cusparseScsrilu02_analysis(
        cusparseHandle_t handle, int m, int nnz,
        const cusparseMatDescr_t descrA, const float *csrSortedValA,
        const int *csrSortedRowPtrA, const int *csrSortedColIndA,
        csrilu02Info_t info, cusparseSolvePolicy_t policy, void *pBuffer)
    cusparseStatus_t cusparseDcsrilu02_analysis(
        cusparseHandle_t handle, int m, int nnz,
        const cusparseMatDescr_t descrA, const double *csrSortedValA,
        const int *csrSortedRowPtrA, const int *csrSortedColIndA,
        csrilu02Info_t info, cusparseSolvePolicy_t policy, void *pBuffer)
    cusparseStatus_t cusparseCcsrilu02_analysis(
        cusparseHandle_t handle, int m, int nnz,
        const cusparseMatDescr_t descrA, const cuComplex *csrSortedValA,
        const int *csrSortedRowPtrA, const int *csrSortedColIndA,
        csrilu02Info_t info, cusparseSolvePolicy_t policy, void *pBuffer)
    cusparseStatus_t cusparseZcsrilu02_analysis(
        cusparseHandle_t handle, int m, int nnz,
        const cusparseMatDescr_t descrA, const cuDoubleComplex *csrSortedValA,
        const int *csrSortedRowPtrA, const int *csrSortedColIndA,
        csrilu02Info_t info, cusparseSolvePolicy_t policy, void *pBuffer)
    cusparseStatus_t cusparseScsrilu02(
        cusparseHandle_t handle, int m, int nnz,
        const cusparseMatDescr_t descrA, float *csrSortedValA_valM,
        const int *csrSortedRowPtrA, const int *csrSortedColIndA,
        csrilu02Info_t info, cusparseSolvePolicy_t policy, void *pBuffer)
    cusparseStatus_t cusparseDcsrilu02(
        cusparseHandle_t handle, int m, int nnz,
        const cusparseMatDescr_t descrA, double *csrSortedValA_valM,
        const int *csrSortedRowPtrA, const int *csrSortedColIndA,
        csrilu02Info_t info, cusparseSolvePolicy_t policy, void *pBuffer)
    cusparseStatus_t cusparseCcsrilu02(
        cusparseHandle_t handle, int m, int nnz,
        const cusparseMatDescr_t descrA, cuComplex *csrSortedValA_valM,
        const int *csrSortedRowPtrA, const int *csrSortedColIndA,
        csrilu02Info_t info, cusparseSolvePolicy_t policy, void *pBuffer)
    cusparseStatus_t cusparseZcsrilu02(
        cusparseHandle_t handle, int m, int nnz,
        const cusparseMatDescr_t descrA, cuDoubleComplex *csrSortedValA_valM,
        const int *csrSortedRowPtrA, const int *csrSortedColIndA,
        csrilu02Info_t info, cusparseSolvePolicy_t policy, void *pBuffer)
    cusparseStatus_t cusparseSbsrilu02_numericBoost(
        cusparseHandle_t handle, bsrilu02Info_t info, int enable_boost,
        double *tol, float *boost_val)
    cusparseStatus_t cusparseDbsrilu02_numericBoost(
        cusparseHandle_t handle, bsrilu02Info_t info, int enable_boost,
        double *tol, double *boost_val)
    cusparseStatus_t cusparseCbsrilu02_numericBoost(
        cusparseHandle_t handle, bsrilu02Info_t info, int enable_boost,
        double *tol, cuComplex *boost_val)
    cusparseStatus_t cusparseZbsrilu02_numericBoost(
        cusparseHandle_t handle, bsrilu02Info_t info, int enable_boost,
        double *tol, cuDoubleComplex *boost_val)
    cusparseStatus_t cusparseXbsrilu02_zeroPivot(
        cusparseHandle_t handle, bsrilu02Info_t info, int *position)
    cusparseStatus_t cusparseSbsrilu02_bufferSize(
        cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
        const cusparseMatDescr_t descrA, float *bsrSortedVal,
        const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
        bsrilu02Info_t info, int *pBufferSizeInBytes)
    cusparseStatus_t cusparseDbsrilu02_bufferSize(
        cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
        const cusparseMatDescr_t descrA, double *bsrSortedVal,
        const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
        bsrilu02Info_t info, int *pBufferSizeInBytes)
    cusparseStatus_t cusparseCbsrilu02_bufferSize(
        cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
        const cusparseMatDescr_t descrA, cuComplex *bsrSortedVal,
        const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
        bsrilu02Info_t info, int *pBufferSizeInBytes)
    cusparseStatus_t cusparseZbsrilu02_bufferSize(
        cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
        const cusparseMatDescr_t descrA, cuDoubleComplex *bsrSortedVal,
        const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
        bsrilu02Info_t info, int *pBufferSizeInBytes)
    cusparseStatus_t cusparseSbsrilu02_analysis(
        cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
        const cusparseMatDescr_t descrA, float *bsrSortedVal,
        const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
        bsrilu02Info_t info, cusparseSolvePolicy_t policy, void *pBuffer)
    cusparseStatus_t cusparseDbsrilu02_analysis(
        cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
        const cusparseMatDescr_t descrA, double *bsrSortedVal,
        const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
        bsrilu02Info_t info, cusparseSolvePolicy_t policy, void *pBuffer)
    cusparseStatus_t cusparseCbsrilu02_analysis(
        cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
        const cusparseMatDescr_t descrA, cuComplex *bsrSortedVal,
        const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
        bsrilu02Info_t info, cusparseSolvePolicy_t policy, void *pBuffer)
    cusparseStatus_t cusparseZbsrilu02_analysis(
        cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
        const cusparseMatDescr_t descrA, cuDoubleComplex *bsrSortedVal,
        const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
        bsrilu02Info_t info, cusparseSolvePolicy_t policy, void *pBuffer)
    cusparseStatus_t cusparseSbsrilu02(
        cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
        const cusparseMatDescr_t descrA, float *bsrSortedVal,
        const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
        bsrilu02Info_t info, cusparseSolvePolicy_t policy, void *pBuffer)
    cusparseStatus_t cusparseDbsrilu02(
        cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
        const cusparseMatDescr_t descrA, double *bsrSortedVal,
        const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
        bsrilu02Info_t info, cusparseSolvePolicy_t policy, void *pBuffer)
    cusparseStatus_t cusparseCbsrilu02(
        cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
        const cusparseMatDescr_t descrA, cuComplex *bsrSortedVal,
        const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
        bsrilu02Info_t info, cusparseSolvePolicy_t policy, void *pBuffer)
    cusparseStatus_t cusparseZbsrilu02(
        cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
        const cusparseMatDescr_t descrA, cuDoubleComplex *bsrSortedVal,
        const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
        bsrilu02Info_t info, cusparseSolvePolicy_t policy, void *pBuffer)
    cusparseStatus_t cusparseXcsric02_zeroPivot(
        cusparseHandle_t handle, csric02Info_t info, int *position)
    cusparseStatus_t cusparseScsric02_bufferSize(
        cusparseHandle_t handle, int m, int nnz,
        const cusparseMatDescr_t descrA, float *csrSortedValA,
        const int *csrSortedRowPtrA, const int *csrSortedColIndA,
        csric02Info_t info, int *pBufferSizeInBytes)
    cusparseStatus_t cusparseDcsric02_bufferSize(
        cusparseHandle_t handle, int m, int nnz,
        const cusparseMatDescr_t descrA, double *csrSortedValA,
        const int *csrSortedRowPtrA, const int *csrSortedColIndA,
        csric02Info_t info, int *pBufferSizeInBytes)
    cusparseStatus_t cusparseCcsric02_bufferSize(
        cusparseHandle_t handle, int m, int nnz,
        const cusparseMatDescr_t descrA, cuComplex *csrSortedValA,
        const int *csrSortedRowPtrA, const int *csrSortedColIndA,
        csric02Info_t info, int *pBufferSizeInBytes)
    cusparseStatus_t cusparseZcsric02_bufferSize(
        cusparseHandle_t handle, int m, int nnz,
        const cusparseMatDescr_t descrA, cuDoubleComplex *csrSortedValA,
        const int *csrSortedRowPtrA, const int *csrSortedColIndA,
        csric02Info_t info, int *pBufferSizeInBytes)
    cusparseStatus_t cusparseScsric02_analysis(
        cusparseHandle_t handle, int m, int nnz,
        const cusparseMatDescr_t descrA, const float *csrSortedValA,
        const int *csrSortedRowPtrA, const int *csrSortedColIndA,
        csric02Info_t info, cusparseSolvePolicy_t policy, void *pBuffer)
    cusparseStatus_t cusparseDcsric02_analysis(
        cusparseHandle_t handle, int m, int nnz,
        const cusparseMatDescr_t descrA, const double *csrSortedValA,
        const int *csrSortedRowPtrA, const int *csrSortedColIndA,
        csric02Info_t info, cusparseSolvePolicy_t policy, void *pBuffer)
    cusparseStatus_t cusparseCcsric02_analysis(
        cusparseHandle_t handle, int m, int nnz,
        const cusparseMatDescr_t descrA, const cuComplex *csrSortedValA,
        const int *csrSortedRowPtrA, const int *csrSortedColIndA,
        csric02Info_t info, cusparseSolvePolicy_t policy, void *pBuffer)
    cusparseStatus_t cusparseZcsric02_analysis(
        cusparseHandle_t handle, int m, int nnz,
        const cusparseMatDescr_t descrA, const cuDoubleComplex *csrSortedValA,
        const int *csrSortedRowPtrA, const int *csrSortedColIndA,
        csric02Info_t info, cusparseSolvePolicy_t policy, void *pBuffer)
    cusparseStatus_t cusparseScsric02(
        cusparseHandle_t handle, int m, int nnz,
        const cusparseMatDescr_t descrA, float *csrSortedValA_valM,
        const int *csrSortedRowPtrA, const int *csrSortedColIndA,
        csric02Info_t info, cusparseSolvePolicy_t policy, void *pBuffer)
    cusparseStatus_t cusparseDcsric02(
        cusparseHandle_t handle, int m, int nnz,
        const cusparseMatDescr_t descrA, double *csrSortedValA_valM,
        const int *csrSortedRowPtrA, const int *csrSortedColIndA,
        csric02Info_t info, cusparseSolvePolicy_t policy, void *pBuffer)
    cusparseStatus_t cusparseCcsric02(
        cusparseHandle_t handle, int m, int nnz,
        const cusparseMatDescr_t descrA, cuComplex *csrSortedValA_valM,
        const int *csrSortedRowPtrA, const int *csrSortedColIndA,
        csric02Info_t info, cusparseSolvePolicy_t policy, void *pBuffer)
    cusparseStatus_t cusparseZcsric02(
        cusparseHandle_t handle, int m, int nnz,
        const cusparseMatDescr_t descrA, cuDoubleComplex *csrSortedValA_valM,
        const int *csrSortedRowPtrA, const int *csrSortedColIndA,
        csric02Info_t info, cusparseSolvePolicy_t policy, void *pBuffer)
    cusparseStatus_t cusparseXbsric02_zeroPivot(
        cusparseHandle_t handle, bsric02Info_t info, int *position)
    cusparseStatus_t cusparseSbsric02_bufferSize(
        cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
        const cusparseMatDescr_t descrA, float *bsrSortedVal,
        const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
        bsric02Info_t info, int *pBufferSizeInBytes)
    cusparseStatus_t cusparseDbsric02_bufferSize(
        cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
        const cusparseMatDescr_t descrA, double *bsrSortedVal,
        const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
        bsric02Info_t info, int *pBufferSizeInBytes)
    cusparseStatus_t cusparseCbsric02_bufferSize(
        cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
        const cusparseMatDescr_t descrA, cuComplex *bsrSortedVal,
        const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
        bsric02Info_t info, int *pBufferSizeInBytes)
    cusparseStatus_t cusparseZbsric02_bufferSize(
        cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
        const cusparseMatDescr_t descrA, cuDoubleComplex *bsrSortedVal,
        const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
        bsric02Info_t info, int *pBufferSizeInBytes)
    cusparseStatus_t cusparseSbsric02_analysis(
        cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
        const cusparseMatDescr_t descrA, const float *bsrSortedVal,
        const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
        bsric02Info_t info, cusparseSolvePolicy_t policy, void *pInputBuffer)
    cusparseStatus_t cusparseDbsric02_analysis(
        cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
        const cusparseMatDescr_t descrA, const double *bsrSortedVal,
        const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
        bsric02Info_t info, cusparseSolvePolicy_t policy, void *pInputBuffer)
    cusparseStatus_t cusparseCbsric02_analysis(
        cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
        const cusparseMatDescr_t descrA, const cuComplex *bsrSortedVal,
        const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
        bsric02Info_t info, cusparseSolvePolicy_t policy, void *pInputBuffer)
    cusparseStatus_t cusparseZbsric02_analysis(
        cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
        const cusparseMatDescr_t descrA, const cuDoubleComplex *bsrSortedVal,
        const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
        bsric02Info_t info, cusparseSolvePolicy_t policy, void *pInputBuffer)
    cusparseStatus_t cusparseSbsric02(
        cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
        const cusparseMatDescr_t descrA, float *bsrSortedVal,
        const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
        bsric02Info_t info, cusparseSolvePolicy_t policy, void *pBuffer)
    cusparseStatus_t cusparseDbsric02(
        cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
        const cusparseMatDescr_t descrA, double *bsrSortedVal,
        const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
        bsric02Info_t info, cusparseSolvePolicy_t policy, void *pBuffer)
    cusparseStatus_t cusparseCbsric02(
        cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
        const cusparseMatDescr_t descrA, cuComplex *bsrSortedVal,
        const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
        bsric02Info_t info, cusparseSolvePolicy_t policy, void *pBuffer)
    cusparseStatus_t cusparseZbsric02(
        cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
        const cusparseMatDescr_t descrA, cuDoubleComplex *bsrSortedVal,
        const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
        bsric02Info_t info, cusparseSolvePolicy_t policy, void *pBuffer)
    cusparseStatus_t cusparseSgtsv2_bufferSizeExt(
        cusparseHandle_t handle, int m, int n, const float *dl, const float *d,
        const float *du, const float *B, int ldb, size_t *bufferSizeInBytes)
    cusparseStatus_t cusparseDgtsv2_bufferSizeExt(
        cusparseHandle_t handle, int m, int n, const double *dl,
        const double *d, const double *du, const double *B, int ldb,
        size_t *bufferSizeInBytes)
    cusparseStatus_t cusparseCgtsv2_bufferSizeExt(
        cusparseHandle_t handle, int m, int n, const cuComplex *dl,
        const cuComplex *d, const cuComplex *du, const cuComplex *B, int ldb,
        size_t *bufferSizeInBytes)
    cusparseStatus_t cusparseZgtsv2_bufferSizeExt(
        cusparseHandle_t handle, int m, int n, const cuDoubleComplex *dl,
        const cuDoubleComplex *d, const cuDoubleComplex *du,
        const cuDoubleComplex *B, int ldb, size_t *bufferSizeInBytes)
    cusparseStatus_t cusparseSgtsv2(
        cusparseHandle_t handle, int m, int n, const float *dl, const float *d,
        const float *du, float *B, int ldb, void *pBuffer)
    cusparseStatus_t cusparseDgtsv2(
        cusparseHandle_t handle, int m, int n, const double *dl,
        const double *d, const double *du, double *B, int ldb, void *pBuffer)
    cusparseStatus_t cusparseCgtsv2(cusparseHandle_t handle, int m, int n,
                                    const cuComplex *dl, const cuComplex *d,
                                    const cuComplex *du, cuComplex *B, int ldb,
                                    void *pBuffer)
    cusparseStatus_t cusparseZgtsv2(
        cusparseHandle_t handle, int m, int n, const cuDoubleComplex *dl,
        const cuDoubleComplex *d, const cuDoubleComplex *du,
        cuDoubleComplex *B, int ldb, void *pBuffer)
    cusparseStatus_t cusparseSgtsv2_nopivot_bufferSizeExt(
        cusparseHandle_t handle, int m, int n, const float *dl, const float *d,
        const float *du, const float *B, int ldb, size_t *bufferSizeInBytes)
    cusparseStatus_t cusparseDgtsv2_nopivot_bufferSizeExt(
        cusparseHandle_t handle, int m, int n, const double *dl,
        const double *d, const double *du, const double *B, int ldb,
        size_t *bufferSizeInBytes)
    cusparseStatus_t cusparseCgtsv2_nopivot_bufferSizeExt(
        cusparseHandle_t handle, int m, int n, const cuComplex *dl,
        const cuComplex *d, const cuComplex *du, const cuComplex *B, int ldb,
        size_t *bufferSizeInBytes)
    cusparseStatus_t cusparseZgtsv2_nopivot_bufferSizeExt(
        cusparseHandle_t handle, int m, int n, const cuDoubleComplex *dl,
        const cuDoubleComplex *d, const cuDoubleComplex *du,
        const cuDoubleComplex *B, int ldb, size_t *bufferSizeInBytes)
    cusparseStatus_t cusparseSgtsv2_nopivot(
        cusparseHandle_t handle, int m, int n, const float *dl, const float *d,
        const float *du, float *B, int ldb, void *pBuffer)
    cusparseStatus_t cusparseDgtsv2_nopivot(
        cusparseHandle_t handle, int m, int n, const double *dl,
        const double *d, const double *du, double *B, int ldb, void *pBuffer)
    cusparseStatus_t cusparseCgtsv2_nopivot(
        cusparseHandle_t handle, int m, int n, const cuComplex *dl,
        const cuComplex *d, const cuComplex *du, cuComplex *B, int ldb,
        void *pBuffer)
    cusparseStatus_t cusparseZgtsv2_nopivot(
        cusparseHandle_t handle, int m, int n, const cuDoubleComplex *dl,
        const cuDoubleComplex *d, const cuDoubleComplex *du,
        cuDoubleComplex *B, int ldb, void *pBuffer)
    cusparseStatus_t cusparseSgtsv2StridedBatch_bufferSizeExt(
        cusparseHandle_t handle, int m, const float *dl, const float *d,
        const float *du, const float *x, int batchCount, int batchStride,
        size_t *bufferSizeInBytes)
    cusparseStatus_t cusparseDgtsv2StridedBatch_bufferSizeExt(
        cusparseHandle_t handle, int m, const double *dl, const double *d,
        const double *du, const double *x, int batchCount, int batchStride,
        size_t *bufferSizeInBytes)
    cusparseStatus_t cusparseCgtsv2StridedBatch_bufferSizeExt(
        cusparseHandle_t handle, int m, const cuComplex *dl,
        const cuComplex *d, const cuComplex *du, const cuComplex *x,
        int batchCount, int batchStride, size_t *bufferSizeInBytes)
    cusparseStatus_t cusparseZgtsv2StridedBatch_bufferSizeExt(
        cusparseHandle_t handle, int m, const cuDoubleComplex *dl,
        const cuDoubleComplex *d, const cuDoubleComplex *du,
        const cuDoubleComplex *x, int batchCount, int batchStride,
        size_t *bufferSizeInBytes)
    cusparseStatus_t cusparseSgtsv2StridedBatch(
        cusparseHandle_t handle, int m, const float *dl, const float *d,
        const float *du, float *x, int batchCount, int batchStride,
        void *pBuffer)
    cusparseStatus_t cusparseDgtsv2StridedBatch(
        cusparseHandle_t handle, int m, const double *dl, const double *d,
        const double *du, double *x, int batchCount, int batchStride,
        void *pBuffer)
    cusparseStatus_t cusparseCgtsv2StridedBatch(
        cusparseHandle_t handle, int m, const cuComplex *dl,
        const cuComplex *d, const cuComplex *du, cuComplex *x, int batchCount,
        int batchStride, void *pBuffer)
    cusparseStatus_t cusparseZgtsv2StridedBatch(
        cusparseHandle_t handle, int m, const cuDoubleComplex *dl,
        const cuDoubleComplex *d, const cuDoubleComplex *du,
        cuDoubleComplex *x, int batchCount, int batchStride, void *pBuffer)
    cusparseStatus_t cusparseSgtsvInterleavedBatch_bufferSizeExt(
        cusparseHandle_t handle, int algo, int m, const float *dl,
        const float *d, const float *du, const float *x, int batchCount,
        size_t *pBufferSizeInBytes)
    cusparseStatus_t cusparseDgtsvInterleavedBatch_bufferSizeExt(
        cusparseHandle_t handle, int algo, int m, const double *dl,
        const double *d, const double *du, const double *x, int batchCount,
        size_t *pBufferSizeInBytes)
    cusparseStatus_t cusparseCgtsvInterleavedBatch_bufferSizeExt(
        cusparseHandle_t handle, int algo, int m, const cuComplex *dl,
        const cuComplex *d, const cuComplex *du, const cuComplex *x,
        int batchCount, size_t *pBufferSizeInBytes)
    cusparseStatus_t cusparseZgtsvInterleavedBatch_bufferSizeExt(
        cusparseHandle_t handle, int algo, int m, const cuDoubleComplex *dl,
        const cuDoubleComplex *d, const cuDoubleComplex *du,
        const cuDoubleComplex *x, int batchCount, size_t *pBufferSizeInBytes)
    cusparseStatus_t cusparseSgtsvInterleavedBatch(
        cusparseHandle_t handle, int algo, int m, float *dl, float *d,
        float *du, float *x, int batchCount, void *pBuffer)
    cusparseStatus_t cusparseDgtsvInterleavedBatch(
        cusparseHandle_t handle, int algo, int m, double *dl, double *d,
        double *du, double *x, int batchCount, void *pBuffer)
    cusparseStatus_t cusparseCgtsvInterleavedBatch(
        cusparseHandle_t handle, int algo, int m, cuComplex *dl, cuComplex *d,
        cuComplex *du, cuComplex *x, int batchCount, void *pBuffer)
    cusparseStatus_t cusparseZgtsvInterleavedBatch(
        cusparseHandle_t handle, int algo, int m, cuDoubleComplex *dl,
        cuDoubleComplex *d, cuDoubleComplex *du, cuDoubleComplex *x,
        int batchCount, void *pBuffer)
    cusparseStatus_t cusparseSgpsvInterleavedBatch_bufferSizeExt(
        cusparseHandle_t handle, int algo, int m, const float *ds,
        const float *dl, const float *d, const float *du, const float *dw,
        const float *x, int batchCount, size_t *pBufferSizeInBytes)
    cusparseStatus_t cusparseDgpsvInterleavedBatch_bufferSizeExt(
        cusparseHandle_t handle, int algo, int m, const double *ds,
        const double *dl, const double *d, const double *du, const double *dw,
        const double *x, int batchCount, size_t *pBufferSizeInBytes)
    cusparseStatus_t cusparseCgpsvInterleavedBatch_bufferSizeExt(
        cusparseHandle_t handle, int algo, int m, const cuComplex *ds,
        const cuComplex *dl, const cuComplex *d, const cuComplex *du,
        const cuComplex *dw, const cuComplex *x, int batchCount,
        size_t *pBufferSizeInBytes)
    cusparseStatus_t cusparseZgpsvInterleavedBatch_bufferSizeExt(
        cusparseHandle_t handle, int algo, int m, const cuDoubleComplex *ds,
        const cuDoubleComplex *dl, const cuDoubleComplex *d,
        const cuDoubleComplex *du, const cuDoubleComplex *dw,
        const cuDoubleComplex *x, int batchCount, size_t *pBufferSizeInBytes)
    cusparseStatus_t cusparseSgpsvInterleavedBatch(
        cusparseHandle_t handle, int algo, int m, float *ds, float *dl,
        float *d, float *du, float *dw, float *x, int batchCount,
        void *pBuffer)
    cusparseStatus_t cusparseDgpsvInterleavedBatch(
        cusparseHandle_t handle, int algo, int m, double *ds, double *dl,
        double *d, double *du, double *dw, double *x, int batchCount,
        void *pBuffer)
    cusparseStatus_t cusparseCgpsvInterleavedBatch(
        cusparseHandle_t handle, int algo, int m, cuComplex *ds, cuComplex *dl,
        cuComplex *d, cuComplex *du, cuComplex *dw, cuComplex *x,
        int batchCount, void *pBuffer)
    cusparseStatus_t cusparseZgpsvInterleavedBatch(
        cusparseHandle_t handle, int algo, int m, cuDoubleComplex *ds,
        cuDoubleComplex *dl, cuDoubleComplex *d, cuDoubleComplex *du,
        cuDoubleComplex *dw, cuDoubleComplex *x, int batchCount, void *pBuffer)


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
    10: 'CUSPARSE_STATUS_NOT_SUPPORTED',
}


class CuSparseError(RuntimeError):

    def __init__(self, int status):
        self.status = status
        super(CuSparseError, self).__init__('%s' % (STATUS[status]))

    def __reduce__(self):
        return (type(self), (self.status,))


@cython.profile(False)
cpdef inline check_status(int status):
    if status != 0:
        raise CuSparseError(status)


@cython.profile(False)
cdef inline cuComplex complex_to_cuda(complex value):
    cdef cuComplex value_cuda
    value_cuda.x = value.real
    value_cuda.y = value.imag
    return value_cuda


@cython.profile(False)
cdef inline cuDoubleComplex double_complex_to_cuda(double complex value):
    cdef cuDoubleComplex value_cuda
    value_cuda.x = value.real
    value_cuda.y = value.imag
    return value_cuda


########################################
# cuSPARSE Helper Function

cpdef size_t create() except? 0:
    cdef Handle handle
    status = cusparseCreate(& handle)
    check_status(status)
    return <size_t >handle


cpdef size_t createMatDescr():
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
# Stream

cpdef setStream(size_t handle, size_t stream):
    status = cusparseSetStream(<Handle>handle, <driver.Stream>stream)
    check_status(status)


cpdef size_t getStream(size_t handle) except? 0:
    cdef driver.Stream stream
    status = cusparseGetStream(<Handle>handle, &stream)
    check_status(status)
    return <size_t>stream


########################################
# cuSPARSE Level1 Function

cpdef sgthr(size_t handle, int nnz, size_t y, size_t xVal, size_t xInd,
            int idxBase):
    setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseSgthr(
        <Handle>handle, nnz, <const float *>y, <float *>xVal,
        <const int *>xInd, <IndexBase>idxBase)
    check_status(status)

cpdef dgthr(size_t handle, int nnz, size_t y, size_t xVal, size_t xInd,
            int idxBase):
    setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseDgthr(
        <Handle>handle, nnz, <const double *>y, <double *>xVal,
        <const int *>xInd, <IndexBase>idxBase)
    check_status(status)

cpdef cgthr(size_t handle, int nnz, size_t y, size_t xVal, size_t xInd,
            int idxBase):
    setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseCgthr(
        <Handle>handle, nnz, <const cuComplex *>y, <cuComplex *>xVal,
        <const int *>xInd, <IndexBase>idxBase)
    check_status(status)

cpdef zgthr(size_t handle, int nnz, size_t y, size_t xVal, size_t xInd,
            int idxBase):
    setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseZgthr(
        <Handle>handle, nnz, <const cuDoubleComplex *>y,
        <cuDoubleComplex *>xVal, <const int *>xInd, <IndexBase>idxBase)
    check_status(status)

########################################
# cuSPARSE Level2 Function

cpdef scsrmv(
        size_t handle, int transA, int m, int n, int nnz,
        size_t alpha, size_t descrA, size_t csrSortedValA,
        size_t csrSortedRowPtrA, size_t csrSortedColIndA,
        size_t x, size_t beta, size_t y):
    setStream(handle, stream_module.get_current_stream_ptr())
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
    setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseDcsrmv(
        <Handle>handle, <Operation>transA, m, n, nnz,
        <const double *>alpha, <MatDescr>descrA, <const double *>csrSortedValA,
        <const int *>csrSortedRowPtrA, <const int *>csrSortedColIndA,
        <const double *>x, <const double *>beta, <double *>y)
    check_status(status)

cpdef ccsrmv(
        size_t handle, int transA, int m, int n, int nnz,
        size_t alpha, size_t descrA, size_t csrSortedValA,
        size_t csrSortedRowPtrA, size_t csrSortedColIndA,
        size_t x, size_t beta, size_t y):
    setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseCcsrmv(
        <Handle>handle, <Operation>transA, m, n, nnz,
        <const cuComplex *>alpha, <MatDescr>descrA,
        <const cuComplex *>csrSortedValA,
        <const int *>csrSortedRowPtrA, <const int *>csrSortedColIndA,
        <const cuComplex *>x, <const cuComplex *>beta, <cuComplex *>y)
    check_status(status)

cpdef zcsrmv(
        size_t handle, int transA, int m, int n, int nnz,
        size_t alpha, size_t descrA, size_t csrSortedValA,
        size_t csrSortedRowPtrA, size_t csrSortedColIndA,
        size_t x, size_t beta, size_t y):
    setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseZcsrmv(
        <Handle>handle, <Operation>transA, m, n, nnz,
        <const cuDoubleComplex *>alpha, <MatDescr>descrA,
        <const cuDoubleComplex *>csrSortedValA,
        <const int *>csrSortedRowPtrA, <const int *>csrSortedColIndA,
        <const cuDoubleComplex *>x, <const cuDoubleComplex *>beta,
        <cuDoubleComplex *>y)
    check_status(status)

cpdef size_t csrmvEx_bufferSize(
        size_t handle, int alg, int transA, int m, int n,
        int nnz, size_t alpha, int alphatype, size_t descrA,
        size_t csrValA, int csrValAtype, size_t csrRowPtrA,
        size_t csrColIndA, size_t x, int xtype, size_t beta,
        int betatype, size_t y, int ytype, int executiontype):
    cdef size_t bufferSizeInBytes
    setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseCsrmvEx_bufferSize(
        <Handle>handle, <AlgMode>alg, <Operation>transA, m,
        n, nnz, <const void *>alpha, <DataType>alphatype,
        <MatDescr>descrA, <const void *>csrValA, <DataType>csrValAtype,
        <const int *>csrRowPtrA, <const int *>csrColIndA,
        <const void *>x, <DataType>xtype, <const void *>beta,
        <DataType>betatype, <void *>y, <DataType>ytype,
        <DataType>executiontype, &bufferSizeInBytes)
    check_status(status)
    return bufferSizeInBytes

cpdef csrmvEx(
        size_t handle, int alg, int transA, int m, int n,
        int nnz, size_t alpha, int alphatype, size_t descrA,
        size_t csrValA, int csrValAtype, size_t csrRowPtrA,
        size_t csrColIndA, size_t x, int xtype, size_t beta,
        int betatype, size_t y, int ytype, int executiontype,
        size_t buffer):
    setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseCsrmvEx(
        <Handle>handle, <AlgMode>alg, <Operation>transA, m,
        n, nnz, <const void *>alpha, <DataType>alphatype,
        <MatDescr>descrA, <const void *>csrValA, <DataType>csrValAtype,
        <const int *>csrRowPtrA, <const int *>csrColIndA,
        <const void *>x, <DataType>xtype, <const void *>beta,
        <DataType>betatype, <void *>y, <DataType>ytype,
        <DataType>executiontype, <void *>buffer)
    check_status(status)

########################################
# cuSPARSE Level3 Function

cpdef scsrmm(
        size_t handle, int transA, int m, int n, int k, int nnz,
        size_t alpha, size_t descrA, size_t csrSortedValA,
        size_t csrSortedRowPtrA, size_t csrSortedColIndA,
        size_t B, int ldb, size_t beta, size_t C, int ldc):
    setStream(handle, stream_module.get_current_stream_ptr())
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
    setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseDcsrmm(
        <Handle>handle, <Operation>transA, m, n, k, nnz,
        <const double *>alpha, <MatDescr>descrA, <const double *>csrSortedValA,
        <const int *>csrSortedRowPtrA, <const int *>csrSortedColIndA,
        <const double *>B, ldb, <const double *>beta, <double *>C, ldc)
    check_status(status)

cpdef ccsrmm(
        size_t handle, int transA, int m, int n, int k, int nnz,
        size_t alpha, size_t descrA, size_t csrSortedValA,
        size_t csrSortedRowPtrA, size_t csrSortedColIndA,
        size_t B, int ldb, size_t beta, size_t C, int ldc):
    setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseCcsrmm(
        <Handle>handle, <Operation>transA, m, n, k, nnz,
        <const cuComplex *>alpha, <MatDescr>descrA,
        <const cuComplex *>csrSortedValA,
        <const int *>csrSortedRowPtrA, <const int *>csrSortedColIndA,
        <const cuComplex *>B, ldb, <const cuComplex *>beta,
        <cuComplex *>C, ldc)
    check_status(status)

cpdef zcsrmm(
        size_t handle, int transA, int m, int n, int k, int nnz,
        size_t alpha, size_t descrA, size_t csrSortedValA,
        size_t csrSortedRowPtrA, size_t csrSortedColIndA,
        size_t B, int ldb, size_t beta, size_t C, int ldc):
    setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseZcsrmm(
        <Handle>handle, <Operation>transA, m, n, k, nnz,
        <const cuDoubleComplex *>alpha, <MatDescr>descrA,
        <const cuDoubleComplex *>csrSortedValA,
        <const int *>csrSortedRowPtrA, <const int *>csrSortedColIndA,
        <const cuDoubleComplex *>B, ldb,
        <const cuDoubleComplex *>beta, <cuDoubleComplex *>C, ldc)
    check_status(status)

cpdef scsrmm2(
        size_t handle, int transA, int transB, int m, int n, int k, int nnz,
        size_t alpha, size_t descrA, size_t csrValA,
        size_t csrRowPtrA, size_t csrColIndA,
        size_t B, int ldb, size_t beta, size_t C, int ldc):
    setStream(handle, stream_module.get_current_stream_ptr())
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
    setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseDcsrmm2(
        <Handle>handle, <Operation>transA, <Operation>transB, m, n, k, nnz,
        <const double *>alpha, <MatDescr>descrA, <const double *>csrValA,
        <const int *>csrRowPtrA, <const int *>csrColIndA,
        <const double *>B, ldb, <const double *>beta, <double *>C, ldc)
    check_status(status)

cpdef ccsrmm2(
        size_t handle, int transA, int transB, int m, int n, int k, int nnz,
        size_t alpha, size_t descrA, size_t csrValA,
        size_t csrRowPtrA, size_t csrColIndA,
        size_t B, int ldb, size_t beta, size_t C, int ldc):
    setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseCcsrmm2(
        <Handle>handle, <Operation>transA, <Operation>transB, m, n, k, nnz,
        <const cuComplex *>alpha, <MatDescr>descrA, <const cuComplex *>csrValA,
        <const int *>csrRowPtrA, <const int *>csrColIndA,
        <const cuComplex *>B, ldb, <const cuComplex *>beta,
        <cuComplex *>C, ldc)
    check_status(status)

cpdef zcsrmm2(
        size_t handle, int transA, int transB, int m, int n, int k, int nnz,
        size_t alpha, size_t descrA, size_t csrValA,
        size_t csrRowPtrA, size_t csrColIndA,
        size_t B, int ldb, size_t beta, size_t C, int ldc):
    setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseZcsrmm2(
        <Handle>handle, <Operation>transA, <Operation>transB, m, n, k, nnz,
        <const cuDoubleComplex *>alpha, <MatDescr>descrA,
        <const cuDoubleComplex *>csrValA,
        <const int *>csrRowPtrA, <const int *>csrColIndA,
        <const cuDoubleComplex *>B, ldb,
        <const cuDoubleComplex *>beta, <cuDoubleComplex *>C, ldc)
    check_status(status)

########################################
# cuSPARSE Extra Function

cpdef xcsrgeamNnz(
        size_t handle, int m, int n, size_t descrA, int nnzA,
        size_t csrRowPtrA, size_t csrColIndA, size_t descrB,
        int nnzB, size_t csrRowPtrB, size_t csrColIndB,
        size_t descrC, size_t csrRowPtrC, size_t nnzTotalDevHostPtr):
    setStream(handle, stream_module.get_current_stream_ptr())
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
    setStream(handle, stream_module.get_current_stream_ptr())
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
    setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseDcsrgeam(
        <Handle>handle, m, n, <const double *>alpha,
        <const MatDescr>descrA, nnzA, <const double *>csrValA,
        <const int *>csrRowPtrA, <const int *>csrColIndA, <const double *>beta,
        <const MatDescr>descrB, nnzB, <const double *>csrValB,
        <const int *>csrRowPtrB, <const int *>csrColIndB,
        <const MatDescr>descrC, <double *>csrValC, <int *>csrRowPtrC,
        <int *>csrColIndC)
    check_status(status)

cpdef ccsrgeam(
        size_t handle, int m, int n, size_t alpha, size_t descrA,
        int nnzA, size_t csrValA, size_t csrRowPtrA,
        size_t csrColIndA, size_t beta, size_t descrB,
        int nnzB, size_t csrValB, size_t csrRowPtrB,
        size_t csrColIndB, size_t descrC, size_t csrValC,
        size_t csrRowPtrC, size_t csrColIndC):
    setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseCcsrgeam(
        <Handle>handle, m, n, <const cuComplex *>alpha,
        <const MatDescr>descrA, nnzA, <const cuComplex *>csrValA,
        <const int *>csrRowPtrA, <const int *>csrColIndA,
        <const cuComplex *>beta,
        <const MatDescr>descrB, nnzB, <const cuComplex *>csrValB,
        <const int *>csrRowPtrB, <const int *>csrColIndB,
        <const MatDescr>descrC, <cuComplex *>csrValC, <int *>csrRowPtrC,
        <int *>csrColIndC)
    check_status(status)

cpdef zcsrgeam(
        size_t handle, int m, int n, size_t alpha, size_t descrA,
        int nnzA, size_t csrValA, size_t csrRowPtrA,
        size_t csrColIndA, size_t beta, size_t descrB,
        int nnzB, size_t csrValB, size_t csrRowPtrB,
        size_t csrColIndB, size_t descrC, size_t csrValC,
        size_t csrRowPtrC, size_t csrColIndC):
    setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseZcsrgeam(
        <Handle>handle, m, n, <const cuDoubleComplex *>alpha,
        <const MatDescr>descrA, nnzA, <const cuDoubleComplex *>csrValA,
        <const int *>csrRowPtrA, <const int *>csrColIndA,
        <const cuDoubleComplex *>beta,
        <const MatDescr>descrB, nnzB, <const cuDoubleComplex *>csrValB,
        <const int *>csrRowPtrB, <const int *>csrColIndB,
        <const MatDescr>descrC, <cuDoubleComplex *>csrValC, <int *>csrRowPtrC,
        <int *>csrColIndC)
    check_status(status)

cpdef xcsrgemmNnz(
        size_t handle, int transA, int transB, int m, int n, int k,
        size_t descrA, int nnzA, size_t csrRowPtrA,
        size_t csrColIndA, size_t descrB, int nnzB,
        size_t csrRowPtrB, size_t csrColIndB,
        size_t descrC, size_t csrRowPtrC, size_t nnzTotalDevHostPtr):
    setStream(handle, stream_module.get_current_stream_ptr())
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
    setStream(handle, stream_module.get_current_stream_ptr())
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
    setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseDcsrgemm(
        <Handle>handle, <Operation>transA, <Operation>transB, m, n, k,
        <const MatDescr>descrA, nnzA, <const double *>csrValA,
        <const int *>csrRowPtrA, <const int *>csrColIndA,
        <const MatDescr>descrB, nnzB, <const double *>csrValB,
        <const int *>csrRowPtrB, <const int *>csrColIndB,
        <const MatDescr>descrC, <double *>csrValC,
        <const int *>csrRowPtrC, <int *>csrColIndC)
    check_status(status)

cpdef ccsrgemm(
        size_t handle, int transA, int transB, int m, int n, int k,
        size_t descrA, const int nnzA, size_t csrValA,
        size_t csrRowPtrA, size_t csrColIndA, size_t descrB,
        const int nnzB, size_t csrValB, size_t csrRowPtrB,
        size_t csrColIndB, size_t descrC, size_t csrValC,
        size_t csrRowPtrC, size_t csrColIndC):
    setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseCcsrgemm(
        <Handle>handle, <Operation>transA, <Operation>transB, m, n, k,
        <const MatDescr>descrA, nnzA, <const cuComplex *>csrValA,
        <const int *>csrRowPtrA, <const int *>csrColIndA,
        <const MatDescr>descrB, nnzB, <const cuComplex *>csrValB,
        <const int *>csrRowPtrB, <const int *>csrColIndB,
        <const MatDescr>descrC, <cuComplex *>csrValC,
        <const int *>csrRowPtrC, <int *>csrColIndC)
    check_status(status)

cpdef zcsrgemm(
        size_t handle, int transA, int transB, int m, int n, int k,
        size_t descrA, const int nnzA, size_t csrValA,
        size_t csrRowPtrA, size_t csrColIndA, size_t descrB,
        const int nnzB, size_t csrValB, size_t csrRowPtrB,
        size_t csrColIndB, size_t descrC, size_t csrValC,
        size_t csrRowPtrC, size_t csrColIndC):
    setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseZcsrgemm(
        <Handle>handle, <Operation>transA, <Operation>transB, m, n, k,
        <const MatDescr>descrA, nnzA, <const cuDoubleComplex *>csrValA,
        <const int *>csrRowPtrA, <const int *>csrColIndA,
        <const MatDescr>descrB, nnzB, <const cuDoubleComplex *>csrValB,
        <const int *>csrRowPtrB, <const int *>csrColIndB,
        <const MatDescr>descrC, <cuDoubleComplex *>csrValC,
        <const int *>csrRowPtrC, <int *>csrColIndC)
    check_status(status)

########################################
# cuSPARSE Format Convrsion

cpdef xcoo2csr(
        size_t handle, size_t cooRowInd, int nnz, int m, size_t csrRowPtr,
        int idxBase):
    setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseXcoo2csr(
        <Handle>handle, <const int *>cooRowInd, nnz, m, <int *>csrRowPtr,
        <IndexBase>idxBase)
    check_status(status)


cpdef scsc2dense(
        size_t handle, int m, int n, size_t descrA,
        size_t cscSortedValA, size_t cscSortedRowIndA,
        size_t cscSortedColPtrA, size_t A, int lda):
    setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseScsc2dense(
        <Handle>handle, m, n, <MatDescr>descrA,
        <const float *>cscSortedValA, <const int *>cscSortedRowIndA,
        <const int *>cscSortedColPtrA, <float *>A, lda)
    check_status(status)


cpdef dcsc2dense(
        size_t handle, int m, int n, size_t descrA,
        size_t cscSortedValA, size_t cscSortedRowIndA,
        size_t cscSortedColPtrA, size_t A, int lda):
    setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseDcsc2dense(
        <Handle>handle, m, n, <MatDescr>descrA,
        <const double *>cscSortedValA, <const int *>cscSortedRowIndA,
        <const int *>cscSortedColPtrA, <double *>A, lda)
    check_status(status)

cpdef ccsc2dense(
        size_t handle, int m, int n, size_t descrA,
        size_t cscSortedValA, size_t cscSortedRowIndA,
        size_t cscSortedColPtrA, size_t A, int lda):
    setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseCcsc2dense(
        <Handle>handle, m, n, <MatDescr>descrA,
        <const cuComplex *>cscSortedValA, <const int *>cscSortedRowIndA,
        <const int *>cscSortedColPtrA, <cuComplex *>A, lda)
    check_status(status)

cpdef zcsc2dense(
        size_t handle, int m, int n, size_t descrA,
        size_t cscSortedValA, size_t cscSortedRowIndA,
        size_t cscSortedColPtrA, size_t A, int lda):
    setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseZcsc2dense(
        <Handle>handle, m, n, <MatDescr>descrA,
        <const cuDoubleComplex *>cscSortedValA, <const int *>cscSortedRowIndA,
        <const int *>cscSortedColPtrA, <cuDoubleComplex *>A, lda)
    check_status(status)

cpdef xcsr2coo(
        size_t handle, size_t csrRowPtr, int nnz, int m, size_t cooRowInd,
        int idxBase):
    setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseXcsr2coo(
        <Handle>handle, <const int *>csrRowPtr, nnz, m, <int *>cooRowInd,
        <IndexBase>idxBase)
    check_status(status)


cpdef scsr2csc(
        size_t handle, int m, int n, int nnz, size_t csrVal,
        size_t csrRowPtr, size_t csrColInd, size_t cscVal,
        size_t cscRowInd, size_t cscColPtr, int copyValues, int idxBase):
    setStream(handle, stream_module.get_current_stream_ptr())
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
    setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseDcsr2csc(
        <Handle>handle, m, n, nnz, <const double *>csrVal,
        <const int *>csrRowPtr, <const int *>csrColInd, <double *>cscVal,
        <int *>cscRowInd, <int *>cscColPtr, <Action>copyValues,
        <IndexBase>idxBase)
    check_status(status)

cpdef ccsr2csc(
        size_t handle, int m, int n, int nnz, size_t csrVal,
        size_t csrRowPtr, size_t csrColInd, size_t cscVal,
        size_t cscRowInd, size_t cscColPtr, int copyValues, int idxBase):
    setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseCcsr2csc(
        <Handle>handle, m, n, nnz, <const cuComplex *>csrVal,
        <const int *>csrRowPtr, <const int *>csrColInd, <cuComplex *>cscVal,
        <int *>cscRowInd, <int *>cscColPtr, <Action>copyValues,
        <IndexBase>idxBase)
    check_status(status)

cpdef zcsr2csc(
        size_t handle, int m, int n, int nnz, size_t csrVal,
        size_t csrRowPtr, size_t csrColInd, size_t cscVal,
        size_t cscRowInd, size_t cscColPtr, int copyValues, int idxBase):
    setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseZcsr2csc(
        <Handle>handle, m, n, nnz, <const cuDoubleComplex *>csrVal,
        <const int *>csrRowPtr, <const int *>csrColInd,
        <cuDoubleComplex *>cscVal,
        <int *>cscRowInd, <int *>cscColPtr, <Action>copyValues,
        <IndexBase>idxBase)
    check_status(status)

cpdef scsr2dense(
        size_t handle, int m, int n, size_t descrA,
        size_t csrSortedValA, size_t csrSortedRowPtrA,
        size_t csrSortedColIndA, size_t A, int lda):
    setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseScsr2dense(
        <Handle>handle, m, n, <MatDescr>descrA,
        <const float *>csrSortedValA, <const int *>csrSortedRowPtrA,
        <const int *>csrSortedColIndA, <float *>A, lda)
    check_status(status)

cpdef dcsr2dense(
        size_t handle, int m, int n, size_t descrA,
        size_t csrSortedValA, size_t csrSortedRowPtrA,
        size_t csrSortedColIndA, size_t A, int lda):
    setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseDcsr2dense(
        <Handle>handle, m, n, <MatDescr>descrA,
        <const double *>csrSortedValA, <const int *>csrSortedRowPtrA,
        <const int *>csrSortedColIndA, <double *>A, lda)
    check_status(status)

cpdef ccsr2dense(
        size_t handle, int m, int n, size_t descrA,
        size_t csrSortedValA, size_t csrSortedRowPtrA,
        size_t csrSortedColIndA, size_t A, int lda):
    setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseCcsr2dense(
        <Handle>handle, m, n, <MatDescr>descrA,
        <const cuComplex *>csrSortedValA, <const int *>csrSortedRowPtrA,
        <const int *>csrSortedColIndA, <cuComplex *>A, lda)
    check_status(status)

cpdef zcsr2dense(
        size_t handle, int m, int n, size_t descrA,
        size_t csrSortedValA, size_t csrSortedRowPtrA,
        size_t csrSortedColIndA, size_t A, int lda):
    setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseZcsr2dense(
        <Handle>handle, m, n, <MatDescr>descrA,
        <const cuDoubleComplex *>csrSortedValA, <const int *>csrSortedRowPtrA,
        <const int *>csrSortedColIndA, <cuDoubleComplex *>A, lda)
    check_status(status)

cpdef snnz_compress(
        size_t handle, int m, size_t descr,
        size_t values, size_t rowPtr, size_t nnzPerRow,
        float tol):
    cdef int nnz_total
    setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseSnnz_compress(
        <Handle>handle, m, <const MatDescr>descr,
        <const float *>values, <const int *>rowPtr, <int *>nnzPerRow,
        &nnz_total, tol)
    check_status(status)
    return nnz_total

cpdef dnnz_compress(
        size_t handle, int m, size_t descr,
        size_t values, size_t rowPtr, size_t nnzPerRow,
        double tol):
    cdef int nnz_total
    setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseDnnz_compress(
        <Handle>handle, m, <const MatDescr>descr,
        <const double *>values, <const int *>rowPtr, <int *>nnzPerRow,
        &nnz_total, tol)
    check_status(status)
    return nnz_total

cpdef cnnz_compress(
        size_t handle, int m, size_t descr,
        size_t values, size_t rowPtr, size_t nnzPerRow,
        complex tol):
    cdef int nnz_total
    setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseCnnz_compress(
        <Handle>handle, m, <const MatDescr>descr,
        <const cuComplex *>values, <const int *>rowPtr, <int *>nnzPerRow,
        &nnz_total, complex_to_cuda(tol))
    check_status(status)
    return nnz_total

cpdef znnz_compress(
        size_t handle, int m, size_t descr,
        size_t values, size_t rowPtr, size_t nnzPerRow,
        double complex tol):
    cdef int nnz_total
    setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseZnnz_compress(
        <Handle>handle, m, <const MatDescr>descr,
        <const cuDoubleComplex *>values, <const int *>rowPtr, <int *>nnzPerRow,
        &nnz_total, double_complex_to_cuda(tol))
    check_status(status)
    return nnz_total

cpdef scsr2csr_compress(
        size_t handle, int m, int n, size_t descrA,
        size_t inVal, size_t inColInd, size_t inRowPtr,
        int inNnz, size_t nnzPerRow, size_t outVal, size_t outColInd,
        size_t outRowPtr, float tol):
    setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseScsr2csr_compress(
        <Handle>handle, m, n, <MatDescr>descrA,
        <const float *>inVal, <const int *>inColInd, <const int *>inRowPtr,
        inNnz, <int *>nnzPerRow, <float *>outVal, <int *>outColInd,
        <int *>outRowPtr, tol)
    check_status(status)


cpdef dcsr2csr_compress(
        size_t handle, int m, int n, size_t descrA,
        size_t inVal, size_t inColInd, size_t inRowPtr,
        int inNnz, size_t nnzPerRow, size_t outVal, size_t outColInd,
        size_t outRowPtr, float tol):
    setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseDcsr2csr_compress(
        <Handle>handle, m, n, <MatDescr>descrA,
        <const double *>inVal, <const int *>inColInd, <const int *>inRowPtr,
        inNnz, <int *>nnzPerRow, <double *>outVal, <int *>outColInd,
        <int *>outRowPtr, tol)
    check_status(status)

cpdef ccsr2csr_compress(
        size_t handle, int m, int n, size_t descrA,
        size_t inVal, size_t inColInd, size_t inRowPtr,
        int inNnz, size_t nnzPerRow, size_t outVal, size_t outColInd,
        size_t outRowPtr, complex tol):
    setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseCcsr2csr_compress(
        <Handle>handle, m, n, <MatDescr>descrA,
        <const cuComplex *>inVal, <const int *>inColInd, <const int *>inRowPtr,
        inNnz, <int *>nnzPerRow, <cuComplex *>outVal, <int *>outColInd,
        <int *>outRowPtr, complex_to_cuda(tol))
    check_status(status)

cpdef zcsr2csr_compress(
        size_t handle, int m, int n, size_t descrA,
        size_t inVal, size_t inColInd, size_t inRowPtr,
        int inNnz, size_t nnzPerRow, size_t outVal, size_t outColInd,
        size_t outRowPtr, double complex tol):
    setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseZcsr2csr_compress(
        <Handle>handle, m, n, <MatDescr>descrA,
        <const cuDoubleComplex *>inVal, <const int *>inColInd,
        <const int *>inRowPtr,
        inNnz, <int *>nnzPerRow, <cuDoubleComplex *>outVal, <int *>outColInd,
        <int *>outRowPtr, double_complex_to_cuda(tol))
    check_status(status)

cpdef sdense2csc(
        size_t handle, int m, int n, size_t descrA, size_t A,
        int lda, size_t nnzPerCol, size_t cscValA, size_t cscRowIndA,
        size_t cscColPtrA):
    setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseSdense2csc(
        <Handle>handle, m, n, <const MatDescr>descrA, <const float *>A,
        lda, <const int *>nnzPerCol, <float *>cscValA, <int *>cscRowIndA,
        <int *>cscColPtrA)
    check_status(status)


cpdef ddense2csc(
        size_t handle, int m, int n, size_t descrA, size_t A,
        int lda, size_t nnzPerCol, size_t cscValA, size_t cscRowIndA,
        size_t cscColPtrA):
    setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseDdense2csc(
        <Handle>handle, m, n, <const MatDescr>descrA, <const double *>A,
        lda, <const int *>nnzPerCol, <double *>cscValA, <int *>cscRowIndA,
        <int *>cscColPtrA)
    check_status(status)

cpdef cdense2csc(
        size_t handle, int m, int n, size_t descrA, size_t A,
        int lda, size_t nnzPerCol, size_t cscValA, size_t cscRowIndA,
        size_t cscColPtrA):
    setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseCdense2csc(
        <Handle>handle, m, n, <const MatDescr>descrA, <const cuComplex *>A,
        lda, <const int *>nnzPerCol, <cuComplex *>cscValA, <int *>cscRowIndA,
        <int *>cscColPtrA)
    check_status(status)

cpdef zdense2csc(
        size_t handle, int m, int n, size_t descrA, size_t A,
        int lda, size_t nnzPerCol, size_t cscValA, size_t cscRowIndA,
        size_t cscColPtrA):
    setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseZdense2csc(
        <Handle>handle, m, n,
        <const MatDescr>descrA, <const cuDoubleComplex *>A,
        lda, <const int *>nnzPerCol,
        <cuDoubleComplex *>cscValA, <int *>cscRowIndA,
        <int *>cscColPtrA)
    check_status(status)

cpdef sdense2csr(
        size_t handle, int m, int n, size_t descrA,
        size_t A, int lda, size_t nnzPerRow, size_t csrValA,
        size_t csrRowPtrA, size_t csrColIndA):
    setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseSdense2csr(
        <Handle>handle, m, n, <MatDescr>descrA,
        <const float *>A, lda, <const int *>nnzPerRow, <float *>csrValA,
        <int *>csrRowPtrA, <int *>csrColIndA)
    check_status(status)


cpdef ddense2csr(
        size_t handle, int m, int n, size_t descrA,
        size_t A, int lda, size_t nnzPerRow, size_t csrValA,
        size_t csrRowPtrA, size_t csrColIndA):
    setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseDdense2csr(
        <Handle>handle, m, n, <MatDescr>descrA,
        <const double *>A, lda, <const int *>nnzPerRow, <double *>csrValA,
        <int *>csrRowPtrA, <int *>csrColIndA)
    check_status(status)

cpdef cdense2csr(
        size_t handle, int m, int n, size_t descrA,
        size_t A, int lda, size_t nnzPerRow, size_t csrValA,
        size_t csrRowPtrA, size_t csrColIndA):
    setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseCdense2csr(
        <Handle>handle, m, n, <MatDescr>descrA,
        <const cuComplex *>A, lda, <const int *>nnzPerRow,
        <cuComplex *>csrValA, <int *>csrRowPtrA, <int *>csrColIndA)
    check_status(status)

cpdef zdense2csr(
        size_t handle, int m, int n, size_t descrA,
        size_t A, int lda, size_t nnzPerRow, size_t csrValA,
        size_t csrRowPtrA, size_t csrColIndA):
    setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseZdense2csr(
        <Handle>handle, m, n, <MatDescr>descrA,
        <const cuDoubleComplex *>A, lda, <const int *>nnzPerRow,
        <cuDoubleComplex *>csrValA, <int *>csrRowPtrA, <int *>csrColIndA)
    check_status(status)

cpdef snnz(
        size_t handle, int dirA, int m, int n, size_t descrA,
        size_t A, int lda, size_t nnzPerRowColumn, size_t nnzTotalDevHostPtr):
    setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseSnnz(
        <Handle>handle, <Direction>dirA, m, n, <const MatDescr>descrA,
        <const float *>A, lda, <int *>nnzPerRowColumn,
        <int *>nnzTotalDevHostPtr)
    check_status(status)


cpdef dnnz(
        size_t handle, int dirA, int m, int n, size_t descrA,
        size_t A, int lda, size_t nnzPerRowColumn, size_t nnzTotalDevHostPtr):
    setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseDnnz(
        <Handle>handle, <Direction>dirA, m, n, <const MatDescr>descrA,
        <const double *>A, lda, <int *>nnzPerRowColumn,
        <int *>nnzTotalDevHostPtr)
    check_status(status)

cpdef cnnz(
        size_t handle, int dirA, int m, int n, size_t descrA,
        size_t A, int lda, size_t nnzPerRowColumn, size_t nnzTotalDevHostPtr):
    setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseCnnz(
        <Handle>handle, <Direction>dirA, m, n, <const MatDescr>descrA,
        <const cuComplex *>A, lda, <int *>nnzPerRowColumn,
        <int *>nnzTotalDevHostPtr)
    check_status(status)

cpdef znnz(
        size_t handle, int dirA, int m, int n, size_t descrA,
        size_t A, int lda, size_t nnzPerRowColumn, size_t nnzTotalDevHostPtr):
    setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseZnnz(
        <Handle>handle, <Direction>dirA, m, n, <const MatDescr>descrA,
        <const cuDoubleComplex *>A, lda, <int *>nnzPerRowColumn,
        <int *>nnzTotalDevHostPtr)
    check_status(status)

cpdef createIdentityPermutation(
        size_t handle, int n, size_t p):
    setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseCreateIdentityPermutation(
        <Handle>handle, n, <int *>p)
    check_status(status)


cpdef size_t xcoosort_bufferSizeExt(
        size_t handle, int m, int n, int nnz, size_t cooRows,
        size_t cooCols):
    cdef size_t bufferSizeInBytes
    setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseXcoosort_bufferSizeExt(
        <Handle>handle, m, n, nnz, <const int *>cooRows,
        <const int *>cooCols, &bufferSizeInBytes)
    check_status(status)
    return bufferSizeInBytes


cpdef xcoosortByRow(
        size_t handle, int m, int n, int nnz, size_t cooRows, size_t cooCols,
        size_t P, size_t pBuffer):
    setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseXcoosortByRow(
        <Handle>handle, m, n, nnz, <int *>cooRows, <int *>cooCols,
        <int *>P, <void *>pBuffer)
    check_status(status)


cpdef size_t xcsrsort_bufferSizeExt(
        size_t handle, int m, int n, int nnz, size_t csrRowPtr,
        size_t csrColInd):
    cdef size_t bufferSizeInBytes
    setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseXcsrsort_bufferSizeExt(
        <Handle>handle, m, n, nnz, <const int *>csrRowPtr,
        <const int *>csrColInd, &bufferSizeInBytes)
    check_status(status)
    return bufferSizeInBytes


cpdef xcsrsort(
        size_t handle, int m, int n, int nnz, size_t descrA,
        size_t csrRowPtr, size_t csrColInd, size_t P, size_t pBuffer):
    setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseXcsrsort(
        <Handle>handle, m, n, nnz, <const MatDescr>descrA,
        <const int *>csrRowPtr, <int *>csrColInd, <int *>P, <void *>pBuffer)
    check_status(status)


cpdef size_t xcscsort_bufferSizeExt(
        size_t handle, int m, int n, int nnz, size_t cscColPtr,
        size_t cscRowInd):
    cdef size_t bufferSizeInBytes
    setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseXcscsort_bufferSizeExt(
        <Handle>handle, m, n, nnz, <const int *>cscColPtr,
        <const int *>cscRowInd, &bufferSizeInBytes)
    check_status(status)
    return bufferSizeInBytes


cpdef xcscsort(
        size_t handle, int m, int n, int nnz, size_t descrA,
        size_t cscColPtr, size_t cscRowInd, size_t P, size_t pBuffer):
    setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseXcscsort(
        <Handle>handle, m, n, nnz, <const MatDescr>descrA,
        <const int *>cscColPtr, <int *>cscRowInd, <int *>P, <void *>pBuffer)
    check_status(status)

########################################
# cuSPARSE PRECONDITIONERS

cpdef size_t createCsrilu02Info():
    cdef csrilu02Info_t info
    with nogil:
        status = cusparseCreateCsrilu02Info(&info)
    check_status(status)
    return <size_t>info

cpdef destroyCsrilu02Info(size_t info):
    with nogil:
        status = cusparseDestroyCsrilu02Info(<csrilu02Info_t>info)
    check_status(status)

cpdef size_t createBsrilu02Info():
    cdef bsrilu02Info_t info
    with nogil:
        status = cusparseCreateBsrilu02Info(&info)
    check_status(status)
    return <size_t>info

cpdef destroyBsrilu02Info(size_t info):
    with nogil:
        status = cusparseDestroyBsrilu02Info(<bsrilu02Info_t>info)
    check_status(status)

cpdef size_t createCsric02Info():
    cdef csric02Info_t info
    with nogil:
        status = cusparseCreateCsric02Info(&info)
    check_status(status)
    return <size_t>info

cpdef destroyCsric02Info(size_t info):
    with nogil:
        status = cusparseDestroyCsric02Info(<csric02Info_t>info)
    check_status(status)

cpdef size_t createBsric02Info():
    cdef bsric02Info_t info
    with nogil:
        status = cusparseCreateBsric02Info(&info)
    check_status(status)
    return <size_t>info

cpdef destroyBsric02Info(size_t info):
    with nogil:
        status = cusparseDestroyBsric02Info(<bsric02Info_t>info)
    check_status(status)

cpdef scsrilu02_numericBoost(size_t handle, size_t info, int enable_boost,
                             size_t tol, size_t boost_val):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseScsrilu02_numericBoost(
            <cusparseHandle_t>handle, <csrilu02Info_t>info, enable_boost,
            <double*>tol, <float*>boost_val)
    check_status(status)

cpdef dcsrilu02_numericBoost(size_t handle, size_t info, int enable_boost,
                             size_t tol, size_t boost_val):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseDcsrilu02_numericBoost(
            <cusparseHandle_t>handle, <csrilu02Info_t>info, enable_boost,
            <double*>tol, <double*>boost_val)
    check_status(status)

cpdef ccsrilu02_numericBoost(size_t handle, size_t info, int enable_boost,
                             size_t tol, size_t boost_val):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseCcsrilu02_numericBoost(
            <cusparseHandle_t>handle, <csrilu02Info_t>info, enable_boost,
            <double*>tol, <cuComplex*>boost_val)
    check_status(status)

cpdef zcsrilu02_numericBoost(size_t handle, size_t info, int enable_boost,
                             size_t tol, size_t boost_val):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseZcsrilu02_numericBoost(
            <cusparseHandle_t>handle, <csrilu02Info_t>info, enable_boost,
            <double*>tol, <cuDoubleComplex*>boost_val)
    check_status(status)

cpdef xcsrilu02_zeroPivot(size_t handle, size_t info, size_t position):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseXcsrilu02_zeroPivot(
            <cusparseHandle_t>handle, <csrilu02Info_t>info, <int*>position)
    check_status(status)

cpdef int scsrilu02_bufferSize(size_t handle, int m, int nnz, size_t descrA,
                               size_t csrSortedValA, size_t csrSortedRowPtrA,
                               size_t csrSortedColIndA, size_t info):
    cdef int pBufferSizeInBytes
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseScsrilu02_bufferSize(
            <cusparseHandle_t>handle, m, nnz, <const cusparseMatDescr_t>descrA,
            <float*>csrSortedValA, <const int*>csrSortedRowPtrA,
            <const int*>csrSortedColIndA, <csrilu02Info_t>info,
            &pBufferSizeInBytes)
    check_status(status)
    return <int>pBufferSizeInBytes

cpdef int dcsrilu02_bufferSize(size_t handle, int m, int nnz, size_t descrA,
                               size_t csrSortedValA, size_t csrSortedRowPtrA,
                               size_t csrSortedColIndA, size_t info):
    cdef int pBufferSizeInBytes
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseDcsrilu02_bufferSize(
            <cusparseHandle_t>handle, m, nnz, <const cusparseMatDescr_t>descrA,
            <double*>csrSortedValA, <const int*>csrSortedRowPtrA,
            <const int*>csrSortedColIndA, <csrilu02Info_t>info,
            &pBufferSizeInBytes)
    check_status(status)
    return <int>pBufferSizeInBytes

cpdef int ccsrilu02_bufferSize(size_t handle, int m, int nnz, size_t descrA,
                               size_t csrSortedValA, size_t csrSortedRowPtrA,
                               size_t csrSortedColIndA, size_t info):
    cdef int pBufferSizeInBytes
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseCcsrilu02_bufferSize(
            <cusparseHandle_t>handle, m, nnz, <const cusparseMatDescr_t>descrA,
            <cuComplex*>csrSortedValA, <const int*>csrSortedRowPtrA,
            <const int*>csrSortedColIndA, <csrilu02Info_t>info,
            &pBufferSizeInBytes)
    check_status(status)
    return <int>pBufferSizeInBytes

cpdef int zcsrilu02_bufferSize(size_t handle, int m, int nnz, size_t descrA,
                               size_t csrSortedValA, size_t csrSortedRowPtrA,
                               size_t csrSortedColIndA, size_t info):
    cdef int pBufferSizeInBytes
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseZcsrilu02_bufferSize(
            <cusparseHandle_t>handle, m, nnz, <const cusparseMatDescr_t>descrA,
            <cuDoubleComplex*>csrSortedValA, <const int*>csrSortedRowPtrA,
            <const int*>csrSortedColIndA, <csrilu02Info_t>info,
            &pBufferSizeInBytes)
    check_status(status)
    return <int>pBufferSizeInBytes

cpdef scsrilu02_analysis(size_t handle, int m, int nnz, size_t descrA,
                         size_t csrSortedValA, size_t csrSortedRowPtrA,
                         size_t csrSortedColIndA, size_t info, int policy,
                         size_t pBuffer):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseScsrilu02_analysis(
            <cusparseHandle_t>handle, m, nnz, <const cusparseMatDescr_t>descrA,
            <const float*>csrSortedValA, <const int*>csrSortedRowPtrA,
            <const int*>csrSortedColIndA, <csrilu02Info_t>info,
            <cusparseSolvePolicy_t>policy, <void*>pBuffer)
    check_status(status)

cpdef dcsrilu02_analysis(size_t handle, int m, int nnz, size_t descrA,
                         size_t csrSortedValA, size_t csrSortedRowPtrA,
                         size_t csrSortedColIndA, size_t info, int policy,
                         size_t pBuffer):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseDcsrilu02_analysis(
            <cusparseHandle_t>handle, m, nnz, <const cusparseMatDescr_t>descrA,
            <const double*>csrSortedValA, <const int*>csrSortedRowPtrA,
            <const int*>csrSortedColIndA, <csrilu02Info_t>info,
            <cusparseSolvePolicy_t>policy, <void*>pBuffer)
    check_status(status)

cpdef ccsrilu02_analysis(size_t handle, int m, int nnz, size_t descrA,
                         size_t csrSortedValA, size_t csrSortedRowPtrA,
                         size_t csrSortedColIndA, size_t info, int policy,
                         size_t pBuffer):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseCcsrilu02_analysis(
            <cusparseHandle_t>handle, m, nnz, <const cusparseMatDescr_t>descrA,
            <const cuComplex*>csrSortedValA, <const int*>csrSortedRowPtrA,
            <const int*>csrSortedColIndA, <csrilu02Info_t>info,
            <cusparseSolvePolicy_t>policy, <void*>pBuffer)
    check_status(status)

cpdef zcsrilu02_analysis(size_t handle, int m, int nnz, size_t descrA,
                         size_t csrSortedValA, size_t csrSortedRowPtrA,
                         size_t csrSortedColIndA, size_t info, int policy,
                         size_t pBuffer):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseZcsrilu02_analysis(
            <cusparseHandle_t>handle, m, nnz, <const cusparseMatDescr_t>descrA,
            <const cuDoubleComplex*>csrSortedValA,
            <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA,
            <csrilu02Info_t>info, <cusparseSolvePolicy_t>policy,
            <void*>pBuffer)
    check_status(status)

cpdef scsrilu02(size_t handle, int m, int nnz, size_t descrA,
                size_t csrSortedValA_valM, size_t csrSortedRowPtrA,
                size_t csrSortedColIndA, size_t info, int policy,
                size_t pBuffer):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseScsrilu02(
            <cusparseHandle_t>handle, m, nnz, <const cusparseMatDescr_t>descrA,
            <float*>csrSortedValA_valM, <const int*>csrSortedRowPtrA,
            <const int*>csrSortedColIndA, <csrilu02Info_t>info,
            <cusparseSolvePolicy_t>policy, <void*>pBuffer)
    check_status(status)

cpdef dcsrilu02(size_t handle, int m, int nnz, size_t descrA,
                size_t csrSortedValA_valM, size_t csrSortedRowPtrA,
                size_t csrSortedColIndA, size_t info, int policy,
                size_t pBuffer):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseDcsrilu02(
            <cusparseHandle_t>handle, m, nnz, <const cusparseMatDescr_t>descrA,
            <double*>csrSortedValA_valM, <const int*>csrSortedRowPtrA,
            <const int*>csrSortedColIndA, <csrilu02Info_t>info,
            <cusparseSolvePolicy_t>policy, <void*>pBuffer)
    check_status(status)

cpdef ccsrilu02(size_t handle, int m, int nnz, size_t descrA,
                size_t csrSortedValA_valM, size_t csrSortedRowPtrA,
                size_t csrSortedColIndA, size_t info, int policy,
                size_t pBuffer):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseCcsrilu02(
            <cusparseHandle_t>handle, m, nnz, <const cusparseMatDescr_t>descrA,
            <cuComplex*>csrSortedValA_valM, <const int*>csrSortedRowPtrA,
            <const int*>csrSortedColIndA, <csrilu02Info_t>info,
            <cusparseSolvePolicy_t>policy, <void*>pBuffer)
    check_status(status)

cpdef zcsrilu02(size_t handle, int m, int nnz, size_t descrA,
                size_t csrSortedValA_valM, size_t csrSortedRowPtrA,
                size_t csrSortedColIndA, size_t info, int policy,
                size_t pBuffer):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseZcsrilu02(
            <cusparseHandle_t>handle, m, nnz, <const cusparseMatDescr_t>descrA,
            <cuDoubleComplex*>csrSortedValA_valM, <const int*>csrSortedRowPtrA,
            <const int*>csrSortedColIndA, <csrilu02Info_t>info,
            <cusparseSolvePolicy_t>policy, <void*>pBuffer)
    check_status(status)

cpdef sbsrilu02_numericBoost(size_t handle, size_t info, int enable_boost,
                             size_t tol, size_t boost_val):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseSbsrilu02_numericBoost(
            <cusparseHandle_t>handle, <bsrilu02Info_t>info, enable_boost,
            <double*>tol, <float*>boost_val)
    check_status(status)

cpdef dbsrilu02_numericBoost(size_t handle, size_t info, int enable_boost,
                             size_t tol, size_t boost_val):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseDbsrilu02_numericBoost(
            <cusparseHandle_t>handle, <bsrilu02Info_t>info, enable_boost,
            <double*>tol, <double*>boost_val)
    check_status(status)

cpdef cbsrilu02_numericBoost(size_t handle, size_t info, int enable_boost,
                             size_t tol, size_t boost_val):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseCbsrilu02_numericBoost(
            <cusparseHandle_t>handle, <bsrilu02Info_t>info, enable_boost,
            <double*>tol, <cuComplex*>boost_val)
    check_status(status)

cpdef zbsrilu02_numericBoost(size_t handle, size_t info, int enable_boost,
                             size_t tol, size_t boost_val):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseZbsrilu02_numericBoost(
            <cusparseHandle_t>handle, <bsrilu02Info_t>info, enable_boost,
            <double*>tol, <cuDoubleComplex*>boost_val)
    check_status(status)

cpdef xbsrilu02_zeroPivot(size_t handle, size_t info, size_t position):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseXbsrilu02_zeroPivot(
            <cusparseHandle_t>handle, <bsrilu02Info_t>info, <int*>position)
    check_status(status)

cpdef int sbsrilu02_bufferSize(size_t handle, int dirA, int mb, int nnzb,
                               size_t descrA, size_t bsrSortedVal,
                               size_t bsrSortedRowPtr, size_t bsrSortedColInd,
                               int blockDim, size_t info):
    cdef int pBufferSizeInBytes
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseSbsrilu02_bufferSize(
            <cusparseHandle_t>handle, <cusparseDirection_t>dirA, mb, nnzb,
            <const cusparseMatDescr_t>descrA, <float*>bsrSortedVal,
            <const int*>bsrSortedRowPtr, <const int*>bsrSortedColInd, blockDim,
            <bsrilu02Info_t>info, &pBufferSizeInBytes)
    check_status(status)
    return <int>pBufferSizeInBytes

cpdef int dbsrilu02_bufferSize(size_t handle, int dirA, int mb, int nnzb,
                               size_t descrA, size_t bsrSortedVal,
                               size_t bsrSortedRowPtr, size_t bsrSortedColInd,
                               int blockDim, size_t info):
    cdef int pBufferSizeInBytes
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseDbsrilu02_bufferSize(
            <cusparseHandle_t>handle, <cusparseDirection_t>dirA, mb, nnzb,
            <const cusparseMatDescr_t>descrA, <double*>bsrSortedVal,
            <const int*>bsrSortedRowPtr, <const int*>bsrSortedColInd, blockDim,
            <bsrilu02Info_t>info, &pBufferSizeInBytes)
    check_status(status)
    return <int>pBufferSizeInBytes

cpdef int cbsrilu02_bufferSize(size_t handle, int dirA, int mb, int nnzb,
                               size_t descrA, size_t bsrSortedVal,
                               size_t bsrSortedRowPtr, size_t bsrSortedColInd,
                               int blockDim, size_t info):
    cdef int pBufferSizeInBytes
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseCbsrilu02_bufferSize(
            <cusparseHandle_t>handle, <cusparseDirection_t>dirA, mb, nnzb,
            <const cusparseMatDescr_t>descrA, <cuComplex*>bsrSortedVal,
            <const int*>bsrSortedRowPtr, <const int*>bsrSortedColInd, blockDim,
            <bsrilu02Info_t>info, &pBufferSizeInBytes)
    check_status(status)
    return <int>pBufferSizeInBytes

cpdef int zbsrilu02_bufferSize(size_t handle, int dirA, int mb, int nnzb,
                               size_t descrA, size_t bsrSortedVal,
                               size_t bsrSortedRowPtr, size_t bsrSortedColInd,
                               int blockDim, size_t info):
    cdef int pBufferSizeInBytes
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseZbsrilu02_bufferSize(
            <cusparseHandle_t>handle, <cusparseDirection_t>dirA, mb, nnzb,
            <const cusparseMatDescr_t>descrA, <cuDoubleComplex*>bsrSortedVal,
            <const int*>bsrSortedRowPtr, <const int*>bsrSortedColInd, blockDim,
            <bsrilu02Info_t>info, &pBufferSizeInBytes)
    check_status(status)
    return <int>pBufferSizeInBytes

cpdef sbsrilu02_analysis(
        size_t handle, int dirA, int mb, int nnzb, size_t descrA,
        size_t bsrSortedVal, size_t bsrSortedRowPtr, size_t bsrSortedColInd,
        int blockDim, size_t info, int policy, size_t pBuffer):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseSbsrilu02_analysis(
            <cusparseHandle_t>handle, <cusparseDirection_t>dirA, mb, nnzb,
            <const cusparseMatDescr_t>descrA, <float*>bsrSortedVal,
            <const int*>bsrSortedRowPtr, <const int*>bsrSortedColInd, blockDim,
            <bsrilu02Info_t>info, <cusparseSolvePolicy_t>policy,
            <void*>pBuffer)
    check_status(status)

cpdef dbsrilu02_analysis(
        size_t handle, int dirA, int mb, int nnzb, size_t descrA,
        size_t bsrSortedVal, size_t bsrSortedRowPtr, size_t bsrSortedColInd,
        int blockDim, size_t info, int policy, size_t pBuffer):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseDbsrilu02_analysis(
            <cusparseHandle_t>handle, <cusparseDirection_t>dirA, mb, nnzb,
            <const cusparseMatDescr_t>descrA, <double*>bsrSortedVal,
            <const int*>bsrSortedRowPtr, <const int*>bsrSortedColInd, blockDim,
            <bsrilu02Info_t>info, <cusparseSolvePolicy_t>policy,
            <void*>pBuffer)
    check_status(status)

cpdef cbsrilu02_analysis(
        size_t handle, int dirA, int mb, int nnzb, size_t descrA,
        size_t bsrSortedVal, size_t bsrSortedRowPtr, size_t bsrSortedColInd,
        int blockDim, size_t info, int policy, size_t pBuffer):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseCbsrilu02_analysis(
            <cusparseHandle_t>handle, <cusparseDirection_t>dirA, mb, nnzb,
            <const cusparseMatDescr_t>descrA, <cuComplex*>bsrSortedVal,
            <const int*>bsrSortedRowPtr, <const int*>bsrSortedColInd, blockDim,
            <bsrilu02Info_t>info, <cusparseSolvePolicy_t>policy,
            <void*>pBuffer)
    check_status(status)

cpdef zbsrilu02_analysis(
        size_t handle, int dirA, int mb, int nnzb, size_t descrA,
        size_t bsrSortedVal, size_t bsrSortedRowPtr, size_t bsrSortedColInd,
        int blockDim, size_t info, int policy, size_t pBuffer):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseZbsrilu02_analysis(
            <cusparseHandle_t>handle, <cusparseDirection_t>dirA, mb, nnzb,
            <const cusparseMatDescr_t>descrA, <cuDoubleComplex*>bsrSortedVal,
            <const int*>bsrSortedRowPtr, <const int*>bsrSortedColInd, blockDim,
            <bsrilu02Info_t>info, <cusparseSolvePolicy_t>policy,
            <void*>pBuffer)
    check_status(status)

cpdef sbsrilu02(size_t handle, int dirA, int mb, int nnzb, size_t descrA,
                size_t bsrSortedVal, size_t bsrSortedRowPtr,
                size_t bsrSortedColInd, int blockDim, size_t info, int policy,
                size_t pBuffer):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseSbsrilu02(
            <cusparseHandle_t>handle, <cusparseDirection_t>dirA, mb, nnzb,
            <const cusparseMatDescr_t>descrA, <float*>bsrSortedVal,
            <const int*>bsrSortedRowPtr, <const int*>bsrSortedColInd, blockDim,
            <bsrilu02Info_t>info, <cusparseSolvePolicy_t>policy,
            <void*>pBuffer)
    check_status(status)

cpdef dbsrilu02(size_t handle, int dirA, int mb, int nnzb, size_t descrA,
                size_t bsrSortedVal, size_t bsrSortedRowPtr,
                size_t bsrSortedColInd, int blockDim, size_t info, int policy,
                size_t pBuffer):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseDbsrilu02(
            <cusparseHandle_t>handle, <cusparseDirection_t>dirA, mb, nnzb,
            <const cusparseMatDescr_t>descrA, <double*>bsrSortedVal,
            <const int*>bsrSortedRowPtr, <const int*>bsrSortedColInd, blockDim,
            <bsrilu02Info_t>info, <cusparseSolvePolicy_t>policy,
            <void*>pBuffer)
    check_status(status)

cpdef cbsrilu02(size_t handle, int dirA, int mb, int nnzb, size_t descrA,
                size_t bsrSortedVal, size_t bsrSortedRowPtr,
                size_t bsrSortedColInd, int blockDim, size_t info, int policy,
                size_t pBuffer):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseCbsrilu02(
            <cusparseHandle_t>handle, <cusparseDirection_t>dirA, mb, nnzb,
            <const cusparseMatDescr_t>descrA, <cuComplex*>bsrSortedVal,
            <const int*>bsrSortedRowPtr, <const int*>bsrSortedColInd, blockDim,
            <bsrilu02Info_t>info, <cusparseSolvePolicy_t>policy,
            <void*>pBuffer)
    check_status(status)

cpdef zbsrilu02(size_t handle, int dirA, int mb, int nnzb, size_t descrA,
                size_t bsrSortedVal, size_t bsrSortedRowPtr,
                size_t bsrSortedColInd, int blockDim, size_t info, int policy,
                size_t pBuffer):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseZbsrilu02(
            <cusparseHandle_t>handle, <cusparseDirection_t>dirA, mb, nnzb,
            <const cusparseMatDescr_t>descrA, <cuDoubleComplex*>bsrSortedVal,
            <const int*>bsrSortedRowPtr, <const int*>bsrSortedColInd, blockDim,
            <bsrilu02Info_t>info, <cusparseSolvePolicy_t>policy,
            <void*>pBuffer)
    check_status(status)

cpdef xcsric02_zeroPivot(size_t handle, size_t info, size_t position):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseXcsric02_zeroPivot(
            <cusparseHandle_t>handle, <csric02Info_t>info, <int*>position)
    check_status(status)

cpdef int scsric02_bufferSize(size_t handle, int m, int nnz, size_t descrA,
                              size_t csrSortedValA, size_t csrSortedRowPtrA,
                              size_t csrSortedColIndA, size_t info):
    cdef int pBufferSizeInBytes
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseScsric02_bufferSize(
            <cusparseHandle_t>handle, m, nnz, <const cusparseMatDescr_t>descrA,
            <float*>csrSortedValA, <const int*>csrSortedRowPtrA,
            <const int*>csrSortedColIndA, <csric02Info_t>info,
            &pBufferSizeInBytes)
    check_status(status)
    return <int>pBufferSizeInBytes

cpdef int dcsric02_bufferSize(size_t handle, int m, int nnz, size_t descrA,
                              size_t csrSortedValA, size_t csrSortedRowPtrA,
                              size_t csrSortedColIndA, size_t info):
    cdef int pBufferSizeInBytes
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseDcsric02_bufferSize(
            <cusparseHandle_t>handle, m, nnz, <const cusparseMatDescr_t>descrA,
            <double*>csrSortedValA, <const int*>csrSortedRowPtrA,
            <const int*>csrSortedColIndA, <csric02Info_t>info,
            &pBufferSizeInBytes)
    check_status(status)
    return <int>pBufferSizeInBytes

cpdef int ccsric02_bufferSize(size_t handle, int m, int nnz, size_t descrA,
                              size_t csrSortedValA, size_t csrSortedRowPtrA,
                              size_t csrSortedColIndA, size_t info):
    cdef int pBufferSizeInBytes
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseCcsric02_bufferSize(
            <cusparseHandle_t>handle, m, nnz, <const cusparseMatDescr_t>descrA,
            <cuComplex*>csrSortedValA, <const int*>csrSortedRowPtrA,
            <const int*>csrSortedColIndA, <csric02Info_t>info,
            &pBufferSizeInBytes)
    check_status(status)
    return <int>pBufferSizeInBytes

cpdef int zcsric02_bufferSize(size_t handle, int m, int nnz, size_t descrA,
                              size_t csrSortedValA, size_t csrSortedRowPtrA,
                              size_t csrSortedColIndA, size_t info):
    cdef int pBufferSizeInBytes
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseZcsric02_bufferSize(
            <cusparseHandle_t>handle, m, nnz, <const cusparseMatDescr_t>descrA,
            <cuDoubleComplex*>csrSortedValA, <const int*>csrSortedRowPtrA,
            <const int*>csrSortedColIndA, <csric02Info_t>info,
            &pBufferSizeInBytes)
    check_status(status)
    return <int>pBufferSizeInBytes

cpdef scsric02_analysis(size_t handle, int m, int nnz, size_t descrA,
                        size_t csrSortedValA, size_t csrSortedRowPtrA,
                        size_t csrSortedColIndA, size_t info, int policy,
                        size_t pBuffer):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseScsric02_analysis(
            <cusparseHandle_t>handle, m, nnz, <const cusparseMatDescr_t>descrA,
            <const float*>csrSortedValA, <const int*>csrSortedRowPtrA,
            <const int*>csrSortedColIndA, <csric02Info_t>info,
            <cusparseSolvePolicy_t>policy, <void*>pBuffer)
    check_status(status)

cpdef dcsric02_analysis(size_t handle, int m, int nnz, size_t descrA,
                        size_t csrSortedValA, size_t csrSortedRowPtrA,
                        size_t csrSortedColIndA, size_t info, int policy,
                        size_t pBuffer):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseDcsric02_analysis(
            <cusparseHandle_t>handle, m, nnz, <const cusparseMatDescr_t>descrA,
            <const double*>csrSortedValA, <const int*>csrSortedRowPtrA,
            <const int*>csrSortedColIndA, <csric02Info_t>info,
            <cusparseSolvePolicy_t>policy, <void*>pBuffer)
    check_status(status)

cpdef ccsric02_analysis(size_t handle, int m, int nnz, size_t descrA,
                        size_t csrSortedValA, size_t csrSortedRowPtrA,
                        size_t csrSortedColIndA, size_t info, int policy,
                        size_t pBuffer):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseCcsric02_analysis(
            <cusparseHandle_t>handle, m, nnz, <const cusparseMatDescr_t>descrA,
            <const cuComplex*>csrSortedValA, <const int*>csrSortedRowPtrA,
            <const int*>csrSortedColIndA, <csric02Info_t>info,
            <cusparseSolvePolicy_t>policy, <void*>pBuffer)
    check_status(status)

cpdef zcsric02_analysis(size_t handle, int m, int nnz, size_t descrA,
                        size_t csrSortedValA, size_t csrSortedRowPtrA,
                        size_t csrSortedColIndA, size_t info, int policy,
                        size_t pBuffer):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseZcsric02_analysis(
            <cusparseHandle_t>handle, m, nnz, <const cusparseMatDescr_t>descrA,
            <const cuDoubleComplex*>csrSortedValA,
            <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA,
            <csric02Info_t>info, <cusparseSolvePolicy_t>policy, <void*>pBuffer)
    check_status(status)

cpdef scsric02(size_t handle, int m, int nnz, size_t descrA,
               size_t csrSortedValA_valM, size_t csrSortedRowPtrA,
               size_t csrSortedColIndA, size_t info, int policy,
               size_t pBuffer):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseScsric02(
            <cusparseHandle_t>handle, m, nnz, <const cusparseMatDescr_t>descrA,
            <float*>csrSortedValA_valM, <const int*>csrSortedRowPtrA,
            <const int*>csrSortedColIndA, <csric02Info_t>info,
            <cusparseSolvePolicy_t>policy, <void*>pBuffer)
    check_status(status)

cpdef dcsric02(size_t handle, int m, int nnz, size_t descrA,
               size_t csrSortedValA_valM, size_t csrSortedRowPtrA,
               size_t csrSortedColIndA, size_t info, int policy,
               size_t pBuffer):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseDcsric02(
            <cusparseHandle_t>handle, m, nnz, <const cusparseMatDescr_t>descrA,
            <double*>csrSortedValA_valM, <const int*>csrSortedRowPtrA,
            <const int*>csrSortedColIndA, <csric02Info_t>info,
            <cusparseSolvePolicy_t>policy, <void*>pBuffer)
    check_status(status)

cpdef ccsric02(size_t handle, int m, int nnz, size_t descrA,
               size_t csrSortedValA_valM, size_t csrSortedRowPtrA,
               size_t csrSortedColIndA, size_t info, int policy,
               size_t pBuffer):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseCcsric02(
            <cusparseHandle_t>handle, m, nnz, <const cusparseMatDescr_t>descrA,
            <cuComplex*>csrSortedValA_valM, <const int*>csrSortedRowPtrA,
            <const int*>csrSortedColIndA, <csric02Info_t>info,
            <cusparseSolvePolicy_t>policy, <void*>pBuffer)
    check_status(status)

cpdef zcsric02(size_t handle, int m, int nnz, size_t descrA,
               size_t csrSortedValA_valM, size_t csrSortedRowPtrA,
               size_t csrSortedColIndA, size_t info, int policy,
               size_t pBuffer):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseZcsric02(
            <cusparseHandle_t>handle, m, nnz, <const cusparseMatDescr_t>descrA,
            <cuDoubleComplex*>csrSortedValA_valM, <const int*>csrSortedRowPtrA,
            <const int*>csrSortedColIndA, <csric02Info_t>info,
            <cusparseSolvePolicy_t>policy, <void*>pBuffer)
    check_status(status)

cpdef xbsric02_zeroPivot(size_t handle, size_t info, size_t position):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseXbsric02_zeroPivot(
            <cusparseHandle_t>handle, <bsric02Info_t>info, <int*>position)
    check_status(status)

cpdef int sbsric02_bufferSize(size_t handle, int dirA, int mb, int nnzb,
                              size_t descrA, size_t bsrSortedVal,
                              size_t bsrSortedRowPtr, size_t bsrSortedColInd,
                              int blockDim, size_t info):
    cdef int pBufferSizeInBytes
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseSbsric02_bufferSize(
            <cusparseHandle_t>handle, <cusparseDirection_t>dirA, mb, nnzb,
            <const cusparseMatDescr_t>descrA, <float*>bsrSortedVal,
            <const int*>bsrSortedRowPtr, <const int*>bsrSortedColInd, blockDim,
            <bsric02Info_t>info, &pBufferSizeInBytes)
    check_status(status)
    return <int>pBufferSizeInBytes

cpdef int dbsric02_bufferSize(size_t handle, int dirA, int mb, int nnzb,
                              size_t descrA, size_t bsrSortedVal,
                              size_t bsrSortedRowPtr, size_t bsrSortedColInd,
                              int blockDim, size_t info):
    cdef int pBufferSizeInBytes
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseDbsric02_bufferSize(
            <cusparseHandle_t>handle, <cusparseDirection_t>dirA, mb, nnzb,
            <const cusparseMatDescr_t>descrA, <double*>bsrSortedVal,
            <const int*>bsrSortedRowPtr, <const int*>bsrSortedColInd, blockDim,
            <bsric02Info_t>info, &pBufferSizeInBytes)
    check_status(status)
    return <int>pBufferSizeInBytes

cpdef int cbsric02_bufferSize(size_t handle, int dirA, int mb, int nnzb,
                              size_t descrA, size_t bsrSortedVal,
                              size_t bsrSortedRowPtr, size_t bsrSortedColInd,
                              int blockDim, size_t info):
    cdef int pBufferSizeInBytes
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseCbsric02_bufferSize(
            <cusparseHandle_t>handle, <cusparseDirection_t>dirA, mb, nnzb,
            <const cusparseMatDescr_t>descrA, <cuComplex*>bsrSortedVal,
            <const int*>bsrSortedRowPtr, <const int*>bsrSortedColInd, blockDim,
            <bsric02Info_t>info, &pBufferSizeInBytes)
    check_status(status)
    return <int>pBufferSizeInBytes

cpdef int zbsric02_bufferSize(size_t handle, int dirA, int mb, int nnzb,
                              size_t descrA, size_t bsrSortedVal,
                              size_t bsrSortedRowPtr, size_t bsrSortedColInd,
                              int blockDim, size_t info):
    cdef int pBufferSizeInBytes
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseZbsric02_bufferSize(
            <cusparseHandle_t>handle, <cusparseDirection_t>dirA, mb, nnzb,
            <const cusparseMatDescr_t>descrA, <cuDoubleComplex*>bsrSortedVal,
            <const int*>bsrSortedRowPtr, <const int*>bsrSortedColInd, blockDim,
            <bsric02Info_t>info, &pBufferSizeInBytes)
    check_status(status)
    return <int>pBufferSizeInBytes

cpdef sbsric02_analysis(
        size_t handle, int dirA, int mb, int nnzb, size_t descrA,
        size_t bsrSortedVal, size_t bsrSortedRowPtr, size_t bsrSortedColInd,
        int blockDim, size_t info, int policy, size_t pInputBuffer):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseSbsric02_analysis(
            <cusparseHandle_t>handle, <cusparseDirection_t>dirA, mb, nnzb,
            <const cusparseMatDescr_t>descrA, <const float*>bsrSortedVal,
            <const int*>bsrSortedRowPtr, <const int*>bsrSortedColInd, blockDim,
            <bsric02Info_t>info, <cusparseSolvePolicy_t>policy,
            <void*>pInputBuffer)
    check_status(status)

cpdef dbsric02_analysis(
        size_t handle, int dirA, int mb, int nnzb, size_t descrA,
        size_t bsrSortedVal, size_t bsrSortedRowPtr, size_t bsrSortedColInd,
        int blockDim, size_t info, int policy, size_t pInputBuffer):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseDbsric02_analysis(
            <cusparseHandle_t>handle, <cusparseDirection_t>dirA, mb, nnzb,
            <const cusparseMatDescr_t>descrA, <const double*>bsrSortedVal,
            <const int*>bsrSortedRowPtr, <const int*>bsrSortedColInd, blockDim,
            <bsric02Info_t>info, <cusparseSolvePolicy_t>policy,
            <void*>pInputBuffer)
    check_status(status)

cpdef cbsric02_analysis(
        size_t handle, int dirA, int mb, int nnzb, size_t descrA,
        size_t bsrSortedVal, size_t bsrSortedRowPtr, size_t bsrSortedColInd,
        int blockDim, size_t info, int policy, size_t pInputBuffer):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseCbsric02_analysis(
            <cusparseHandle_t>handle, <cusparseDirection_t>dirA, mb, nnzb,
            <const cusparseMatDescr_t>descrA, <const cuComplex*>bsrSortedVal,
            <const int*>bsrSortedRowPtr, <const int*>bsrSortedColInd, blockDim,
            <bsric02Info_t>info, <cusparseSolvePolicy_t>policy,
            <void*>pInputBuffer)
    check_status(status)

cpdef zbsric02_analysis(
        size_t handle, int dirA, int mb, int nnzb, size_t descrA,
        size_t bsrSortedVal, size_t bsrSortedRowPtr, size_t bsrSortedColInd,
        int blockDim, size_t info, int policy, size_t pInputBuffer):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseZbsric02_analysis(
            <cusparseHandle_t>handle, <cusparseDirection_t>dirA, mb, nnzb,
            <const cusparseMatDescr_t>descrA,
            <const cuDoubleComplex*>bsrSortedVal, <const int*>bsrSortedRowPtr,
            <const int*>bsrSortedColInd, blockDim, <bsric02Info_t>info,
            <cusparseSolvePolicy_t>policy, <void*>pInputBuffer)
    check_status(status)

cpdef sbsric02(size_t handle, int dirA, int mb, int nnzb, size_t descrA,
               size_t bsrSortedVal, size_t bsrSortedRowPtr,
               size_t bsrSortedColInd, int blockDim, size_t info, int policy,
               size_t pBuffer):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseSbsric02(
            <cusparseHandle_t>handle, <cusparseDirection_t>dirA, mb, nnzb,
            <const cusparseMatDescr_t>descrA, <float*>bsrSortedVal,
            <const int*>bsrSortedRowPtr, <const int*>bsrSortedColInd, blockDim,
            <bsric02Info_t>info, <cusparseSolvePolicy_t>policy, <void*>pBuffer)
    check_status(status)

cpdef dbsric02(size_t handle, int dirA, int mb, int nnzb, size_t descrA,
               size_t bsrSortedVal, size_t bsrSortedRowPtr,
               size_t bsrSortedColInd, int blockDim, size_t info, int policy,
               size_t pBuffer):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseDbsric02(
            <cusparseHandle_t>handle, <cusparseDirection_t>dirA, mb, nnzb,
            <const cusparseMatDescr_t>descrA, <double*>bsrSortedVal,
            <const int*>bsrSortedRowPtr, <const int*>bsrSortedColInd, blockDim,
            <bsric02Info_t>info, <cusparseSolvePolicy_t>policy, <void*>pBuffer)
    check_status(status)

cpdef cbsric02(size_t handle, int dirA, int mb, int nnzb, size_t descrA,
               size_t bsrSortedVal, size_t bsrSortedRowPtr,
               size_t bsrSortedColInd, int blockDim, size_t info, int policy,
               size_t pBuffer):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseCbsric02(
            <cusparseHandle_t>handle, <cusparseDirection_t>dirA, mb, nnzb,
            <const cusparseMatDescr_t>descrA, <cuComplex*>bsrSortedVal,
            <const int*>bsrSortedRowPtr, <const int*>bsrSortedColInd, blockDim,
            <bsric02Info_t>info, <cusparseSolvePolicy_t>policy, <void*>pBuffer)
    check_status(status)

cpdef zbsric02(size_t handle, int dirA, int mb, int nnzb, size_t descrA,
               size_t bsrSortedVal, size_t bsrSortedRowPtr,
               size_t bsrSortedColInd, int blockDim, size_t info, int policy,
               size_t pBuffer):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseZbsric02(
            <cusparseHandle_t>handle, <cusparseDirection_t>dirA, mb, nnzb,
            <const cusparseMatDescr_t>descrA, <cuDoubleComplex*>bsrSortedVal,
            <const int*>bsrSortedRowPtr, <const int*>bsrSortedColInd, blockDim,
            <bsric02Info_t>info, <cusparseSolvePolicy_t>policy, <void*>pBuffer)
    check_status(status)

cpdef sgtsv2_bufferSizeExt(size_t handle, int m, int n, size_t dl, size_t d,
                           size_t du, size_t B, int ldb,
                           size_t bufferSizeInBytes):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseSgtsv2_bufferSizeExt(
            <cusparseHandle_t>handle, m, n, <const float*>dl, <const float*>d,
            <const float*>du, <const float*>B, ldb, <size_t*>bufferSizeInBytes)
    check_status(status)

cpdef dgtsv2_bufferSizeExt(size_t handle, int m, int n, size_t dl, size_t d,
                           size_t du, size_t B, int ldb,
                           size_t bufferSizeInBytes):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseDgtsv2_bufferSizeExt(
            <cusparseHandle_t>handle, m, n, <const double*>dl,
            <const double*>d, <const double*>du, <const double*>B, ldb,
            <size_t*>bufferSizeInBytes)
    check_status(status)

cpdef cgtsv2_bufferSizeExt(size_t handle, int m, int n, size_t dl, size_t d,
                           size_t du, size_t B, int ldb,
                           size_t bufferSizeInBytes):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseCgtsv2_bufferSizeExt(
            <cusparseHandle_t>handle, m, n, <const cuComplex*>dl,
            <const cuComplex*>d, <const cuComplex*>du, <const cuComplex*>B,
            ldb, <size_t*>bufferSizeInBytes)
    check_status(status)

cpdef zgtsv2_bufferSizeExt(size_t handle, int m, int n, size_t dl, size_t d,
                           size_t du, size_t B, int ldb,
                           size_t bufferSizeInBytes):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseZgtsv2_bufferSizeExt(
            <cusparseHandle_t>handle, m, n, <const cuDoubleComplex*>dl,
            <const cuDoubleComplex*>d, <const cuDoubleComplex*>du,
            <const cuDoubleComplex*>B, ldb, <size_t*>bufferSizeInBytes)
    check_status(status)

cpdef sgtsv2(size_t handle, int m, int n, size_t dl, size_t d, size_t du,
             size_t B, int ldb, size_t pBuffer):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseSgtsv2(
            <cusparseHandle_t>handle, m, n, <const float*>dl, <const float*>d,
            <const float*>du, <float*>B, ldb, <void*>pBuffer)
    check_status(status)

cpdef dgtsv2(size_t handle, int m, int n, size_t dl, size_t d, size_t du,
             size_t B, int ldb, size_t pBuffer):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseDgtsv2(<cusparseHandle_t>handle, m, n,
                                <const double*>dl, <const double*>d,
                                <const double*>du, <double*>B, ldb,
                                <void*>pBuffer)
    check_status(status)

cpdef cgtsv2(size_t handle, int m, int n, size_t dl, size_t d, size_t du,
             size_t B, int ldb, size_t pBuffer):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseCgtsv2(<cusparseHandle_t>handle, m, n,
                                <const cuComplex*>dl, <const cuComplex*>d,
                                <const cuComplex*>du, <cuComplex*>B, ldb,
                                <void*>pBuffer)
    check_status(status)

cpdef zgtsv2(size_t handle, int m, int n, size_t dl, size_t d, size_t du,
             size_t B, int ldb, size_t pBuffer):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseZgtsv2(
            <cusparseHandle_t>handle, m, n, <const cuDoubleComplex*>dl,
            <const cuDoubleComplex*>d, <const cuDoubleComplex*>du,
            <cuDoubleComplex*>B, ldb, <void*>pBuffer)
    check_status(status)

cpdef sgtsv2_nopivot_bufferSizeExt(size_t handle, int m, int n, size_t dl,
                                   size_t d, size_t du, size_t B, int ldb,
                                   size_t bufferSizeInBytes):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseSgtsv2_nopivot_bufferSizeExt(
            <cusparseHandle_t>handle, m, n, <const float*>dl, <const float*>d,
            <const float*>du, <const float*>B, ldb, <size_t*>bufferSizeInBytes)
    check_status(status)

cpdef dgtsv2_nopivot_bufferSizeExt(size_t handle, int m, int n, size_t dl,
                                   size_t d, size_t du, size_t B, int ldb,
                                   size_t bufferSizeInBytes):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseDgtsv2_nopivot_bufferSizeExt(
            <cusparseHandle_t>handle, m, n, <const double*>dl,
            <const double*>d, <const double*>du, <const double*>B, ldb,
            <size_t*>bufferSizeInBytes)
    check_status(status)

cpdef cgtsv2_nopivot_bufferSizeExt(size_t handle, int m, int n, size_t dl,
                                   size_t d, size_t du, size_t B, int ldb,
                                   size_t bufferSizeInBytes):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseCgtsv2_nopivot_bufferSizeExt(
            <cusparseHandle_t>handle, m, n, <const cuComplex*>dl,
            <const cuComplex*>d, <const cuComplex*>du, <const cuComplex*>B,
            ldb, <size_t*>bufferSizeInBytes)
    check_status(status)

cpdef zgtsv2_nopivot_bufferSizeExt(size_t handle, int m, int n, size_t dl,
                                   size_t d, size_t du, size_t B, int ldb,
                                   size_t bufferSizeInBytes):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseZgtsv2_nopivot_bufferSizeExt(
            <cusparseHandle_t>handle, m, n, <const cuDoubleComplex*>dl,
            <const cuDoubleComplex*>d, <const cuDoubleComplex*>du,
            <const cuDoubleComplex*>B, ldb, <size_t*>bufferSizeInBytes)
    check_status(status)

cpdef sgtsv2_nopivot(size_t handle, int m, int n, size_t dl, size_t d,
                     size_t du, size_t B, int ldb, size_t pBuffer):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseSgtsv2_nopivot(
            <cusparseHandle_t>handle, m, n, <const float*>dl, <const float*>d,
            <const float*>du, <float*>B, ldb, <void*>pBuffer)
    check_status(status)

cpdef dgtsv2_nopivot(size_t handle, int m, int n, size_t dl, size_t d,
                     size_t du, size_t B, int ldb, size_t pBuffer):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseDgtsv2_nopivot(<cusparseHandle_t>handle, m, n,
                                        <const double*>dl, <const double*>d,
                                        <const double*>du, <double*>B, ldb,
                                        <void*>pBuffer)
    check_status(status)

cpdef cgtsv2_nopivot(size_t handle, int m, int n, size_t dl, size_t d,
                     size_t du, size_t B, int ldb, size_t pBuffer):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseCgtsv2_nopivot(
            <cusparseHandle_t>handle, m, n, <const cuComplex*>dl,
            <const cuComplex*>d, <const cuComplex*>du, <cuComplex*>B, ldb,
            <void*>pBuffer)
    check_status(status)

cpdef zgtsv2_nopivot(size_t handle, int m, int n, size_t dl, size_t d,
                     size_t du, size_t B, int ldb, size_t pBuffer):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseZgtsv2_nopivot(
            <cusparseHandle_t>handle, m, n, <const cuDoubleComplex*>dl,
            <const cuDoubleComplex*>d, <const cuDoubleComplex*>du,
            <cuDoubleComplex*>B, ldb, <void*>pBuffer)
    check_status(status)

cpdef sgtsv2StridedBatch_bufferSizeExt(
        size_t handle, int m, size_t dl, size_t d, size_t du, size_t x,
        int batchCount, int batchStride, size_t bufferSizeInBytes):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseSgtsv2StridedBatch_bufferSizeExt(
            <cusparseHandle_t>handle, m, <const float*>dl, <const float*>d,
            <const float*>du, <const float*>x, batchCount, batchStride,
            <size_t*>bufferSizeInBytes)
    check_status(status)

cpdef dgtsv2StridedBatch_bufferSizeExt(
        size_t handle, int m, size_t dl, size_t d, size_t du, size_t x,
        int batchCount, int batchStride, size_t bufferSizeInBytes):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseDgtsv2StridedBatch_bufferSizeExt(
            <cusparseHandle_t>handle, m, <const double*>dl, <const double*>d,
            <const double*>du, <const double*>x, batchCount, batchStride,
            <size_t*>bufferSizeInBytes)
    check_status(status)

cpdef cgtsv2StridedBatch_bufferSizeExt(
        size_t handle, int m, size_t dl, size_t d, size_t du, size_t x,
        int batchCount, int batchStride, size_t bufferSizeInBytes):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseCgtsv2StridedBatch_bufferSizeExt(
            <cusparseHandle_t>handle, m, <const cuComplex*>dl,
            <const cuComplex*>d, <const cuComplex*>du, <const cuComplex*>x,
            batchCount, batchStride, <size_t*>bufferSizeInBytes)
    check_status(status)

cpdef zgtsv2StridedBatch_bufferSizeExt(
        size_t handle, int m, size_t dl, size_t d, size_t du, size_t x,
        int batchCount, int batchStride, size_t bufferSizeInBytes):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseZgtsv2StridedBatch_bufferSizeExt(
            <cusparseHandle_t>handle, m, <const cuDoubleComplex*>dl,
            <const cuDoubleComplex*>d, <const cuDoubleComplex*>du,
            <const cuDoubleComplex*>x, batchCount, batchStride,
            <size_t*>bufferSizeInBytes)
    check_status(status)

cpdef sgtsv2StridedBatch(size_t handle, int m, size_t dl, size_t d, size_t du,
                         size_t x, int batchCount, int batchStride,
                         size_t pBuffer):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseSgtsv2StridedBatch(
            <cusparseHandle_t>handle, m, <const float*>dl, <const float*>d,
            <const float*>du, <float*>x, batchCount, batchStride,
            <void*>pBuffer)
    check_status(status)

cpdef dgtsv2StridedBatch(size_t handle, int m, size_t dl, size_t d, size_t du,
                         size_t x, int batchCount, int batchStride,
                         size_t pBuffer):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseDgtsv2StridedBatch(
            <cusparseHandle_t>handle, m, <const double*>dl, <const double*>d,
            <const double*>du, <double*>x, batchCount, batchStride,
            <void*>pBuffer)
    check_status(status)

cpdef cgtsv2StridedBatch(size_t handle, int m, size_t dl, size_t d, size_t du,
                         size_t x, int batchCount, int batchStride,
                         size_t pBuffer):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseCgtsv2StridedBatch(
            <cusparseHandle_t>handle, m, <const cuComplex*>dl,
            <const cuComplex*>d, <const cuComplex*>du, <cuComplex*>x,
            batchCount, batchStride, <void*>pBuffer)
    check_status(status)

cpdef zgtsv2StridedBatch(size_t handle, int m, size_t dl, size_t d, size_t du,
                         size_t x, int batchCount, int batchStride,
                         size_t pBuffer):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseZgtsv2StridedBatch(
            <cusparseHandle_t>handle, m, <const cuDoubleComplex*>dl,
            <const cuDoubleComplex*>d, <const cuDoubleComplex*>du,
            <cuDoubleComplex*>x, batchCount, batchStride, <void*>pBuffer)
    check_status(status)

cpdef size_t sgtsvInterleavedBatch_bufferSizeExt(
        size_t handle, int algo, int m, size_t dl, size_t d, size_t du,
        size_t x, int batchCount):
    cdef size_t pBufferSizeInBytes
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseSgtsvInterleavedBatch_bufferSizeExt(
            <cusparseHandle_t>handle, algo, m, <const float*>dl,
            <const float*>d, <const float*>du, <const float*>x, batchCount,
            &pBufferSizeInBytes)
    check_status(status)
    return <size_t>pBufferSizeInBytes

cpdef size_t dgtsvInterleavedBatch_bufferSizeExt(
        size_t handle, int algo, int m, size_t dl, size_t d, size_t du,
        size_t x, int batchCount):
    cdef size_t pBufferSizeInBytes
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseDgtsvInterleavedBatch_bufferSizeExt(
            <cusparseHandle_t>handle, algo, m, <const double*>dl,
            <const double*>d, <const double*>du, <const double*>x, batchCount,
            &pBufferSizeInBytes)
    check_status(status)
    return <size_t>pBufferSizeInBytes

cpdef size_t cgtsvInterleavedBatch_bufferSizeExt(
        size_t handle, int algo, int m, size_t dl, size_t d, size_t du,
        size_t x, int batchCount):
    cdef size_t pBufferSizeInBytes
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseCgtsvInterleavedBatch_bufferSizeExt(
            <cusparseHandle_t>handle, algo, m, <const cuComplex*>dl,
            <const cuComplex*>d, <const cuComplex*>du, <const cuComplex*>x,
            batchCount, &pBufferSizeInBytes)
    check_status(status)
    return <size_t>pBufferSizeInBytes

cpdef size_t zgtsvInterleavedBatch_bufferSizeExt(
        size_t handle, int algo, int m, size_t dl, size_t d, size_t du,
        size_t x, int batchCount):
    cdef size_t pBufferSizeInBytes
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseZgtsvInterleavedBatch_bufferSizeExt(
            <cusparseHandle_t>handle, algo, m, <const cuDoubleComplex*>dl,
            <const cuDoubleComplex*>d, <const cuDoubleComplex*>du,
            <const cuDoubleComplex*>x, batchCount, &pBufferSizeInBytes)
    check_status(status)
    return <size_t>pBufferSizeInBytes

cpdef sgtsvInterleavedBatch(size_t handle, int algo, int m, size_t dl,
                            size_t d, size_t du, size_t x, int batchCount,
                            size_t pBuffer):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseSgtsvInterleavedBatch(
            <cusparseHandle_t>handle, algo, m, <float*>dl, <float*>d,
            <float*>du, <float*>x, batchCount, <void*>pBuffer)
    check_status(status)

cpdef dgtsvInterleavedBatch(size_t handle, int algo, int m, size_t dl,
                            size_t d, size_t du, size_t x, int batchCount,
                            size_t pBuffer):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseDgtsvInterleavedBatch(
            <cusparseHandle_t>handle, algo, m, <double*>dl, <double*>d,
            <double*>du, <double*>x, batchCount, <void*>pBuffer)
    check_status(status)

cpdef cgtsvInterleavedBatch(size_t handle, int algo, int m, size_t dl,
                            size_t d, size_t du, size_t x, int batchCount,
                            size_t pBuffer):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseCgtsvInterleavedBatch(
            <cusparseHandle_t>handle, algo, m, <cuComplex*>dl, <cuComplex*>d,
            <cuComplex*>du, <cuComplex*>x, batchCount, <void*>pBuffer)
    check_status(status)

cpdef zgtsvInterleavedBatch(size_t handle, int algo, int m, size_t dl,
                            size_t d, size_t du, size_t x, int batchCount,
                            size_t pBuffer):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseZgtsvInterleavedBatch(
            <cusparseHandle_t>handle, algo, m, <cuDoubleComplex*>dl,
            <cuDoubleComplex*>d, <cuDoubleComplex*>du, <cuDoubleComplex*>x,
            batchCount, <void*>pBuffer)
    check_status(status)

cpdef size_t sgpsvInterleavedBatch_bufferSizeExt(
        size_t handle, int algo, int m, size_t ds, size_t dl, size_t d,
        size_t du, size_t dw, size_t x, int batchCount):
    cdef size_t pBufferSizeInBytes
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseSgpsvInterleavedBatch_bufferSizeExt(
            <cusparseHandle_t>handle, algo, m, <const float*>ds,
            <const float*>dl, <const float*>d, <const float*>du,
            <const float*>dw, <const float*>x, batchCount, &pBufferSizeInBytes)
    check_status(status)
    return <size_t>pBufferSizeInBytes

cpdef size_t dgpsvInterleavedBatch_bufferSizeExt(
        size_t handle, int algo, int m, size_t ds, size_t dl, size_t d,
        size_t du, size_t dw, size_t x, int batchCount):
    cdef size_t pBufferSizeInBytes
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseDgpsvInterleavedBatch_bufferSizeExt(
            <cusparseHandle_t>handle, algo, m, <const double*>ds,
            <const double*>dl, <const double*>d, <const double*>du,
            <const double*>dw, <const double*>x, batchCount,
            &pBufferSizeInBytes)
    check_status(status)
    return <size_t>pBufferSizeInBytes

cpdef size_t cgpsvInterleavedBatch_bufferSizeExt(
        size_t handle, int algo, int m, size_t ds, size_t dl, size_t d,
        size_t du, size_t dw, size_t x, int batchCount):
    cdef size_t pBufferSizeInBytes
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseCgpsvInterleavedBatch_bufferSizeExt(
            <cusparseHandle_t>handle, algo, m, <const cuComplex*>ds,
            <const cuComplex*>dl, <const cuComplex*>d, <const cuComplex*>du,
            <const cuComplex*>dw, <const cuComplex*>x, batchCount,
            &pBufferSizeInBytes)
    check_status(status)
    return <size_t>pBufferSizeInBytes

cpdef size_t zgpsvInterleavedBatch_bufferSizeExt(
        size_t handle, int algo, int m, size_t ds, size_t dl, size_t d,
        size_t du, size_t dw, size_t x, int batchCount):
    cdef size_t pBufferSizeInBytes
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseZgpsvInterleavedBatch_bufferSizeExt(
            <cusparseHandle_t>handle, algo, m, <const cuDoubleComplex*>ds,
            <const cuDoubleComplex*>dl, <const cuDoubleComplex*>d,
            <const cuDoubleComplex*>du, <const cuDoubleComplex*>dw,
            <const cuDoubleComplex*>x, batchCount, &pBufferSizeInBytes)
    check_status(status)
    return <size_t>pBufferSizeInBytes

cpdef sgpsvInterleavedBatch(size_t handle, int algo, int m, size_t ds,
                            size_t dl, size_t d, size_t du, size_t dw,
                            size_t x, int batchCount, size_t pBuffer):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseSgpsvInterleavedBatch(
            <cusparseHandle_t>handle, algo, m, <float*>ds, <float*>dl,
            <float*>d, <float*>du, <float*>dw, <float*>x, batchCount,
            <void*>pBuffer)
    check_status(status)

cpdef dgpsvInterleavedBatch(size_t handle, int algo, int m, size_t ds,
                            size_t dl, size_t d, size_t du, size_t dw,
                            size_t x, int batchCount, size_t pBuffer):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseDgpsvInterleavedBatch(
            <cusparseHandle_t>handle, algo, m, <double*>ds, <double*>dl,
            <double*>d, <double*>du, <double*>dw, <double*>x, batchCount,
            <void*>pBuffer)
    check_status(status)

cpdef cgpsvInterleavedBatch(size_t handle, int algo, int m, size_t ds,
                            size_t dl, size_t d, size_t du, size_t dw,
                            size_t x, int batchCount, size_t pBuffer):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseCgpsvInterleavedBatch(
            <cusparseHandle_t>handle, algo, m, <cuComplex*>ds, <cuComplex*>dl,
            <cuComplex*>d, <cuComplex*>du, <cuComplex*>dw, <cuComplex*>x,
            batchCount, <void*>pBuffer)
    check_status(status)

cpdef zgpsvInterleavedBatch(size_t handle, int algo, int m, size_t ds,
                            size_t dl, size_t d, size_t du, size_t dw,
                            size_t x, int batchCount, size_t pBuffer):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusparseZgpsvInterleavedBatch(
            <cusparseHandle_t>handle, algo, m, <cuDoubleComplex*>ds,
            <cuDoubleComplex*>dl, <cuDoubleComplex*>d, <cuDoubleComplex*>du,
            <cuDoubleComplex*>dw, <cuDoubleComplex*>x, batchCount,
            <void*>pBuffer)
    check_status(status)
