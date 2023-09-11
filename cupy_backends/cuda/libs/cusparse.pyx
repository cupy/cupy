import sys as _sys  # no-cython-lint
cimport cython  # NOQA

from cupy_backends.cuda.api cimport runtime
from cupy_backends.cuda.api.runtime cimport _is_hip_environment
from cupy_backends.cuda cimport stream as stream_module
from cupy_backends.cuda._softlink cimport SoftLink


cdef extern from '../../cupy_complex.h':
    ctypedef struct cuComplex 'cuComplex':
        float x, y

    ctypedef struct cuDoubleComplex 'cuDoubleComplex':
        double x, y

cdef extern from '../../cupy_sparse.h' nogil:
    ctypedef void* Stream 'cudaStream_t'

    # Version
    cusparseStatus_t cusparseGetVersion(cusparseHandle_t handle, int* version)

    # Error handling
    const char* cusparseGetErrorName(Status status)
    const char* cusparseGetErrorString(Status status)

    # cuSPARSE Helper Function
    Status cusparseCreate(Handle *handle)
    Status cusparseCreateMatDescr(MatDescr descr)
    Status cusparseDestroy(Handle handle)
    Status cusparseDestroyMatDescr(MatDescr descr)
    Status cusparseSetMatIndexBase(MatDescr descr, IndexBase base)
    Status cusparseSetMatType(MatDescr descr, MatrixType type)
    Status cusparseSetMatFillMode(MatDescr descrA, FillMode fillMode)
    Status cusparseSetMatDiagType(MatDescr descrA, DiagType diagType)
    Status cusparseSetPointerMode(Handle handle, PointerMode mode)

    # Stream
    Status cusparseSetStream(Handle handle, Stream streamId)
    Status cusparseGetStream(Handle handle, Stream* streamId)

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

    Status cusparseCreateCsrsv2Info(csrsv2Info_t* info)
    Status cusparseDestroyCsrsv2Info(csrsv2Info_t info)

    Status cusparseScsrsv2_bufferSize(
        Handle handle, Operation transA, int m, int nnz, const MatDescr descrA,
        float* csrSortedValA, const int* csrSortedRowPtrA,
        const int* csrSortedColIndA, csrsv2Info_t info,
        int* pBufferSizeInBytes)
    Status cusparseDcsrsv2_bufferSize(
        Handle handle, Operation transA, int m, int nnz, const MatDescr descrA,
        double* csrSortedValA, const int* csrSortedRowPtrA,
        const int* csrSortedColIndA, csrsv2Info_t info,
        int* pBufferSizeInBytes)
    Status cusparseCcsrsv2_bufferSize(
        Handle handle, Operation transA, int m, int nnz, const MatDescr descrA,
        cuComplex* csrSortedValA, const int* csrSortedRowPtrA,
        const int* csrSortedColIndA, csrsv2Info_t info,
        int* pBufferSizeInBytes)
    Status cusparseZcsrsv2_bufferSize(
        Handle handle, Operation transA, int m, int nnz, const MatDescr descrA,
        cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA,
        const int* csrSortedColIndA, csrsv2Info_t info,
        int* pBufferSizeInBytes)

    Status cusparseScsrsv2_analysis(
        Handle handle, Operation transA, int m, int nnz, const MatDescr descrA,
        const float* csrSortedValA, const int* csrSortedRowPtrA,
        const int* csrSortedColIndA, csrsv2Info_t info,
        cusparseSolvePolicy_t policy, void* pBuffer)
    Status cusparseDcsrsv2_analysis(
        Handle handle, Operation transA, int m, int nnz, const MatDescr descrA,
        const double* csrSortedValA, const int* csrSortedRowPtrA,
        const int* csrSortedColIndA, csrsv2Info_t info,
        cusparseSolvePolicy_t policy, void* pBuffer)
    Status cusparseCcsrsv2_analysis(
        Handle handle, Operation transA, int m, int nnz, const MatDescr descrA,
        const cuComplex* csrSortedValA, const int* csrSortedRowPtrA,
        const int* csrSortedColIndA, csrsv2Info_t info,
        cusparseSolvePolicy_t policy, void* pBuffer)
    Status cusparseZcsrsv2_analysis(
        Handle handle, Operation transA, int m, int nnz, const MatDescr descrA,
        const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA,
        const int* csrSortedColIndA, csrsv2Info_t info,
        cusparseSolvePolicy_t policy, void* pBuffer)

    Status cusparseScsrsv2_solve(
        Handle handle, Operation transA, int m, int nnz,
        const float* alpha, const MatDescr descrA,
        const float* csrSortedValA, const int* csrSortedRowPtrA,
        const int* csrSortedColIndA, csrsv2Info_t info,
        const float* x, float* y,
        cusparseSolvePolicy_t policy, void* pBuffer)
    Status cusparseDcsrsv2_solve(
        Handle handle, Operation transA, int m, int nnz,
        const double* alpha, const MatDescr descrA,
        const double* csrSortedValA, const int* csrSortedRowPtrA,
        const int* csrSortedColIndA, csrsv2Info_t info,
        const double* x, double* y,
        cusparseSolvePolicy_t policy, void* pBuffer)
    Status cusparseCcsrsv2_solve(
        Handle handle, Operation transA, int m, int nnz,
        const cuComplex* alpha, const MatDescr descrA,
        const cuComplex* csrSortedValA, const int* csrSortedRowPtrA,
        const int* csrSortedColIndA, csrsv2Info_t info,
        const cuComplex* x, cuComplex* y,
        cusparseSolvePolicy_t policy, void* pBuffer)
    Status cusparseZcsrsv2_solve(
        Handle handle, Operation transA, int m, int nnz,
        const cuDoubleComplex* alpha, const MatDescr descrA,
        const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA,
        const int* csrSortedColIndA, csrsv2Info_t info,
        const cuDoubleComplex* x, cuDoubleComplex* y,
        cusparseSolvePolicy_t policy, void* pBuffer)

    Status cusparseXcsrsv2_zeroPivot(
        Handle handle, csrsv2Info_t info, int* position)

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

    Status cusparseCreateCsrsm2Info(csrsm2Info_t* info)
    Status cusparseDestroyCsrsm2Info(csrsm2Info_t info)

    Status cusparseScsrsm2_bufferSizeExt(
        Handle handle, int algo, Operation transA, Operation transB, int m,
        int nrhs, int nnz, const float* alpha, const MatDescr descrA,
        const float* csrSortedValA, const int* csrSortedRowPtrA,
        const int* csrSortedColIndA, const float* B,
        int ldb, csrsm2Info_t info, cusparseSolvePolicy_t policy,
        size_t* pBufferSize)
    Status cusparseDcsrsm2_bufferSizeExt(
        Handle handle, int algo, Operation transA, Operation transB, int m,
        int nrhs, int nnz, const double* alpha, const MatDescr descrA,
        const double* csrSortedValA, const int* csrSortedRowPtrA,
        const int* csrSortedColIndA, const double* B,
        int ldb, csrsm2Info_t info, cusparseSolvePolicy_t policy,
        size_t* pBufferSize)
    Status cusparseCcsrsm2_bufferSizeExt(
        Handle handle, int algo, Operation transA, Operation transB, int m,
        int nrhs, int nnz, const cuComplex* alpha, const MatDescr descrA,
        const cuComplex* csrSortedValA, const int* csrSortedRowPtrA,
        const int* csrSortedColIndA, const cuComplex* B,
        int ldb, csrsm2Info_t info, cusparseSolvePolicy_t policy,
        size_t* pBufferSize)
    Status cusparseZcsrsm2_bufferSizeExt(
        Handle handle, int algo, Operation transA, Operation transB, int m,
        int nrhs, int nnz, const cuDoubleComplex* alpha, const MatDescr descrA,
        const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA,
        const int* csrSortedColIndA, const cuDoubleComplex* B,
        int ldb, csrsm2Info_t info, cusparseSolvePolicy_t policy,
        size_t* pBufferSize)

    Status cusparseScsrsm2_analysis(
        Handle handle, int algo, Operation transA, Operation transB, int m,
        int nrhs, int nnz, const float* alpha, const MatDescr descrA,
        const float* csrSortedValA, const int* csrSortedRowPtrA,
        const int* csrSortedColIndA, const float* B,
        int ldb, csrsm2Info_t info, cusparseSolvePolicy_t policy,
        void* pBuffer)
    Status cusparseDcsrsm2_analysis(
        Handle handle, int algo, Operation transA, Operation transB, int m,
        int nrhs, int nnz, const double* alpha, const MatDescr descrA,
        const double* csrSortedValA, const int* csrSortedRowPtrA,
        const int* csrSortedColIndA, const double* B,
        int ldb, csrsm2Info_t info, cusparseSolvePolicy_t policy,
        void* pBuffer)
    Status cusparseCcsrsm2_analysis(
        Handle handle, int algo, Operation transA, Operation transB, int m,
        int nrhs, int nnz, const cuComplex* alpha, const MatDescr descrA,
        const cuComplex* csrSortedValA, const int* csrSortedRowPtrA,
        const int* csrSortedColIndA, const cuComplex* B,
        int ldb, csrsm2Info_t info, cusparseSolvePolicy_t policy,
        void* pBuffer)
    Status cusparseZcsrsm2_analysis(
        Handle handle, int algo, Operation transA, Operation transB, int m,
        int nrhs, int nnz, const cuDoubleComplex* alpha, const MatDescr descrA,
        const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA,
        const int* csrSortedColIndA, const cuDoubleComplex* B,
        int ldb, csrsm2Info_t info, cusparseSolvePolicy_t policy,
        void* pBuffer)

    Status cusparseScsrsm2_solve(
        Handle handle, int algo, Operation transA, Operation transB, int m,
        int nrhs, int nnz, const float* alpha, const MatDescr descrA,
        const float* csrSortedValA, const int* csrSortedRowPtrA,
        const int* csrSortedColIndA, float* B,
        int ldb, csrsm2Info_t info, cusparseSolvePolicy_t policy,
        void* pBuffer)
    Status cusparseDcsrsm2_solve(
        Handle handle, int algo, Operation transA, Operation transB, int m,
        int nrhs, int nnz, const double* alpha, const MatDescr descrA,
        const double* csrSortedValA, const int* csrSortedRowPtrA,
        const int* csrSortedColIndA, double* B,
        int ldb, csrsm2Info_t info, cusparseSolvePolicy_t policy,
        void* pBuffer)
    Status cusparseCcsrsm2_solve(
        Handle handle, int algo, Operation transA, Operation transB, int m,
        int nrhs, int nnz, const cuComplex* alpha, const MatDescr descrA,
        const cuComplex* csrSortedValA, const int* csrSortedRowPtrA,
        const int* csrSortedColIndA, cuComplex* B,
        int ldb, csrsm2Info_t info, cusparseSolvePolicy_t policy,
        void* pBuffer)
    Status cusparseZcsrsm2_solve(
        Handle handle, int algo, Operation transA, Operation transB, int m,
        int nrhs, int nnz, const cuDoubleComplex* alpha, const MatDescr descrA,
        const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA,
        const int* csrSortedColIndA, cuDoubleComplex* B,
        int ldb, csrsm2Info_t info, cusparseSolvePolicy_t policy,
        void* pBuffer)

    Status cusparseXcsrsm2_zeroPivot(
        Handle handle, csrsm2Info_t info, int* position)

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

    Status cusparseScsrgeam2_bufferSizeExt(
        Handle handle, int m, int n, const float *alpha, const MatDescr descrA,
        int nnzA, const float *csrValA, const int *csrRowPtrA,
        const int *csrColIndA, const float *beta, const MatDescr descrB,
        int nnzB, const float *csrValB, const int *csrRowPtrB,
        const int *csrColIndB, const MatDescr descrC, float *csrValC,
        int *csrRowPtrC, int *csrColIndC, size_t *pBufferSize)

    Status cusparseDcsrgeam2_bufferSizeExt(
        Handle handle, int m, int n, const double *alpha,
        const MatDescr descrA,
        int nnzA, const double *csrValA, const int *csrRowPtrA,
        const int *csrColIndA, const double *beta, const MatDescr descrB,
        int nnzB, const double *csrValB, const int *csrRowPtrB,
        const int *csrColIndB, const MatDescr descrC, double *csrValC,
        int *csrRowPtrC, int *csrColIndC, size_t *pBufferSize)

    Status cusparseCcsrgeam2_bufferSizeExt(
        Handle handle, int m, int n, const cuComplex *alpha,
        const MatDescr descrA,
        int nnzA, const cuComplex *csrValA, const int *csrRowPtrA,
        const int *csrColIndA, const cuComplex *beta, const MatDescr descrB,
        int nnzB, const cuComplex *csrValB, const int *csrRowPtrB,
        const int *csrColIndB, const MatDescr descrC, cuComplex *csrValC,
        int *csrRowPtrC, int *csrColIndC, size_t *pBufferSize)

    Status cusparseZcsrgeam2_bufferSizeExt(
        Handle handle, int m, int n, const cuDoubleComplex *alpha,
        const MatDescr descrA,
        int nnzA, const cuDoubleComplex *csrValA, const int *csrRowPtrA,
        const int *csrColIndA, const cuDoubleComplex *beta,
        const MatDescr descrB,
        int nnzB, const cuDoubleComplex *csrValB, const int *csrRowPtrB,
        const int *csrColIndB, const MatDescr descrC,
        cuDoubleComplex *csrValC, int *csrRowPtrC, int *csrColIndC,
        size_t *pBufferSize)

    Status cusparseXcsrgeam2Nnz(
        Handle handle, int m, int n, const MatDescr descrA, int nnzA,
        const int *csrRowPtrA, const int *csrColIndA, const MatDescr descrB,
        int nnzB, const int *csrRowPtrB, const int *csrColIndB,
        const MatDescr descrC, int *csrRowPtrC, int *nnzTotalDevHostPtr,
        void *workspace)

    Status cusparseScsrgeam2(
        Handle handle, int m, int n, const float *alpha, const MatDescr descrA,
        int nnzA, const float *csrValA, const int *csrRowPtrA,
        const int *csrColIndA, const float *beta, const MatDescr descrB,
        int nnzB, const float *csrValB, const int *csrRowPtrB,
        const int *csrColIndB, const MatDescr descrC, float *csrValC,
        int *csrRowPtrC, int *csrColIndC, void *pBuffer)

    Status cusparseDcsrgeam2(
        Handle handle, int m, int n, const double *alpha,
        const MatDescr descrA,
        int nnzA, const double *csrValA, const int *csrRowPtrA,
        const int *csrColIndA, const double *beta, const MatDescr descrB,
        int nnzB, const double *csrValB, const int *csrRowPtrB,
        const int *csrColIndB, const MatDescr descrC, double *csrValC,
        int *csrRowPtrC, int *csrColIndC, void *pBuffer)

    Status cusparseCcsrgeam2(
        Handle handle, int m, int n, const cuComplex *alpha,
        const MatDescr descrA,
        int nnzA, const cuComplex *csrValA, const int *csrRowPtrA,
        const int *csrColIndA, const cuComplex *beta, const MatDescr descrB,
        int nnzB, const cuComplex *csrValB, const int *csrRowPtrB,
        const int *csrColIndB, const MatDescr descrC, cuComplex *csrValC,
        int *csrRowPtrC, int *csrColIndC, void *pBuffer)

    Status cusparseZcsrgeam2(
        Handle handle, int m, int n, const cuDoubleComplex *alpha,
        const MatDescr descrA,
        int nnzA, const cuDoubleComplex *csrValA, const int *csrRowPtrA,
        const int *csrColIndA, const cuDoubleComplex *beta,
        const MatDescr descrB,
        int nnzB, const cuDoubleComplex *csrValB, const int *csrRowPtrB,
        const int *csrColIndB, const MatDescr descrC,
        cuDoubleComplex *csrValC, int *csrRowPtrC, int *csrColIndC,
        void *pBuffer)

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

    cusparseStatus_t cusparseCreateCsrgemm2Info(csrgemm2Info_t *info)
    cusparseStatus_t cusparseDestroyCsrgemm2Info(csrgemm2Info_t info)

    Status cusparseScsrgemm2_bufferSizeExt(
        Handle handle, int m, int n, int k, const float *alpha,
        const MatDescr descrA, int nnzA, const int *csrRowPtrA,
        const int *csrColIndA, const MatDescr descrB, int nnzB,
        const int *csrRowPtrB, const int *csrColIndB, const float *beta,
        const MatDescr descrD, int nnzD, const int *csrRowPtrD,
        const int *csrColIndD, const csrgemm2Info_t info, size_t* pBufferSize)

    Status cusparseDcsrgemm2_bufferSizeExt(
        Handle handle, int m, int n, int k, const double *alpha,
        const MatDescr descrA, int nnzA, const int *csrRowPtrA,
        const int *csrColIndA, const MatDescr descrB, int nnzB,
        const int *csrRowPtrB, const int *csrColIndB, const double *beta,
        const MatDescr descrD, int nnzD, const int *csrRowPtrD,
        const int *csrColIndD, const csrgemm2Info_t info, size_t* pBufferSize)

    Status cusparseCcsrgemm2_bufferSizeExt(
        Handle handle, int m, int n, int k, const cuComplex *alpha,
        const MatDescr descrA, int nnzA, const int *csrRowPtrA,
        const int *csrColIndA, const MatDescr descrB, int nnzB,
        const int *csrRowPtrB, const int *csrColIndB, const cuComplex *beta,
        const MatDescr descrD, int nnzD, const int *csrRowPtrD,
        const int *csrColIndD, const csrgemm2Info_t info, size_t* pBufferSize)

    Status cusparseZcsrgemm2_bufferSizeExt(
        Handle handle, int m, int n, int k, const cuDoubleComplex *alpha,
        const MatDescr descrA, int nnzA, const int *csrRowPtrA,
        const int *csrColIndA, const MatDescr descrB, int nnzB,
        const int *csrRowPtrB, const int *csrColIndB,
        const cuDoubleComplex *beta, const MatDescr descrD, int nnzD,
        const int *csrRowPtrD, const int *csrColIndD,
        const csrgemm2Info_t info, size_t* pBufferSize)

    Status cusparseXcsrgemm2Nnz(
        Handle handle, int m, int n, int k, const MatDescr descrA, int nnzA,
        const int *csrRowPtrA, const int *csrColIndA, const MatDescr descrB,
        int nnzB, const int *csrRowPtrB, const int *csrColIndB,
        const MatDescr descrD, int nnzD, const int *csrRowPtrD,
        const int *csrColIndD, const MatDescr descrC, int *csrRowPtrC,
        int *nnzTotalDevHostPtr, const csrgemm2Info_t info, void* pBuffer)

    Status cusparseScsrgemm2(
        Handle handle, int m, int n, int k, const float *alpha,
        const MatDescr descrA, int nnzA, const float *csrValA,
        const int *csrRowPtrA, const int *csrColIndA, const MatDescr descrB,
        int nnzB, const float *csrValB, const int *csrRowPtrB,
        const int *csrColIndB, const float *beta, const MatDescr descrD,
        int nnzD, const float *csrValD, const int *csrRowPtrD,
        const int *csrColIndD, const MatDescr descrC, float *csrValC,
        const int *csrRowPtrC, int *csrColIndC, const csrgemm2Info_t info,
        void* pBuffer)

    Status cusparseDcsrgemm2(
        Handle handle, int m, int n, int k, const double *alpha,
        const MatDescr descrA, int nnzA, const double *csrValA,
        const int *csrRowPtrA, const int *csrColIndA, const MatDescr descrB,
        int nnzB, const double *csrValB, const int *csrRowPtrB,
        const int *csrColIndB, const double *beta, const MatDescr descrD,
        int nnzD, const double *csrValD, const int *csrRowPtrD,
        const int *csrColIndD, const MatDescr descrC, double *csrValC,
        const int *csrRowPtrC, int *csrColIndC, const csrgemm2Info_t info,
        void* pBuffer)

    Status cusparseCcsrgemm2(
        Handle handle, int m, int n, int k, const cuComplex *alpha,
        const MatDescr descrA, int nnzA, const cuComplex *csrValA,
        const int *csrRowPtrA, const int *csrColIndA, const MatDescr descrB,
        int nnzB, const cuComplex *csrValB, const int *csrRowPtrB,
        const int *csrColIndB, const cuComplex *beta, const MatDescr descrD,
        int nnzD, const cuComplex *csrValD, const int *csrRowPtrD,
        const int *csrColIndD, const MatDescr descrC, cuComplex *csrValC,
        const int *csrRowPtrC, int *csrColIndC, const csrgemm2Info_t info,
        void* pBuffer)

    Status cusparseZcsrgemm2(
        Handle handle, int m, int n, int k, const cuDoubleComplex *alpha,
        const MatDescr descrA, int nnzA, const cuDoubleComplex *csrValA,
        const int *csrRowPtrA, const int *csrColIndA, const MatDescr descrB,
        int nnzB, const cuDoubleComplex *csrValB, const int *csrRowPtrB,
        const int *csrColIndB, const cuDoubleComplex *beta,
        const MatDescr descrD, int nnzD, const cuDoubleComplex *csrValD,
        const int *csrRowPtrD, const int *csrColIndD, const MatDescr descrC,
        cuDoubleComplex *csrValC, const int *csrRowPtrC, int *csrColIndC,
        const csrgemm2Info_t info, void* pBuffer)

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

    Status cusparseXcoosortByColumn(
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

    # Sparse Vector APIs
    Status cusparseCreateSpVec(SpVecDescr* spVecDescr, int64_t size,
                               int64_t nnz, void* indices, void* values,
                               IndexType idxType, IndexBase idxBase,
                               DataType valueType)
    Status cusparseDestroySpVec(SpVecDescr spVecDescr)
    Status cusparseSpVecGet(SpVecDescr spVecDescr, int64_t* size, int64_t* nnz,
                            void** indices, void** values, IndexType* idxType,
                            IndexBase* idxBase, DataType* valueType)
    Status cusparseSpVecGetIndexBase(SpVecDescr spVecDescr, IndexBase* idxBae)
    Status cusparseSpVecGetValues(SpVecDescr spVecDescr, void** values)
    Status cusparseSpVecSetValues(SpVecDescr spVecDescr, void* values)

    # Sparse Matrix APIs
    Status cusparseCreateCoo(SpMatDescr* spMatDescr, int64_t rows,
                             int64_t cols, int64_t nnz, void* cooRowInd,
                             void* cooColInd, void* cooValues,
                             IndexType cooIdxType, IndexBase idxBase,
                             DataType valueType)
    Status cusparseCreateCooAoS(SpMatDescr* spMatDescr, int64_t rows,
                                int64_t cols, int64_t nnz, void* cooInd,
                                void* cooValues, IndexType cooIdxType,
                                IndexBase idxBase, DataType valueType)
    Status cusparseCreateCsr(SpMatDescr* spMatDescr, int64_t rows,
                             int64_t cols, int64_t nnz, void* csrRowOffsets,
                             void* csrColind, void* csrValues,
                             IndexType csrRowOffsetsType,
                             IndexType csrColIndType, IndexBase idxBase,
                             DataType valueType)
    Status cusparseDestroySpMat(SpMatDescr spMatDescr)
    Status cusparseCooGet(SpMatDescr spMatDescr, int64_t* rows, int64_t* cols,
                          int64_t* nnz, void** cooRowInd, void** cooColInd,
                          void** cooValues, IndexType* idxType,
                          IndexBase* idxBase, DataType* valueType)
    Status cusparseCooAoSGet(SpMatDescr spMatDescr, int64_t* rows,
                             int64_t* cols, int64_t* nnz, void** cooInd,
                             void** cooValues, IndexType* idxType,
                             IndexBase* idxBase, DataType* valueType)
    Status cusparseCsrGet(SpMatDescr spMatDescr, int64_t* rows, int64_t* cols,
                          int64_t* nnz, void** csrRowOffsets, void** csrColInd,
                          void** csrValues, IndexType* csrRowOffsetsType,
                          IndexType* csrColIndType, IndexBase* idxBase,
                          DataType* valueType)
    Status cusparseCsrSetPointers(SpMatDescr spMatDescr, void* csrRowOffsets,
                                  void* csrColInd, void* csrValues)
    Status cusparseSpMatGetFormat(SpMatDescr spMatDescr, Format* format)
    Status cusparseSpMatGetIndexBase(SpMatDescr spMatDescr, IndexBase* idxBase)
    Status cusparseSpMatGetValues(SpMatDescr spMatDescr, void** values)
    Status cusparseSpMatSetValues(SpMatDescr spMatDescr, void* values)
    Status cusparseSpMatGetSize(SpMatDescr spMatDescr, int64_t* rows,
                                int64_t* cols, int64_t* nnz)
    Status cusparseSpMatGetStridedBatch(SpMatDescr spMatDescr, int* batchCount)
    Status cusparseSpMatSetStridedBatch(SpMatDescr spMatDescr, int batchCount)

    # Dense Vector APIs
    Status cusparseCreateDnVec(DnVecDescr *dnVecDescr, int64_t size,
                               void* values, DataType valueType)
    Status cusparseDestroyDnVec(DnVecDescr dnVecDescr)
    Status cusparseDnVecGet(DnVecDescr dnVecDescr, int64_t* size,
                            void** values, DataType* valueType)
    Status cusparseDnVecGetValues(DnVecDescr dnVecDescr, void** values)
    Status cusparseDnVecSetValues(DnVecDescr dnVecDescr, void* values)

    # Dense Matrix APIs
    Status cusparseCreateDnMat(DnMatDescr* dnMatDescr, int64_t rows,
                               int64_t cols, int64_t ld, void* values,
                               DataType valueType, Order order)
    Status cusparseDestroyDnMat(DnMatDescr dnVecDescr)
    Status cusparseDnMatGet(DnMatDescr dnMatDescr, int64_t* rows,
                            int64_t* cols, int64_t* ld, void** values,
                            DataType* valueType, Order* order)
    Status cusparseDnMatGetValues(DnMatDescr spMatDescr, void** values)
    Status cusparseDnMatSetValues(DnMatDescr spMatDescr, void* values)
    Status cusparseDnMatGetStridedBatch(DnMatDescr dnMatDescr, int* batchCount,
                                        int64_t *batchStride)
    Status cusparseDnMatSetStridedBatch(DnMatDescr dnMatDescr, int batchCount,
                                        int64_t batchStride)

    # Generic API Functions
    Status cusparseSpVV_bufferSize(Handle handle, Operation opX,
                                   SpVecDescr vecX, DnVecDescr vecY,
                                   void* result, DataType computeType,
                                   size_t* bufferSize)
    Status cusparseSpVV(Handle handle, Operation opX, SpVecDescr vecX,
                        DnVecDescr vecY, void* result, DataType computeType,
                        void* externalBuffer)
    Status cusparseSpMV_bufferSize(Handle handle, Operation opA, void* alpha,
                                   SpMatDescr matA, DnVecDescr vecX,
                                   void* beta, DnVecDescr vecY,
                                   DataType computeType, SpMVAlg alg,
                                   size_t* bufferSize)
    Status cusparseSpMV(Handle handle, Operation opA, void* alpha,
                        SpMatDescr matA, DnVecDescr vecX, void* beta,
                        DnVecDescr vecY, DataType computeType, SpMVAlg alg,
                        void* externalBuffer)
    Status cusparseSpMM_bufferSize(Handle handle, Operation opA, Operation opB,
                                   void* alpha, SpMatDescr matA,
                                   DnMatDescr matB, void* beta,
                                   DnMatDescr matC, DataType computeType,
                                   SpMMAlg alg, size_t* bufferSize)
    Status cusparseSpMM(Handle handle, Operation opA, Operation opB,
                        void* alpha, SpMatDescr matA, DnMatDescr matB,
                        void* beta, DnMatDescr matC, DataType computeType,
                        SpMMAlg alg, void* externalBuffer)
    Status cusparseConstrainedGeMM_bufferSize(
        Handle handle, Operation opA, Operation opB, void* alpha,
        DnMatDescr matA, DnMatDescr matB, void* beta, SpMatDescr matC,
        DataType computeType, size_t* bufferSize)
    Status cusparseConstrainedGeMM(
        Handle handle, Operation opA, Operation opB, void* alpha,
        DnMatDescr matA, DnMatDescr matB, void* beta, SpMatDescr matC,
        DataType computeType, void* externalBuffer)

    Status cusparseSpGEMM_createDescr(SpGEMMDescr* spgemmDescr)
    Status cusparseSpGEMM_destroyDescr(SpGEMMDescr spgemmDescr)
    Status cusparseSpGEMM_workEstimation(
        Handle handle, Operation opA, Operation opB, const void* alpha,
        SpMatDescr matA, SpMatDescr matB, const void* beta, SpMatDescr matC,
        DataType computeType, SpGEMMAlg alg, SpGEMMDescr spgemmDescr,
        size_t* bufferSize1, void* externalBuffer1)
    Status cusparseSpGEMM_compute(
        Handle handle, Operation opA, Operation opB, const void* alpha,
        SpMatDescr matA, SpMatDescr matB, const void* beta, SpMatDescr matC,
        DataType computeType, SpGEMMAlg alg, SpGEMMDescr spgemmDescr,
        size_t* bufferSize2, void* externalBuffer2)
    Status cusparseSpGEMM_copy(
        Handle handle, Operation opA, Operation opB, const void* alpha,
        SpMatDescr matA, SpMatDescr matB, const void* beta, SpMatDescr matC,
        DataType computeType, SpGEMMAlg alg, SpGEMMDescr spgemmDescr)
    Status cusparseGather(Handle handle, DnVecDescr vecY, SpVecDescr vecX)

    # CSR2CSC
    Status cusparseCsr2cscEx2_bufferSize(
        Handle handle, int m, int n, int nnz, const void* csrVal,
        const int* csrRowPtr, const int* csrColInd, void* cscVal,
        int* cscColPtr, int* cscRowInd, DataType valType, Action copyValues,
        IndexBase idxBase, Csr2CscAlg alg, size_t* bufferSize)

    Status cusparseCsr2cscEx2(
        Handle handle, int m, int n, int nnz, const void* csrVal,
        const int* csrRowPtr, const int* csrColInd, void* cscVal,
        int* cscColPtr, int* cscRowInd, DataType valType, Action copyValues,
        IndexBase idxBase, Csr2CscAlg alg, void* buffer)

    # Build-time version
    int CUSPARSE_VERSION

ctypedef Status (*f_type)(...) nogil  # NOQA
IF 11010 <= CUPY_CUDA_VERSION < 12000:
    if _sys.platform == 'linux':
        _libname = 'libcusparse.so.11'
    else:
        _libname = 'cusparse64_11.dll'
ELIF 12000 <= CUPY_CUDA_VERSION < 13000:
    if _sys.platform == 'linux':
        _libname = 'libcusparse.so.12'
    else:
        _libname = 'cusparse64_12.dll'
ELIF 0 < CUPY_HIP_VERSION:
    _libname = __file__
ELSE:
    _libname = None

cdef SoftLink _lib = SoftLink(_libname, 'cusparse')
# cuSPARSE 11.6+ (CUDA 11.3.1+)
cdef f_type cusparseSpSM_createDescr = <f_type>_lib.get('SpSM_createDescr')
cdef f_type cusparseSpSM_destroyDescr = <f_type>_lib.get('SpSM_destroyDescr')
cdef f_type cusparseSpSM_bufferSize = <f_type>_lib.get('SpSM_bufferSize')
cdef f_type cusparseSpSM_analysis = <f_type>_lib.get('SpSM_analysis')
cdef f_type cusparseSpSM_solve = <f_type>_lib.get('SpSM_solve')
# cuSPARSE 11.5+ (CUDA 11.3.0+)
cdef f_type cusparseSpMatSetAttribute = <f_type>_lib.get('SpMatSetAttribute')
# cuSPARSE 11.3.1+ (CUDA 11.2.0+)
cdef f_type cusparseCreateCsc = <f_type>_lib.get('CreateCsc')
# cuSPARSE 11.3+ (CUDA 11.1.1+)
# Note: CUDA 11.1.0 contains cuSPARSE 11.2.0.275
cdef f_type cusparseSparseToDense_bufferSize = <f_type>_lib.get('SparseToDense_bufferSize')  # NOQA
cdef f_type cusparseSparseToDense = <f_type>_lib.get('SparseToDense')
cdef f_type cusparseDenseToSparse_bufferSize = <f_type>_lib.get('DenseToSparse_bufferSize')  # NOQA
cdef f_type cusparseDenseToSparse_analysis = <f_type>_lib.get('DenseToSparse_analysis')  # NOQA
cdef f_type cusparseDenseToSparse_convert = <f_type>_lib.get('DenseToSparse_convert')  # NOQA

cdef dict HIP_STATUS = {
    0: b'HIPSPARSE_STATUS_SUCCESS',
    1: b'HIPSPARSE_STATUS_NOT_INITIALIZED',
    2: b'HIPSPARSE_STATUS_ALLOC_FAILED',
    3: b'HIPSPARSE_STATUS_INVALID_VALUE',
    4: b'HIPSPARSE_STATUS_ARCH_MISMATCH',
    5: b'HIPSPARSE_STATUS_MAPPING_ERROR',
    6: b'HIPSPARSE_STATUS_EXECUTION_FAILED',
    7: b'HIPSPARSE_STATUS_INTERNAL_ERROR',
    8: b'HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED',
    9: b'HIPSPARSE_STATUS_ZERO_PIVOT',
    10: b'HIPSPARSE_STATUS_NOT_SUPPORTED',
    11: b'HIPSPARSE_STATUS_INSUFFICIENT_RESOURCES',
}


cdef class SpVecAttributes:

    def __init__(self, int64_t size, int64_t nnz,
                 intptr_t idx, intptr_t values,
                 IndexType idxType, IndexBase idxBase, DataType valueType):
        self.size = size
        self.nnz = nnz
        self.idx = idx
        self.values = values
        self.idxType = idxType
        self.idxBase = idxBase
        self.valueType = valueType


cdef class CooAttributes:

    def __init__(self, int64_t rows, int64_t cols, int64_t nnz,
                 intptr_t rowIdx, intptr_t colIdx, intptr_t values,
                 IndexType idxType, IndexBase idxBase, DataType valueType):
        self.rows = rows
        self.cols = cols
        self.nnz = nnz
        self.rowIdx = rowIdx
        self.colIdx = colIdx
        self.values = values
        self.idxType = idxType
        self.idxBase = idxBase
        self.valueType = valueType


cdef class CooAoSAttributes:

    def __init__(self, int64_t rows, int64_t cols, int64_t nnz,
                 intptr_t ind, intptr_t values,
                 IndexType idxType, IndexBase idxBase, DataType valueType):
        self.rows = rows
        self.cols = cols
        self.nnz = nnz
        self.ind = ind
        self.values = values
        self.idxType = idxType
        self.idxBase = idxBase
        self.valueType = valueType


cdef class CsrAttributes:

    def __init__(self, int64_t rows, int64_t cols, int64_t nnz,
                 intptr_t rowOffsets, intptr_t colIdx, intptr_t values,
                 IndexType rowOffsetType, IndexType colIdxType,
                 IndexBase idxBase, DataType valueType):
        self.rows = rows
        self.cols = cols
        self.nnz = nnz
        self.rowOffsets = rowOffsets
        self.colIdx = colIdx
        self.values = values
        self.rowOffsetType = rowOffsetType
        self.colIdxType = colIdxType
        self.idxBase = idxBase
        self.valueType = valueType


cdef class DnVecAttributes:

    def __init__(self, int64_t size, intptr_t values, DataType valueType):
        self.size = size
        self.values = values
        self.valueType = valueType


cdef class DnMatAttributes:

    def __init__(self, int64_t rows, int64_t cols, int64_t ld,
                 intptr_t values, DataType valueType, Order order):
        self.rows = rows
        self.cols = cols
        self.ld = ld
        self.values = values
        self.valueType = valueType
        self.order = order


cdef class DnMatBatchAttributes:

    def __init__(self, int count, int64_t stride):
        self.count = count
        self.stride = stride


class CuSparseError(RuntimeError):

    def __init__(self, Status status):
        self.status = status
        cdef bytes name
        cdef bytes msg
        if _is_hip_environment:
            name = HIP_STATUS[status]
            msg = name
        else:
            name = cusparseGetErrorName(status)
            msg = cusparseGetErrorString(status)
        super().__init__(f'{name.decode()}: {msg.decode()}')

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


cpdef int getVersion(intptr_t handle) except? -1:
    cdef int version
    status = cusparseGetVersion(<Handle>handle, &version)
    check_status(status)
    return version


def get_build_version():
    return CUSPARSE_VERSION


########################################
# cuSPARSE Helper Function

cpdef intptr_t create() except? 0:
    cdef Handle handle
    status = cusparseCreate(& handle)
    check_status(status)
    return <intptr_t>handle


cpdef size_t createMatDescr() except? -1:
    cdef MatDescr desc
    status = cusparseCreateMatDescr(& desc)
    check_status(status)
    return <size_t>desc


cpdef void destroy(intptr_t handle) except *:
    status = cusparseDestroy(<Handle >handle)
    check_status(status)


cpdef void destroyMatDescr(size_t descr) except *:
    status = cusparseDestroyMatDescr(<MatDescr>descr)
    check_status(status)


cpdef void setMatIndexBase(size_t descr, base) except *:
    status = cusparseSetMatIndexBase(<MatDescr>descr, base)
    check_status(status)


cpdef void setMatType(size_t descr, typ) except *:
    status = cusparseSetMatType(<MatDescr>descr, typ)
    check_status(status)

cpdef void setMatFillMode(size_t descrA, int fillMode) except *:
    status = cusparseSetMatFillMode(<MatDescr>descrA, <FillMode>fillMode)
    check_status(status)

cpdef void setMatDiagType(size_t descrA, int diagType) except *:
    status = cusparseSetMatDiagType(<MatDescr>descrA, <DiagType>diagType)
    check_status(status)

cpdef void setPointerMode(intptr_t handle, int mode) except *:
    status = cusparseSetPointerMode(<Handle>handle, <PointerMode>mode)
    check_status(status)

cpdef void spMatSetAttribute(
        size_t spMatDescr, int attribute, int data) except *:
    # Assuming the value of attribute is an enum value, whose underlying type
    # As for CUDA 11.7, the types of all the sparse matrix descriptor
    # attributes are enums, whose underlying type is always int in C.
    status = cusparseSpMatSetAttribute(
        <SpMatDescr>spMatDescr, <SpMatAttribute>attribute, <void*>&data,
        sizeof(int))
    check_status(status)


########################################
# Stream

cpdef void setStream(intptr_t handle, size_t stream) except *:
    # TODO(leofang): It seems most of cuSPARSE APIs support stream capture (as
    # of CUDA 11.5) under certain conditions, see
    # https://docs.nvidia.com/cuda/cusparse/index.html#optimization-notes
    # Before we come up with a robust strategy to test the support conditions,
    # we disable this functionality.
    if not runtime._is_hip_environment and runtime.streamIsCapturing(stream):
        raise NotImplementedError(
            'calling cuSPARSE API during stream capture is currently '
            'unsupported')

    status = cusparseSetStream(<Handle>handle, <Stream>stream)
    check_status(status)


cpdef size_t getStream(intptr_t handle) except? -1:
    cdef Stream stream
    status = cusparseGetStream(<Handle>handle, &stream)
    check_status(status)
    return <size_t>stream


cdef void _setStream(intptr_t handle) except *:
    """Set current stream"""
    setStream(handle, stream_module.get_current_stream_ptr())


########################################
# cuSPARSE Level1 Function

cpdef void sgthr(
        intptr_t handle, int nnz, size_t y, size_t xVal, size_t xInd,
        int idxBase) except *:
    _setStream(handle)
    status = cusparseSgthr(
        <Handle>handle, nnz, <const float *>y, <float *>xVal,
        <const int *>xInd, <IndexBase>idxBase)
    check_status(status)

cpdef void dgthr(
        intptr_t handle, int nnz, size_t y, size_t xVal, size_t xInd,
        int idxBase) except *:
    _setStream(handle)
    status = cusparseDgthr(
        <Handle>handle, nnz, <const double *>y, <double *>xVal,
        <const int *>xInd, <IndexBase>idxBase)
    check_status(status)

cpdef void cgthr(
        intptr_t handle, int nnz, size_t y, size_t xVal, size_t xInd,
        int idxBase) except *:
    _setStream(handle)
    status = cusparseCgthr(
        <Handle>handle, nnz, <const cuComplex *>y, <cuComplex *>xVal,
        <const int *>xInd, <IndexBase>idxBase)
    check_status(status)

cpdef void zgthr(
        intptr_t handle, int nnz, size_t y, size_t xVal, size_t xInd,
        int idxBase) except *:
    _setStream(handle)
    status = cusparseZgthr(
        <Handle>handle, nnz, <const cuDoubleComplex *>y,
        <cuDoubleComplex *>xVal, <const int *>xInd, <IndexBase>idxBase)
    check_status(status)

########################################
# cuSPARSE Level2 Function

cpdef void scsrmv(
        intptr_t handle, int transA, int m, int n, int nnz,
        size_t alpha, size_t descrA, size_t csrSortedValA,
        size_t csrSortedRowPtrA, size_t csrSortedColIndA,
        size_t x, size_t beta, size_t y) except *:
    _setStream(handle)
    status = cusparseScsrmv(
        <Handle>handle, <Operation>transA, m, n, nnz,
        <const float *>alpha, <MatDescr>descrA, <const float *>csrSortedValA,
        <const int *>csrSortedRowPtrA, <const int *>csrSortedColIndA,
        <const float *>x, <const float *>beta, <float *>y)
    check_status(status)

cpdef void dcsrmv(
        intptr_t handle, int transA, int m, int n, int nnz,
        size_t alpha, size_t descrA, size_t csrSortedValA,
        size_t csrSortedRowPtrA, size_t csrSortedColIndA,
        size_t x, size_t beta, size_t y) except *:
    _setStream(handle)
    status = cusparseDcsrmv(
        <Handle>handle, <Operation>transA, m, n, nnz,
        <const double *>alpha, <MatDescr>descrA, <const double *>csrSortedValA,
        <const int *>csrSortedRowPtrA, <const int *>csrSortedColIndA,
        <const double *>x, <const double *>beta, <double *>y)
    check_status(status)

cpdef void ccsrmv(
        intptr_t handle, int transA, int m, int n, int nnz,
        size_t alpha, size_t descrA, size_t csrSortedValA,
        size_t csrSortedRowPtrA, size_t csrSortedColIndA,
        size_t x, size_t beta, size_t y) except *:
    _setStream(handle)
    status = cusparseCcsrmv(
        <Handle>handle, <Operation>transA, m, n, nnz,
        <const cuComplex *>alpha, <MatDescr>descrA,
        <const cuComplex *>csrSortedValA,
        <const int *>csrSortedRowPtrA, <const int *>csrSortedColIndA,
        <const cuComplex *>x, <const cuComplex *>beta, <cuComplex *>y)
    check_status(status)

cpdef void zcsrmv(
        intptr_t handle, int transA, int m, int n, int nnz,
        size_t alpha, size_t descrA, size_t csrSortedValA,
        size_t csrSortedRowPtrA, size_t csrSortedColIndA,
        size_t x, size_t beta, size_t y) except *:
    _setStream(handle)
    status = cusparseZcsrmv(
        <Handle>handle, <Operation>transA, m, n, nnz,
        <const cuDoubleComplex *>alpha, <MatDescr>descrA,
        <const cuDoubleComplex *>csrSortedValA,
        <const int *>csrSortedRowPtrA, <const int *>csrSortedColIndA,
        <const cuDoubleComplex *>x, <const cuDoubleComplex *>beta,
        <cuDoubleComplex *>y)
    check_status(status)

cpdef size_t csrmvEx_bufferSize(
        intptr_t handle, int alg, int transA, int m, int n,
        int nnz, size_t alpha, int alphatype, size_t descrA,
        size_t csrValA, int csrValAtype, size_t csrRowPtrA,
        size_t csrColIndA, size_t x, int xtype, size_t beta,
        int betatype, size_t y, int ytype, int executiontype) except? -1:
    cdef size_t bufferSizeInBytes
    _setStream(handle)
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

cpdef void csrmvEx(
        intptr_t handle, int alg, int transA, int m, int n,
        int nnz, size_t alpha, int alphatype, size_t descrA,
        size_t csrValA, int csrValAtype, size_t csrRowPtrA,
        size_t csrColIndA, size_t x, int xtype, size_t beta,
        int betatype, size_t y, int ytype, int executiontype,
        size_t buffer) except *:
    _setStream(handle)
    status = cusparseCsrmvEx(
        <Handle>handle, <AlgMode>alg, <Operation>transA, m,
        n, nnz, <const void *>alpha, <DataType>alphatype,
        <MatDescr>descrA, <const void *>csrValA, <DataType>csrValAtype,
        <const int *>csrRowPtrA, <const int *>csrColIndA,
        <const void *>x, <DataType>xtype, <const void *>beta,
        <DataType>betatype, <void *>y, <DataType>ytype,
        <DataType>executiontype, <void *>buffer)
    check_status(status)

cpdef size_t createCsrsv2Info() except? -1:
    cdef csrsv2Info_t info
    status = cusparseCreateCsrsv2Info(&info)
    check_status(status)
    return <size_t>info

cpdef void destroyCsrsv2Info(size_t info) except *:
    status = cusparseDestroyCsrsv2Info(<csrsv2Info_t>info)
    check_status(status)

cpdef int scsrsv2_bufferSize(
        intptr_t handle, int transA, int m, int nnz, size_t descrA,
        size_t csrSortedValA, size_t csrSortedRowPtrA,
        size_t csrSortedColIndA, size_t info) except? -1:
    cdef int bufferSize
    _setStream(handle)
    status = cusparseScsrsv2_bufferSize(
        <Handle>handle, <Operation>transA, m, nnz, <const MatDescr>descrA,
        <float*>csrSortedValA, <const int*>csrSortedRowPtrA,
        <const int*>csrSortedColIndA, <csrsv2Info_t>info, &bufferSize)
    check_status(status)
    return bufferSize

cpdef int dcsrsv2_bufferSize(
        intptr_t handle, int transA, int m, int nnz, size_t descrA,
        size_t csrSortedValA, size_t csrSortedRowPtrA,
        size_t csrSortedColIndA, size_t info) except? -1:
    cdef int bufferSize
    _setStream(handle)
    status = cusparseDcsrsv2_bufferSize(
        <Handle>handle, <Operation>transA, m, nnz, <const MatDescr>descrA,
        <double*>csrSortedValA, <const int*>csrSortedRowPtrA,
        <const int*>csrSortedColIndA, <csrsv2Info_t>info, &bufferSize)
    check_status(status)
    return bufferSize

cpdef int ccsrsv2_bufferSize(
        intptr_t handle, int transA, int m, int nnz, size_t descrA,
        size_t csrSortedValA, size_t csrSortedRowPtrA,
        size_t csrSortedColIndA, size_t info) except? -1:
    cdef int bufferSize
    _setStream(handle)
    status = cusparseCcsrsv2_bufferSize(
        <Handle>handle, <Operation>transA, m, nnz, <const MatDescr>descrA,
        <cuComplex*>csrSortedValA, <const int*>csrSortedRowPtrA,
        <const int*>csrSortedColIndA, <csrsv2Info_t>info, &bufferSize)
    check_status(status)
    return bufferSize

cpdef int zcsrsv2_bufferSize(
        intptr_t handle, int transA, int m, int nnz, size_t descrA,
        size_t csrSortedValA, size_t csrSortedRowPtrA,
        size_t csrSortedColIndA, size_t info) except? -1:
    cdef int bufferSize
    _setStream(handle)
    status = cusparseZcsrsv2_bufferSize(
        <Handle>handle, <Operation>transA, m, nnz, <const MatDescr>descrA,
        <cuDoubleComplex*>csrSortedValA, <const int*>csrSortedRowPtrA,
        <const int*>csrSortedColIndA, <csrsv2Info_t>info, &bufferSize)
    check_status(status)
    return bufferSize

cpdef void scsrsv2_analysis(
        intptr_t handle, int transA, int m, int nnz, size_t descrA,
        size_t csrSortedValA, size_t csrSortedRowPtrA,
        size_t csrSortedColIndA, size_t info, int policy,
        size_t pBuffer) except *:
    _setStream(handle)
    status = cusparseScsrsv2_analysis(
        <Handle>handle, <Operation>transA, m, nnz, <const MatDescr>descrA,
        <const float*>csrSortedValA, <const int*>csrSortedRowPtrA,
        <const int*>csrSortedColIndA, <csrsv2Info_t>info,
        <cusparseSolvePolicy_t>policy, <void*>pBuffer)
    check_status(status)

cpdef void dcsrsv2_analysis(
        intptr_t handle, int transA, int m, int nnz, size_t descrA,
        size_t csrSortedValA, size_t csrSortedRowPtrA,
        size_t csrSortedColIndA, size_t info, int policy,
        size_t pBuffer) except *:
    _setStream(handle)
    status = cusparseDcsrsv2_analysis(
        <Handle>handle, <Operation>transA, m, nnz, <const MatDescr>descrA,
        <const double*>csrSortedValA, <const int*>csrSortedRowPtrA,
        <const int*>csrSortedColIndA, <csrsv2Info_t>info,
        <cusparseSolvePolicy_t>policy, <void*>pBuffer)
    check_status(status)

cpdef void ccsrsv2_analysis(
        intptr_t handle, int transA, int m, int nnz, size_t descrA,
        size_t csrSortedValA, size_t csrSortedRowPtrA,
        size_t csrSortedColIndA, size_t info, int policy,
        size_t pBuffer) except *:
    _setStream(handle)
    status = cusparseCcsrsv2_analysis(
        <Handle>handle, <Operation>transA, m, nnz, <const MatDescr>descrA,
        <const cuComplex*>csrSortedValA, <const int*>csrSortedRowPtrA,
        <const int*>csrSortedColIndA, <csrsv2Info_t>info,
        <cusparseSolvePolicy_t>policy, <void*>pBuffer)
    check_status(status)

cpdef void zcsrsv2_analysis(
        intptr_t handle, int transA, int m, int nnz, size_t descrA,
        size_t csrSortedValA, size_t csrSortedRowPtrA,
        size_t csrSortedColIndA, size_t info, int policy,
        size_t pBuffer) except *:
    _setStream(handle)
    status = cusparseZcsrsv2_analysis(
        <Handle>handle, <Operation>transA, m, nnz, <const MatDescr>descrA,
        <const cuDoubleComplex*>csrSortedValA, <const int*>csrSortedRowPtrA,
        <const int*>csrSortedColIndA, <csrsv2Info_t>info,
        <cusparseSolvePolicy_t>policy, <void*>pBuffer)
    check_status(status)

cpdef void scsrsv2_solve(
        intptr_t handle, int transA, int m, int nnz, size_t alpha,
        size_t descrA, size_t csrSortedValA, size_t csrSortedRowPtrA,
        size_t csrSortedColIndA, size_t info, size_t x, size_t y, int policy,
        size_t pBuffer) except *:
    _setStream(handle)
    status = cusparseScsrsv2_solve(
        <Handle>handle, <Operation>transA, m, nnz,
        <const float*>alpha, <const MatDescr>descrA,
        <const float*>csrSortedValA, <const int*>csrSortedRowPtrA,
        <const int*>csrSortedColIndA, <csrsv2Info_t>info,
        <const float*>x, <float*>y,
        <cusparseSolvePolicy_t>policy, <void*>pBuffer)
    check_status(status)

cpdef void dcsrsv2_solve(
        intptr_t handle, int transA, int m, int nnz, size_t alpha,
        size_t descrA, size_t csrSortedValA, size_t csrSortedRowPtrA,
        size_t csrSortedColIndA, size_t info, size_t x, size_t y, int policy,
        size_t pBuffer) except *:
    _setStream(handle)
    status = cusparseDcsrsv2_solve(
        <Handle>handle, <Operation>transA, m, nnz,
        <const double*>alpha, <const MatDescr>descrA,
        <const double*>csrSortedValA, <const int*>csrSortedRowPtrA,
        <const int*>csrSortedColIndA, <csrsv2Info_t>info,
        <const double*>x, <double*>y,
        <cusparseSolvePolicy_t>policy, <void*>pBuffer)
    check_status(status)

cpdef void ccsrsv2_solve(
        intptr_t handle, int transA, int m, int nnz, size_t alpha,
        size_t descrA, size_t csrSortedValA, size_t csrSortedRowPtrA,
        size_t csrSortedColIndA, size_t info, size_t x, size_t y, int policy,
        size_t pBuffer) except *:
    _setStream(handle)
    status = cusparseCcsrsv2_solve(
        <Handle>handle, <Operation>transA, m, nnz,
        <const cuComplex*>alpha, <const MatDescr>descrA,
        <const cuComplex*>csrSortedValA, <const int*>csrSortedRowPtrA,
        <const int*>csrSortedColIndA, <csrsv2Info_t>info,
        <const cuComplex*>x, <cuComplex*>y,
        <cusparseSolvePolicy_t>policy, <void*>pBuffer)
    check_status(status)

cpdef void zcsrsv2_solve(
        intptr_t handle, int transA, int m, int nnz, size_t alpha,
        size_t descrA, size_t csrSortedValA, size_t csrSortedRowPtrA,
        size_t csrSortedColIndA, size_t info, size_t x, size_t y, int policy,
        size_t pBuffer) except *:
    _setStream(handle)
    status = cusparseZcsrsv2_solve(
        <Handle>handle, <Operation>transA, m, nnz,
        <const cuDoubleComplex*>alpha, <const MatDescr>descrA,
        <const cuDoubleComplex*>csrSortedValA, <const int*>csrSortedRowPtrA,
        <const int*>csrSortedColIndA, <csrsv2Info_t>info,
        <const cuDoubleComplex*>x, <cuDoubleComplex*>y,
        <cusparseSolvePolicy_t>policy, <void*>pBuffer)
    check_status(status)

cpdef void xcsrsv2_zeroPivot(
        intptr_t handle, size_t info, size_t position) except *:
    _setStream(handle)
    status = cusparseXcsrsv2_zeroPivot(
        <Handle>handle, <csrsv2Info_t>info, <int*>position)
    check_status(status)

########################################
# cuSPARSE Level3 Function

cpdef void scsrmm(
        intptr_t handle, int transA, int m, int n, int k, int nnz,
        size_t alpha, size_t descrA, size_t csrSortedValA,
        size_t csrSortedRowPtrA, size_t csrSortedColIndA,
        size_t B, int ldb, size_t beta, size_t C, int ldc) except *:
    _setStream(handle)
    status = cusparseScsrmm(
        <Handle>handle, <Operation>transA, m, n, k, nnz,
        <const float *>alpha, <MatDescr>descrA, <const float *>csrSortedValA,
        <const int *>csrSortedRowPtrA, <const int *>csrSortedColIndA,
        <const float *>B, ldb, <const float *>beta, <float *>C, ldc)
    check_status(status)

cpdef void dcsrmm(
        intptr_t handle, int transA, int m, int n, int k, int nnz,
        size_t alpha, size_t descrA, size_t csrSortedValA,
        size_t csrSortedRowPtrA, size_t csrSortedColIndA,
        size_t B, int ldb, size_t beta, size_t C, int ldc) except *:
    _setStream(handle)
    status = cusparseDcsrmm(
        <Handle>handle, <Operation>transA, m, n, k, nnz,
        <const double *>alpha, <MatDescr>descrA, <const double *>csrSortedValA,
        <const int *>csrSortedRowPtrA, <const int *>csrSortedColIndA,
        <const double *>B, ldb, <const double *>beta, <double *>C, ldc)
    check_status(status)

cpdef void ccsrmm(
        intptr_t handle, int transA, int m, int n, int k, int nnz,
        size_t alpha, size_t descrA, size_t csrSortedValA,
        size_t csrSortedRowPtrA, size_t csrSortedColIndA,
        size_t B, int ldb, size_t beta, size_t C, int ldc) except *:
    _setStream(handle)
    status = cusparseCcsrmm(
        <Handle>handle, <Operation>transA, m, n, k, nnz,
        <const cuComplex *>alpha, <MatDescr>descrA,
        <const cuComplex *>csrSortedValA,
        <const int *>csrSortedRowPtrA, <const int *>csrSortedColIndA,
        <const cuComplex *>B, ldb, <const cuComplex *>beta,
        <cuComplex *>C, ldc)
    check_status(status)

cpdef void zcsrmm(
        intptr_t handle, int transA, int m, int n, int k, int nnz,
        size_t alpha, size_t descrA, size_t csrSortedValA,
        size_t csrSortedRowPtrA, size_t csrSortedColIndA,
        size_t B, int ldb, size_t beta, size_t C, int ldc) except *:
    _setStream(handle)
    status = cusparseZcsrmm(
        <Handle>handle, <Operation>transA, m, n, k, nnz,
        <const cuDoubleComplex *>alpha, <MatDescr>descrA,
        <const cuDoubleComplex *>csrSortedValA,
        <const int *>csrSortedRowPtrA, <const int *>csrSortedColIndA,
        <const cuDoubleComplex *>B, ldb,
        <const cuDoubleComplex *>beta, <cuDoubleComplex *>C, ldc)
    check_status(status)

cpdef void scsrmm2(
        intptr_t handle, int transA, int transB, int m, int n, int k, int nnz,
        size_t alpha, size_t descrA, size_t csrValA,
        size_t csrRowPtrA, size_t csrColIndA,
        size_t B, int ldb, size_t beta, size_t C, int ldc) except *:
    _setStream(handle)
    status = cusparseScsrmm2(
        <Handle>handle, <Operation>transA, <Operation>transB, m, n, k, nnz,
        <const float *>alpha, <MatDescr>descrA, <const float *>csrValA,
        <const int *>csrRowPtrA, <const int *>csrColIndA,
        <const float *>B, ldb, <const float *>beta, <float *>C, ldc)
    check_status(status)

cpdef void dcsrmm2(
        intptr_t handle, int transA, int transB, int m, int n, int k, int nnz,
        size_t alpha, size_t descrA, size_t csrValA,
        size_t csrRowPtrA, size_t csrColIndA,
        size_t B, int ldb, size_t beta, size_t C, int ldc) except *:
    _setStream(handle)
    status = cusparseDcsrmm2(
        <Handle>handle, <Operation>transA, <Operation>transB, m, n, k, nnz,
        <const double *>alpha, <MatDescr>descrA, <const double *>csrValA,
        <const int *>csrRowPtrA, <const int *>csrColIndA,
        <const double *>B, ldb, <const double *>beta, <double *>C, ldc)
    check_status(status)

cpdef void ccsrmm2(
        intptr_t handle, int transA, int transB, int m, int n, int k, int nnz,
        size_t alpha, size_t descrA, size_t csrValA,
        size_t csrRowPtrA, size_t csrColIndA,
        size_t B, int ldb, size_t beta, size_t C, int ldc) except *:
    _setStream(handle)
    status = cusparseCcsrmm2(
        <Handle>handle, <Operation>transA, <Operation>transB, m, n, k, nnz,
        <const cuComplex *>alpha, <MatDescr>descrA, <const cuComplex *>csrValA,
        <const int *>csrRowPtrA, <const int *>csrColIndA,
        <const cuComplex *>B, ldb, <const cuComplex *>beta,
        <cuComplex *>C, ldc)
    check_status(status)

cpdef void zcsrmm2(
        intptr_t handle, int transA, int transB, int m, int n, int k, int nnz,
        size_t alpha, size_t descrA, size_t csrValA,
        size_t csrRowPtrA, size_t csrColIndA,
        size_t B, int ldb, size_t beta, size_t C, int ldc) except *:
    _setStream(handle)
    status = cusparseZcsrmm2(
        <Handle>handle, <Operation>transA, <Operation>transB, m, n, k, nnz,
        <const cuDoubleComplex *>alpha, <MatDescr>descrA,
        <const cuDoubleComplex *>csrValA,
        <const int *>csrRowPtrA, <const int *>csrColIndA,
        <const cuDoubleComplex *>B, ldb,
        <const cuDoubleComplex *>beta, <cuDoubleComplex *>C, ldc)
    check_status(status)

cpdef size_t createCsrsm2Info() except? -1:
    cdef csrsm2Info_t info
    status = cusparseCreateCsrsm2Info(&info)
    check_status(status)
    return <size_t>info

cpdef void destroyCsrsm2Info(size_t info) except *:
    status = cusparseDestroyCsrsm2Info(<csrsm2Info_t>info)
    check_status(status)

cpdef size_t scsrsm2_bufferSizeExt(
        intptr_t handle, int algo, int transA, int transB, int m, int nrhs,
        int nnz, size_t alpha, size_t descrA, size_t csrSortedValA,
        size_t csrSortedRowPtrA, size_t csrSortedColIndA, size_t B, int ldb,
        size_t info, int policy) except? -1:
    cdef size_t bufferSize
    _setStream(handle)
    status = cusparseScsrsm2_bufferSizeExt(
        <Handle>handle, algo, <Operation>transA, <Operation>transB, m, nrhs,
        nnz, <const float*>alpha, <const MatDescr>descrA,
        <const float*>csrSortedValA, <const int*>csrSortedRowPtrA,
        <const int*>csrSortedColIndA, <float*>B, ldb,
        <csrsm2Info_t>info, <cusparseSolvePolicy_t>policy, &bufferSize)
    check_status(status)
    return bufferSize

cpdef size_t dcsrsm2_bufferSizeExt(
        intptr_t handle, int algo, int transA, int transB, int m, int nrhs,
        int nnz, size_t alpha, size_t descrA, size_t csrSortedValA,
        size_t csrSortedRowPtrA, size_t csrSortedColIndA, size_t B, int ldb,
        size_t info, int policy) except? -1:
    cdef size_t bufferSize
    _setStream(handle)
    status = cusparseDcsrsm2_bufferSizeExt(
        <Handle>handle, algo, <Operation>transA, <Operation>transB, m, nrhs,
        nnz, <const double*>alpha, <const MatDescr>descrA,
        <const double*>csrSortedValA, <const int*>csrSortedRowPtrA,
        <const int*>csrSortedColIndA, <double*>B, ldb,
        <csrsm2Info_t>info, <cusparseSolvePolicy_t>policy, &bufferSize)
    check_status(status)
    return bufferSize

cpdef size_t ccsrsm2_bufferSizeExt(
        intptr_t handle, int algo, int transA, int transB, int m, int nrhs,
        int nnz, size_t alpha, size_t descrA, size_t csrSortedValA,
        size_t csrSortedRowPtrA, size_t csrSortedColIndA, size_t B, int ldb,
        size_t info, int policy) except? -1:
    cdef size_t bufferSize
    _setStream(handle)
    status = cusparseCcsrsm2_bufferSizeExt(
        <Handle>handle, algo, <Operation>transA, <Operation>transB, m, nrhs,
        nnz, <const cuComplex*>alpha, <const MatDescr>descrA,
        <const cuComplex*>csrSortedValA, <const int*>csrSortedRowPtrA,
        <const int*>csrSortedColIndA, <cuComplex*>B, ldb,
        <csrsm2Info_t>info, <cusparseSolvePolicy_t>policy, &bufferSize)
    check_status(status)
    return bufferSize

cpdef size_t zcsrsm2_bufferSizeExt(
        intptr_t handle, int algo, int transA, int transB, int m, int nrhs,
        int nnz, size_t alpha, size_t descrA, size_t csrSortedValA,
        size_t csrSortedRowPtrA, size_t csrSortedColIndA, size_t B, int ldb,
        size_t info, int policy) except? -1:
    cdef size_t bufferSize
    _setStream(handle)
    status = cusparseZcsrsm2_bufferSizeExt(
        <Handle>handle, algo, <Operation>transA, <Operation>transB, m, nrhs,
        nnz, <const cuDoubleComplex*>alpha, <const MatDescr>descrA,
        <const cuDoubleComplex*>csrSortedValA, <const int*>csrSortedRowPtrA,
        <const int*>csrSortedColIndA, <cuDoubleComplex*>B, ldb,
        <csrsm2Info_t>info, <cusparseSolvePolicy_t>policy, &bufferSize)
    check_status(status)
    return bufferSize

cpdef void scsrsm2_analysis(
        intptr_t handle, int algo, int transA, int transB, int m, int nrhs,
        int nnz, size_t alpha, size_t descrA, size_t csrSortedValA,
        size_t csrSortedRowPtrA, size_t csrSortedColIndA, size_t B, int ldb,
        size_t info, int policy, size_t pBuffer) except *:
    _setStream(handle)
    status = cusparseScsrsm2_analysis(
        <Handle>handle, algo, <Operation>transA, <Operation>transB, m, nrhs,
        nnz, <const float*>alpha, <const MatDescr>descrA,
        <const float*>csrSortedValA, <const int*>csrSortedRowPtrA,
        <const int*>csrSortedColIndA, <float*>B, ldb,
        <csrsm2Info_t>info, <cusparseSolvePolicy_t>policy, <void*>pBuffer)
    check_status(status)

cpdef void dcsrsm2_analysis(
        intptr_t handle, int algo, int transA, int transB, int m, int nrhs,
        int nnz, size_t alpha, size_t descrA, size_t csrSortedValA,
        size_t csrSortedRowPtrA, size_t csrSortedColIndA, size_t B, int ldb,
        size_t info, int policy, size_t pBuffer) except *:
    _setStream(handle)
    status = cusparseDcsrsm2_analysis(
        <Handle>handle, algo, <Operation>transA, <Operation>transB, m, nrhs,
        nnz, <const double*>alpha, <const MatDescr>descrA,
        <const double*>csrSortedValA, <const int*>csrSortedRowPtrA,
        <const int*>csrSortedColIndA, <double*>B, ldb,
        <csrsm2Info_t>info, <cusparseSolvePolicy_t>policy, <void*>pBuffer)
    check_status(status)

cpdef void ccsrsm2_analysis(
        intptr_t handle, int algo, int transA, int transB, int m, int nrhs,
        int nnz, size_t alpha, size_t descrA, size_t csrSortedValA,
        size_t csrSortedRowPtrA, size_t csrSortedColIndA, size_t B, int ldb,
        size_t info, int policy, size_t pBuffer) except *:
    _setStream(handle)
    status = cusparseCcsrsm2_analysis(
        <Handle>handle, algo, <Operation>transA, <Operation>transB, m, nrhs,
        nnz, <const cuComplex*>alpha, <const MatDescr>descrA,
        <const cuComplex*>csrSortedValA, <const int*>csrSortedRowPtrA,
        <const int*>csrSortedColIndA, <cuComplex*>B, ldb,
        <csrsm2Info_t>info, <cusparseSolvePolicy_t>policy, <void*>pBuffer)
    check_status(status)

cpdef void zcsrsm2_analysis(
        intptr_t handle, int algo, int transA, int transB, int m, int nrhs,
        int nnz, size_t alpha, size_t descrA, size_t csrSortedValA,
        size_t csrSortedRowPtrA, size_t csrSortedColIndA, size_t B, int ldb,
        size_t info, int policy, size_t pBuffer) except *:
    _setStream(handle)
    status = cusparseZcsrsm2_analysis(
        <Handle>handle, algo, <Operation>transA, <Operation>transB, m, nrhs,
        nnz, <const cuDoubleComplex*>alpha, <const MatDescr>descrA,
        <const cuDoubleComplex*>csrSortedValA, <const int*>csrSortedRowPtrA,
        <const int*>csrSortedColIndA, <cuDoubleComplex*>B, ldb,
        <csrsm2Info_t>info, <cusparseSolvePolicy_t>policy, <void*>pBuffer)
    check_status(status)

cpdef void scsrsm2_solve(
        intptr_t handle, int algo, int transA, int transB, int m, int nrhs,
        int nnz, size_t alpha, size_t descrA, size_t csrSortedValA,
        size_t csrSortedRowPtrA, size_t csrSortedColIndA, size_t B, int ldb,
        size_t info, int policy, size_t pBuffer) except *:
    _setStream(handle)
    status = cusparseScsrsm2_solve(
        <Handle>handle, algo, <Operation>transA, <Operation>transB, m, nrhs,
        nnz, <const float*>alpha, <const MatDescr>descrA,
        <const float*>csrSortedValA, <const int*>csrSortedRowPtrA,
        <const int*>csrSortedColIndA, <float*>B, ldb,
        <csrsm2Info_t>info, <cusparseSolvePolicy_t>policy, <void*>pBuffer)
    check_status(status)

cpdef void dcsrsm2_solve(
        intptr_t handle, int algo, int transA, int transB, int m, int nrhs,
        int nnz, size_t alpha, size_t descrA, size_t csrSortedValA,
        size_t csrSortedRowPtrA, size_t csrSortedColIndA, size_t B, int ldb,
        size_t info, int policy, size_t pBuffer) except *:
    _setStream(handle)
    status = cusparseDcsrsm2_solve(
        <Handle>handle, algo, <Operation>transA, <Operation>transB, m, nrhs,
        nnz, <const double*>alpha, <const MatDescr>descrA,
        <const double*>csrSortedValA, <const int*>csrSortedRowPtrA,
        <const int*>csrSortedColIndA, <double*>B, ldb,
        <csrsm2Info_t>info, <cusparseSolvePolicy_t>policy, <void*>pBuffer)
    check_status(status)

cpdef void ccsrsm2_solve(
        intptr_t handle, int algo, int transA, int transB, int m, int nrhs,
        int nnz, size_t alpha, size_t descrA, size_t csrSortedValA,
        size_t csrSortedRowPtrA, size_t csrSortedColIndA, size_t B, int ldb,
        size_t info, int policy, size_t pBuffer) except *:
    _setStream(handle)
    status = cusparseCcsrsm2_solve(
        <Handle>handle, algo, <Operation>transA, <Operation>transB, m, nrhs,
        nnz, <const cuComplex*>alpha, <const MatDescr>descrA,
        <const cuComplex*>csrSortedValA, <const int*>csrSortedRowPtrA,
        <const int*>csrSortedColIndA, <cuComplex*>B, ldb,
        <csrsm2Info_t>info, <cusparseSolvePolicy_t>policy, <void*>pBuffer)
    check_status(status)

cpdef void zcsrsm2_solve(
        intptr_t handle, int algo, int transA, int transB, int m, int nrhs,
        int nnz, size_t alpha, size_t descrA, size_t csrSortedValA,
        size_t csrSortedRowPtrA, size_t csrSortedColIndA, size_t B, int ldb,
        size_t info, int policy, size_t pBuffer) except *:
    _setStream(handle)
    status = cusparseZcsrsm2_solve(
        <Handle>handle, algo, <Operation>transA, <Operation>transB, m, nrhs,
        nnz, <const cuDoubleComplex*>alpha, <const MatDescr>descrA,
        <const cuDoubleComplex*>csrSortedValA, <const int*>csrSortedRowPtrA,
        <const int*>csrSortedColIndA, <cuDoubleComplex*>B, ldb,
        <csrsm2Info_t>info, <cusparseSolvePolicy_t>policy, <void*>pBuffer)
    check_status(status)

cpdef void xcsrsm2_zeroPivot(
        intptr_t handle, size_t info, size_t position) except *:
    _setStream(handle)
    status = cusparseXcsrsm2_zeroPivot(
        <Handle>handle, <csrsm2Info_t>info, <int*>position)
    check_status(status)

########################################
# cuSPARSE Extra Function

cpdef void xcsrgeamNnz(
        intptr_t handle, int m, int n, size_t descrA, int nnzA,
        size_t csrRowPtrA, size_t csrColIndA, size_t descrB,
        int nnzB, size_t csrRowPtrB, size_t csrColIndB,
        size_t descrC, size_t csrRowPtrC, size_t nnzTotalDevHostPtr) except *:
    _setStream(handle)
    status = cusparseXcsrgeamNnz(
        <Handle>handle, m, n, <const MatDescr>descrA, nnzA,
        <const int *>csrRowPtrA, <const int *>csrColIndA,
        <const MatDescr>descrB, nnzB, <const int *>csrRowPtrB,
        <const int *>csrColIndB, <const MatDescr>descrC, <int *>csrRowPtrC,
        <int *>nnzTotalDevHostPtr)
    check_status(status)

cpdef void scsrgeam(
        intptr_t handle, int m, int n, size_t alpha, size_t descrA,
        int nnzA, size_t csrValA, size_t csrRowPtrA,
        size_t csrColIndA, size_t beta, size_t descrB,
        int nnzB, size_t csrValB, size_t csrRowPtrB,
        size_t csrColIndB, size_t descrC, size_t csrValC,
        size_t csrRowPtrC, size_t csrColIndC) except *:
    _setStream(handle)
    status = cusparseScsrgeam(
        <Handle>handle, m, n, <const float *>alpha,
        <const MatDescr>descrA, nnzA, <const float *>csrValA,
        <const int *>csrRowPtrA, <const int *>csrColIndA, <const float *>beta,
        <const MatDescr>descrB, nnzB, <const float *>csrValB,
        <const int *>csrRowPtrB, <const int *>csrColIndB,
        <const MatDescr>descrC, <float *>csrValC, <int *>csrRowPtrC,
        <int *>csrColIndC)
    check_status(status)

cpdef void dcsrgeam(
        intptr_t handle, int m, int n, size_t alpha, size_t descrA,
        int nnzA, size_t csrValA, size_t csrRowPtrA,
        size_t csrColIndA, size_t beta, size_t descrB,
        int nnzB, size_t csrValB, size_t csrRowPtrB,
        size_t csrColIndB, size_t descrC, size_t csrValC,
        size_t csrRowPtrC, size_t csrColIndC) except *:
    _setStream(handle)
    status = cusparseDcsrgeam(
        <Handle>handle, m, n, <const double *>alpha,
        <const MatDescr>descrA, nnzA, <const double *>csrValA,
        <const int *>csrRowPtrA, <const int *>csrColIndA, <const double *>beta,
        <const MatDescr>descrB, nnzB, <const double *>csrValB,
        <const int *>csrRowPtrB, <const int *>csrColIndB,
        <const MatDescr>descrC, <double *>csrValC, <int *>csrRowPtrC,
        <int *>csrColIndC)
    check_status(status)

cpdef void ccsrgeam(
        intptr_t handle, int m, int n, size_t alpha, size_t descrA,
        int nnzA, size_t csrValA, size_t csrRowPtrA,
        size_t csrColIndA, size_t beta, size_t descrB,
        int nnzB, size_t csrValB, size_t csrRowPtrB,
        size_t csrColIndB, size_t descrC, size_t csrValC,
        size_t csrRowPtrC, size_t csrColIndC) except *:
    _setStream(handle)
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

cpdef void zcsrgeam(
        intptr_t handle, int m, int n, size_t alpha, size_t descrA,
        int nnzA, size_t csrValA, size_t csrRowPtrA,
        size_t csrColIndA, size_t beta, size_t descrB,
        int nnzB, size_t csrValB, size_t csrRowPtrB,
        size_t csrColIndB, size_t descrC, size_t csrValC,
        size_t csrRowPtrC, size_t csrColIndC) except *:
    _setStream(handle)
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

cpdef size_t scsrgeam2_bufferSizeExt(
        intptr_t handle, int m, int n, size_t alpha, size_t descrA,
        int nnzA, size_t csrValA, size_t csrRowPtrA,
        size_t csrColIndA, size_t beta, size_t descrB,
        int nnzB, size_t csrValB, size_t csrRowPtrB,
        size_t csrColIndB, size_t descrC, size_t csrValC,
        size_t csrRowPtrC, size_t csrColIndC) except? -1:
    cdef size_t bufferSize
    _setStream(handle)
    status = cusparseScsrgeam2_bufferSizeExt(
        <Handle>handle, m, n, <const float *>alpha,
        <const MatDescr>descrA, nnzA, <const float *>csrValA,
        <const int *>csrRowPtrA, <const int *>csrColIndA, <const float *>beta,
        <const MatDescr>descrB, nnzB, <const float *>csrValB,
        <const int *>csrRowPtrB, <const int *>csrColIndB,
        <const MatDescr>descrC, <float *>csrValC, <int *>csrRowPtrC,
        <int *>csrColIndC, &bufferSize)
    check_status(status)
    return bufferSize

cpdef size_t dcsrgeam2_bufferSizeExt(
        intptr_t handle, int m, int n, size_t alpha, size_t descrA,
        int nnzA, size_t csrValA, size_t csrRowPtrA,
        size_t csrColIndA, size_t beta, size_t descrB,
        int nnzB, size_t csrValB, size_t csrRowPtrB,
        size_t csrColIndB, size_t descrC, size_t csrValC,
        size_t csrRowPtrC, size_t csrColIndC) except? -1:
    cdef size_t bufferSize
    _setStream(handle)
    status = cusparseDcsrgeam2_bufferSizeExt(
        <Handle>handle, m, n, <const double *>alpha,
        <const MatDescr>descrA, nnzA, <const double *>csrValA,
        <const int *>csrRowPtrA, <const int *>csrColIndA, <const double *>beta,
        <const MatDescr>descrB, nnzB, <const double *>csrValB,
        <const int *>csrRowPtrB, <const int *>csrColIndB,
        <const MatDescr>descrC, <double *>csrValC, <int *>csrRowPtrC,
        <int *>csrColIndC, &bufferSize)
    check_status(status)
    return bufferSize

cpdef size_t ccsrgeam2_bufferSizeExt(
        intptr_t handle, int m, int n, size_t alpha, size_t descrA,
        int nnzA, size_t csrValA, size_t csrRowPtrA,
        size_t csrColIndA, size_t beta, size_t descrB,
        int nnzB, size_t csrValB, size_t csrRowPtrB,
        size_t csrColIndB, size_t descrC, size_t csrValC,
        size_t csrRowPtrC, size_t csrColIndC) except? -1:
    cdef size_t bufferSize
    _setStream(handle)
    status = cusparseCcsrgeam2_bufferSizeExt(
        <Handle>handle, m, n, <const cuComplex *>alpha,
        <const MatDescr>descrA, nnzA, <const cuComplex *>csrValA,
        <const int *>csrRowPtrA, <const int *>csrColIndA,
        <const cuComplex *>beta,
        <const MatDescr>descrB, nnzB, <const cuComplex *>csrValB,
        <const int *>csrRowPtrB, <const int *>csrColIndB,
        <const MatDescr>descrC, <cuComplex *>csrValC, <int *>csrRowPtrC,
        <int *>csrColIndC, &bufferSize)
    check_status(status)
    return bufferSize

cpdef size_t zcsrgeam2_bufferSizeExt(
        intptr_t handle, int m, int n, size_t alpha, size_t descrA,
        int nnzA, size_t csrValA, size_t csrRowPtrA,
        size_t csrColIndA, size_t beta, size_t descrB,
        int nnzB, size_t csrValB, size_t csrRowPtrB,
        size_t csrColIndB, size_t descrC, size_t csrValC,
        size_t csrRowPtrC, size_t csrColIndC) except? -1:
    cdef size_t bufferSize
    _setStream(handle)
    status = cusparseZcsrgeam2_bufferSizeExt(
        <Handle>handle, m, n, <const cuDoubleComplex *>alpha,
        <const MatDescr>descrA, nnzA, <const cuDoubleComplex *>csrValA,
        <const int *>csrRowPtrA, <const int *>csrColIndA,
        <const cuDoubleComplex *>beta,
        <const MatDescr>descrB, nnzB, <const cuDoubleComplex *>csrValB,
        <const int *>csrRowPtrB, <const int *>csrColIndB,
        <const MatDescr>descrC, <cuDoubleComplex *>csrValC, <int *>csrRowPtrC,
        <int *>csrColIndC, &bufferSize)
    check_status(status)
    return bufferSize

cpdef void xcsrgeam2Nnz(
        intptr_t handle, int m, int n, size_t descrA, int nnzA,
        size_t csrRowPtrA, size_t csrColIndA, size_t descrB,
        int nnzB, size_t csrRowPtrB, size_t csrColIndB,
        size_t descrC, size_t csrRowPtrC, size_t nnzTotalDevHostPtr,
        size_t workspace) except *:
    _setStream(handle)
    status = cusparseXcsrgeam2Nnz(
        <Handle>handle, m, n, <const MatDescr>descrA, nnzA,
        <const int *>csrRowPtrA, <const int *>csrColIndA,
        <const MatDescr>descrB, nnzB, <const int *>csrRowPtrB,
        <const int *>csrColIndB, <const MatDescr>descrC, <int *>csrRowPtrC,
        <int *>nnzTotalDevHostPtr, <void*> workspace)
    check_status(status)

cpdef void scsrgeam2(
        intptr_t handle, int m, int n, size_t alpha, size_t descrA,
        int nnzA, size_t csrValA, size_t csrRowPtrA,
        size_t csrColIndA, size_t beta, size_t descrB,
        int nnzB, size_t csrValB, size_t csrRowPtrB,
        size_t csrColIndB, size_t descrC, size_t csrValC,
        size_t csrRowPtrC, size_t csrColIndC, size_t buffer) except *:
    _setStream(handle)
    status = cusparseScsrgeam2(
        <Handle>handle, m, n, <const float *>alpha,
        <const MatDescr>descrA, nnzA, <const float *>csrValA,
        <const int *>csrRowPtrA, <const int *>csrColIndA, <const float *>beta,
        <const MatDescr>descrB, nnzB, <const float *>csrValB,
        <const int *>csrRowPtrB, <const int *>csrColIndB,
        <const MatDescr>descrC, <float *>csrValC, <int *>csrRowPtrC,
        <int *>csrColIndC, <void*>buffer)
    check_status(status)

cpdef void dcsrgeam2(
        intptr_t handle, int m, int n, size_t alpha, size_t descrA,
        int nnzA, size_t csrValA, size_t csrRowPtrA,
        size_t csrColIndA, size_t beta, size_t descrB,
        int nnzB, size_t csrValB, size_t csrRowPtrB,
        size_t csrColIndB, size_t descrC, size_t csrValC,
        size_t csrRowPtrC, size_t csrColIndC, size_t buffer) except *:
    _setStream(handle)
    status = cusparseDcsrgeam2(
        <Handle>handle, m, n, <const double *>alpha,
        <const MatDescr>descrA, nnzA, <const double *>csrValA,
        <const int *>csrRowPtrA, <const int *>csrColIndA, <const double *>beta,
        <const MatDescr>descrB, nnzB, <const double *>csrValB,
        <const int *>csrRowPtrB, <const int *>csrColIndB,
        <const MatDescr>descrC, <double *>csrValC, <int *>csrRowPtrC,
        <int *>csrColIndC, <void*>buffer)
    check_status(status)

cpdef void ccsrgeam2(
        intptr_t handle, int m, int n, size_t alpha, size_t descrA,
        int nnzA, size_t csrValA, size_t csrRowPtrA,
        size_t csrColIndA, size_t beta, size_t descrB,
        int nnzB, size_t csrValB, size_t csrRowPtrB,
        size_t csrColIndB, size_t descrC, size_t csrValC,
        size_t csrRowPtrC, size_t csrColIndC, size_t buffer) except *:
    _setStream(handle)
    status = cusparseCcsrgeam2(
        <Handle>handle, m, n, <const cuComplex *>alpha,
        <const MatDescr>descrA, nnzA, <const cuComplex *>csrValA,
        <const int *>csrRowPtrA, <const int *>csrColIndA,
        <const cuComplex *>beta,
        <const MatDescr>descrB, nnzB, <const cuComplex *>csrValB,
        <const int *>csrRowPtrB, <const int *>csrColIndB,
        <const MatDescr>descrC, <cuComplex *>csrValC, <int *>csrRowPtrC,
        <int *>csrColIndC, <void*>buffer)
    check_status(status)

cpdef void zcsrgeam2(
        intptr_t handle, int m, int n, size_t alpha, size_t descrA,
        int nnzA, size_t csrValA, size_t csrRowPtrA,
        size_t csrColIndA, size_t beta, size_t descrB,
        int nnzB, size_t csrValB, size_t csrRowPtrB,
        size_t csrColIndB, size_t descrC, size_t csrValC,
        size_t csrRowPtrC, size_t csrColIndC, size_t buffer) except *:
    _setStream(handle)
    status = cusparseZcsrgeam2(
        <Handle>handle, m, n, <const cuDoubleComplex *>alpha,
        <const MatDescr>descrA, nnzA, <const cuDoubleComplex *>csrValA,
        <const int *>csrRowPtrA, <const int *>csrColIndA,
        <const cuDoubleComplex *>beta,
        <const MatDescr>descrB, nnzB, <const cuDoubleComplex *>csrValB,
        <const int *>csrRowPtrB, <const int *>csrColIndB,
        <const MatDescr>descrC, <cuDoubleComplex *>csrValC, <int *>csrRowPtrC,
        <int *>csrColIndC, <void*>buffer)
    check_status(status)

cpdef void xcsrgemmNnz(
        intptr_t handle, int transA, int transB, int m, int n, int k,
        size_t descrA, int nnzA, size_t csrRowPtrA,
        size_t csrColIndA, size_t descrB, int nnzB,
        size_t csrRowPtrB, size_t csrColIndB,
        size_t descrC, size_t csrRowPtrC, size_t nnzTotalDevHostPtr) except *:
    _setStream(handle)
    status = cusparseXcsrgemmNnz(
        <Handle>handle, <Operation>transA, <Operation>transB, m, n, k,
        <const MatDescr>descrA, nnzA, <const int *>csrRowPtrA,
        <const int *>csrColIndA, <const MatDescr>descrB, nnzB,
        <const int *>csrRowPtrB, <const int *>csrColIndB,
        <const MatDescr>descrC, <int *>csrRowPtrC, <int *>nnzTotalDevHostPtr)
    check_status(status)

cpdef void scsrgemm(
        intptr_t handle, int transA, int transB, int m, int n, int k,
        size_t descrA, const int nnzA, size_t csrValA,
        size_t csrRowPtrA, size_t csrColIndA, size_t descrB,
        const int nnzB, size_t csrValB, size_t csrRowPtrB,
        size_t csrColIndB, size_t descrC, size_t csrValC,
        size_t csrRowPtrC, size_t csrColIndC) except *:
    _setStream(handle)
    status = cusparseScsrgemm(
        <Handle>handle, <Operation>transA, <Operation>transB, m, n, k,
        <const MatDescr>descrA, nnzA, <const float *>csrValA,
        <const int *>csrRowPtrA, <const int *>csrColIndA,
        <const MatDescr>descrB, nnzB, <const float *>csrValB,
        <const int *>csrRowPtrB, <const int *>csrColIndB,
        <const MatDescr>descrC, <float *>csrValC,
        <const int *>csrRowPtrC, <int *>csrColIndC)
    check_status(status)

cpdef void dcsrgemm(
        intptr_t handle, int transA, int transB, int m, int n, int k,
        size_t descrA, const int nnzA, size_t csrValA,
        size_t csrRowPtrA, size_t csrColIndA, size_t descrB,
        const int nnzB, size_t csrValB, size_t csrRowPtrB,
        size_t csrColIndB, size_t descrC, size_t csrValC,
        size_t csrRowPtrC, size_t csrColIndC) except *:
    _setStream(handle)
    status = cusparseDcsrgemm(
        <Handle>handle, <Operation>transA, <Operation>transB, m, n, k,
        <const MatDescr>descrA, nnzA, <const double *>csrValA,
        <const int *>csrRowPtrA, <const int *>csrColIndA,
        <const MatDescr>descrB, nnzB, <const double *>csrValB,
        <const int *>csrRowPtrB, <const int *>csrColIndB,
        <const MatDescr>descrC, <double *>csrValC,
        <const int *>csrRowPtrC, <int *>csrColIndC)
    check_status(status)

cpdef void ccsrgemm(
        intptr_t handle, int transA, int transB, int m, int n, int k,
        size_t descrA, const int nnzA, size_t csrValA,
        size_t csrRowPtrA, size_t csrColIndA, size_t descrB,
        const int nnzB, size_t csrValB, size_t csrRowPtrB,
        size_t csrColIndB, size_t descrC, size_t csrValC,
        size_t csrRowPtrC, size_t csrColIndC) except *:
    _setStream(handle)
    status = cusparseCcsrgemm(
        <Handle>handle, <Operation>transA, <Operation>transB, m, n, k,
        <const MatDescr>descrA, nnzA, <const cuComplex *>csrValA,
        <const int *>csrRowPtrA, <const int *>csrColIndA,
        <const MatDescr>descrB, nnzB, <const cuComplex *>csrValB,
        <const int *>csrRowPtrB, <const int *>csrColIndB,
        <const MatDescr>descrC, <cuComplex *>csrValC,
        <const int *>csrRowPtrC, <int *>csrColIndC)
    check_status(status)

cpdef void zcsrgemm(
        intptr_t handle, int transA, int transB, int m, int n, int k,
        size_t descrA, const int nnzA, size_t csrValA,
        size_t csrRowPtrA, size_t csrColIndA, size_t descrB,
        const int nnzB, size_t csrValB, size_t csrRowPtrB,
        size_t csrColIndB, size_t descrC, size_t csrValC,
        size_t csrRowPtrC, size_t csrColIndC) except *:
    _setStream(handle)
    status = cusparseZcsrgemm(
        <Handle>handle, <Operation>transA, <Operation>transB, m, n, k,
        <const MatDescr>descrA, nnzA, <const cuDoubleComplex *>csrValA,
        <const int *>csrRowPtrA, <const int *>csrColIndA,
        <const MatDescr>descrB, nnzB, <const cuDoubleComplex *>csrValB,
        <const int *>csrRowPtrB, <const int *>csrColIndB,
        <const MatDescr>descrC, <cuDoubleComplex *>csrValC,
        <const int *>csrRowPtrC, <int *>csrColIndC)
    check_status(status)

cpdef size_t createCsrgemm2Info() except? -1:
    cdef csrgemm2Info_t info
    with nogil:
        status = cusparseCreateCsrgemm2Info(&info)
    check_status(status)
    return <size_t>info

cpdef void destroyCsrgemm2Info(size_t info) except *:
    with nogil:
        status = cusparseDestroyCsrgemm2Info(<csrgemm2Info_t>info)
    check_status(status)

cpdef size_t scsrgemm2_bufferSizeExt(
        intptr_t handle, int m, int n, int k,
        size_t alpha,
        size_t descrA, int nnzA, size_t csrRowPtrA, size_t csrColIndA,
        size_t descrB, int nnzB, size_t csrRowPtrB, size_t csrColIndB,
        size_t beta,
        size_t descrD, int nnzD, size_t csrRowPtrD, size_t csrColIndD,
        size_t info) except? -1:
    cdef size_t bufferSize
    status = cusparseScsrgemm2_bufferSizeExt(
        <Handle>handle, m, n, k,
        <float*>alpha,
        <MatDescr>descrA, nnzA, <int*>csrRowPtrA, <int*>csrColIndA,
        <MatDescr>descrB, nnzB, <int*>csrRowPtrB, <int*>csrColIndB,
        <float*>beta,
        <MatDescr>descrD, nnzD, <int*>csrRowPtrD, <int*>csrColIndD,
        <csrgemm2Info_t>info, &bufferSize)
    check_status(status)
    return bufferSize

cpdef size_t dcsrgemm2_bufferSizeExt(
        intptr_t handle, int m, int n, int k,
        size_t alpha,
        size_t descrA, int nnzA, size_t csrRowPtrA, size_t csrColIndA,
        size_t descrB, int nnzB, size_t csrRowPtrB, size_t csrColIndB,
        size_t beta,
        size_t descrD, int nnzD, size_t csrRowPtrD, size_t csrColIndD,
        size_t info) except? -1:
    cdef size_t bufferSize
    status = cusparseDcsrgemm2_bufferSizeExt(
        <Handle>handle, m, n, k,
        <double*>alpha,
        <MatDescr>descrA, nnzA, <int*>csrRowPtrA, <int*>csrColIndA,
        <MatDescr>descrB, nnzB, <int*>csrRowPtrB, <int*>csrColIndB,
        <double*>beta,
        <MatDescr>descrD, nnzD, <int*>csrRowPtrD, <int*>csrColIndD,
        <csrgemm2Info_t>info, &bufferSize)
    check_status(status)
    return bufferSize

cpdef size_t ccsrgemm2_bufferSizeExt(
        intptr_t handle, int m, int n, int k,
        size_t alpha,
        size_t descrA, int nnzA, size_t csrRowPtrA, size_t csrColIndA,
        size_t descrB, int nnzB, size_t csrRowPtrB, size_t csrColIndB,
        size_t beta,
        size_t descrD, int nnzD, size_t csrRowPtrD, size_t csrColIndD,
        size_t info) except? -1:
    cdef size_t bufferSize
    status = cusparseCcsrgemm2_bufferSizeExt(
        <Handle>handle, m, n, k,
        <cuComplex*>alpha,
        <MatDescr>descrA, nnzA, <int*>csrRowPtrA, <int*>csrColIndA,
        <MatDescr>descrB, nnzB, <int*>csrRowPtrB, <int*>csrColIndB,
        <cuComplex*>beta,
        <MatDescr>descrD, nnzD, <int*>csrRowPtrD, <int*>csrColIndD,
        <csrgemm2Info_t>info, &bufferSize)
    check_status(status)
    return bufferSize

cpdef size_t zcsrgemm2_bufferSizeExt(
        intptr_t handle, int m, int n, int k,
        size_t alpha,
        size_t descrA, int nnzA, size_t csrRowPtrA, size_t csrColIndA,
        size_t descrB, int nnzB, size_t csrRowPtrB, size_t csrColIndB,
        size_t beta,
        size_t descrD, int nnzD, size_t csrRowPtrD, size_t csrColIndD,
        size_t info) except? -1:
    cdef size_t bufferSize
    status = cusparseZcsrgemm2_bufferSizeExt(
        <Handle>handle, m, n, k,
        <cuDoubleComplex*>alpha,
        <MatDescr>descrA, nnzA, <int*>csrRowPtrA, <int*>csrColIndA,
        <MatDescr>descrB, nnzB, <int*>csrRowPtrB, <int*>csrColIndB,
        <cuDoubleComplex*>beta,
        <MatDescr>descrD, nnzD, <int*>csrRowPtrD, <int*>csrColIndD,
        <csrgemm2Info_t>info, &bufferSize)
    check_status(status)
    return bufferSize

cpdef void xcsrgemm2Nnz(
        intptr_t handle, int m, int n, int k,
        size_t descrA, int nnzA, size_t csrRowPtrA, size_t csrColIndA,
        size_t descrB, int nnzB, size_t csrRowPtrB, size_t csrColIndB,
        size_t descrD, int nnzD, size_t csrRowPtrD, size_t csrColIndD,
        size_t descrC, size_t csrRowPtrC,
        intptr_t nnzTotalDevHostPtr, size_t info, intptr_t pBuffer) except *:
    _setStream(handle)
    status = cusparseXcsrgemm2Nnz(
        <Handle>handle, m, n, k,
        <MatDescr>descrA, nnzA, <int*>csrRowPtrA, <int*>csrColIndA,
        <MatDescr>descrB, nnzB, <int*>csrRowPtrB, <int*>csrColIndB,
        <MatDescr>descrD, nnzD, <int*>csrRowPtrD, <int*>csrColIndD,
        <MatDescr>descrC, <int*>csrRowPtrC,
        <int*>nnzTotalDevHostPtr, <csrgemm2Info_t>info, <void*>pBuffer)
    check_status(status)

cpdef void scsrgemm2(
        intptr_t handle, int m, int n, int k, size_t alpha, size_t descrA,
        int nnzA, size_t csrValA, size_t csrRowPtrA, size_t csrColIndA,
        size_t descrB, int nnzB, size_t csrValB, size_t csrRowPtrB,
        size_t csrColIndB, size_t beta, size_t descrD, int nnzD,
        size_t csrValD, size_t csrRowPtrD, size_t csrColIndD, size_t descrC,
        size_t csrValC, size_t csrRowPtrC, size_t csrColIndC, size_t info,
        intptr_t pBuffer) except *:
    _setStream(handle)
    status = cusparseScsrgemm2(
        <Handle>handle, m, n, k, <float*>alpha, <MatDescr>descrA, nnzA,
        <float*>csrValA, <int*>csrRowPtrA, <int*>csrColIndA, <MatDescr>descrB,
        nnzB, <float*>csrValB, <int*>csrRowPtrB, <int*>csrColIndB,
        <float*>beta, <MatDescr>descrD, nnzD, <float*>csrValD,
        <int*>csrRowPtrD, <int*>csrColIndD, <MatDescr>descrC, <float*>csrValC,
        <int*>csrRowPtrC, <int*>csrColIndC, <csrgemm2Info_t>info,
        <void*>pBuffer)
    check_status(status)

cpdef void dcsrgemm2(
        intptr_t handle, int m, int n, int k, size_t alpha, size_t descrA,
        int nnzA, size_t csrValA, size_t csrRowPtrA, size_t csrColIndA,
        size_t descrB, int nnzB, size_t csrValB, size_t csrRowPtrB,
        size_t csrColIndB, size_t beta, size_t descrD, int nnzD,
        size_t csrValD, size_t csrRowPtrD, size_t csrColIndD, size_t descrC,
        size_t csrValC, size_t csrRowPtrC, size_t csrColIndC, size_t info,
        intptr_t pBuffer) except *:
    _setStream(handle)
    status = cusparseDcsrgemm2(
        <Handle>handle, m, n, k, <double*>alpha, <MatDescr>descrA, nnzA,
        <double*>csrValA, <int*>csrRowPtrA, <int*>csrColIndA, <MatDescr>descrB,
        nnzB, <double*>csrValB, <int*>csrRowPtrB, <int*>csrColIndB,
        <double*>beta, <MatDescr>descrD, nnzD, <double*>csrValD,
        <int*>csrRowPtrD, <int*>csrColIndD, <MatDescr>descrC, <double*>csrValC,
        <int*>csrRowPtrC, <int*>csrColIndC, <csrgemm2Info_t>info,
        <void*>pBuffer)
    check_status(status)

cpdef void ccsrgemm2(
        intptr_t handle, int m, int n, int k, size_t alpha, size_t descrA,
        int nnzA, size_t csrValA, size_t csrRowPtrA, size_t csrColIndA,
        size_t descrB, int nnzB, size_t csrValB, size_t csrRowPtrB,
        size_t csrColIndB, size_t beta, size_t descrD, int nnzD,
        size_t csrValD, size_t csrRowPtrD, size_t csrColIndD, size_t descrC,
        size_t csrValC, size_t csrRowPtrC, size_t csrColIndC, size_t info,
        intptr_t pBuffer) except *:
    _setStream(handle)
    status = cusparseCcsrgemm2(
        <Handle>handle, m, n, k, <cuComplex*>alpha, <MatDescr>descrA, nnzA,
        <cuComplex*>csrValA, <int*>csrRowPtrA, <int*>csrColIndA,
        <MatDescr>descrB, nnzB, <cuComplex*>csrValB, <int*>csrRowPtrB,
        <int*>csrColIndB, <cuComplex*>beta, <MatDescr>descrD, nnzD,
        <cuComplex*>csrValD, <int*>csrRowPtrD, <int*>csrColIndD,
        <MatDescr>descrC, <cuComplex*>csrValC, <int*>csrRowPtrC,
        <int*>csrColIndC, <csrgemm2Info_t>info, <void*>pBuffer)
    check_status(status)

cpdef void zcsrgemm2(
        intptr_t handle, int m, int n, int k, size_t alpha, size_t descrA,
        int nnzA, size_t csrValA, size_t csrRowPtrA, size_t csrColIndA,
        size_t descrB, int nnzB, size_t csrValB, size_t csrRowPtrB,
        size_t csrColIndB, size_t beta, size_t descrD, int nnzD,
        size_t csrValD, size_t csrRowPtrD, size_t csrColIndD, size_t descrC,
        size_t csrValC, size_t csrRowPtrC, size_t csrColIndC, size_t info,
        intptr_t pBuffer) except *:
    _setStream(handle)
    status = cusparseZcsrgemm2(
        <Handle>handle, m, n, k, <cuDoubleComplex*>alpha, <MatDescr>descrA,
        nnzA, <cuDoubleComplex*>csrValA, <int*>csrRowPtrA, <int*>csrColIndA,
        <MatDescr>descrB, nnzB, <cuDoubleComplex*>csrValB, <int*>csrRowPtrB,
        <int*>csrColIndB, <cuDoubleComplex*>beta, <MatDescr>descrD, nnzD,
        <cuDoubleComplex*>csrValD, <int*>csrRowPtrD, <int*>csrColIndD,
        <MatDescr>descrC, <cuDoubleComplex*>csrValC, <int*>csrRowPtrC,
        <int*>csrColIndC, <csrgemm2Info_t>info, <void*>pBuffer)
    check_status(status)

########################################
# cuSPARSE Format Convrsion

cpdef void xcoo2csr(
        intptr_t handle, size_t cooRowInd, int nnz, int m, size_t csrRowPtr,
        int idxBase) except *:
    _setStream(handle)
    status = cusparseXcoo2csr(
        <Handle>handle, <const int *>cooRowInd, nnz, m, <int *>csrRowPtr,
        <IndexBase>idxBase)
    check_status(status)


cpdef void scsc2dense(
        intptr_t handle, int m, int n, size_t descrA,
        size_t cscSortedValA, size_t cscSortedRowIndA,
        size_t cscSortedColPtrA, size_t A, int lda) except *:
    _setStream(handle)
    status = cusparseScsc2dense(
        <Handle>handle, m, n, <MatDescr>descrA,
        <const float *>cscSortedValA, <const int *>cscSortedRowIndA,
        <const int *>cscSortedColPtrA, <float *>A, lda)
    check_status(status)


cpdef void dcsc2dense(
        intptr_t handle, int m, int n, size_t descrA,
        size_t cscSortedValA, size_t cscSortedRowIndA,
        size_t cscSortedColPtrA, size_t A, int lda) except *:
    _setStream(handle)
    status = cusparseDcsc2dense(
        <Handle>handle, m, n, <MatDescr>descrA,
        <const double *>cscSortedValA, <const int *>cscSortedRowIndA,
        <const int *>cscSortedColPtrA, <double *>A, lda)
    check_status(status)

cpdef void ccsc2dense(
        intptr_t handle, int m, int n, size_t descrA,
        size_t cscSortedValA, size_t cscSortedRowIndA,
        size_t cscSortedColPtrA, size_t A, int lda) except *:
    _setStream(handle)
    status = cusparseCcsc2dense(
        <Handle>handle, m, n, <MatDescr>descrA,
        <const cuComplex *>cscSortedValA, <const int *>cscSortedRowIndA,
        <const int *>cscSortedColPtrA, <cuComplex *>A, lda)
    check_status(status)

cpdef void zcsc2dense(
        intptr_t handle, int m, int n, size_t descrA,
        size_t cscSortedValA, size_t cscSortedRowIndA,
        size_t cscSortedColPtrA, size_t A, int lda) except *:
    _setStream(handle)
    status = cusparseZcsc2dense(
        <Handle>handle, m, n, <MatDescr>descrA,
        <const cuDoubleComplex *>cscSortedValA, <const int *>cscSortedRowIndA,
        <const int *>cscSortedColPtrA, <cuDoubleComplex *>A, lda)
    check_status(status)

cpdef void xcsr2coo(
        intptr_t handle, size_t csrRowPtr, int nnz, int m, size_t cooRowInd,
        int idxBase) except *:
    _setStream(handle)
    status = cusparseXcsr2coo(
        <Handle>handle, <const int *>csrRowPtr, nnz, m, <int *>cooRowInd,
        <IndexBase>idxBase)
    check_status(status)


cpdef void scsr2csc(
        intptr_t handle, int m, int n, int nnz, size_t csrVal,
        size_t csrRowPtr, size_t csrColInd, size_t cscVal,
        size_t cscRowInd, size_t cscColPtr, int copyValues,
        int idxBase) except *:
    _setStream(handle)
    status = cusparseScsr2csc(
        <Handle>handle, m, n, nnz, <const float *>csrVal,
        <const int *>csrRowPtr, <const int *>csrColInd, <float *>cscVal,
        <int *>cscRowInd, <int *>cscColPtr, <Action>copyValues,
        <IndexBase>idxBase)
    check_status(status)


cpdef void dcsr2csc(
        intptr_t handle, int m, int n, int nnz, size_t csrVal,
        size_t csrRowPtr, size_t csrColInd, size_t cscVal,
        size_t cscRowInd, size_t cscColPtr, int copyValues,
        int idxBase) except *:
    _setStream(handle)
    status = cusparseDcsr2csc(
        <Handle>handle, m, n, nnz, <const double *>csrVal,
        <const int *>csrRowPtr, <const int *>csrColInd, <double *>cscVal,
        <int *>cscRowInd, <int *>cscColPtr, <Action>copyValues,
        <IndexBase>idxBase)
    check_status(status)

cpdef void ccsr2csc(
        intptr_t handle, int m, int n, int nnz, size_t csrVal,
        size_t csrRowPtr, size_t csrColInd, size_t cscVal,
        size_t cscRowInd, size_t cscColPtr, int copyValues,
        int idxBase) except *:
    _setStream(handle)
    status = cusparseCcsr2csc(
        <Handle>handle, m, n, nnz, <const cuComplex *>csrVal,
        <const int *>csrRowPtr, <const int *>csrColInd, <cuComplex *>cscVal,
        <int *>cscRowInd, <int *>cscColPtr, <Action>copyValues,
        <IndexBase>idxBase)
    check_status(status)

cpdef void zcsr2csc(
        intptr_t handle, int m, int n, int nnz, size_t csrVal,
        size_t csrRowPtr, size_t csrColInd, size_t cscVal,
        size_t cscRowInd, size_t cscColPtr, int copyValues,
        int idxBase) except *:
    _setStream(handle)
    status = cusparseZcsr2csc(
        <Handle>handle, m, n, nnz, <const cuDoubleComplex *>csrVal,
        <const int *>csrRowPtr, <const int *>csrColInd,
        <cuDoubleComplex *>cscVal,
        <int *>cscRowInd, <int *>cscColPtr, <Action>copyValues,
        <IndexBase>idxBase)
    check_status(status)

cpdef void scsr2dense(
        intptr_t handle, int m, int n, size_t descrA,
        size_t csrSortedValA, size_t csrSortedRowPtrA,
        size_t csrSortedColIndA, size_t A, int lda) except *:
    _setStream(handle)
    status = cusparseScsr2dense(
        <Handle>handle, m, n, <MatDescr>descrA,
        <const float *>csrSortedValA, <const int *>csrSortedRowPtrA,
        <const int *>csrSortedColIndA, <float *>A, lda)
    check_status(status)

cpdef void dcsr2dense(
        intptr_t handle, int m, int n, size_t descrA,
        size_t csrSortedValA, size_t csrSortedRowPtrA,
        size_t csrSortedColIndA, size_t A, int lda) except *:
    _setStream(handle)
    status = cusparseDcsr2dense(
        <Handle>handle, m, n, <MatDescr>descrA,
        <const double *>csrSortedValA, <const int *>csrSortedRowPtrA,
        <const int *>csrSortedColIndA, <double *>A, lda)
    check_status(status)

cpdef void ccsr2dense(
        intptr_t handle, int m, int n, size_t descrA,
        size_t csrSortedValA, size_t csrSortedRowPtrA,
        size_t csrSortedColIndA, size_t A, int lda) except *:
    _setStream(handle)
    status = cusparseCcsr2dense(
        <Handle>handle, m, n, <MatDescr>descrA,
        <const cuComplex *>csrSortedValA, <const int *>csrSortedRowPtrA,
        <const int *>csrSortedColIndA, <cuComplex *>A, lda)
    check_status(status)

cpdef void zcsr2dense(
        intptr_t handle, int m, int n, size_t descrA,
        size_t csrSortedValA, size_t csrSortedRowPtrA,
        size_t csrSortedColIndA, size_t A, int lda) except *:
    _setStream(handle)
    status = cusparseZcsr2dense(
        <Handle>handle, m, n, <MatDescr>descrA,
        <const cuDoubleComplex *>csrSortedValA, <const int *>csrSortedRowPtrA,
        <const int *>csrSortedColIndA, <cuDoubleComplex *>A, lda)
    check_status(status)

cpdef int snnz_compress(
        intptr_t handle, int m, size_t descr,
        size_t values, size_t rowPtr, size_t nnzPerRow,
        float tol) except? -1:
    cdef int nnz_total
    _setStream(handle)
    status = cusparseSnnz_compress(
        <Handle>handle, m, <const MatDescr>descr,
        <const float *>values, <const int *>rowPtr, <int *>nnzPerRow,
        &nnz_total, tol)
    check_status(status)
    return nnz_total

cpdef int dnnz_compress(
        intptr_t handle, int m, size_t descr,
        size_t values, size_t rowPtr, size_t nnzPerRow,
        double tol) except? -1:
    cdef int nnz_total
    _setStream(handle)
    status = cusparseDnnz_compress(
        <Handle>handle, m, <const MatDescr>descr,
        <const double *>values, <const int *>rowPtr, <int *>nnzPerRow,
        &nnz_total, tol)
    check_status(status)
    return nnz_total

cpdef int cnnz_compress(
        intptr_t handle, int m, size_t descr,
        size_t values, size_t rowPtr, size_t nnzPerRow,
        complex tol) except? -1:
    cdef int nnz_total
    _setStream(handle)
    status = cusparseCnnz_compress(
        <Handle>handle, m, <const MatDescr>descr,
        <const cuComplex *>values, <const int *>rowPtr, <int *>nnzPerRow,
        &nnz_total, complex_to_cuda(tol))
    check_status(status)
    return nnz_total

cpdef int znnz_compress(
        intptr_t handle, int m, size_t descr,
        size_t values, size_t rowPtr, size_t nnzPerRow,
        double complex tol) except? -1:
    cdef int nnz_total
    _setStream(handle)
    status = cusparseZnnz_compress(
        <Handle>handle, m, <const MatDescr>descr,
        <const cuDoubleComplex *>values, <const int *>rowPtr, <int *>nnzPerRow,
        &nnz_total, double_complex_to_cuda(tol))
    check_status(status)
    return nnz_total

cpdef void scsr2csr_compress(
        intptr_t handle, int m, int n, size_t descrA,
        size_t inVal, size_t inColInd, size_t inRowPtr,
        int inNnz, size_t nnzPerRow, size_t outVal, size_t outColInd,
        size_t outRowPtr, float tol) except *:
    _setStream(handle)
    status = cusparseScsr2csr_compress(
        <Handle>handle, m, n, <MatDescr>descrA,
        <const float *>inVal, <const int *>inColInd, <const int *>inRowPtr,
        inNnz, <int *>nnzPerRow, <float *>outVal, <int *>outColInd,
        <int *>outRowPtr, tol)
    check_status(status)


cpdef void dcsr2csr_compress(
        intptr_t handle, int m, int n, size_t descrA,
        size_t inVal, size_t inColInd, size_t inRowPtr,
        int inNnz, size_t nnzPerRow, size_t outVal, size_t outColInd,
        size_t outRowPtr, float tol) except *:
    _setStream(handle)
    status = cusparseDcsr2csr_compress(
        <Handle>handle, m, n, <MatDescr>descrA,
        <const double *>inVal, <const int *>inColInd, <const int *>inRowPtr,
        inNnz, <int *>nnzPerRow, <double *>outVal, <int *>outColInd,
        <int *>outRowPtr, tol)
    check_status(status)

cpdef void ccsr2csr_compress(
        intptr_t handle, int m, int n, size_t descrA,
        size_t inVal, size_t inColInd, size_t inRowPtr,
        int inNnz, size_t nnzPerRow, size_t outVal, size_t outColInd,
        size_t outRowPtr, complex tol) except *:
    _setStream(handle)
    status = cusparseCcsr2csr_compress(
        <Handle>handle, m, n, <MatDescr>descrA,
        <const cuComplex *>inVal, <const int *>inColInd, <const int *>inRowPtr,
        inNnz, <int *>nnzPerRow, <cuComplex *>outVal, <int *>outColInd,
        <int *>outRowPtr, complex_to_cuda(tol))
    check_status(status)

cpdef void zcsr2csr_compress(
        intptr_t handle, int m, int n, size_t descrA,
        size_t inVal, size_t inColInd, size_t inRowPtr,
        int inNnz, size_t nnzPerRow, size_t outVal, size_t outColInd,
        size_t outRowPtr, double complex tol) except *:
    _setStream(handle)
    status = cusparseZcsr2csr_compress(
        <Handle>handle, m, n, <MatDescr>descrA,
        <const cuDoubleComplex *>inVal, <const int *>inColInd,
        <const int *>inRowPtr,
        inNnz, <int *>nnzPerRow, <cuDoubleComplex *>outVal, <int *>outColInd,
        <int *>outRowPtr, double_complex_to_cuda(tol))
    check_status(status)

cpdef void sdense2csc(
        intptr_t handle, int m, int n, size_t descrA, size_t A,
        int lda, size_t nnzPerCol, size_t cscValA, size_t cscRowIndA,
        size_t cscColPtrA) except *:
    _setStream(handle)
    status = cusparseSdense2csc(
        <Handle>handle, m, n, <const MatDescr>descrA, <const float *>A,
        lda, <const int *>nnzPerCol, <float *>cscValA, <int *>cscRowIndA,
        <int *>cscColPtrA)
    check_status(status)


cpdef void ddense2csc(
        intptr_t handle, int m, int n, size_t descrA, size_t A,
        int lda, size_t nnzPerCol, size_t cscValA, size_t cscRowIndA,
        size_t cscColPtrA) except *:
    _setStream(handle)
    status = cusparseDdense2csc(
        <Handle>handle, m, n, <const MatDescr>descrA, <const double *>A,
        lda, <const int *>nnzPerCol, <double *>cscValA, <int *>cscRowIndA,
        <int *>cscColPtrA)
    check_status(status)

cpdef void cdense2csc(
        intptr_t handle, int m, int n, size_t descrA, size_t A,
        int lda, size_t nnzPerCol, size_t cscValA, size_t cscRowIndA,
        size_t cscColPtrA) except *:
    _setStream(handle)
    status = cusparseCdense2csc(
        <Handle>handle, m, n, <const MatDescr>descrA, <const cuComplex *>A,
        lda, <const int *>nnzPerCol, <cuComplex *>cscValA, <int *>cscRowIndA,
        <int *>cscColPtrA)
    check_status(status)

cpdef void zdense2csc(
        intptr_t handle, int m, int n, size_t descrA, size_t A,
        int lda, size_t nnzPerCol, size_t cscValA, size_t cscRowIndA,
        size_t cscColPtrA) except *:
    _setStream(handle)
    status = cusparseZdense2csc(
        <Handle>handle, m, n,
        <const MatDescr>descrA, <const cuDoubleComplex *>A,
        lda, <const int *>nnzPerCol,
        <cuDoubleComplex *>cscValA, <int *>cscRowIndA,
        <int *>cscColPtrA)
    check_status(status)

cpdef void sdense2csr(
        intptr_t handle, int m, int n, size_t descrA,
        size_t A, int lda, size_t nnzPerRow, size_t csrValA,
        size_t csrRowPtrA, size_t csrColIndA) except *:
    _setStream(handle)
    status = cusparseSdense2csr(
        <Handle>handle, m, n, <MatDescr>descrA,
        <const float *>A, lda, <const int *>nnzPerRow, <float *>csrValA,
        <int *>csrRowPtrA, <int *>csrColIndA)
    check_status(status)


cpdef void ddense2csr(
        intptr_t handle, int m, int n, size_t descrA,
        size_t A, int lda, size_t nnzPerRow, size_t csrValA,
        size_t csrRowPtrA, size_t csrColIndA) except *:
    _setStream(handle)
    status = cusparseDdense2csr(
        <Handle>handle, m, n, <MatDescr>descrA,
        <const double *>A, lda, <const int *>nnzPerRow, <double *>csrValA,
        <int *>csrRowPtrA, <int *>csrColIndA)
    check_status(status)

cpdef void cdense2csr(
        intptr_t handle, int m, int n, size_t descrA,
        size_t A, int lda, size_t nnzPerRow, size_t csrValA,
        size_t csrRowPtrA, size_t csrColIndA) except *:
    _setStream(handle)
    status = cusparseCdense2csr(
        <Handle>handle, m, n, <MatDescr>descrA,
        <const cuComplex *>A, lda, <const int *>nnzPerRow,
        <cuComplex *>csrValA, <int *>csrRowPtrA, <int *>csrColIndA)
    check_status(status)

cpdef void zdense2csr(
        intptr_t handle, int m, int n, size_t descrA,
        size_t A, int lda, size_t nnzPerRow, size_t csrValA,
        size_t csrRowPtrA, size_t csrColIndA) except *:
    _setStream(handle)
    status = cusparseZdense2csr(
        <Handle>handle, m, n, <MatDescr>descrA,
        <const cuDoubleComplex *>A, lda, <const int *>nnzPerRow,
        <cuDoubleComplex *>csrValA, <int *>csrRowPtrA, <int *>csrColIndA)
    check_status(status)

cpdef void snnz(
        intptr_t handle, int dirA, int m, int n, size_t descrA,
        size_t A, int lda, size_t nnzPerRowColumn,
        size_t nnzTotalDevHostPtr) except *:
    _setStream(handle)
    status = cusparseSnnz(
        <Handle>handle, <Direction>dirA, m, n, <const MatDescr>descrA,
        <const float *>A, lda, <int *>nnzPerRowColumn,
        <int *>nnzTotalDevHostPtr)
    check_status(status)


cpdef void dnnz(
        intptr_t handle, int dirA, int m, int n, size_t descrA,
        size_t A, int lda, size_t nnzPerRowColumn,
        size_t nnzTotalDevHostPtr) except *:
    _setStream(handle)
    status = cusparseDnnz(
        <Handle>handle, <Direction>dirA, m, n, <const MatDescr>descrA,
        <const double *>A, lda, <int *>nnzPerRowColumn,
        <int *>nnzTotalDevHostPtr)
    check_status(status)

cpdef void cnnz(
        intptr_t handle, int dirA, int m, int n, size_t descrA,
        size_t A, int lda, size_t nnzPerRowColumn,
        size_t nnzTotalDevHostPtr) except *:
    _setStream(handle)
    status = cusparseCnnz(
        <Handle>handle, <Direction>dirA, m, n, <const MatDescr>descrA,
        <const cuComplex *>A, lda, <int *>nnzPerRowColumn,
        <int *>nnzTotalDevHostPtr)
    check_status(status)

cpdef void znnz(
        intptr_t handle, int dirA, int m, int n, size_t descrA,
        size_t A, int lda, size_t nnzPerRowColumn,
        size_t nnzTotalDevHostPtr) except *:
    _setStream(handle)
    status = cusparseZnnz(
        <Handle>handle, <Direction>dirA, m, n, <const MatDescr>descrA,
        <const cuDoubleComplex *>A, lda, <int *>nnzPerRowColumn,
        <int *>nnzTotalDevHostPtr)
    check_status(status)

cpdef void createIdentityPermutation(
        intptr_t handle, int n, size_t p) except *:
    _setStream(handle)
    status = cusparseCreateIdentityPermutation(
        <Handle>handle, n, <int *>p)
    check_status(status)


cpdef size_t xcoosort_bufferSizeExt(
        intptr_t handle, int m, int n, int nnz, size_t cooRows,
        size_t cooCols) except? -1:
    cdef size_t bufferSizeInBytes
    _setStream(handle)
    status = cusparseXcoosort_bufferSizeExt(
        <Handle>handle, m, n, nnz, <const int *>cooRows,
        <const int *>cooCols, &bufferSizeInBytes)
    check_status(status)
    return bufferSizeInBytes


cpdef void xcoosortByRow(
        intptr_t handle, int m, int n, int nnz, size_t cooRows, size_t cooCols,
        size_t P, size_t pBuffer) except *:
    _setStream(handle)
    status = cusparseXcoosortByRow(
        <Handle>handle, m, n, nnz, <int *>cooRows, <int *>cooCols,
        <int *>P, <void *>pBuffer)
    check_status(status)


cpdef void xcoosortByColumn(
        intptr_t handle, int m, int n, int nnz, size_t cooRows, size_t cooCols,
        size_t P, size_t pBuffer) except *:
    _setStream(handle)
    status = cusparseXcoosortByColumn(
        <Handle>handle, m, n, nnz, <int *>cooRows, <int *>cooCols,
        <int *>P, <void *>pBuffer)
    check_status(status)


cpdef size_t xcsrsort_bufferSizeExt(
        intptr_t handle, int m, int n, int nnz, size_t csrRowPtr,
        size_t csrColInd) except? -1:
    cdef size_t bufferSizeInBytes
    _setStream(handle)
    status = cusparseXcsrsort_bufferSizeExt(
        <Handle>handle, m, n, nnz, <const int *>csrRowPtr,
        <const int *>csrColInd, &bufferSizeInBytes)
    check_status(status)
    return bufferSizeInBytes


cpdef void xcsrsort(
        intptr_t handle, int m, int n, int nnz, size_t descrA,
        size_t csrRowPtr, size_t csrColInd, size_t P, size_t pBuffer) except *:
    _setStream(handle)
    status = cusparseXcsrsort(
        <Handle>handle, m, n, nnz, <const MatDescr>descrA,
        <const int *>csrRowPtr, <int *>csrColInd, <int *>P, <void *>pBuffer)
    check_status(status)


cpdef size_t xcscsort_bufferSizeExt(
        intptr_t handle, int m, int n, int nnz, size_t cscColPtr,
        size_t cscRowInd) except? -1:
    cdef size_t bufferSizeInBytes
    _setStream(handle)
    status = cusparseXcscsort_bufferSizeExt(
        <Handle>handle, m, n, nnz, <const int *>cscColPtr,
        <const int *>cscRowInd, &bufferSizeInBytes)
    check_status(status)
    return bufferSizeInBytes


cpdef void xcscsort(
        intptr_t handle, int m, int n, int nnz, size_t descrA,
        size_t cscColPtr, size_t cscRowInd, size_t P, size_t pBuffer) except *:
    _setStream(handle)
    status = cusparseXcscsort(
        <Handle>handle, m, n, nnz, <const MatDescr>descrA,
        <const int *>cscColPtr, <int *>cscRowInd, <int *>P, <void *>pBuffer)
    check_status(status)

########################################
# cuSPARSE PRECONDITIONERS

cpdef size_t createCsrilu02Info() except? -1:
    cdef csrilu02Info_t info
    with nogil:
        status = cusparseCreateCsrilu02Info(&info)
    check_status(status)
    return <size_t>info

cpdef void destroyCsrilu02Info(size_t info) except *:
    with nogil:
        status = cusparseDestroyCsrilu02Info(<csrilu02Info_t>info)
    check_status(status)

cpdef size_t createBsrilu02Info() except? -1:
    cdef bsrilu02Info_t info
    with nogil:
        status = cusparseCreateBsrilu02Info(&info)
    check_status(status)
    return <size_t>info

cpdef void destroyBsrilu02Info(size_t info) except *:
    with nogil:
        status = cusparseDestroyBsrilu02Info(<bsrilu02Info_t>info)
    check_status(status)

cpdef size_t createCsric02Info() except? -1:
    cdef csric02Info_t info
    with nogil:
        status = cusparseCreateCsric02Info(&info)
    check_status(status)
    return <size_t>info

cpdef void destroyCsric02Info(size_t info) except *:
    with nogil:
        status = cusparseDestroyCsric02Info(<csric02Info_t>info)
    check_status(status)

cpdef size_t createBsric02Info() except? -1:
    cdef bsric02Info_t info
    with nogil:
        status = cusparseCreateBsric02Info(&info)
    check_status(status)
    return <size_t>info

cpdef void destroyBsric02Info(size_t info) except *:
    with nogil:
        status = cusparseDestroyBsric02Info(<bsric02Info_t>info)
    check_status(status)

cpdef void scsrilu02_numericBoost(
        intptr_t handle, size_t info, int enable_boost,
        size_t tol, size_t boost_val) except *:
    _setStream(handle)
    with nogil:
        status = cusparseScsrilu02_numericBoost(
            <cusparseHandle_t>handle, <csrilu02Info_t>info, enable_boost,
            <double*>tol, <float*>boost_val)
    check_status(status)

cpdef void dcsrilu02_numericBoost(
        intptr_t handle, size_t info, int enable_boost,
        size_t tol, size_t boost_val) except *:
    _setStream(handle)
    with nogil:
        status = cusparseDcsrilu02_numericBoost(
            <cusparseHandle_t>handle, <csrilu02Info_t>info, enable_boost,
            <double*>tol, <double*>boost_val)
    check_status(status)

cpdef void ccsrilu02_numericBoost(
        intptr_t handle, size_t info, int enable_boost,
        size_t tol, size_t boost_val) except *:
    _setStream(handle)
    with nogil:
        status = cusparseCcsrilu02_numericBoost(
            <cusparseHandle_t>handle, <csrilu02Info_t>info, enable_boost,
            <double*>tol, <cuComplex*>boost_val)
    check_status(status)

cpdef void zcsrilu02_numericBoost(
        intptr_t handle, size_t info, int enable_boost,
        size_t tol, size_t boost_val) except *:
    _setStream(handle)
    with nogil:
        status = cusparseZcsrilu02_numericBoost(
            <cusparseHandle_t>handle, <csrilu02Info_t>info, enable_boost,
            <double*>tol, <cuDoubleComplex*>boost_val)
    check_status(status)

cpdef void xcsrilu02_zeroPivot(
        intptr_t handle, size_t info, size_t position) except *:
    _setStream(handle)
    with nogil:
        status = cusparseXcsrilu02_zeroPivot(
            <cusparseHandle_t>handle, <csrilu02Info_t>info, <int*>position)
    check_status(status)

cpdef int scsrilu02_bufferSize(
        intptr_t handle, int m, int nnz, size_t descrA,
        size_t csrSortedValA, size_t csrSortedRowPtrA,
        size_t csrSortedColIndA, size_t info) except? -1:
    cdef int pBufferSizeInBytes
    _setStream(handle)
    with nogil:
        status = cusparseScsrilu02_bufferSize(
            <cusparseHandle_t>handle, m, nnz, <const cusparseMatDescr_t>descrA,
            <float*>csrSortedValA, <const int*>csrSortedRowPtrA,
            <const int*>csrSortedColIndA, <csrilu02Info_t>info,
            &pBufferSizeInBytes)
    check_status(status)
    return <int>pBufferSizeInBytes

cpdef int dcsrilu02_bufferSize(
        intptr_t handle, int m, int nnz, size_t descrA,
        size_t csrSortedValA, size_t csrSortedRowPtrA,
        size_t csrSortedColIndA, size_t info) except? -1:
    cdef int pBufferSizeInBytes
    _setStream(handle)
    with nogil:
        status = cusparseDcsrilu02_bufferSize(
            <cusparseHandle_t>handle, m, nnz, <const cusparseMatDescr_t>descrA,
            <double*>csrSortedValA, <const int*>csrSortedRowPtrA,
            <const int*>csrSortedColIndA, <csrilu02Info_t>info,
            &pBufferSizeInBytes)
    check_status(status)
    return <int>pBufferSizeInBytes

cpdef int ccsrilu02_bufferSize(
        intptr_t handle, int m, int nnz, size_t descrA,
        size_t csrSortedValA, size_t csrSortedRowPtrA,
        size_t csrSortedColIndA, size_t info) except? -1:
    cdef int pBufferSizeInBytes
    _setStream(handle)
    with nogil:
        status = cusparseCcsrilu02_bufferSize(
            <cusparseHandle_t>handle, m, nnz, <const cusparseMatDescr_t>descrA,
            <cuComplex*>csrSortedValA, <const int*>csrSortedRowPtrA,
            <const int*>csrSortedColIndA, <csrilu02Info_t>info,
            &pBufferSizeInBytes)
    check_status(status)
    return <int>pBufferSizeInBytes

cpdef int zcsrilu02_bufferSize(
        intptr_t handle, int m, int nnz, size_t descrA,
        size_t csrSortedValA, size_t csrSortedRowPtrA,
        size_t csrSortedColIndA, size_t info) except? -1:
    cdef int pBufferSizeInBytes
    _setStream(handle)
    with nogil:
        status = cusparseZcsrilu02_bufferSize(
            <cusparseHandle_t>handle, m, nnz, <const cusparseMatDescr_t>descrA,
            <cuDoubleComplex*>csrSortedValA, <const int*>csrSortedRowPtrA,
            <const int*>csrSortedColIndA, <csrilu02Info_t>info,
            &pBufferSizeInBytes)
    check_status(status)
    return <int>pBufferSizeInBytes

cpdef void scsrilu02_analysis(intptr_t handle, int m, int nnz, size_t descrA,
                              size_t csrSortedValA, size_t csrSortedRowPtrA,
                              size_t csrSortedColIndA, size_t info, int policy,
                              size_t pBuffer) except *:
    _setStream(handle)
    with nogil:
        status = cusparseScsrilu02_analysis(
            <cusparseHandle_t>handle, m, nnz, <const cusparseMatDescr_t>descrA,
            <const float*>csrSortedValA, <const int*>csrSortedRowPtrA,
            <const int*>csrSortedColIndA, <csrilu02Info_t>info,
            <cusparseSolvePolicy_t>policy, <void*>pBuffer)
    check_status(status)

cpdef void dcsrilu02_analysis(intptr_t handle, int m, int nnz, size_t descrA,
                              size_t csrSortedValA, size_t csrSortedRowPtrA,
                              size_t csrSortedColIndA, size_t info, int policy,
                              size_t pBuffer) except *:
    _setStream(handle)
    with nogil:
        status = cusparseDcsrilu02_analysis(
            <cusparseHandle_t>handle, m, nnz, <const cusparseMatDescr_t>descrA,
            <const double*>csrSortedValA, <const int*>csrSortedRowPtrA,
            <const int*>csrSortedColIndA, <csrilu02Info_t>info,
            <cusparseSolvePolicy_t>policy, <void*>pBuffer)
    check_status(status)

cpdef void ccsrilu02_analysis(intptr_t handle, int m, int nnz, size_t descrA,
                              size_t csrSortedValA, size_t csrSortedRowPtrA,
                              size_t csrSortedColIndA, size_t info, int policy,
                              size_t pBuffer) except *:
    _setStream(handle)
    with nogil:
        status = cusparseCcsrilu02_analysis(
            <cusparseHandle_t>handle, m, nnz, <const cusparseMatDescr_t>descrA,
            <const cuComplex*>csrSortedValA, <const int*>csrSortedRowPtrA,
            <const int*>csrSortedColIndA, <csrilu02Info_t>info,
            <cusparseSolvePolicy_t>policy, <void*>pBuffer)
    check_status(status)

cpdef void zcsrilu02_analysis(intptr_t handle, int m, int nnz, size_t descrA,
                              size_t csrSortedValA, size_t csrSortedRowPtrA,
                              size_t csrSortedColIndA, size_t info, int policy,
                              size_t pBuffer) except *:
    _setStream(handle)
    with nogil:
        status = cusparseZcsrilu02_analysis(
            <cusparseHandle_t>handle, m, nnz, <const cusparseMatDescr_t>descrA,
            <const cuDoubleComplex*>csrSortedValA,
            <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA,
            <csrilu02Info_t>info, <cusparseSolvePolicy_t>policy,
            <void*>pBuffer)
    check_status(status)

cpdef void scsrilu02(intptr_t handle, int m, int nnz, size_t descrA,
                     size_t csrSortedValA_valM, size_t csrSortedRowPtrA,
                     size_t csrSortedColIndA, size_t info, int policy,
                     size_t pBuffer) except *:
    _setStream(handle)
    with nogil:
        status = cusparseScsrilu02(
            <cusparseHandle_t>handle, m, nnz, <const cusparseMatDescr_t>descrA,
            <float*>csrSortedValA_valM, <const int*>csrSortedRowPtrA,
            <const int*>csrSortedColIndA, <csrilu02Info_t>info,
            <cusparseSolvePolicy_t>policy, <void*>pBuffer)
    check_status(status)

cpdef void dcsrilu02(intptr_t handle, int m, int nnz, size_t descrA,
                     size_t csrSortedValA_valM, size_t csrSortedRowPtrA,
                     size_t csrSortedColIndA, size_t info, int policy,
                     size_t pBuffer) except *:
    _setStream(handle)
    with nogil:
        status = cusparseDcsrilu02(
            <cusparseHandle_t>handle, m, nnz, <const cusparseMatDescr_t>descrA,
            <double*>csrSortedValA_valM, <const int*>csrSortedRowPtrA,
            <const int*>csrSortedColIndA, <csrilu02Info_t>info,
            <cusparseSolvePolicy_t>policy, <void*>pBuffer)
    check_status(status)

cpdef void ccsrilu02(intptr_t handle, int m, int nnz, size_t descrA,
                     size_t csrSortedValA_valM, size_t csrSortedRowPtrA,
                     size_t csrSortedColIndA, size_t info, int policy,
                     size_t pBuffer) except *:
    _setStream(handle)
    with nogil:
        status = cusparseCcsrilu02(
            <cusparseHandle_t>handle, m, nnz, <const cusparseMatDescr_t>descrA,
            <cuComplex*>csrSortedValA_valM, <const int*>csrSortedRowPtrA,
            <const int*>csrSortedColIndA, <csrilu02Info_t>info,
            <cusparseSolvePolicy_t>policy, <void*>pBuffer)
    check_status(status)

cpdef void zcsrilu02(intptr_t handle, int m, int nnz, size_t descrA,
                     size_t csrSortedValA_valM, size_t csrSortedRowPtrA,
                     size_t csrSortedColIndA, size_t info, int policy,
                     size_t pBuffer) except *:
    _setStream(handle)
    with nogil:
        status = cusparseZcsrilu02(
            <cusparseHandle_t>handle, m, nnz, <const cusparseMatDescr_t>descrA,
            <cuDoubleComplex*>csrSortedValA_valM, <const int*>csrSortedRowPtrA,
            <const int*>csrSortedColIndA, <csrilu02Info_t>info,
            <cusparseSolvePolicy_t>policy, <void*>pBuffer)
    check_status(status)

cpdef void sbsrilu02_numericBoost(
        intptr_t handle, size_t info, int enable_boost,
        size_t tol, size_t boost_val) except *:
    _setStream(handle)
    with nogil:
        status = cusparseSbsrilu02_numericBoost(
            <cusparseHandle_t>handle, <bsrilu02Info_t>info, enable_boost,
            <double*>tol, <float*>boost_val)
    check_status(status)

cpdef void dbsrilu02_numericBoost(
        intptr_t handle, size_t info, int enable_boost,
        size_t tol, size_t boost_val) except *:
    _setStream(handle)
    with nogil:
        status = cusparseDbsrilu02_numericBoost(
            <cusparseHandle_t>handle, <bsrilu02Info_t>info, enable_boost,
            <double*>tol, <double*>boost_val)
    check_status(status)

cpdef void cbsrilu02_numericBoost(
        intptr_t handle, size_t info, int enable_boost,
        size_t tol, size_t boost_val) except *:
    _setStream(handle)
    with nogil:
        status = cusparseCbsrilu02_numericBoost(
            <cusparseHandle_t>handle, <bsrilu02Info_t>info, enable_boost,
            <double*>tol, <cuComplex*>boost_val)
    check_status(status)

cpdef void zbsrilu02_numericBoost(
        intptr_t handle, size_t info, int enable_boost,
        size_t tol, size_t boost_val) except *:
    _setStream(handle)
    with nogil:
        status = cusparseZbsrilu02_numericBoost(
            <cusparseHandle_t>handle, <bsrilu02Info_t>info, enable_boost,
            <double*>tol, <cuDoubleComplex*>boost_val)
    check_status(status)

cpdef void xbsrilu02_zeroPivot(
        intptr_t handle, size_t info, size_t position) except *:
    _setStream(handle)
    with nogil:
        status = cusparseXbsrilu02_zeroPivot(
            <cusparseHandle_t>handle, <bsrilu02Info_t>info, <int*>position)
    check_status(status)

cpdef int sbsrilu02_bufferSize(intptr_t handle, int dirA, int mb, int nnzb,
                               size_t descrA, size_t bsrSortedVal,
                               size_t bsrSortedRowPtr, size_t bsrSortedColInd,
                               int blockDim, size_t info) except? -1:
    cdef int pBufferSizeInBytes
    _setStream(handle)
    with nogil:
        status = cusparseSbsrilu02_bufferSize(
            <cusparseHandle_t>handle, <cusparseDirection_t>dirA, mb, nnzb,
            <const cusparseMatDescr_t>descrA, <float*>bsrSortedVal,
            <const int*>bsrSortedRowPtr, <const int*>bsrSortedColInd, blockDim,
            <bsrilu02Info_t>info, &pBufferSizeInBytes)
    check_status(status)
    return <int>pBufferSizeInBytes

cpdef int dbsrilu02_bufferSize(intptr_t handle, int dirA, int mb, int nnzb,
                               size_t descrA, size_t bsrSortedVal,
                               size_t bsrSortedRowPtr, size_t bsrSortedColInd,
                               int blockDim, size_t info) except? -1:
    cdef int pBufferSizeInBytes
    _setStream(handle)
    with nogil:
        status = cusparseDbsrilu02_bufferSize(
            <cusparseHandle_t>handle, <cusparseDirection_t>dirA, mb, nnzb,
            <const cusparseMatDescr_t>descrA, <double*>bsrSortedVal,
            <const int*>bsrSortedRowPtr, <const int*>bsrSortedColInd, blockDim,
            <bsrilu02Info_t>info, &pBufferSizeInBytes)
    check_status(status)
    return <int>pBufferSizeInBytes

cpdef int cbsrilu02_bufferSize(intptr_t handle, int dirA, int mb, int nnzb,
                               size_t descrA, size_t bsrSortedVal,
                               size_t bsrSortedRowPtr, size_t bsrSortedColInd,
                               int blockDim, size_t info) except? -1:
    cdef int pBufferSizeInBytes
    _setStream(handle)
    with nogil:
        status = cusparseCbsrilu02_bufferSize(
            <cusparseHandle_t>handle, <cusparseDirection_t>dirA, mb, nnzb,
            <const cusparseMatDescr_t>descrA, <cuComplex*>bsrSortedVal,
            <const int*>bsrSortedRowPtr, <const int*>bsrSortedColInd, blockDim,
            <bsrilu02Info_t>info, &pBufferSizeInBytes)
    check_status(status)
    return <int>pBufferSizeInBytes

cpdef int zbsrilu02_bufferSize(intptr_t handle, int dirA, int mb, int nnzb,
                               size_t descrA, size_t bsrSortedVal,
                               size_t bsrSortedRowPtr, size_t bsrSortedColInd,
                               int blockDim, size_t info) except? -1:
    cdef int pBufferSizeInBytes
    _setStream(handle)
    with nogil:
        status = cusparseZbsrilu02_bufferSize(
            <cusparseHandle_t>handle, <cusparseDirection_t>dirA, mb, nnzb,
            <const cusparseMatDescr_t>descrA, <cuDoubleComplex*>bsrSortedVal,
            <const int*>bsrSortedRowPtr, <const int*>bsrSortedColInd, blockDim,
            <bsrilu02Info_t>info, &pBufferSizeInBytes)
    check_status(status)
    return <int>pBufferSizeInBytes

cpdef void sbsrilu02_analysis(
        intptr_t handle, int dirA, int mb, int nnzb, size_t descrA,
        size_t bsrSortedVal, size_t bsrSortedRowPtr, size_t bsrSortedColInd,
        int blockDim, size_t info, int policy, size_t pBuffer) except *:
    _setStream(handle)
    with nogil:
        status = cusparseSbsrilu02_analysis(
            <cusparseHandle_t>handle, <cusparseDirection_t>dirA, mb, nnzb,
            <const cusparseMatDescr_t>descrA, <float*>bsrSortedVal,
            <const int*>bsrSortedRowPtr, <const int*>bsrSortedColInd, blockDim,
            <bsrilu02Info_t>info, <cusparseSolvePolicy_t>policy,
            <void*>pBuffer)
    check_status(status)

cpdef void dbsrilu02_analysis(
        intptr_t handle, int dirA, int mb, int nnzb, size_t descrA,
        size_t bsrSortedVal, size_t bsrSortedRowPtr, size_t bsrSortedColInd,
        int blockDim, size_t info, int policy, size_t pBuffer) except *:
    _setStream(handle)
    with nogil:
        status = cusparseDbsrilu02_analysis(
            <cusparseHandle_t>handle, <cusparseDirection_t>dirA, mb, nnzb,
            <const cusparseMatDescr_t>descrA, <double*>bsrSortedVal,
            <const int*>bsrSortedRowPtr, <const int*>bsrSortedColInd, blockDim,
            <bsrilu02Info_t>info, <cusparseSolvePolicy_t>policy,
            <void*>pBuffer)
    check_status(status)

cpdef void cbsrilu02_analysis(
        intptr_t handle, int dirA, int mb, int nnzb, size_t descrA,
        size_t bsrSortedVal, size_t bsrSortedRowPtr, size_t bsrSortedColInd,
        int blockDim, size_t info, int policy, size_t pBuffer) except *:
    _setStream(handle)
    with nogil:
        status = cusparseCbsrilu02_analysis(
            <cusparseHandle_t>handle, <cusparseDirection_t>dirA, mb, nnzb,
            <const cusparseMatDescr_t>descrA, <cuComplex*>bsrSortedVal,
            <const int*>bsrSortedRowPtr, <const int*>bsrSortedColInd, blockDim,
            <bsrilu02Info_t>info, <cusparseSolvePolicy_t>policy,
            <void*>pBuffer)
    check_status(status)

cpdef void zbsrilu02_analysis(
        intptr_t handle, int dirA, int mb, int nnzb, size_t descrA,
        size_t bsrSortedVal, size_t bsrSortedRowPtr, size_t bsrSortedColInd,
        int blockDim, size_t info, int policy, size_t pBuffer) except *:
    _setStream(handle)
    with nogil:
        status = cusparseZbsrilu02_analysis(
            <cusparseHandle_t>handle, <cusparseDirection_t>dirA, mb, nnzb,
            <const cusparseMatDescr_t>descrA, <cuDoubleComplex*>bsrSortedVal,
            <const int*>bsrSortedRowPtr, <const int*>bsrSortedColInd, blockDim,
            <bsrilu02Info_t>info, <cusparseSolvePolicy_t>policy,
            <void*>pBuffer)
    check_status(status)

cpdef void sbsrilu02(
        intptr_t handle, int dirA, int mb, int nnzb, size_t descrA,
        size_t bsrSortedVal, size_t bsrSortedRowPtr,
        size_t bsrSortedColInd, int blockDim, size_t info, int policy,
        size_t pBuffer) except *:
    _setStream(handle)
    with nogil:
        status = cusparseSbsrilu02(
            <cusparseHandle_t>handle, <cusparseDirection_t>dirA, mb, nnzb,
            <const cusparseMatDescr_t>descrA, <float*>bsrSortedVal,
            <const int*>bsrSortedRowPtr, <const int*>bsrSortedColInd, blockDim,
            <bsrilu02Info_t>info, <cusparseSolvePolicy_t>policy,
            <void*>pBuffer)
    check_status(status)

cpdef void dbsrilu02(
        intptr_t handle, int dirA, int mb, int nnzb, size_t descrA,
        size_t bsrSortedVal, size_t bsrSortedRowPtr,
        size_t bsrSortedColInd, int blockDim, size_t info, int policy,
        size_t pBuffer) except *:
    _setStream(handle)
    with nogil:
        status = cusparseDbsrilu02(
            <cusparseHandle_t>handle, <cusparseDirection_t>dirA, mb, nnzb,
            <const cusparseMatDescr_t>descrA, <double*>bsrSortedVal,
            <const int*>bsrSortedRowPtr, <const int*>bsrSortedColInd, blockDim,
            <bsrilu02Info_t>info, <cusparseSolvePolicy_t>policy,
            <void*>pBuffer)
    check_status(status)

cpdef void cbsrilu02(
        intptr_t handle, int dirA, int mb, int nnzb, size_t descrA,
        size_t bsrSortedVal, size_t bsrSortedRowPtr,
        size_t bsrSortedColInd, int blockDim, size_t info, int policy,
        size_t pBuffer) except *:
    _setStream(handle)
    with nogil:
        status = cusparseCbsrilu02(
            <cusparseHandle_t>handle, <cusparseDirection_t>dirA, mb, nnzb,
            <const cusparseMatDescr_t>descrA, <cuComplex*>bsrSortedVal,
            <const int*>bsrSortedRowPtr, <const int*>bsrSortedColInd, blockDim,
            <bsrilu02Info_t>info, <cusparseSolvePolicy_t>policy,
            <void*>pBuffer)
    check_status(status)

cpdef void zbsrilu02(
        intptr_t handle, int dirA, int mb, int nnzb, size_t descrA,
        size_t bsrSortedVal, size_t bsrSortedRowPtr,
        size_t bsrSortedColInd, int blockDim, size_t info, int policy,
        size_t pBuffer) except *:
    _setStream(handle)
    with nogil:
        status = cusparseZbsrilu02(
            <cusparseHandle_t>handle, <cusparseDirection_t>dirA, mb, nnzb,
            <const cusparseMatDescr_t>descrA, <cuDoubleComplex*>bsrSortedVal,
            <const int*>bsrSortedRowPtr, <const int*>bsrSortedColInd, blockDim,
            <bsrilu02Info_t>info, <cusparseSolvePolicy_t>policy,
            <void*>pBuffer)
    check_status(status)

cpdef void xcsric02_zeroPivot(
        intptr_t handle, size_t info, size_t position) except *:
    _setStream(handle)
    with nogil:
        status = cusparseXcsric02_zeroPivot(
            <cusparseHandle_t>handle, <csric02Info_t>info, <int*>position)
    check_status(status)

cpdef int scsric02_bufferSize(intptr_t handle, int m, int nnz, size_t descrA,
                              size_t csrSortedValA, size_t csrSortedRowPtrA,
                              size_t csrSortedColIndA, size_t info) except? -1:
    cdef int pBufferSizeInBytes
    _setStream(handle)
    with nogil:
        status = cusparseScsric02_bufferSize(
            <cusparseHandle_t>handle, m, nnz, <const cusparseMatDescr_t>descrA,
            <float*>csrSortedValA, <const int*>csrSortedRowPtrA,
            <const int*>csrSortedColIndA, <csric02Info_t>info,
            &pBufferSizeInBytes)
    check_status(status)
    return <int>pBufferSizeInBytes

cpdef int dcsric02_bufferSize(intptr_t handle, int m, int nnz, size_t descrA,
                              size_t csrSortedValA, size_t csrSortedRowPtrA,
                              size_t csrSortedColIndA, size_t info) except? -1:
    cdef int pBufferSizeInBytes
    _setStream(handle)
    with nogil:
        status = cusparseDcsric02_bufferSize(
            <cusparseHandle_t>handle, m, nnz, <const cusparseMatDescr_t>descrA,
            <double*>csrSortedValA, <const int*>csrSortedRowPtrA,
            <const int*>csrSortedColIndA, <csric02Info_t>info,
            &pBufferSizeInBytes)
    check_status(status)
    return <int>pBufferSizeInBytes

cpdef int ccsric02_bufferSize(intptr_t handle, int m, int nnz, size_t descrA,
                              size_t csrSortedValA, size_t csrSortedRowPtrA,
                              size_t csrSortedColIndA, size_t info) except? -1:
    cdef int pBufferSizeInBytes
    _setStream(handle)
    with nogil:
        status = cusparseCcsric02_bufferSize(
            <cusparseHandle_t>handle, m, nnz, <const cusparseMatDescr_t>descrA,
            <cuComplex*>csrSortedValA, <const int*>csrSortedRowPtrA,
            <const int*>csrSortedColIndA, <csric02Info_t>info,
            &pBufferSizeInBytes)
    check_status(status)
    return <int>pBufferSizeInBytes

cpdef int zcsric02_bufferSize(intptr_t handle, int m, int nnz, size_t descrA,
                              size_t csrSortedValA, size_t csrSortedRowPtrA,
                              size_t csrSortedColIndA, size_t info) except? -1:
    cdef int pBufferSizeInBytes
    _setStream(handle)
    with nogil:
        status = cusparseZcsric02_bufferSize(
            <cusparseHandle_t>handle, m, nnz, <const cusparseMatDescr_t>descrA,
            <cuDoubleComplex*>csrSortedValA, <const int*>csrSortedRowPtrA,
            <const int*>csrSortedColIndA, <csric02Info_t>info,
            &pBufferSizeInBytes)
    check_status(status)
    return <int>pBufferSizeInBytes

cpdef void scsric02_analysis(
        intptr_t handle, int m, int nnz, size_t descrA,
        size_t csrSortedValA, size_t csrSortedRowPtrA,
        size_t csrSortedColIndA, size_t info, int policy,
        size_t pBuffer) except *:
    _setStream(handle)
    with nogil:
        status = cusparseScsric02_analysis(
            <cusparseHandle_t>handle, m, nnz, <const cusparseMatDescr_t>descrA,
            <const float*>csrSortedValA, <const int*>csrSortedRowPtrA,
            <const int*>csrSortedColIndA, <csric02Info_t>info,
            <cusparseSolvePolicy_t>policy, <void*>pBuffer)
    check_status(status)

cpdef void dcsric02_analysis(
        intptr_t handle, int m, int nnz, size_t descrA,
        size_t csrSortedValA, size_t csrSortedRowPtrA,
        size_t csrSortedColIndA, size_t info, int policy,
        size_t pBuffer) except *:
    _setStream(handle)
    with nogil:
        status = cusparseDcsric02_analysis(
            <cusparseHandle_t>handle, m, nnz, <const cusparseMatDescr_t>descrA,
            <const double*>csrSortedValA, <const int*>csrSortedRowPtrA,
            <const int*>csrSortedColIndA, <csric02Info_t>info,
            <cusparseSolvePolicy_t>policy, <void*>pBuffer)
    check_status(status)

cpdef void ccsric02_analysis(
        intptr_t handle, int m, int nnz, size_t descrA,
        size_t csrSortedValA, size_t csrSortedRowPtrA,
        size_t csrSortedColIndA, size_t info, int policy,
        size_t pBuffer) except *:
    _setStream(handle)
    with nogil:
        status = cusparseCcsric02_analysis(
            <cusparseHandle_t>handle, m, nnz, <const cusparseMatDescr_t>descrA,
            <const cuComplex*>csrSortedValA, <const int*>csrSortedRowPtrA,
            <const int*>csrSortedColIndA, <csric02Info_t>info,
            <cusparseSolvePolicy_t>policy, <void*>pBuffer)
    check_status(status)

cpdef void zcsric02_analysis(
        intptr_t handle, int m, int nnz, size_t descrA,
        size_t csrSortedValA, size_t csrSortedRowPtrA,
        size_t csrSortedColIndA, size_t info, int policy,
        size_t pBuffer) except *:
    _setStream(handle)
    with nogil:
        status = cusparseZcsric02_analysis(
            <cusparseHandle_t>handle, m, nnz, <const cusparseMatDescr_t>descrA,
            <const cuDoubleComplex*>csrSortedValA,
            <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA,
            <csric02Info_t>info, <cusparseSolvePolicy_t>policy, <void*>pBuffer)
    check_status(status)

cpdef void scsric02(
        intptr_t handle, int m, int nnz, size_t descrA,
        size_t csrSortedValA_valM, size_t csrSortedRowPtrA,
        size_t csrSortedColIndA, size_t info, int policy,
        size_t pBuffer) except *:
    _setStream(handle)
    with nogil:
        status = cusparseScsric02(
            <cusparseHandle_t>handle, m, nnz, <const cusparseMatDescr_t>descrA,
            <float*>csrSortedValA_valM, <const int*>csrSortedRowPtrA,
            <const int*>csrSortedColIndA, <csric02Info_t>info,
            <cusparseSolvePolicy_t>policy, <void*>pBuffer)
    check_status(status)

cpdef void dcsric02(
        intptr_t handle, int m, int nnz, size_t descrA,
        size_t csrSortedValA_valM, size_t csrSortedRowPtrA,
        size_t csrSortedColIndA, size_t info, int policy,
        size_t pBuffer) except *:
    _setStream(handle)
    with nogil:
        status = cusparseDcsric02(
            <cusparseHandle_t>handle, m, nnz, <const cusparseMatDescr_t>descrA,
            <double*>csrSortedValA_valM, <const int*>csrSortedRowPtrA,
            <const int*>csrSortedColIndA, <csric02Info_t>info,
            <cusparseSolvePolicy_t>policy, <void*>pBuffer)
    check_status(status)

cpdef void ccsric02(
        intptr_t handle, int m, int nnz, size_t descrA,
        size_t csrSortedValA_valM, size_t csrSortedRowPtrA,
        size_t csrSortedColIndA, size_t info, int policy,
        size_t pBuffer) except *:
    _setStream(handle)
    with nogil:
        status = cusparseCcsric02(
            <cusparseHandle_t>handle, m, nnz, <const cusparseMatDescr_t>descrA,
            <cuComplex*>csrSortedValA_valM, <const int*>csrSortedRowPtrA,
            <const int*>csrSortedColIndA, <csric02Info_t>info,
            <cusparseSolvePolicy_t>policy, <void*>pBuffer)
    check_status(status)

cpdef void zcsric02(
        intptr_t handle, int m, int nnz, size_t descrA,
        size_t csrSortedValA_valM, size_t csrSortedRowPtrA,
        size_t csrSortedColIndA, size_t info, int policy,
        size_t pBuffer) except *:
    _setStream(handle)
    with nogil:
        status = cusparseZcsric02(
            <cusparseHandle_t>handle, m, nnz, <const cusparseMatDescr_t>descrA,
            <cuDoubleComplex*>csrSortedValA_valM, <const int*>csrSortedRowPtrA,
            <const int*>csrSortedColIndA, <csric02Info_t>info,
            <cusparseSolvePolicy_t>policy, <void*>pBuffer)
    check_status(status)

cpdef void xbsric02_zeroPivot(
        intptr_t handle, size_t info, size_t position) except *:
    _setStream(handle)
    with nogil:
        status = cusparseXbsric02_zeroPivot(
            <cusparseHandle_t>handle, <bsric02Info_t>info, <int*>position)
    check_status(status)

cpdef int sbsric02_bufferSize(intptr_t handle, int dirA, int mb, int nnzb,
                              size_t descrA, size_t bsrSortedVal,
                              size_t bsrSortedRowPtr, size_t bsrSortedColInd,
                              int blockDim, size_t info) except? -1:
    cdef int pBufferSizeInBytes
    _setStream(handle)
    with nogil:
        status = cusparseSbsric02_bufferSize(
            <cusparseHandle_t>handle, <cusparseDirection_t>dirA, mb, nnzb,
            <const cusparseMatDescr_t>descrA, <float*>bsrSortedVal,
            <const int*>bsrSortedRowPtr, <const int*>bsrSortedColInd, blockDim,
            <bsric02Info_t>info, &pBufferSizeInBytes)
    check_status(status)
    return <int>pBufferSizeInBytes

cpdef int dbsric02_bufferSize(intptr_t handle, int dirA, int mb, int nnzb,
                              size_t descrA, size_t bsrSortedVal,
                              size_t bsrSortedRowPtr, size_t bsrSortedColInd,
                              int blockDim, size_t info) except? -1:
    cdef int pBufferSizeInBytes
    _setStream(handle)
    with nogil:
        status = cusparseDbsric02_bufferSize(
            <cusparseHandle_t>handle, <cusparseDirection_t>dirA, mb, nnzb,
            <const cusparseMatDescr_t>descrA, <double*>bsrSortedVal,
            <const int*>bsrSortedRowPtr, <const int*>bsrSortedColInd, blockDim,
            <bsric02Info_t>info, &pBufferSizeInBytes)
    check_status(status)
    return <int>pBufferSizeInBytes

cpdef int cbsric02_bufferSize(intptr_t handle, int dirA, int mb, int nnzb,
                              size_t descrA, size_t bsrSortedVal,
                              size_t bsrSortedRowPtr, size_t bsrSortedColInd,
                              int blockDim, size_t info) except? -1:
    cdef int pBufferSizeInBytes
    _setStream(handle)
    with nogil:
        status = cusparseCbsric02_bufferSize(
            <cusparseHandle_t>handle, <cusparseDirection_t>dirA, mb, nnzb,
            <const cusparseMatDescr_t>descrA, <cuComplex*>bsrSortedVal,
            <const int*>bsrSortedRowPtr, <const int*>bsrSortedColInd, blockDim,
            <bsric02Info_t>info, &pBufferSizeInBytes)
    check_status(status)
    return <int>pBufferSizeInBytes

cpdef int zbsric02_bufferSize(intptr_t handle, int dirA, int mb, int nnzb,
                              size_t descrA, size_t bsrSortedVal,
                              size_t bsrSortedRowPtr, size_t bsrSortedColInd,
                              int blockDim, size_t info) except? -1:
    cdef int pBufferSizeInBytes
    _setStream(handle)
    with nogil:
        status = cusparseZbsric02_bufferSize(
            <cusparseHandle_t>handle, <cusparseDirection_t>dirA, mb, nnzb,
            <const cusparseMatDescr_t>descrA, <cuDoubleComplex*>bsrSortedVal,
            <const int*>bsrSortedRowPtr, <const int*>bsrSortedColInd, blockDim,
            <bsric02Info_t>info, &pBufferSizeInBytes)
    check_status(status)
    return <int>pBufferSizeInBytes

cpdef void sbsric02_analysis(
        intptr_t handle, int dirA, int mb, int nnzb, size_t descrA,
        size_t bsrSortedVal, size_t bsrSortedRowPtr, size_t bsrSortedColInd,
        int blockDim, size_t info, int policy, size_t pInputBuffer) except *:
    _setStream(handle)
    with nogil:
        status = cusparseSbsric02_analysis(
            <cusparseHandle_t>handle, <cusparseDirection_t>dirA, mb, nnzb,
            <const cusparseMatDescr_t>descrA, <const float*>bsrSortedVal,
            <const int*>bsrSortedRowPtr, <const int*>bsrSortedColInd, blockDim,
            <bsric02Info_t>info, <cusparseSolvePolicy_t>policy,
            <void*>pInputBuffer)
    check_status(status)

cpdef void dbsric02_analysis(
        intptr_t handle, int dirA, int mb, int nnzb, size_t descrA,
        size_t bsrSortedVal, size_t bsrSortedRowPtr, size_t bsrSortedColInd,
        int blockDim, size_t info, int policy, size_t pInputBuffer) except *:
    _setStream(handle)
    with nogil:
        status = cusparseDbsric02_analysis(
            <cusparseHandle_t>handle, <cusparseDirection_t>dirA, mb, nnzb,
            <const cusparseMatDescr_t>descrA, <const double*>bsrSortedVal,
            <const int*>bsrSortedRowPtr, <const int*>bsrSortedColInd, blockDim,
            <bsric02Info_t>info, <cusparseSolvePolicy_t>policy,
            <void*>pInputBuffer)
    check_status(status)

cpdef void cbsric02_analysis(
        intptr_t handle, int dirA, int mb, int nnzb, size_t descrA,
        size_t bsrSortedVal, size_t bsrSortedRowPtr, size_t bsrSortedColInd,
        int blockDim, size_t info, int policy, size_t pInputBuffer) except *:
    _setStream(handle)
    with nogil:
        status = cusparseCbsric02_analysis(
            <cusparseHandle_t>handle, <cusparseDirection_t>dirA, mb, nnzb,
            <const cusparseMatDescr_t>descrA, <const cuComplex*>bsrSortedVal,
            <const int*>bsrSortedRowPtr, <const int*>bsrSortedColInd, blockDim,
            <bsric02Info_t>info, <cusparseSolvePolicy_t>policy,
            <void*>pInputBuffer)
    check_status(status)

cpdef void zbsric02_analysis(
        intptr_t handle, int dirA, int mb, int nnzb, size_t descrA,
        size_t bsrSortedVal, size_t bsrSortedRowPtr, size_t bsrSortedColInd,
        int blockDim, size_t info, int policy, size_t pInputBuffer) except *:
    _setStream(handle)
    with nogil:
        status = cusparseZbsric02_analysis(
            <cusparseHandle_t>handle, <cusparseDirection_t>dirA, mb, nnzb,
            <const cusparseMatDescr_t>descrA,
            <const cuDoubleComplex*>bsrSortedVal, <const int*>bsrSortedRowPtr,
            <const int*>bsrSortedColInd, blockDim, <bsric02Info_t>info,
            <cusparseSolvePolicy_t>policy, <void*>pInputBuffer)
    check_status(status)

cpdef void sbsric02(
        intptr_t handle, int dirA, int mb, int nnzb, size_t descrA,
        size_t bsrSortedVal, size_t bsrSortedRowPtr,
        size_t bsrSortedColInd, int blockDim, size_t info, int policy,
        size_t pBuffer) except *:
    _setStream(handle)
    with nogil:
        status = cusparseSbsric02(
            <cusparseHandle_t>handle, <cusparseDirection_t>dirA, mb, nnzb,
            <const cusparseMatDescr_t>descrA, <float*>bsrSortedVal,
            <const int*>bsrSortedRowPtr, <const int*>bsrSortedColInd, blockDim,
            <bsric02Info_t>info, <cusparseSolvePolicy_t>policy, <void*>pBuffer)
    check_status(status)

cpdef void dbsric02(
        intptr_t handle, int dirA, int mb, int nnzb, size_t descrA,
        size_t bsrSortedVal, size_t bsrSortedRowPtr,
        size_t bsrSortedColInd, int blockDim, size_t info, int policy,
        size_t pBuffer) except *:
    _setStream(handle)
    with nogil:
        status = cusparseDbsric02(
            <cusparseHandle_t>handle, <cusparseDirection_t>dirA, mb, nnzb,
            <const cusparseMatDescr_t>descrA, <double*>bsrSortedVal,
            <const int*>bsrSortedRowPtr, <const int*>bsrSortedColInd, blockDim,
            <bsric02Info_t>info, <cusparseSolvePolicy_t>policy, <void*>pBuffer)
    check_status(status)

cpdef void cbsric02(
        intptr_t handle, int dirA, int mb, int nnzb, size_t descrA,
        size_t bsrSortedVal, size_t bsrSortedRowPtr,
        size_t bsrSortedColInd, int blockDim, size_t info, int policy,
        size_t pBuffer) except *:
    _setStream(handle)
    with nogil:
        status = cusparseCbsric02(
            <cusparseHandle_t>handle, <cusparseDirection_t>dirA, mb, nnzb,
            <const cusparseMatDescr_t>descrA, <cuComplex*>bsrSortedVal,
            <const int*>bsrSortedRowPtr, <const int*>bsrSortedColInd, blockDim,
            <bsric02Info_t>info, <cusparseSolvePolicy_t>policy, <void*>pBuffer)
    check_status(status)

cpdef void zbsric02(
        intptr_t handle, int dirA, int mb, int nnzb, size_t descrA,
        size_t bsrSortedVal, size_t bsrSortedRowPtr,
        size_t bsrSortedColInd, int blockDim, size_t info, int policy,
        size_t pBuffer) except *:
    _setStream(handle)
    with nogil:
        status = cusparseZbsric02(
            <cusparseHandle_t>handle, <cusparseDirection_t>dirA, mb, nnzb,
            <const cusparseMatDescr_t>descrA, <cuDoubleComplex*>bsrSortedVal,
            <const int*>bsrSortedRowPtr, <const int*>bsrSortedColInd, blockDim,
            <bsric02Info_t>info, <cusparseSolvePolicy_t>policy, <void*>pBuffer)
    check_status(status)

cpdef size_t sgtsv2_bufferSizeExt(
        intptr_t handle, int m, int n, size_t dl, size_t d,
        size_t du, size_t B, int ldb) except? -1:
    cdef size_t bufferSizeInBytes
    _setStream(handle)
    with nogil:
        status = cusparseSgtsv2_bufferSizeExt(
            <cusparseHandle_t>handle, m, n, <const float*>dl, <const float*>d,
            <const float*>du, <const float*>B, ldb, &bufferSizeInBytes)
    check_status(status)
    return bufferSizeInBytes

cpdef size_t dgtsv2_bufferSizeExt(
        intptr_t handle, int m, int n, size_t dl, size_t d,
        size_t du, size_t B, int ldb) except? -1:
    cdef size_t bufferSizeInBytes
    _setStream(handle)
    with nogil:
        status = cusparseDgtsv2_bufferSizeExt(
            <cusparseHandle_t>handle, m, n, <const double*>dl,
            <const double*>d, <const double*>du, <const double*>B, ldb,
            &bufferSizeInBytes)
    check_status(status)
    return bufferSizeInBytes

cpdef size_t cgtsv2_bufferSizeExt(
        intptr_t handle, int m, int n, size_t dl, size_t d,
        size_t du, size_t B, int ldb) except? -1:
    cdef size_t bufferSizeInBytes
    _setStream(handle)
    with nogil:
        status = cusparseCgtsv2_bufferSizeExt(
            <cusparseHandle_t>handle, m, n, <const cuComplex*>dl,
            <const cuComplex*>d, <const cuComplex*>du, <const cuComplex*>B,
            ldb, &bufferSizeInBytes)
    check_status(status)
    return bufferSizeInBytes

cpdef size_t zgtsv2_bufferSizeExt(
        intptr_t handle, int m, int n, size_t dl, size_t d,
        size_t du, size_t B, int ldb) except? -1:
    cdef size_t bufferSizeInBytes
    _setStream(handle)
    with nogil:
        status = cusparseZgtsv2_bufferSizeExt(
            <cusparseHandle_t>handle, m, n, <const cuDoubleComplex*>dl,
            <const cuDoubleComplex*>d, <const cuDoubleComplex*>du,
            <const cuDoubleComplex*>B, ldb, &bufferSizeInBytes)
    check_status(status)
    return bufferSizeInBytes

cpdef void sgtsv2(
        intptr_t handle, int m, int n, size_t dl, size_t d, size_t du,
        size_t B, int ldb, size_t pBuffer) except *:
    _setStream(handle)
    with nogil:
        status = cusparseSgtsv2(
            <cusparseHandle_t>handle, m, n, <const float*>dl, <const float*>d,
            <const float*>du, <float*>B, ldb, <void*>pBuffer)
    check_status(status)

cpdef void dgtsv2(
        intptr_t handle, int m, int n, size_t dl, size_t d, size_t du,
        size_t B, int ldb, size_t pBuffer) except *:
    _setStream(handle)
    with nogil:
        status = cusparseDgtsv2(<cusparseHandle_t>handle, m, n,
                                <const double*>dl, <const double*>d,
                                <const double*>du, <double*>B, ldb,
                                <void*>pBuffer)
    check_status(status)

cpdef void cgtsv2(
        intptr_t handle, int m, int n, size_t dl, size_t d, size_t du,
        size_t B, int ldb, size_t pBuffer) except *:
    _setStream(handle)
    with nogil:
        status = cusparseCgtsv2(<cusparseHandle_t>handle, m, n,
                                <const cuComplex*>dl, <const cuComplex*>d,
                                <const cuComplex*>du, <cuComplex*>B, ldb,
                                <void*>pBuffer)
    check_status(status)

cpdef void zgtsv2(
        intptr_t handle, int m, int n, size_t dl, size_t d, size_t du,
        size_t B, int ldb, size_t pBuffer) except *:
    _setStream(handle)
    with nogil:
        status = cusparseZgtsv2(
            <cusparseHandle_t>handle, m, n, <const cuDoubleComplex*>dl,
            <const cuDoubleComplex*>d, <const cuDoubleComplex*>du,
            <cuDoubleComplex*>B, ldb, <void*>pBuffer)
    check_status(status)

cpdef size_t sgtsv2_nopivot_bufferSizeExt(
        intptr_t handle, int m, int n, size_t dl,
        size_t d, size_t du, size_t B, int ldb) except? -1:
    cdef size_t pBufferSizeInBytes
    _setStream(handle)
    with nogil:
        status = cusparseSgtsv2_nopivot_bufferSizeExt(
            <cusparseHandle_t>handle, m, n, <const float*>dl, <const float*>d,
            <const float*>du, <const float*>B, ldb, &pBufferSizeInBytes)
    check_status(status)
    return pBufferSizeInBytes

cpdef size_t dgtsv2_nopivot_bufferSizeExt(
        intptr_t handle, int m, int n, size_t dl,
        size_t d, size_t du, size_t B, int ldb) except? -1:
    cdef size_t pBufferSizeInBytes
    _setStream(handle)
    with nogil:
        status = cusparseDgtsv2_nopivot_bufferSizeExt(
            <cusparseHandle_t>handle, m, n, <const double*>dl,
            <const double*>d, <const double*>du, <const double*>B, ldb,
            &pBufferSizeInBytes)
    check_status(status)
    return pBufferSizeInBytes

cpdef size_t cgtsv2_nopivot_bufferSizeExt(
        intptr_t handle, int m, int n, size_t dl,
        size_t d, size_t du, size_t B, int ldb) except? -1:
    cdef size_t pBufferSizeInBytes
    _setStream(handle)
    with nogil:
        status = cusparseCgtsv2_nopivot_bufferSizeExt(
            <cusparseHandle_t>handle, m, n, <const cuComplex*>dl,
            <const cuComplex*>d, <const cuComplex*>du, <const cuComplex*>B,
            ldb, &pBufferSizeInBytes)
    check_status(status)
    return pBufferSizeInBytes

cpdef size_t zgtsv2_nopivot_bufferSizeExt(
        intptr_t handle, int m, int n, size_t dl,
        size_t d, size_t du, size_t B, int ldb) except? -1:
    cdef size_t pBufferSizeInBytes
    _setStream(handle)
    with nogil:
        status = cusparseZgtsv2_nopivot_bufferSizeExt(
            <cusparseHandle_t>handle, m, n, <const cuDoubleComplex*>dl,
            <const cuDoubleComplex*>d, <const cuDoubleComplex*>du,
            <const cuDoubleComplex*>B, ldb, &pBufferSizeInBytes)
    check_status(status)
    return pBufferSizeInBytes

cpdef void sgtsv2_nopivot(
        intptr_t handle, int m, int n, size_t dl, size_t d,
        size_t du, size_t B, int ldb, size_t pBuffer) except *:
    _setStream(handle)
    with nogil:
        status = cusparseSgtsv2_nopivot(
            <cusparseHandle_t>handle, m, n, <const float*>dl, <const float*>d,
            <const float*>du, <float*>B, ldb, <void*>pBuffer)
    check_status(status)

cpdef void dgtsv2_nopivot(
        intptr_t handle, int m, int n, size_t dl, size_t d,
        size_t du, size_t B, int ldb, size_t pBuffer) except *:
    _setStream(handle)
    with nogil:
        status = cusparseDgtsv2_nopivot(<cusparseHandle_t>handle, m, n,
                                        <const double*>dl, <const double*>d,
                                        <const double*>du, <double*>B, ldb,
                                        <void*>pBuffer)
    check_status(status)

cpdef void cgtsv2_nopivot(
        intptr_t handle, int m, int n, size_t dl, size_t d,
        size_t du, size_t B, int ldb, size_t pBuffer) except *:
    _setStream(handle)
    with nogil:
        status = cusparseCgtsv2_nopivot(
            <cusparseHandle_t>handle, m, n, <const cuComplex*>dl,
            <const cuComplex*>d, <const cuComplex*>du, <cuComplex*>B, ldb,
            <void*>pBuffer)
    check_status(status)

cpdef void zgtsv2_nopivot(
        intptr_t handle, int m, int n, size_t dl, size_t d,
        size_t du, size_t B, int ldb, size_t pBuffer) except *:
    _setStream(handle)
    with nogil:
        status = cusparseZgtsv2_nopivot(
            <cusparseHandle_t>handle, m, n, <const cuDoubleComplex*>dl,
            <const cuDoubleComplex*>d, <const cuDoubleComplex*>du,
            <cuDoubleComplex*>B, ldb, <void*>pBuffer)
    check_status(status)

cpdef size_t sgtsv2StridedBatch_bufferSizeExt(
        intptr_t handle, int m, size_t dl, size_t d, size_t du, size_t x,
        int batchCount, int batchStride) except? -1:
    cdef size_t bufferSizeInBytes
    _setStream(handle)
    with nogil:
        status = cusparseSgtsv2StridedBatch_bufferSizeExt(
            <cusparseHandle_t>handle, m, <const float*>dl, <const float*>d,
            <const float*>du, <const float*>x, batchCount, batchStride,
            &bufferSizeInBytes)
    check_status(status)
    return bufferSizeInBytes

cpdef size_t dgtsv2StridedBatch_bufferSizeExt(
        intptr_t handle, int m, size_t dl, size_t d, size_t du, size_t x,
        int batchCount, int batchStride):
    cdef size_t bufferSizeInBytes
    _setStream(handle)
    with nogil:
        status = cusparseDgtsv2StridedBatch_bufferSizeExt(
            <cusparseHandle_t>handle, m, <const double*>dl, <const double*>d,
            <const double*>du, <const double*>x, batchCount, batchStride,
            &bufferSizeInBytes)
    check_status(status)
    return bufferSizeInBytes

cpdef size_t cgtsv2StridedBatch_bufferSizeExt(
        intptr_t handle, int m, size_t dl, size_t d, size_t du, size_t x,
        int batchCount, int batchStride) except? -1:
    cdef size_t bufferSizeInBytes
    _setStream(handle)
    with nogil:
        status = cusparseCgtsv2StridedBatch_bufferSizeExt(
            <cusparseHandle_t>handle, m, <const cuComplex*>dl,
            <const cuComplex*>d, <const cuComplex*>du, <const cuComplex*>x,
            batchCount, batchStride, &bufferSizeInBytes)
    check_status(status)
    return bufferSizeInBytes

cpdef size_t zgtsv2StridedBatch_bufferSizeExt(
        intptr_t handle, int m, size_t dl, size_t d, size_t du, size_t x,
        int batchCount, int batchStride) except? -1:
    cdef size_t bufferSizeInBytes
    _setStream(handle)
    with nogil:
        status = cusparseZgtsv2StridedBatch_bufferSizeExt(
            <cusparseHandle_t>handle, m, <const cuDoubleComplex*>dl,
            <const cuDoubleComplex*>d, <const cuDoubleComplex*>du,
            <const cuDoubleComplex*>x, batchCount, batchStride,
            &bufferSizeInBytes)
    check_status(status)
    return bufferSizeInBytes

cpdef void sgtsv2StridedBatch(
        intptr_t handle, int m, size_t dl, size_t d,
        size_t du, size_t x, int batchCount, int batchStride,
        size_t pBuffer) except *:
    _setStream(handle)
    with nogil:
        status = cusparseSgtsv2StridedBatch(
            <cusparseHandle_t>handle, m, <const float*>dl, <const float*>d,
            <const float*>du, <float*>x, batchCount, batchStride,
            <void*>pBuffer)
    check_status(status)

cpdef void dgtsv2StridedBatch(
        intptr_t handle, int m, size_t dl, size_t d,
        size_t du, size_t x, int batchCount, int batchStride,
        size_t pBuffer) except *:
    _setStream(handle)
    with nogil:
        status = cusparseDgtsv2StridedBatch(
            <cusparseHandle_t>handle, m, <const double*>dl, <const double*>d,
            <const double*>du, <double*>x, batchCount, batchStride,
            <void*>pBuffer)
    check_status(status)

cpdef void cgtsv2StridedBatch(
        intptr_t handle, int m, size_t dl, size_t d,
        size_t du, size_t x, int batchCount, int batchStride,
        size_t pBuffer) except *:
    _setStream(handle)
    with nogil:
        status = cusparseCgtsv2StridedBatch(
            <cusparseHandle_t>handle, m, <const cuComplex*>dl,
            <const cuComplex*>d, <const cuComplex*>du, <cuComplex*>x,
            batchCount, batchStride, <void*>pBuffer)
    check_status(status)

cpdef void zgtsv2StridedBatch(
        intptr_t handle, int m, size_t dl, size_t d,
        size_t du, size_t x, int batchCount, int batchStride,
        size_t pBuffer) except *:
    _setStream(handle)
    with nogil:
        status = cusparseZgtsv2StridedBatch(
            <cusparseHandle_t>handle, m, <const cuDoubleComplex*>dl,
            <const cuDoubleComplex*>d, <const cuDoubleComplex*>du,
            <cuDoubleComplex*>x, batchCount, batchStride, <void*>pBuffer)
    check_status(status)

cpdef size_t sgtsvInterleavedBatch_bufferSizeExt(
        intptr_t handle, int algo, int m, size_t dl, size_t d, size_t du,
        size_t x, int batchCount) except? -1:
    cdef size_t pBufferSizeInBytes
    _setStream(handle)
    with nogil:
        status = cusparseSgtsvInterleavedBatch_bufferSizeExt(
            <cusparseHandle_t>handle, algo, m, <const float*>dl,
            <const float*>d, <const float*>du, <const float*>x, batchCount,
            &pBufferSizeInBytes)
    check_status(status)
    return <size_t>pBufferSizeInBytes

cpdef size_t dgtsvInterleavedBatch_bufferSizeExt(
        intptr_t handle, int algo, int m, size_t dl, size_t d, size_t du,
        size_t x, int batchCount) except? -1:
    cdef size_t pBufferSizeInBytes
    _setStream(handle)
    with nogil:
        status = cusparseDgtsvInterleavedBatch_bufferSizeExt(
            <cusparseHandle_t>handle, algo, m, <const double*>dl,
            <const double*>d, <const double*>du, <const double*>x, batchCount,
            &pBufferSizeInBytes)
    check_status(status)
    return <size_t>pBufferSizeInBytes

cpdef size_t cgtsvInterleavedBatch_bufferSizeExt(
        intptr_t handle, int algo, int m, size_t dl, size_t d, size_t du,
        size_t x, int batchCount) except? -1:
    cdef size_t pBufferSizeInBytes
    _setStream(handle)
    with nogil:
        status = cusparseCgtsvInterleavedBatch_bufferSizeExt(
            <cusparseHandle_t>handle, algo, m, <const cuComplex*>dl,
            <const cuComplex*>d, <const cuComplex*>du, <const cuComplex*>x,
            batchCount, &pBufferSizeInBytes)
    check_status(status)
    return <size_t>pBufferSizeInBytes

cpdef size_t zgtsvInterleavedBatch_bufferSizeExt(
        intptr_t handle, int algo, int m, size_t dl, size_t d, size_t du,
        size_t x, int batchCount) except? -1:
    cdef size_t pBufferSizeInBytes
    _setStream(handle)
    with nogil:
        status = cusparseZgtsvInterleavedBatch_bufferSizeExt(
            <cusparseHandle_t>handle, algo, m, <const cuDoubleComplex*>dl,
            <const cuDoubleComplex*>d, <const cuDoubleComplex*>du,
            <const cuDoubleComplex*>x, batchCount, &pBufferSizeInBytes)
    check_status(status)
    return <size_t>pBufferSizeInBytes

cpdef void sgtsvInterleavedBatch(
        intptr_t handle, int algo, int m, size_t dl,
        size_t d, size_t du, size_t x, int batchCount,
        size_t pBuffer) except *:
    _setStream(handle)
    with nogil:
        status = cusparseSgtsvInterleavedBatch(
            <cusparseHandle_t>handle, algo, m, <float*>dl, <float*>d,
            <float*>du, <float*>x, batchCount, <void*>pBuffer)
    check_status(status)

cpdef void dgtsvInterleavedBatch(
        intptr_t handle, int algo, int m, size_t dl,
        size_t d, size_t du, size_t x, int batchCount,
        size_t pBuffer) except *:
    _setStream(handle)
    with nogil:
        status = cusparseDgtsvInterleavedBatch(
            <cusparseHandle_t>handle, algo, m, <double*>dl, <double*>d,
            <double*>du, <double*>x, batchCount, <void*>pBuffer)
    check_status(status)

cpdef void cgtsvInterleavedBatch(
        intptr_t handle, int algo, int m, size_t dl,
        size_t d, size_t du, size_t x, int batchCount,
        size_t pBuffer) except *:
    _setStream(handle)
    with nogil:
        status = cusparseCgtsvInterleavedBatch(
            <cusparseHandle_t>handle, algo, m, <cuComplex*>dl, <cuComplex*>d,
            <cuComplex*>du, <cuComplex*>x, batchCount, <void*>pBuffer)
    check_status(status)

cpdef void zgtsvInterleavedBatch(
        intptr_t handle, int algo, int m, size_t dl,
        size_t d, size_t du, size_t x, int batchCount,
        size_t pBuffer) except *:
    _setStream(handle)
    with nogil:
        status = cusparseZgtsvInterleavedBatch(
            <cusparseHandle_t>handle, algo, m, <cuDoubleComplex*>dl,
            <cuDoubleComplex*>d, <cuDoubleComplex*>du, <cuDoubleComplex*>x,
            batchCount, <void*>pBuffer)
    check_status(status)

cpdef size_t sgpsvInterleavedBatch_bufferSizeExt(
        intptr_t handle, int algo, int m, size_t ds, size_t dl, size_t d,
        size_t du, size_t dw, size_t x, int batchCount) except? -1:
    cdef size_t pBufferSizeInBytes
    _setStream(handle)
    with nogil:
        status = cusparseSgpsvInterleavedBatch_bufferSizeExt(
            <cusparseHandle_t>handle, algo, m, <const float*>ds,
            <const float*>dl, <const float*>d, <const float*>du,
            <const float*>dw, <const float*>x, batchCount, &pBufferSizeInBytes)
    check_status(status)
    return <size_t>pBufferSizeInBytes

cpdef size_t dgpsvInterleavedBatch_bufferSizeExt(
        intptr_t handle, int algo, int m, size_t ds, size_t dl, size_t d,
        size_t du, size_t dw, size_t x, int batchCount) except? -1:
    cdef size_t pBufferSizeInBytes
    _setStream(handle)
    with nogil:
        status = cusparseDgpsvInterleavedBatch_bufferSizeExt(
            <cusparseHandle_t>handle, algo, m, <const double*>ds,
            <const double*>dl, <const double*>d, <const double*>du,
            <const double*>dw, <const double*>x, batchCount,
            &pBufferSizeInBytes)
    check_status(status)
    return <size_t>pBufferSizeInBytes

cpdef size_t cgpsvInterleavedBatch_bufferSizeExt(
        intptr_t handle, int algo, int m, size_t ds, size_t dl, size_t d,
        size_t du, size_t dw, size_t x, int batchCount) except? -1:
    cdef size_t pBufferSizeInBytes
    _setStream(handle)
    with nogil:
        status = cusparseCgpsvInterleavedBatch_bufferSizeExt(
            <cusparseHandle_t>handle, algo, m, <const cuComplex*>ds,
            <const cuComplex*>dl, <const cuComplex*>d, <const cuComplex*>du,
            <const cuComplex*>dw, <const cuComplex*>x, batchCount,
            &pBufferSizeInBytes)
    check_status(status)
    return <size_t>pBufferSizeInBytes

cpdef size_t zgpsvInterleavedBatch_bufferSizeExt(
        intptr_t handle, int algo, int m, size_t ds, size_t dl, size_t d,
        size_t du, size_t dw, size_t x, int batchCount) except? -1:
    cdef size_t pBufferSizeInBytes
    _setStream(handle)
    with nogil:
        status = cusparseZgpsvInterleavedBatch_bufferSizeExt(
            <cusparseHandle_t>handle, algo, m, <const cuDoubleComplex*>ds,
            <const cuDoubleComplex*>dl, <const cuDoubleComplex*>d,
            <const cuDoubleComplex*>du, <const cuDoubleComplex*>dw,
            <const cuDoubleComplex*>x, batchCount, &pBufferSizeInBytes)
    check_status(status)
    return <size_t>pBufferSizeInBytes

cpdef void sgpsvInterleavedBatch(
        intptr_t handle, int algo, int m, size_t ds,
        size_t dl, size_t d, size_t du, size_t dw,
        size_t x, int batchCount, size_t pBuffer) except *:
    _setStream(handle)
    with nogil:
        status = cusparseSgpsvInterleavedBatch(
            <cusparseHandle_t>handle, algo, m, <float*>ds, <float*>dl,
            <float*>d, <float*>du, <float*>dw, <float*>x, batchCount,
            <void*>pBuffer)
    check_status(status)

cpdef void dgpsvInterleavedBatch(
        intptr_t handle, int algo, int m, size_t ds,
        size_t dl, size_t d, size_t du, size_t dw,
        size_t x, int batchCount, size_t pBuffer) except *:
    _setStream(handle)
    with nogil:
        status = cusparseDgpsvInterleavedBatch(
            <cusparseHandle_t>handle, algo, m, <double*>ds, <double*>dl,
            <double*>d, <double*>du, <double*>dw, <double*>x, batchCount,
            <void*>pBuffer)
    check_status(status)

cpdef void cgpsvInterleavedBatch(
        intptr_t handle, int algo, int m, size_t ds,
        size_t dl, size_t d, size_t du, size_t dw,
        size_t x, int batchCount, size_t pBuffer) except *:
    _setStream(handle)
    with nogil:
        status = cusparseCgpsvInterleavedBatch(
            <cusparseHandle_t>handle, algo, m, <cuComplex*>ds, <cuComplex*>dl,
            <cuComplex*>d, <cuComplex*>du, <cuComplex*>dw, <cuComplex*>x,
            batchCount, <void*>pBuffer)
    check_status(status)

cpdef void zgpsvInterleavedBatch(
        intptr_t handle, int algo, int m, size_t ds,
        size_t dl, size_t d, size_t du, size_t dw,
        size_t x, int batchCount, size_t pBuffer) except *:
    _setStream(handle)
    with nogil:
        status = cusparseZgpsvInterleavedBatch(
            <cusparseHandle_t>handle, algo, m, <cuDoubleComplex*>ds,
            <cuDoubleComplex*>dl, <cuDoubleComplex*>d, <cuDoubleComplex*>du,
            <cuDoubleComplex*>dw, <cuDoubleComplex*>x, batchCount,
            <void*>pBuffer)
    check_status(status)

############################################################
# Sparse Vector APIs

cpdef size_t createSpVec(int64_t size, int64_t nnz, intptr_t indices,
                         intptr_t values, IndexType idxType, IndexBase idxBase,
                         DataType valueType) except? -1:
    cdef SpVecDescr desc
    status = cusparseCreateSpVec(&desc, size, nnz, <void*>indices,
                                 <void*>values, idxType, idxBase, valueType)
    check_status(status)
    return <size_t>desc

cpdef void destroySpVec(size_t desc) except *:
    status = cusparseDestroySpVec(<SpVecDescr>desc)
    check_status(status)

cpdef SpVecAttributes spVecGet(size_t desc):
    cdef int64_t size, nnz
    cdef intptr_t indices, values
    cdef IndexType idxType
    cdef IndexBase idxBase
    cdef DataType valueType
    status = cusparseSpVecGet(<SpVecDescr>desc, &size, &nnz,
                              <void**>&indices, <void**>&values,
                              &idxType, &idxBase, &valueType)
    check_status(status)
    return SpVecAttributes(size, nnz, indices, values, idxType, idxBase,
                           valueType)

cpdef int spVecGetIndexBase(size_t desc) except? -1:
    cdef IndexBase idxBase
    status = cusparseSpVecGetIndexBase(<SpVecDescr>desc, &idxBase)
    check_status(status)
    return <int>idxBase

cpdef intptr_t spVecGetValues(size_t desc) except? -1:
    cdef intptr_t values
    status = cusparseSpVecGetValues(<SpVecDescr>desc, <void**>&values)
    check_status(status)
    return values

cpdef void spVecSetValues(size_t desc, intptr_t values) except *:
    status = cusparseSpVecSetValues(<SpVecDescr>desc, <void*>values)
    check_status(status)

############################################################
# Sparse Matrix APIs

cpdef size_t createCoo(int64_t rows, int64_t cols, int64_t nnz,
                       intptr_t cooRowInd, intptr_t cooColInd,
                       intptr_t cooValues, IndexType cooIdxType,
                       IndexBase idxBase, DataType valueType) except? -1:
    cdef SpMatDescr desc
    status = cusparseCreateCoo(&desc, rows, cols, nnz, <void*>cooRowInd,
                               <void*>cooColInd, <void*>cooValues,
                               cooIdxType, idxBase, valueType)
    check_status(status)
    return <size_t>desc

cpdef size_t createCooAoS(int64_t rows, int64_t cols, int64_t nnz,
                          intptr_t cooInd, intptr_t cooValues,
                          IndexType cooIdxType, IndexBase idxBase,
                          DataType valueType) except? -1:
    cdef SpMatDescr desc
    status = cusparseCreateCooAoS(&desc, rows, cols, nnz, <void*>cooInd,
                                  <void*>cooValues, cooIdxType, idxBase,
                                  valueType)
    check_status(status)
    return <size_t>desc

cpdef size_t createCsr(int64_t rows, int64_t cols, int64_t nnz,
                       intptr_t csrRowOffsets, intptr_t csrColind,
                       intptr_t csrValues, IndexType csrRowOffsetsType,
                       IndexType csrColIndType, IndexBase idxBase,
                       DataType valueType) except? -1:
    cdef SpMatDescr desc
    status = cusparseCreateCsr(&desc, rows, cols, nnz,
                               <void*>csrRowOffsets, <void*>csrColind,
                               <void*>csrValues, csrRowOffsetsType,
                               csrColIndType, idxBase, valueType)
    check_status(status)
    return <size_t>desc

cpdef size_t createCsc(int64_t rows, int64_t cols, int64_t nnz,
                       intptr_t cscColOffsets, intptr_t cscRowInd,
                       intptr_t cscValues, IndexType cscColOffsetsType,
                       IndexType cscRowIndType, IndexBase idxBase,
                       DataType valueType) except? -1:
    cdef SpMatDescr desc
    status = cusparseCreateCsc(&desc, rows, cols, nnz,
                               <void*>cscColOffsets, <void*>cscRowInd,
                               <void*>cscValues, cscColOffsetsType,
                               cscRowIndType, idxBase, valueType)
    check_status(status)
    return <size_t>desc

cpdef void destroySpMat(size_t desc) except *:
    status = cusparseDestroySpMat(<SpMatDescr>desc)
    check_status(status)

cpdef CooAttributes cooGet(size_t desc):
    cdef int64_t rows, cols, nnz
    cdef intptr_t rowInd, colInd, values,
    cdef IndexType idxType,
    cdef IndexBase idxBase,
    cdef DataType valueType
    status = cusparseCooGet(<SpMatDescr>desc, &rows, &cols, &nnz,
                            <void**>&rowInd, <void**>&colInd, <void**>&values,
                            &idxType, &idxBase, &valueType)
    check_status(status)
    return CooAttributes(rows, cols, nnz, rowInd, colInd, values,
                         idxType, idxBase, valueType)

cpdef CooAoSAttributes cooAoSGet(size_t desc):
    cdef int64_t rows, cols, nnz
    cdef intptr_t ind, values,
    cdef IndexType idxType,
    cdef IndexBase idxBase,
    cdef DataType valueType
    status = cusparseCooAoSGet(<SpMatDescr>desc, &rows, &cols, &nnz,
                               <void**>&ind, <void**>&values,
                               &idxType, &idxBase, &valueType)
    check_status(status)
    return CooAoSAttributes(rows, cols, nnz, ind, values,
                            idxType, idxBase, valueType)

cpdef CsrAttributes csrGet(size_t desc):
    cdef int64_t rows, cols, nnz
    cdef intptr_t rowOffsets, colInd, values,
    cdef IndexType rowOffsetsType, colIndType
    cdef IndexBase idxBase
    cdef DataType valueType
    status = cusparseCsrGet(<SpMatDescr>desc, &rows, &cols, &nnz,
                            <void**>&rowOffsets, <void**>&colInd,
                            <void**>&values, &rowOffsetsType, &colIndType,
                            &idxBase, &valueType)
    check_status(status)
    return CsrAttributes(rows, cols, nnz, rowOffsets, colInd, values,
                         rowOffsetsType, colIndType, idxBase, valueType)

cpdef void csrSetPointers(
        size_t desc, size_t csrRowOffsets, size_t csrColInd,
        size_t csrValues) except *:
    status = cusparseCsrSetPointers(<SpMatDescr>desc, <void*>csrRowOffsets,
                                    <void*>csrColInd, <void*>csrValues)
    check_status(status)

cpdef int spMatGetFormat(size_t desc) except? -1:
    cdef Format format
    status = cusparseSpMatGetFormat(<SpMatDescr>desc, &format)
    check_status(status)
    return <int>format

cpdef int spMatGetIndexBase(size_t desc) except? -1:
    cdef IndexBase idxBase
    status = cusparseSpMatGetIndexBase(<SpMatDescr>desc, &idxBase)
    check_status(status)
    return <int>idxBase

cpdef intptr_t spMatGetValues(size_t desc) except? -1:
    cdef intptr_t values
    status = cusparseSpMatGetValues(<SpMatDescr>desc, <void**>&values)
    check_status(status)
    return values

cpdef void spMatSetValues(size_t desc, intptr_t values) except *:
    status = cusparseSpMatSetValues(<SpMatDescr>desc, <void*>values)
    check_status(status)

cpdef void spMatGetSize(
        size_t desc, size_t rows, size_t cols, size_t nnz) except *:
    status = cusparseSpMatGetSize(<SpMatDescr>desc, <int64_t*>rows,
                                  <int64_t*>cols, <int64_t*>nnz)
    check_status(status)

cpdef int spMatGetStridedBatch(size_t desc) except? -1:
    cdef int batchCount
    status = cusparseSpMatGetStridedBatch(<SpMatDescr>desc, &batchCount)
    check_status(status)
    return batchCount

cpdef void spMatSetStridedBatch(size_t desc, int batchCount) except *:
    status = cusparseSpMatSetStridedBatch(<SpMatDescr>desc, batchCount)
    check_status(status)

############################################################
# Dense Vector APIs

cpdef size_t createDnVec(int64_t size, intptr_t values,
                         DataType valueType) except? -1:
    cdef DnVecDescr desc
    status = cusparseCreateDnVec(&desc, size, <void*>values, valueType)
    check_status(status)
    return <size_t>desc

cpdef void destroyDnVec(size_t desc) except *:
    status = cusparseDestroyDnVec(<DnVecDescr>desc)
    check_status(status)

cpdef DnVecAttributes dnVecGet(size_t desc):
    cdef int64_t size
    cdef intptr_t values
    cdef DataType valueType
    status = cusparseDnVecGet(<DnVecDescr>desc, &size, <void**>&values,
                              &valueType)
    check_status(status)
    return DnVecAttributes(size, values, valueType)

cpdef intptr_t dnVecGetValues(size_t desc) except? -1:
    cdef intptr_t values
    status = cusparseDnVecGetValues(<DnVecDescr>desc, <void**>&values)
    check_status(status)
    return values

cpdef void dnVecSetValues(size_t desc, intptr_t values) except *:
    status = cusparseDnVecSetValues(<DnVecDescr>desc, <void*>values)
    check_status(status)

############################################################
# Dense Matrix APIs

cpdef size_t createDnMat(int64_t rows, int64_t cols, int64_t ld,
                         intptr_t values, DataType valueType,
                         Order order) except? -1:
    cdef DnMatDescr desc
    status = cusparseCreateDnMat(&desc, rows, cols, ld, <void*>values,
                                 valueType, order)
    check_status(status)
    return <size_t>desc

cpdef void destroyDnMat(size_t desc) except *:
    status = cusparseDestroyDnMat(<DnMatDescr>desc)
    check_status(status)

cpdef DnMatAttributes dnMatGet(size_t desc):
    cdef int64_t rows, cols, ld
    cdef intptr_t values,
    cdef DataType valueType
    cdef Order order
    status = cusparseDnMatGet(<DnMatDescr>desc, &rows, &cols, &ld,
                              <void**>&values, &valueType, &order)
    check_status(status)
    return DnMatAttributes(rows, cols, ld, values, valueType, order)

cpdef intptr_t dnMatGetValues(size_t desc) except? -1:
    cdef intptr_t values
    status = cusparseDnMatGetValues(<DnMatDescr>desc, <void**>&values)
    check_status(status)
    return values

cpdef void dnMatSetValues(size_t desc, intptr_t values) except *:
    status = cusparseDnMatSetValues(<DnMatDescr>desc, <void*>values)
    check_status(status)

cpdef DnMatBatchAttributes dnMatGetStridedBatch(size_t desc):
    cdef int batchCount
    cdef int64_t batchStride
    status = cusparseDnMatGetStridedBatch(<DnMatDescr>desc, &batchCount,
                                          &batchStride)
    check_status(status)
    return DnMatBatchAttributes(batchCount, batchStride)

cpdef void dnMatSetStridedBatch(
        size_t desc, int batchCount, int64_t batchStride) except *:
    status = cusparseDnMatSetStridedBatch(<DnMatDescr>desc, batchCount,
                                          batchStride)
    check_status(status)

############################################################
# Generic API Functions

cpdef size_t spVV_bufferSize(intptr_t handle, Operation opX,
                             size_t vecX, size_t vecY,
                             intptr_t result, DataType computeType) except? -1:
    cdef size_t bufferSize
    status = cusparseSpVV_bufferSize(<Handle>handle, opX,
                                     <SpVecDescr>vecX, <DnVecDescr>vecY,
                                     <void*>result, computeType, &bufferSize)
    check_status(status)
    return bufferSize

cpdef void spVV(
        intptr_t handle, Operation opX, size_t vecX, size_t vecY,
        intptr_t result, DataType computeType,
        intptr_t externalBuffer) except *:
    _setStream(handle)
    status = cusparseSpVV(<Handle>handle, opX, <SpVecDescr>vecX,
                          <DnVecDescr>vecY, <void*>result, computeType,
                          <void*>externalBuffer)
    check_status(status)

cpdef size_t spMV_bufferSize(intptr_t handle, Operation opA, intptr_t alpha,
                             size_t matA, size_t vecX, intptr_t beta,
                             size_t vecY, DataType computeType,
                             SpMVAlg alg) except? -1:
    cdef size_t bufferSize
    status = cusparseSpMV_bufferSize(<Handle>handle, opA, <void*>alpha,
                                     <SpMatDescr>matA, <DnVecDescr>vecX,
                                     <void*>beta, <DnVecDescr>vecY,
                                     computeType, alg, &bufferSize)
    check_status(status)
    return bufferSize

cpdef void spMV(
        intptr_t handle, Operation opA, intptr_t alpha, size_t matA,
        size_t vecX, intptr_t beta, size_t vecY, DataType computeType,
        SpMVAlg alg, intptr_t externalBuffer) except *:
    _setStream(handle)
    status = cusparseSpMV(<Handle>handle, opA, <void*>alpha, <SpMatDescr>matA,
                          <DnVecDescr>vecX, <void*>beta, <DnVecDescr>vecY,
                          computeType, alg, <void*>externalBuffer)
    check_status(status)

cpdef size_t spSM_createDescr() except? -1:
    cdef SpSMDescr descr
    status = cusparseSpSM_createDescr(&descr)
    check_status(status)
    return <size_t>descr

cpdef void spSM_destroyDescr(size_t descr) except *:
    status = cusparseSpSM_destroyDescr(<SpSMDescr>descr)
    check_status(status)

cpdef size_t spSM_bufferSize(intptr_t handle, Operation opA, Operation opB,
                             intptr_t alpha, size_t matA, size_t matB,
                             size_t matC, DataType computeType, SpSMAlg alg,
                             size_t spsmDescr) except? -1:
    cdef size_t bufferSize
    status = cusparseSpSM_bufferSize(<Handle>handle, opA, opB, <void*>alpha,
                                     <SpMatDescr>matA, <DnMatDescr>matB,
                                     <DnMatDescr>matC, computeType, alg,
                                     <SpSMDescr>spsmDescr, &bufferSize)
    check_status(status)
    return bufferSize

cpdef void spSM_analysis(
        intptr_t handle, Operation opA, Operation opB,
        intptr_t alpha, size_t matA, size_t matB, size_t matC,
        DataType computeType, SpSMAlg alg, size_t spsmDescr,
        intptr_t externalBuffer) except *:
    _setStream(handle)
    status = cusparseSpSM_analysis(<Handle> handle, opA, opB, <void*>alpha,
                                   <SpMatDescr>matA, <DnMatDescr>matB,
                                   <DnMatDescr>matC, computeType, alg,
                                   <SpSMDescr>spsmDescr, <void*>externalBuffer)
    check_status(status)

cpdef void spSM_solve(
        intptr_t handle, Operation opA, Operation opB, intptr_t alpha,
        size_t matA, size_t matB, size_t matC, DataType computeType,
        SpSMAlg alg, size_t spsmDescr, intptr_t externalBuffer=0) except *:
    _setStream(handle)
    IF CUPY_HIP_VERSION > 0:
        # hipsparseSpSM_solve has the extra `externalBuffer` parameter that
        # cusparseSpSM_solve does not require.
        status = cusparseSpSM_solve(<Handle> handle, opA, opB, <void*>alpha,
                                    <SpMatDescr>matA, <DnMatDescr>matB,
                                    <DnMatDescr>matC, computeType, alg,
                                    <SpSMDescr>spsmDescr,
                                    <void*>externalBuffer)
    ELSE:
        status = cusparseSpSM_solve(<Handle> handle, opA, opB, <void*>alpha,
                                    <SpMatDescr>matA, <DnMatDescr>matB,
                                    <DnMatDescr>matC, computeType, alg,
                                    <SpSMDescr>spsmDescr)
    check_status(status)

cpdef size_t spMM_bufferSize(intptr_t handle, Operation opA, Operation opB,
                             intptr_t alpha, size_t matA, size_t matB,
                             intptr_t beta, size_t matC, DataType computeType,
                             SpMMAlg alg) except? -1:
    cdef size_t bufferSize
    status = cusparseSpMM_bufferSize(<Handle>handle, opA, opB, <void*>alpha,
                                     <SpMatDescr>matA, <DnMatDescr>matB,
                                     <void*>beta, <DnMatDescr>matC,
                                     computeType, alg, &bufferSize)
    check_status(status)
    return bufferSize

cpdef void spMM(
        intptr_t handle, Operation opA, Operation opB, intptr_t alpha,
        size_t matA, size_t matB, intptr_t beta, size_t matC,
        DataType computeType, SpMMAlg alg, intptr_t externalBuffer) except *:
    _setStream(handle)
    status = cusparseSpMM(<Handle>handle, opA, opB, <void*>alpha,
                          <SpMatDescr>matA, <DnMatDescr>matB, <void*>beta,
                          <DnMatDescr>matC, computeType, alg,
                          <void*>externalBuffer)
    check_status(status)

cpdef size_t constrainedGeMM_bufferSize(intptr_t handle, Operation opA,
                                        Operation opB, intptr_t alpha,
                                        size_t matA, size_t matB,
                                        intptr_t beta, size_t matC,
                                        DataType computeType) except? -1:
    cdef size_t bufferSize
    status = cusparseConstrainedGeMM_bufferSize(
        <Handle>handle, opA, opB, <void*>alpha, <DnMatDescr>matA,
        <DnMatDescr>matB, <void*>beta, <SpMatDescr>matC, computeType,
        &bufferSize)
    check_status(status)
    return bufferSize

cpdef void constrainedGeMM(
        intptr_t handle, Operation opA, Operation opB,
        intptr_t alpha, size_t matA, size_t matB, intptr_t beta,
        size_t matC, DataType computeType,
        intptr_t externalBuffer) except *:
    _setStream(handle)
    status = cusparseConstrainedGeMM(
        <Handle>handle, opA, opB, <void*>alpha, <DnMatDescr>matA,
        <DnMatDescr>matB, <void*>beta, <SpMatDescr>matC, computeType,
        <void*>externalBuffer)
    check_status(status)

cpdef size_t spGEMM_createDescr() except? -1:
    cdef SpGEMMDescr descr
    status = cusparseSpGEMM_createDescr(&descr)
    check_status(status)
    return <size_t>descr

cpdef void spGEMM_destroyDescr(size_t descr) except *:
    status = cusparseSpGEMM_destroyDescr(<SpGEMMDescr>descr)
    check_status(status)

cpdef size_t spGEMM_workEstimation(
        intptr_t handle, Operation opA, Operation opB, intptr_t alpha,
        size_t matA, size_t matB, intptr_t beta, size_t matC,
        DataType computeType, int alg, size_t spgemmDescr,
        size_t bufferSize, intptr_t externalBuffer1) except? -1:
    cdef size_t bufferSize1 = bufferSize
    status = cusparseSpGEMM_workEstimation(
        <Handle>handle, opA, opB, <const void*>alpha, <SpMatDescr>matA,
        <SpMatDescr>matB, <const void*>beta, <SpMatDescr>matC, computeType,
        <SpGEMMAlg>alg, <SpGEMMDescr>spgemmDescr, &bufferSize1,
        <void*>externalBuffer1)
    check_status(status)
    return bufferSize1

cpdef size_t spGEMM_compute(
        intptr_t handle, Operation opA, Operation opB, intptr_t alpha,
        size_t matA, size_t matB, intptr_t beta, size_t matC,
        DataType computeType, int alg, size_t spgemmDescr,
        size_t bufferSize, intptr_t externalBuffer2) except? -1:
    cdef size_t bufferSize2 = bufferSize
    status = cusparseSpGEMM_compute(
        <Handle>handle, opA, opB, <const void*>alpha, <SpMatDescr>matA,
        <SpMatDescr>matB, <const void*>beta, <SpMatDescr>matC, computeType,
        <SpGEMMAlg>alg, <SpGEMMDescr>spgemmDescr, &bufferSize2,
        <void*>externalBuffer2)
    check_status(status)
    return bufferSize2

cpdef void spGEMM_copy(
        intptr_t handle, Operation opA, Operation opB, intptr_t alpha,
        size_t matA, size_t matB, intptr_t beta, size_t matC,
        DataType computeType, int alg, size_t spgemmDescr) except *:
    status = cusparseSpGEMM_copy(
        <Handle>handle, opA, opB, <const void*>alpha, <SpMatDescr>matA,
        <SpMatDescr>matB, <const void*>beta, <SpMatDescr>matC, computeType,
        <SpGEMMAlg>alg, <SpGEMMDescr>spgemmDescr)
    check_status(status)

cpdef void gather(intptr_t handle, size_t vecY, size_t vecX) except *:
    status = cusparseGather(<Handle>handle, <DnVecDescr>vecY, <SpVecDescr>vecX)
    check_status(status)

cpdef size_t sparseToDense_bufferSize(intptr_t handle, size_t matA,
                                      size_t matB, int alg) except? -1:
    cdef size_t bufferSize
    status = cusparseSparseToDense_bufferSize(
        <Handle>handle, <SpMatDescr>matA, <DnMatDescr>matB,
        <cusparseSparseToDenseAlg_t>alg, &bufferSize)
    check_status(status)
    return bufferSize

cpdef void sparseToDense(
        intptr_t handle, size_t matA, size_t matB, int alg,
        intptr_t buffer) except *:
    _setStream(handle)
    status = cusparseSparseToDense(
        <Handle>handle, <SpMatDescr>matA, <DnMatDescr>matB,
        <cusparseSparseToDenseAlg_t>alg, <void*>buffer)
    check_status(status)

cpdef size_t denseToSparse_bufferSize(intptr_t handle, size_t matA,
                                      size_t matB, int alg) except? -1:
    cdef size_t bufferSize
    status = cusparseDenseToSparse_bufferSize(
        <Handle>handle, <DnMatDescr>matA, <SpMatDescr>matB,
        <cusparseDenseToSparseAlg_t>alg, &bufferSize)
    check_status(status)
    return bufferSize

cpdef void denseToSparse_analysis(
        intptr_t handle, size_t matA, size_t matB,
        int alg, intptr_t buffer) except *:
    _setStream(handle)
    status = cusparseDenseToSparse_analysis(
        <Handle>handle, <DnMatDescr>matA, <SpMatDescr>matB,
        <cusparseDenseToSparseAlg_t>alg, <void*>buffer)
    check_status(status)

cpdef void denseToSparse_convert(
        intptr_t handle, size_t matA, size_t matB,
        int alg, intptr_t buffer) except *:
    _setStream(handle)
    status = cusparseDenseToSparse_convert(
        <Handle>handle, <DnMatDescr>matA, <SpMatDescr>matB,
        <cusparseDenseToSparseAlg_t>alg, <void*>buffer)
    check_status(status)

# CSR2CSC
cpdef size_t csr2cscEx2_bufferSize(
        intptr_t handle, int m, int n, int nnz, intptr_t csrVal,
        intptr_t csrRowPtr, intptr_t csrColInd, intptr_t cscVal,
        intptr_t cscColPtr, intptr_t cscRowInd, DataType valType,
        Action copyValues, IndexBase idxBase, Csr2CscAlg alg) except? -1:
    cdef size_t bufferSize
    status = cusparseCsr2cscEx2_bufferSize(
        <Handle>handle, m, n, nnz, <const void*>csrVal, <const int*>csrRowPtr,
        <const int*>csrColInd, <void*>cscVal, <int*>cscColPtr, <int*>cscRowInd,
        valType, copyValues, idxBase, alg, &bufferSize)
    check_status(status)
    return bufferSize

cpdef void csr2cscEx2(
        intptr_t handle, int m, int n, int nnz, intptr_t csrVal,
        intptr_t csrRowPtr, intptr_t csrColInd, intptr_t cscVal,
        intptr_t cscColPtr, intptr_t cscRowInd, DataType valType,
        Action copyValues, IndexBase idxBase, Csr2CscAlg alg,
        intptr_t buffer) except *:
    setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseCsr2cscEx2(
        <Handle>handle, m, n, nnz, <const void*>csrVal, <const int*>csrRowPtr,
        <const int*>csrColInd, <void*>cscVal, <int*>cscColPtr, <int*>cscRowInd,
        valType, copyValues, idxBase, alg, <void*>buffer)
    check_status(status)
