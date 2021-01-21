# This code was automatically generated. Do not modify it directly.

cimport cython  # NOQA

from cupy_backends.cuda.api cimport driver
from cupy_backends.cuda.api.runtime cimport DataType
from cupy_backends.cuda cimport stream as stream_module


cdef extern from '../../cupy_complex.h':
    ctypedef struct cuComplex 'cuComplex':
        float x, y

    ctypedef struct cuDoubleComplex 'cuDoubleComplex':
        double x, y

cdef extern from '../../cupy_sparse.h' nogil:

    # cuSPARSE Management Function
    Status cusparseCreate(Handle* handle)
    Status cusparseDestroy(Handle handle)
    Status cusparseGetVersion(Handle handle, int* version)
    Status cusparseSetPointerMode(Handle handle, PointerMode mode)
    Status cusparseGetStream(Handle handle, driver.Stream* streamId)
    Status cusparseSetStream(Handle handle, driver.Stream streamId)

    # cuSPARSE Helper Function
    Status cusparseCreateMatDescr(MatDescr* descrA)
    Status cusparseDestroyMatDescr(MatDescr descrA)
    Status cusparseSetMatDiagType(MatDescr descrA, DiagType diagType)
    Status cusparseSetMatFillMode(MatDescr descrA, FillMode fillMode)
    Status cusparseSetMatIndexBase(MatDescr descrA, IndexBase base)
    Status cusparseSetMatType(MatDescr descrA, MatrixType type)
    Status cusparseCreateCsrsv2Info(csrsv2Info_t* info)
    Status cusparseDestroyCsrsv2Info(csrsv2Info_t info)
    Status cusparseCreateCsrsm2Info(csrsm2Info_t* info)
    Status cusparseDestroyCsrsm2Info(csrsm2Info_t info)
    Status cusparseCreateCsric02Info(csric02Info_t* info)
    Status cusparseDestroyCsric02Info(csric02Info_t info)
    Status cusparseCreateCsrilu02Info(csrilu02Info_t* info)
    Status cusparseDestroyCsrilu02Info(csrilu02Info_t info)
    Status cusparseCreateBsric02Info(bsric02Info_t* info)
    Status cusparseDestroyBsric02Info(bsric02Info_t info)
    Status cusparseCreateBsrilu02Info(bsrilu02Info_t* info)
    Status cusparseDestroyBsrilu02Info(bsrilu02Info_t info)
    Status cusparseCreateCsrgemm2Info(csrgemm2Info_t* info)
    Status cusparseDestroyCsrgemm2Info(csrgemm2Info_t info)

    # cuSPARSE Level 1 Function
    Status cusparseSgthr(Handle handle, int nnz, const float* y, float* xVal, const int* xInd, IndexBase idxBase)
    Status cusparseDgthr(Handle handle, int nnz, const double* y, double* xVal, const int* xInd, IndexBase idxBase)
    Status cusparseCgthr(Handle handle, int nnz, const cuComplex* y, cuComplex* xVal, const int* xInd, IndexBase idxBase)
    Status cusparseZgthr(Handle handle, int nnz, const cuDoubleComplex* y, cuDoubleComplex* xVal, const int* xInd, IndexBase idxBase)

    # cuSPARSE Level 2 Function
    # REMOVED
    Status cusparseScsrmv(Handle handle, Operation transA, int m, int n, int nnz, const float* alpha, const MatDescr descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const float* x, const float* beta, float* y)
    # REMOVED
    Status cusparseDcsrmv(Handle handle, Operation transA, int m, int n, int nnz, const double* alpha, const MatDescr descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const double* x, const double* beta, double* y)
    # REMOVED
    Status cusparseCcsrmv(Handle handle, Operation transA, int m, int n, int nnz, const cuComplex* alpha, const MatDescr descrA, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cuComplex* x, const cuComplex* beta, cuComplex* y)
    # REMOVED
    Status cusparseZcsrmv(Handle handle, Operation transA, int m, int n, int nnz, const cuDoubleComplex* alpha, const MatDescr descrA, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cuDoubleComplex* x, const cuDoubleComplex* beta, cuDoubleComplex* y)
    Status cusparseCsrmvEx_bufferSize(Handle handle, AlgMode alg, Operation transA, int m, int n, int nnz, const void* alpha, DataType alphatype, const MatDescr descrA, const void* csrValA, DataType csrValAtype, const int* csrRowPtrA, const int* csrColIndA, const void* x, DataType xtype, const void* beta, DataType betatype, void* y, DataType ytype, DataType executiontype, size_t* bufferSizeInBytes)
    Status cusparseCsrmvEx(Handle handle, AlgMode alg, Operation transA, int m, int n, int nnz, const void* alpha, DataType alphatype, const MatDescr descrA, const void* csrValA, DataType csrValAtype, const int* csrRowPtrA, const int* csrColIndA, const void* x, DataType xtype, const void* beta, DataType betatype, void* y, DataType ytype, DataType executiontype, void* buffer)
    Status cusparseScsrsv2_bufferSize(Handle handle, Operation transA, int m, int nnz, const MatDescr descrA, float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrsv2Info_t info, int* pBufferSizeInBytes)
    Status cusparseDcsrsv2_bufferSize(Handle handle, Operation transA, int m, int nnz, const MatDescr descrA, double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrsv2Info_t info, int* pBufferSizeInBytes)
    Status cusparseCcsrsv2_bufferSize(Handle handle, Operation transA, int m, int nnz, const MatDescr descrA, cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrsv2Info_t info, int* pBufferSizeInBytes)
    Status cusparseZcsrsv2_bufferSize(Handle handle, Operation transA, int m, int nnz, const MatDescr descrA, cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrsv2Info_t info, int* pBufferSizeInBytes)
    Status cusparseScsrsv2_analysis(Handle handle, Operation transA, int m, int nnz, const MatDescr descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrsv2Info_t info, SolvePolicy policy, void* pBuffer)
    Status cusparseDcsrsv2_analysis(Handle handle, Operation transA, int m, int nnz, const MatDescr descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrsv2Info_t info, SolvePolicy policy, void* pBuffer)
    Status cusparseCcsrsv2_analysis(Handle handle, Operation transA, int m, int nnz, const MatDescr descrA, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrsv2Info_t info, SolvePolicy policy, void* pBuffer)
    Status cusparseZcsrsv2_analysis(Handle handle, Operation transA, int m, int nnz, const MatDescr descrA, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrsv2Info_t info, SolvePolicy policy, void* pBuffer)
    Status cusparseScsrsv2_solve(Handle handle, Operation transA, int m, int nnz, const float* alpha, const MatDescr descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrsv2Info_t info, const float* f, float* x, SolvePolicy policy, void* pBuffer)
    Status cusparseDcsrsv2_solve(Handle handle, Operation transA, int m, int nnz, const double* alpha, const MatDescr descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrsv2Info_t info, const double* f, double* x, SolvePolicy policy, void* pBuffer)
    Status cusparseCcsrsv2_solve(Handle handle, Operation transA, int m, int nnz, const cuComplex* alpha, const MatDescr descrA, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrsv2Info_t info, const cuComplex* f, cuComplex* x, SolvePolicy policy, void* pBuffer)
    Status cusparseZcsrsv2_solve(Handle handle, Operation transA, int m, int nnz, const cuDoubleComplex* alpha, const MatDescr descrA, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrsv2Info_t info, const cuDoubleComplex* f, cuDoubleComplex* x, SolvePolicy policy, void* pBuffer)
    Status cusparseXcsrsv2_zeroPivot(Handle handle, csrsv2Info_t info, int* position)

    # cuSPARSE Level 3 Function
    # REMOVED
    Status cusparseScsrmm(Handle handle, Operation transA, int m, int n, int k, int nnz, const float* alpha, const MatDescr descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const float* B, int ldb, const float* beta, float* C, int ldc)
    # REMOVED
    Status cusparseDcsrmm(Handle handle, Operation transA, int m, int n, int k, int nnz, const double* alpha, const MatDescr descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const double* B, int ldb, const double* beta, double* C, int ldc)
    # REMOVED
    Status cusparseCcsrmm(Handle handle, Operation transA, int m, int n, int k, int nnz, const cuComplex* alpha, const MatDescr descrA, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cuComplex* B, int ldb, const cuComplex* beta, cuComplex* C, int ldc)
    # REMOVED
    Status cusparseZcsrmm(Handle handle, Operation transA, int m, int n, int k, int nnz, const cuDoubleComplex* alpha, const MatDescr descrA, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cuDoubleComplex* B, int ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc)
    # REMOVED
    Status cusparseScsrmm2(Handle handle, Operation transA, Operation transB, int m, int n, int k, int nnz, const float* alpha, const MatDescr descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const float* B, int ldb, const float* beta, float* C, int ldc)
    # REMOVED
    Status cusparseDcsrmm2(Handle handle, Operation transA, Operation transB, int m, int n, int k, int nnz, const double* alpha, const MatDescr descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const double* B, int ldb, const double* beta, double* C, int ldc)
    # REMOVED
    Status cusparseCcsrmm2(Handle handle, Operation transA, Operation transB, int m, int n, int k, int nnz, const cuComplex* alpha, const MatDescr descrA, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cuComplex* B, int ldb, const cuComplex* beta, cuComplex* C, int ldc)
    # REMOVED
    Status cusparseZcsrmm2(Handle handle, Operation transA, Operation transB, int m, int n, int k, int nnz, const cuDoubleComplex* alpha, const MatDescr descrA, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cuDoubleComplex* B, int ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc)
    Status cusparseScsrsm2_bufferSizeExt(Handle handle, int algo, Operation transA, Operation transB, int m, int nrhs, int nnz, const float* alpha, const MatDescr descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const float* B, int ldb, csrsm2Info_t info, SolvePolicy policy, size_t* pBufferSize)
    Status cusparseDcsrsm2_bufferSizeExt(Handle handle, int algo, Operation transA, Operation transB, int m, int nrhs, int nnz, const double* alpha, const MatDescr descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const double* B, int ldb, csrsm2Info_t info, SolvePolicy policy, size_t* pBufferSize)
    Status cusparseCcsrsm2_bufferSizeExt(Handle handle, int algo, Operation transA, Operation transB, int m, int nrhs, int nnz, const cuComplex* alpha, const MatDescr descrA, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cuComplex* B, int ldb, csrsm2Info_t info, SolvePolicy policy, size_t* pBufferSize)
    Status cusparseZcsrsm2_bufferSizeExt(Handle handle, int algo, Operation transA, Operation transB, int m, int nrhs, int nnz, const cuDoubleComplex* alpha, const MatDescr descrA, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cuDoubleComplex* B, int ldb, csrsm2Info_t info, SolvePolicy policy, size_t* pBufferSize)
    Status cusparseScsrsm2_analysis(Handle handle, int algo, Operation transA, Operation transB, int m, int nrhs, int nnz, const float* alpha, const MatDescr descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const float* B, int ldb, csrsm2Info_t info, SolvePolicy policy, void* pBuffer)
    Status cusparseDcsrsm2_analysis(Handle handle, int algo, Operation transA, Operation transB, int m, int nrhs, int nnz, const double* alpha, const MatDescr descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const double* B, int ldb, csrsm2Info_t info, SolvePolicy policy, void* pBuffer)
    Status cusparseCcsrsm2_analysis(Handle handle, int algo, Operation transA, Operation transB, int m, int nrhs, int nnz, const cuComplex* alpha, const MatDescr descrA, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cuComplex* B, int ldb, csrsm2Info_t info, SolvePolicy policy, void* pBuffer)
    Status cusparseZcsrsm2_analysis(Handle handle, int algo, Operation transA, Operation transB, int m, int nrhs, int nnz, const cuDoubleComplex* alpha, const MatDescr descrA, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cuDoubleComplex* B, int ldb, csrsm2Info_t info, SolvePolicy policy, void* pBuffer)
    Status cusparseScsrsm2_solve(Handle handle, int algo, Operation transA, Operation transB, int m, int nrhs, int nnz, const float* alpha, const MatDescr descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, float* B, int ldb, csrsm2Info_t info, SolvePolicy policy, void* pBuffer)
    Status cusparseDcsrsm2_solve(Handle handle, int algo, Operation transA, Operation transB, int m, int nrhs, int nnz, const double* alpha, const MatDescr descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, double* B, int ldb, csrsm2Info_t info, SolvePolicy policy, void* pBuffer)
    Status cusparseCcsrsm2_solve(Handle handle, int algo, Operation transA, Operation transB, int m, int nrhs, int nnz, const cuComplex* alpha, const MatDescr descrA, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, cuComplex* B, int ldb, csrsm2Info_t info, SolvePolicy policy, void* pBuffer)
    Status cusparseZcsrsm2_solve(Handle handle, int algo, Operation transA, Operation transB, int m, int nrhs, int nnz, const cuDoubleComplex* alpha, const MatDescr descrA, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, cuDoubleComplex* B, int ldb, csrsm2Info_t info, SolvePolicy policy, void* pBuffer)
    Status cusparseXcsrsm2_zeroPivot(Handle handle, csrsm2Info_t info, int* position)

    # cuSPARSE Extra Function
    # REMOVED
    Status cusparseXcsrgeamNnz(Handle handle, int m, int n, const MatDescr descrA, int nnzA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const MatDescr descrB, int nnzB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const MatDescr descrC, int* csrSortedRowPtrC, int* nnzTotalDevHostPtr)
    # REMOVED
    Status cusparseScsrgeam(Handle handle, int m, int n, const float* alpha, const MatDescr descrA, int nnzA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const float* beta, const MatDescr descrB, int nnzB, const float* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const MatDescr descrC, float* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC)
    # REMOVED
    Status cusparseDcsrgeam(Handle handle, int m, int n, const double* alpha, const MatDescr descrA, int nnzA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const double* beta, const MatDescr descrB, int nnzB, const double* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const MatDescr descrC, double* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC)
    # REMOVED
    Status cusparseCcsrgeam(Handle handle, int m, int n, const cuComplex* alpha, const MatDescr descrA, int nnzA, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cuComplex* beta, const MatDescr descrB, int nnzB, const cuComplex* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const MatDescr descrC, cuComplex* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC)
    # REMOVED
    Status cusparseZcsrgeam(Handle handle, int m, int n, const cuDoubleComplex* alpha, const MatDescr descrA, int nnzA, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cuDoubleComplex* beta, const MatDescr descrB, int nnzB, const cuDoubleComplex* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const MatDescr descrC, cuDoubleComplex* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC)
    Status cusparseScsrgeam2_bufferSizeExt(Handle handle, int m, int n, const float* alpha, const MatDescr descrA, int nnzA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const float* beta, const MatDescr descrB, int nnzB, const float* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const MatDescr descrC, const float* csrSortedValC, const int* csrSortedRowPtrC, const int* csrSortedColIndC, size_t* pBufferSizeInBytes)
    Status cusparseDcsrgeam2_bufferSizeExt(Handle handle, int m, int n, const double* alpha, const MatDescr descrA, int nnzA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const double* beta, const MatDescr descrB, int nnzB, const double* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const MatDescr descrC, const double* csrSortedValC, const int* csrSortedRowPtrC, const int* csrSortedColIndC, size_t* pBufferSizeInBytes)
    Status cusparseCcsrgeam2_bufferSizeExt(Handle handle, int m, int n, const cuComplex* alpha, const MatDescr descrA, int nnzA, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cuComplex* beta, const MatDescr descrB, int nnzB, const cuComplex* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const MatDescr descrC, const cuComplex* csrSortedValC, const int* csrSortedRowPtrC, const int* csrSortedColIndC, size_t* pBufferSizeInBytes)
    Status cusparseZcsrgeam2_bufferSizeExt(Handle handle, int m, int n, const cuDoubleComplex* alpha, const MatDescr descrA, int nnzA, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cuDoubleComplex* beta, const MatDescr descrB, int nnzB, const cuDoubleComplex* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const MatDescr descrC, const cuDoubleComplex* csrSortedValC, const int* csrSortedRowPtrC, const int* csrSortedColIndC, size_t* pBufferSizeInBytes)
    Status cusparseXcsrgeam2Nnz(Handle handle, int m, int n, const MatDescr descrA, int nnzA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const MatDescr descrB, int nnzB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const MatDescr descrC, int* csrSortedRowPtrC, int* nnzTotalDevHostPtr, void* workspace)
    Status cusparseScsrgeam2(Handle handle, int m, int n, const float* alpha, const MatDescr descrA, int nnzA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const float* beta, const MatDescr descrB, int nnzB, const float* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const MatDescr descrC, float* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC, void* pBuffer)
    Status cusparseDcsrgeam2(Handle handle, int m, int n, const double* alpha, const MatDescr descrA, int nnzA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const double* beta, const MatDescr descrB, int nnzB, const double* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const MatDescr descrC, double* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC, void* pBuffer)
    Status cusparseCcsrgeam2(Handle handle, int m, int n, const cuComplex* alpha, const MatDescr descrA, int nnzA, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cuComplex* beta, const MatDescr descrB, int nnzB, const cuComplex* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const MatDescr descrC, cuComplex* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC, void* pBuffer)
    Status cusparseZcsrgeam2(Handle handle, int m, int n, const cuDoubleComplex* alpha, const MatDescr descrA, int nnzA, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cuDoubleComplex* beta, const MatDescr descrB, int nnzB, const cuDoubleComplex* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const MatDescr descrC, cuDoubleComplex* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC, void* pBuffer)
    # REMOVED
    Status cusparseXcsrgemmNnz(Handle handle, Operation transA, Operation transB, int m, int n, int k, const MatDescr descrA, const int nnzA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const MatDescr descrB, const int nnzB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const MatDescr descrC, int* csrSortedRowPtrC, int* nnzTotalDevHostPtr)
    # REMOVED
    Status cusparseScsrgemm(Handle handle, Operation transA, Operation transB, int m, int n, int k, const MatDescr descrA, const int nnzA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const MatDescr descrB, const int nnzB, const float* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const MatDescr descrC, float* csrSortedValC, const int* csrSortedRowPtrC, int* csrSortedColIndC)
    # REMOVED
    Status cusparseDcsrgemm(Handle handle, Operation transA, Operation transB, int m, int n, int k, const MatDescr descrA, int nnzA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const MatDescr descrB, int nnzB, const double* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const MatDescr descrC, double* csrSortedValC, const int* csrSortedRowPtrC, int* csrSortedColIndC)
    # REMOVED
    Status cusparseCcsrgemm(Handle handle, Operation transA, Operation transB, int m, int n, int k, const MatDescr descrA, int nnzA, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const MatDescr descrB, int nnzB, const cuComplex* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const MatDescr descrC, cuComplex* csrSortedValC, const int* csrSortedRowPtrC, int* csrSortedColIndC)
    # REMOVED
    Status cusparseZcsrgemm(Handle handle, Operation transA, Operation transB, int m, int n, int k, const MatDescr descrA, int nnzA, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const MatDescr descrB, int nnzB, const cuDoubleComplex* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const MatDescr descrC, cuDoubleComplex* csrSortedValC, const int* csrSortedRowPtrC, int* csrSortedColIndC)
    Status cusparseScsrgemm2_bufferSizeExt(Handle handle, int m, int n, int k, const float* alpha, const MatDescr descrA, int nnzA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const MatDescr descrB, int nnzB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const float* beta, const MatDescr descrD, int nnzD, const int* csrSortedRowPtrD, const int* csrSortedColIndD, csrgemm2Info_t info, size_t* pBufferSizeInBytes)
    Status cusparseDcsrgemm2_bufferSizeExt(Handle handle, int m, int n, int k, const double* alpha, const MatDescr descrA, int nnzA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const MatDescr descrB, int nnzB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const double* beta, const MatDescr descrD, int nnzD, const int* csrSortedRowPtrD, const int* csrSortedColIndD, csrgemm2Info_t info, size_t* pBufferSizeInBytes)
    Status cusparseCcsrgemm2_bufferSizeExt(Handle handle, int m, int n, int k, const cuComplex* alpha, const MatDescr descrA, int nnzA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const MatDescr descrB, int nnzB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cuComplex* beta, const MatDescr descrD, int nnzD, const int* csrSortedRowPtrD, const int* csrSortedColIndD, csrgemm2Info_t info, size_t* pBufferSizeInBytes)
    Status cusparseZcsrgemm2_bufferSizeExt(Handle handle, int m, int n, int k, const cuDoubleComplex* alpha, const MatDescr descrA, int nnzA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const MatDescr descrB, int nnzB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cuDoubleComplex* beta, const MatDescr descrD, int nnzD, const int* csrSortedRowPtrD, const int* csrSortedColIndD, csrgemm2Info_t info, size_t* pBufferSizeInBytes)
    Status cusparseXcsrgemm2Nnz(Handle handle, int m, int n, int k, const MatDescr descrA, int nnzA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const MatDescr descrB, int nnzB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const MatDescr descrD, int nnzD, const int* csrSortedRowPtrD, const int* csrSortedColIndD, const MatDescr descrC, int* csrSortedRowPtrC, int* nnzTotalDevHostPtr, const csrgemm2Info_t info, void* pBuffer)
    Status cusparseScsrgemm2(Handle handle, int m, int n, int k, const float* alpha, const MatDescr descrA, int nnzA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const MatDescr descrB, int nnzB, const float* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const float* beta, const MatDescr descrD, int nnzD, const float* csrSortedValD, const int* csrSortedRowPtrD, const int* csrSortedColIndD, const MatDescr descrC, float* csrSortedValC, const int* csrSortedRowPtrC, int* csrSortedColIndC, const csrgemm2Info_t info, void* pBuffer)
    Status cusparseDcsrgemm2(Handle handle, int m, int n, int k, const double* alpha, const MatDescr descrA, int nnzA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const MatDescr descrB, int nnzB, const double* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const double* beta, const MatDescr descrD, int nnzD, const double* csrSortedValD, const int* csrSortedRowPtrD, const int* csrSortedColIndD, const MatDescr descrC, double* csrSortedValC, const int* csrSortedRowPtrC, int* csrSortedColIndC, const csrgemm2Info_t info, void* pBuffer)
    Status cusparseCcsrgemm2(Handle handle, int m, int n, int k, const cuComplex* alpha, const MatDescr descrA, int nnzA, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const MatDescr descrB, int nnzB, const cuComplex* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cuComplex* beta, const MatDescr descrD, int nnzD, const cuComplex* csrSortedValD, const int* csrSortedRowPtrD, const int* csrSortedColIndD, const MatDescr descrC, cuComplex* csrSortedValC, const int* csrSortedRowPtrC, int* csrSortedColIndC, const csrgemm2Info_t info, void* pBuffer)
    Status cusparseZcsrgemm2(Handle handle, int m, int n, int k, const cuDoubleComplex* alpha, const MatDescr descrA, int nnzA, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const MatDescr descrB, int nnzB, const cuDoubleComplex* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cuDoubleComplex* beta, const MatDescr descrD, int nnzD, const cuDoubleComplex* csrSortedValD, const int* csrSortedRowPtrD, const int* csrSortedColIndD, const MatDescr descrC, cuDoubleComplex* csrSortedValC, const int* csrSortedRowPtrC, int* csrSortedColIndC, const csrgemm2Info_t info, void* pBuffer)

    # cuSPARSE Preconditioners - Incomplete Cholesky Factorization: level 0
    Status cusparseScsric02_bufferSize(Handle handle, int m, int nnz, const MatDescr descrA, float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csric02Info_t info, int* pBufferSizeInBytes)
    Status cusparseDcsric02_bufferSize(Handle handle, int m, int nnz, const MatDescr descrA, double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csric02Info_t info, int* pBufferSizeInBytes)
    Status cusparseCcsric02_bufferSize(Handle handle, int m, int nnz, const MatDescr descrA, cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csric02Info_t info, int* pBufferSizeInBytes)
    Status cusparseZcsric02_bufferSize(Handle handle, int m, int nnz, const MatDescr descrA, cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csric02Info_t info, int* pBufferSizeInBytes)
    Status cusparseScsric02_analysis(Handle handle, int m, int nnz, const MatDescr descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csric02Info_t info, SolvePolicy policy, void* pBuffer)
    Status cusparseDcsric02_analysis(Handle handle, int m, int nnz, const MatDescr descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csric02Info_t info, SolvePolicy policy, void* pBuffer)
    Status cusparseCcsric02_analysis(Handle handle, int m, int nnz, const MatDescr descrA, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csric02Info_t info, SolvePolicy policy, void* pBuffer)
    Status cusparseZcsric02_analysis(Handle handle, int m, int nnz, const MatDescr descrA, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csric02Info_t info, SolvePolicy policy, void* pBuffer)
    Status cusparseScsric02(Handle handle, int m, int nnz, const MatDescr descrA, float* csrSortedValA_valM, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csric02Info_t info, SolvePolicy policy, void* pBuffer)
    Status cusparseDcsric02(Handle handle, int m, int nnz, const MatDescr descrA, double* csrSortedValA_valM, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csric02Info_t info, SolvePolicy policy, void* pBuffer)
    Status cusparseCcsric02(Handle handle, int m, int nnz, const MatDescr descrA, cuComplex* csrSortedValA_valM, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csric02Info_t info, SolvePolicy policy, void* pBuffer)
    Status cusparseZcsric02(Handle handle, int m, int nnz, const MatDescr descrA, cuDoubleComplex* csrSortedValA_valM, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csric02Info_t info, SolvePolicy policy, void* pBuffer)
    Status cusparseXcsric02_zeroPivot(Handle handle, csric02Info_t info, int* position)
    Status cusparseSbsric02_bufferSize(Handle handle, Direction dirA, int mb, int nnzb, const MatDescr descrA, float* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsric02Info_t info, int* pBufferSizeInBytes)
    Status cusparseDbsric02_bufferSize(Handle handle, Direction dirA, int mb, int nnzb, const MatDescr descrA, double* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsric02Info_t info, int* pBufferSizeInBytes)
    Status cusparseCbsric02_bufferSize(Handle handle, Direction dirA, int mb, int nnzb, const MatDescr descrA, cuComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsric02Info_t info, int* pBufferSizeInBytes)
    Status cusparseZbsric02_bufferSize(Handle handle, Direction dirA, int mb, int nnzb, const MatDescr descrA, cuDoubleComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsric02Info_t info, int* pBufferSizeInBytes)
    Status cusparseSbsric02_analysis(Handle handle, Direction dirA, int mb, int nnzb, const MatDescr descrA, const float* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsric02Info_t info, SolvePolicy policy, void* pInputBuffer)
    Status cusparseDbsric02_analysis(Handle handle, Direction dirA, int mb, int nnzb, const MatDescr descrA, const double* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsric02Info_t info, SolvePolicy policy, void* pInputBuffer)
    Status cusparseCbsric02_analysis(Handle handle, Direction dirA, int mb, int nnzb, const MatDescr descrA, const cuComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsric02Info_t info, SolvePolicy policy, void* pInputBuffer)
    Status cusparseZbsric02_analysis(Handle handle, Direction dirA, int mb, int nnzb, const MatDescr descrA, const cuDoubleComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsric02Info_t info, SolvePolicy policy, void* pInputBuffer)
    Status cusparseSbsric02(Handle handle, Direction dirA, int mb, int nnzb, const MatDescr descrA, float* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsric02Info_t info, SolvePolicy policy, void* pBuffer)
    Status cusparseDbsric02(Handle handle, Direction dirA, int mb, int nnzb, const MatDescr descrA, double* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsric02Info_t info, SolvePolicy policy, void* pBuffer)
    Status cusparseCbsric02(Handle handle, Direction dirA, int mb, int nnzb, const MatDescr descrA, cuComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsric02Info_t info, SolvePolicy policy, void* pBuffer)
    Status cusparseZbsric02(Handle handle, Direction dirA, int mb, int nnzb, const MatDescr descrA, cuDoubleComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsric02Info_t info, SolvePolicy policy, void* pBuffer)
    Status cusparseXbsric02_zeroPivot(Handle handle, bsric02Info_t info, int* position)

    # cuSPARSE Preconditioners - Incomplete LU Factorization: level 0
    Status cusparseScsrilu02_numericBoost(Handle handle, csrilu02Info_t info, int enable_boost, double* tol, float* boost_val)
    Status cusparseDcsrilu02_numericBoost(Handle handle, csrilu02Info_t info, int enable_boost, double* tol, double* boost_val)
    Status cusparseCcsrilu02_numericBoost(Handle handle, csrilu02Info_t info, int enable_boost, double* tol, cuComplex* boost_val)
    Status cusparseZcsrilu02_numericBoost(Handle handle, csrilu02Info_t info, int enable_boost, double* tol, cuDoubleComplex* boost_val)
    Status cusparseScsrilu02_bufferSize(Handle handle, int m, int nnz, const MatDescr descrA, float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrilu02Info_t info, int* pBufferSizeInBytes)
    Status cusparseDcsrilu02_bufferSize(Handle handle, int m, int nnz, const MatDescr descrA, double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrilu02Info_t info, int* pBufferSizeInBytes)
    Status cusparseCcsrilu02_bufferSize(Handle handle, int m, int nnz, const MatDescr descrA, cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrilu02Info_t info, int* pBufferSizeInBytes)
    Status cusparseZcsrilu02_bufferSize(Handle handle, int m, int nnz, const MatDescr descrA, cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrilu02Info_t info, int* pBufferSizeInBytes)
    Status cusparseScsrilu02_analysis(Handle handle, int m, int nnz, const MatDescr descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrilu02Info_t info, SolvePolicy policy, void* pBuffer)
    Status cusparseDcsrilu02_analysis(Handle handle, int m, int nnz, const MatDescr descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrilu02Info_t info, SolvePolicy policy, void* pBuffer)
    Status cusparseCcsrilu02_analysis(Handle handle, int m, int nnz, const MatDescr descrA, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrilu02Info_t info, SolvePolicy policy, void* pBuffer)
    Status cusparseZcsrilu02_analysis(Handle handle, int m, int nnz, const MatDescr descrA, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrilu02Info_t info, SolvePolicy policy, void* pBuffer)
    Status cusparseScsrilu02(Handle handle, int m, int nnz, const MatDescr descrA, float* csrSortedValA_valM, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrilu02Info_t info, SolvePolicy policy, void* pBuffer)
    Status cusparseDcsrilu02(Handle handle, int m, int nnz, const MatDescr descrA, double* csrSortedValA_valM, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrilu02Info_t info, SolvePolicy policy, void* pBuffer)
    Status cusparseCcsrilu02(Handle handle, int m, int nnz, const MatDescr descrA, cuComplex* csrSortedValA_valM, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrilu02Info_t info, SolvePolicy policy, void* pBuffer)
    Status cusparseZcsrilu02(Handle handle, int m, int nnz, const MatDescr descrA, cuDoubleComplex* csrSortedValA_valM, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrilu02Info_t info, SolvePolicy policy, void* pBuffer)
    Status cusparseXcsrilu02_zeroPivot(Handle handle, csrilu02Info_t info, int* position)
    Status cusparseSbsrilu02_numericBoost(Handle handle, bsrilu02Info_t info, int enable_boost, double* tol, float* boost_val)
    Status cusparseDbsrilu02_numericBoost(Handle handle, bsrilu02Info_t info, int enable_boost, double* tol, double* boost_val)
    Status cusparseCbsrilu02_numericBoost(Handle handle, bsrilu02Info_t info, int enable_boost, double* tol, cuComplex* boost_val)
    Status cusparseZbsrilu02_numericBoost(Handle handle, bsrilu02Info_t info, int enable_boost, double* tol, cuDoubleComplex* boost_val)
    Status cusparseSbsrilu02_bufferSize(Handle handle, Direction dirA, int mb, int nnzb, const MatDescr descrA, float* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsrilu02Info_t info, int* pBufferSizeInBytes)
    Status cusparseDbsrilu02_bufferSize(Handle handle, Direction dirA, int mb, int nnzb, const MatDescr descrA, double* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsrilu02Info_t info, int* pBufferSizeInBytes)
    Status cusparseCbsrilu02_bufferSize(Handle handle, Direction dirA, int mb, int nnzb, const MatDescr descrA, cuComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsrilu02Info_t info, int* pBufferSizeInBytes)
    Status cusparseZbsrilu02_bufferSize(Handle handle, Direction dirA, int mb, int nnzb, const MatDescr descrA, cuDoubleComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsrilu02Info_t info, int* pBufferSizeInBytes)
    Status cusparseSbsrilu02_analysis(Handle handle, Direction dirA, int mb, int nnzb, const MatDescr descrA, float* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsrilu02Info_t info, SolvePolicy policy, void* pBuffer)
    Status cusparseDbsrilu02_analysis(Handle handle, Direction dirA, int mb, int nnzb, const MatDescr descrA, double* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsrilu02Info_t info, SolvePolicy policy, void* pBuffer)
    Status cusparseCbsrilu02_analysis(Handle handle, Direction dirA, int mb, int nnzb, const MatDescr descrA, cuComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsrilu02Info_t info, SolvePolicy policy, void* pBuffer)
    Status cusparseZbsrilu02_analysis(Handle handle, Direction dirA, int mb, int nnzb, const MatDescr descrA, cuDoubleComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsrilu02Info_t info, SolvePolicy policy, void* pBuffer)
    Status cusparseSbsrilu02(Handle handle, Direction dirA, int mb, int nnzb, const MatDescr descrA, float* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsrilu02Info_t info, SolvePolicy policy, void* pBuffer)
    Status cusparseDbsrilu02(Handle handle, Direction dirA, int mb, int nnzb, const MatDescr descrA, double* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsrilu02Info_t info, SolvePolicy policy, void* pBuffer)
    Status cusparseCbsrilu02(Handle handle, Direction dirA, int mb, int nnzb, const MatDescr descrA, cuComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsrilu02Info_t info, SolvePolicy policy, void* pBuffer)
    Status cusparseZbsrilu02(Handle handle, Direction dirA, int mb, int nnzb, const MatDescr descrA, cuDoubleComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsrilu02Info_t info, SolvePolicy policy, void* pBuffer)
    Status cusparseXbsrilu02_zeroPivot(Handle handle, bsrilu02Info_t info, int* position)

    # cuSPARSE Preconditioners - Tridiagonal Solve
    Status cusparseSgtsv2_bufferSizeExt(Handle handle, int m, int n, const float* dl, const float* d, const float* du, const float* B, int ldb, size_t* bufferSizeInBytes)
    Status cusparseDgtsv2_bufferSizeExt(Handle handle, int m, int n, const double* dl, const double* d, const double* du, const double* B, int ldb, size_t* bufferSizeInBytes)
    Status cusparseCgtsv2_bufferSizeExt(Handle handle, int m, int n, const cuComplex* dl, const cuComplex* d, const cuComplex* du, const cuComplex* B, int ldb, size_t* bufferSizeInBytes)
    Status cusparseZgtsv2_bufferSizeExt(Handle handle, int m, int n, const cuDoubleComplex* dl, const cuDoubleComplex* d, const cuDoubleComplex* du, const cuDoubleComplex* B, int ldb, size_t* bufferSizeInBytes)
    Status cusparseSgtsv2(Handle handle, int m, int n, const float* dl, const float* d, const float* du, float* B, int ldb, void* pBuffer)
    Status cusparseDgtsv2(Handle handle, int m, int n, const double* dl, const double* d, const double* du, double* B, int ldb, void* pBuffer)
    Status cusparseCgtsv2(Handle handle, int m, int n, const cuComplex* dl, const cuComplex* d, const cuComplex* du, cuComplex* B, int ldb, void* pBuffer)
    Status cusparseZgtsv2(Handle handle, int m, int n, const cuDoubleComplex* dl, const cuDoubleComplex* d, const cuDoubleComplex* du, cuDoubleComplex* B, int ldb, void* pBuffer)
    Status cusparseSgtsv2_nopivot_bufferSizeExt(Handle handle, int m, int n, const float* dl, const float* d, const float* du, const float* B, int ldb, size_t* bufferSizeInBytes)
    Status cusparseDgtsv2_nopivot_bufferSizeExt(Handle handle, int m, int n, const double* dl, const double* d, const double* du, const double* B, int ldb, size_t* bufferSizeInBytes)
    Status cusparseCgtsv2_nopivot_bufferSizeExt(Handle handle, int m, int n, const cuComplex* dl, const cuComplex* d, const cuComplex* du, const cuComplex* B, int ldb, size_t* bufferSizeInBytes)
    Status cusparseZgtsv2_nopivot_bufferSizeExt(Handle handle, int m, int n, const cuDoubleComplex* dl, const cuDoubleComplex* d, const cuDoubleComplex* du, const cuDoubleComplex* B, int ldb, size_t* bufferSizeInBytes)
    Status cusparseSgtsv2_nopivot(Handle handle, int m, int n, const float* dl, const float* d, const float* du, float* B, int ldb, void* pBuffer)
    Status cusparseDgtsv2_nopivot(Handle handle, int m, int n, const double* dl, const double* d, const double* du, double* B, int ldb, void* pBuffer)
    Status cusparseCgtsv2_nopivot(Handle handle, int m, int n, const cuComplex* dl, const cuComplex* d, const cuComplex* du, cuComplex* B, int ldb, void* pBuffer)
    Status cusparseZgtsv2_nopivot(Handle handle, int m, int n, const cuDoubleComplex* dl, const cuDoubleComplex* d, const cuDoubleComplex* du, cuDoubleComplex* B, int ldb, void* pBuffer)

    # cuSPARSE Preconditioners - Batched Tridiagonal Solve
    Status cusparseSgtsv2StridedBatch_bufferSizeExt(Handle handle, int m, const float* dl, const float* d, const float* du, const float* x, int batchCount, int batchStride, size_t* bufferSizeInBytes)
    Status cusparseDgtsv2StridedBatch_bufferSizeExt(Handle handle, int m, const double* dl, const double* d, const double* du, const double* x, int batchCount, int batchStride, size_t* bufferSizeInBytes)
    Status cusparseCgtsv2StridedBatch_bufferSizeExt(Handle handle, int m, const cuComplex* dl, const cuComplex* d, const cuComplex* du, const cuComplex* x, int batchCount, int batchStride, size_t* bufferSizeInBytes)
    Status cusparseZgtsv2StridedBatch_bufferSizeExt(Handle handle, int m, const cuDoubleComplex* dl, const cuDoubleComplex* d, const cuDoubleComplex* du, const cuDoubleComplex* x, int batchCount, int batchStride, size_t* bufferSizeInBytes)
    Status cusparseSgtsv2StridedBatch(Handle handle, int m, const float* dl, const float* d, const float* du, float* x, int batchCount, int batchStride, void* pBuffer)
    Status cusparseDgtsv2StridedBatch(Handle handle, int m, const double* dl, const double* d, const double* du, double* x, int batchCount, int batchStride, void* pBuffer)
    Status cusparseCgtsv2StridedBatch(Handle handle, int m, const cuComplex* dl, const cuComplex* d, const cuComplex* du, cuComplex* x, int batchCount, int batchStride, void* pBuffer)
    Status cusparseZgtsv2StridedBatch(Handle handle, int m, const cuDoubleComplex* dl, const cuDoubleComplex* d, const cuDoubleComplex* du, cuDoubleComplex* x, int batchCount, int batchStride, void* pBuffer)
    Status cusparseSgtsvInterleavedBatch_bufferSizeExt(Handle handle, int algo, int m, const float* dl, const float* d, const float* du, const float* x, int batchCount, size_t* pBufferSizeInBytes)
    Status cusparseDgtsvInterleavedBatch_bufferSizeExt(Handle handle, int algo, int m, const double* dl, const double* d, const double* du, const double* x, int batchCount, size_t* pBufferSizeInBytes)
    Status cusparseCgtsvInterleavedBatch_bufferSizeExt(Handle handle, int algo, int m, const cuComplex* dl, const cuComplex* d, const cuComplex* du, const cuComplex* x, int batchCount, size_t* pBufferSizeInBytes)
    Status cusparseZgtsvInterleavedBatch_bufferSizeExt(Handle handle, int algo, int m, const cuDoubleComplex* dl, const cuDoubleComplex* d, const cuDoubleComplex* du, const cuDoubleComplex* x, int batchCount, size_t* pBufferSizeInBytes)
    Status cusparseSgtsvInterleavedBatch(Handle handle, int algo, int m, float* dl, float* d, float* du, float* x, int batchCount, void* pBuffer)
    Status cusparseDgtsvInterleavedBatch(Handle handle, int algo, int m, double* dl, double* d, double* du, double* x, int batchCount, void* pBuffer)
    Status cusparseCgtsvInterleavedBatch(Handle handle, int algo, int m, cuComplex* dl, cuComplex* d, cuComplex* du, cuComplex* x, int batchCount, void* pBuffer)
    Status cusparseZgtsvInterleavedBatch(Handle handle, int algo, int m, cuDoubleComplex* dl, cuDoubleComplex* d, cuDoubleComplex* du, cuDoubleComplex* x, int batchCount, void* pBuffer)

    # cuSPARSE Preconditioners - Batched Pentadiagonal Solve
    Status cusparseSgpsvInterleavedBatch_bufferSizeExt(Handle handle, int algo, int m, const float* ds, const float* dl, const float* d, const float* du, const float* dw, const float* x, int batchCount, size_t* pBufferSizeInBytes)
    Status cusparseDgpsvInterleavedBatch_bufferSizeExt(Handle handle, int algo, int m, const double* ds, const double* dl, const double* d, const double* du, const double* dw, const double* x, int batchCount, size_t* pBufferSizeInBytes)
    Status cusparseCgpsvInterleavedBatch_bufferSizeExt(Handle handle, int algo, int m, const cuComplex* ds, const cuComplex* dl, const cuComplex* d, const cuComplex* du, const cuComplex* dw, const cuComplex* x, int batchCount, size_t* pBufferSizeInBytes)
    Status cusparseZgpsvInterleavedBatch_bufferSizeExt(Handle handle, int algo, int m, const cuDoubleComplex* ds, const cuDoubleComplex* dl, const cuDoubleComplex* d, const cuDoubleComplex* du, const cuDoubleComplex* dw, const cuDoubleComplex* x, int batchCount, size_t* pBufferSizeInBytes)
    Status cusparseSgpsvInterleavedBatch(Handle handle, int algo, int m, float* ds, float* dl, float* d, float* du, float* dw, float* x, int batchCount, void* pBuffer)
    Status cusparseDgpsvInterleavedBatch(Handle handle, int algo, int m, double* ds, double* dl, double* d, double* du, double* dw, double* x, int batchCount, void* pBuffer)
    Status cusparseCgpsvInterleavedBatch(Handle handle, int algo, int m, cuComplex* ds, cuComplex* dl, cuComplex* d, cuComplex* du, cuComplex* dw, cuComplex* x, int batchCount, void* pBuffer)
    Status cusparseZgpsvInterleavedBatch(Handle handle, int algo, int m, cuDoubleComplex* ds, cuDoubleComplex* dl, cuDoubleComplex* d, cuDoubleComplex* du, cuDoubleComplex* dw, cuDoubleComplex* x, int batchCount, void* pBuffer)

    # cuSPARSE Reorderings

    # cuSPARSE Format Conversion
    Status cusparseXcoo2csr(Handle handle, const int* cooRowInd, int nnz, int m, int* csrSortedRowPtr, IndexBase idxBase)
    Status cusparseScsc2dense(Handle handle, int m, int n, const MatDescr descrA, const float* cscSortedValA, const int* cscSortedRowIndA, const int* cscSortedColPtrA, float* A, int lda)
    Status cusparseDcsc2dense(Handle handle, int m, int n, const MatDescr descrA, const double* cscSortedValA, const int* cscSortedRowIndA, const int* cscSortedColPtrA, double* A, int lda)
    Status cusparseCcsc2dense(Handle handle, int m, int n, const MatDescr descrA, const cuComplex* cscSortedValA, const int* cscSortedRowIndA, const int* cscSortedColPtrA, cuComplex* A, int lda)
    Status cusparseZcsc2dense(Handle handle, int m, int n, const MatDescr descrA, const cuDoubleComplex* cscSortedValA, const int* cscSortedRowIndA, const int* cscSortedColPtrA, cuDoubleComplex* A, int lda)
    Status cusparseXcsr2coo(Handle handle, const int* csrSortedRowPtr, int nnz, int m, int* cooRowInd, IndexBase idxBase)
    # REMOVED
    Status cusparseScsr2csc(Handle handle, int m, int n, int nnz, const float* csrSortedVal, const int* csrSortedRowPtr, const int* csrSortedColInd, float* cscSortedVal, int* cscSortedRowInd, int* cscSortedColPtr, Action copyValues, IndexBase idxBase)
    # REMOVED
    Status cusparseDcsr2csc(Handle handle, int m, int n, int nnz, const double* csrSortedVal, const int* csrSortedRowPtr, const int* csrSortedColInd, double* cscSortedVal, int* cscSortedRowInd, int* cscSortedColPtr, Action copyValues, IndexBase idxBase)
    # REMOVED
    Status cusparseCcsr2csc(Handle handle, int m, int n, int nnz, const cuComplex* csrSortedVal, const int* csrSortedRowPtr, const int* csrSortedColInd, cuComplex* cscSortedVal, int* cscSortedRowInd, int* cscSortedColPtr, Action copyValues, IndexBase idxBase)
    # REMOVED
    Status cusparseZcsr2csc(Handle handle, int m, int n, int nnz, const cuDoubleComplex* csrSortedVal, const int* csrSortedRowPtr, const int* csrSortedColInd, cuDoubleComplex* cscSortedVal, int* cscSortedRowInd, int* cscSortedColPtr, Action copyValues, IndexBase idxBase)
    Status cusparseCsr2cscEx2_bufferSize(Handle handle, int m, int n, int nnz, const void* csrVal, const int* csrRowPtr, const int* csrColInd, void* cscVal, int* cscColPtr, int* cscRowInd, DataType valType, Action copyValues, IndexBase idxBase, Csr2CscAlg alg, size_t* bufferSize)
    Status cusparseCsr2cscEx2(Handle handle, int m, int n, int nnz, const void* csrVal, const int* csrRowPtr, const int* csrColInd, void* cscVal, int* cscColPtr, int* cscRowInd, DataType valType, Action copyValues, IndexBase idxBase, Csr2CscAlg alg, void* buffer)
    Status cusparseScsr2dense(Handle handle, int m, int n, const MatDescr descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, float* A, int lda)
    Status cusparseDcsr2dense(Handle handle, int m, int n, const MatDescr descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, double* A, int lda)
    Status cusparseCcsr2dense(Handle handle, int m, int n, const MatDescr descrA, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, cuComplex* A, int lda)
    Status cusparseZcsr2dense(Handle handle, int m, int n, const MatDescr descrA, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, cuDoubleComplex* A, int lda)
    Status cusparseSnnz_compress(Handle handle, int m, const MatDescr descr, const float* csrSortedValA, const int* csrSortedRowPtrA, int* nnzPerRow, int* nnzC, float tol)
    Status cusparseDnnz_compress(Handle handle, int m, const MatDescr descr, const double* csrSortedValA, const int* csrSortedRowPtrA, int* nnzPerRow, int* nnzC, double tol)
    Status cusparseCnnz_compress(Handle handle, int m, const MatDescr descr, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA, int* nnzPerRow, int* nnzC, cuComplex tol)
    Status cusparseZnnz_compress(Handle handle, int m, const MatDescr descr, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, int* nnzPerRow, int* nnzC, cuDoubleComplex tol)
    Status cusparseScsr2csr_compress(Handle handle, int m, int n, const MatDescr descrA, const float* csrSortedValA, const int* csrSortedColIndA, const int* csrSortedRowPtrA, int nnzA, const int* nnzPerRow, float* csrSortedValC, int* csrSortedColIndC, int* csrSortedRowPtrC, float tol)
    Status cusparseDcsr2csr_compress(Handle handle, int m, int n, const MatDescr descrA, const double* csrSortedValA, const int* csrSortedColIndA, const int* csrSortedRowPtrA, int nnzA, const int* nnzPerRow, double* csrSortedValC, int* csrSortedColIndC, int* csrSortedRowPtrC, double tol)
    Status cusparseCcsr2csr_compress(Handle handle, int m, int n, const MatDescr descrA, const cuComplex* csrSortedValA, const int* csrSortedColIndA, const int* csrSortedRowPtrA, int nnzA, const int* nnzPerRow, cuComplex* csrSortedValC, int* csrSortedColIndC, int* csrSortedRowPtrC, cuComplex tol)
    Status cusparseZcsr2csr_compress(Handle handle, int m, int n, const MatDescr descrA, const cuDoubleComplex* csrSortedValA, const int* csrSortedColIndA, const int* csrSortedRowPtrA, int nnzA, const int* nnzPerRow, cuDoubleComplex* csrSortedValC, int* csrSortedColIndC, int* csrSortedRowPtrC, cuDoubleComplex tol)
    Status cusparseSdense2csc(Handle handle, int m, int n, const MatDescr descrA, const float* A, int lda, const int* nnzPerCol, float* cscSortedValA, int* cscSortedRowIndA, int* cscSortedColPtrA)
    Status cusparseDdense2csc(Handle handle, int m, int n, const MatDescr descrA, const double* A, int lda, const int* nnzPerCol, double* cscSortedValA, int* cscSortedRowIndA, int* cscSortedColPtrA)
    Status cusparseCdense2csc(Handle handle, int m, int n, const MatDescr descrA, const cuComplex* A, int lda, const int* nnzPerCol, cuComplex* cscSortedValA, int* cscSortedRowIndA, int* cscSortedColPtrA)
    Status cusparseZdense2csc(Handle handle, int m, int n, const MatDescr descrA, const cuDoubleComplex* A, int lda, const int* nnzPerCol, cuDoubleComplex* cscSortedValA, int* cscSortedRowIndA, int* cscSortedColPtrA)
    Status cusparseSdense2csr(Handle handle, int m, int n, const MatDescr descrA, const float* A, int lda, const int* nnzPerRow, float* csrSortedValA, int* csrSortedRowPtrA, int* csrSortedColIndA)
    Status cusparseDdense2csr(Handle handle, int m, int n, const MatDescr descrA, const double* A, int lda, const int* nnzPerRow, double* csrSortedValA, int* csrSortedRowPtrA, int* csrSortedColIndA)
    Status cusparseCdense2csr(Handle handle, int m, int n, const MatDescr descrA, const cuComplex* A, int lda, const int* nnzPerRow, cuComplex* csrSortedValA, int* csrSortedRowPtrA, int* csrSortedColIndA)
    Status cusparseZdense2csr(Handle handle, int m, int n, const MatDescr descrA, const cuDoubleComplex* A, int lda, const int* nnzPerRow, cuDoubleComplex* csrSortedValA, int* csrSortedRowPtrA, int* csrSortedColIndA)
    Status cusparseSnnz(Handle handle, Direction dirA, int m, int n, const MatDescr descrA, const float* A, int lda, int* nnzPerRowCol, int* nnzTotalDevHostPtr)
    Status cusparseDnnz(Handle handle, Direction dirA, int m, int n, const MatDescr descrA, const double* A, int lda, int* nnzPerRowCol, int* nnzTotalDevHostPtr)
    Status cusparseCnnz(Handle handle, Direction dirA, int m, int n, const MatDescr descrA, const cuComplex* A, int lda, int* nnzPerRowCol, int* nnzTotalDevHostPtr)
    Status cusparseZnnz(Handle handle, Direction dirA, int m, int n, const MatDescr descrA, const cuDoubleComplex* A, int lda, int* nnzPerRowCol, int* nnzTotalDevHostPtr)
    Status cusparseCreateIdentityPermutation(Handle handle, int n, int* p)
    Status cusparseXcoosort_bufferSizeExt(Handle handle, int m, int n, int nnz, const int* cooRowsA, const int* cooColsA, size_t* pBufferSizeInBytes)
    Status cusparseXcoosortByRow(Handle handle, int m, int n, int nnz, int* cooRowsA, int* cooColsA, int* P, void* pBuffer)
    Status cusparseXcoosortByColumn(Handle handle, int m, int n, int nnz, int* cooRowsA, int* cooColsA, int* P, void* pBuffer)
    Status cusparseXcsrsort_bufferSizeExt(Handle handle, int m, int n, int nnz, const int* csrRowPtrA, const int* csrColIndA, size_t* pBufferSizeInBytes)
    Status cusparseXcsrsort(Handle handle, int m, int n, int nnz, const MatDescr descrA, const int* csrRowPtrA, int* csrColIndA, int* P, void* pBuffer)
    Status cusparseXcscsort_bufferSizeExt(Handle handle, int m, int n, int nnz, const int* cscColPtrA, const int* cscRowIndA, size_t* pBufferSizeInBytes)
    Status cusparseXcscsort(Handle handle, int m, int n, int nnz, const MatDescr descrA, const int* cscColPtrA, int* cscRowIndA, int* P, void* pBuffer)

    # cuSPARSE Generic API - Sparse Vector APIs
    Status cusparseCreateSpVec(SpVecDescr* spVecDescr, int64_t size, int64_t nnz, void* indices, void* values, IndexType idxType, IndexBase idxBase, DataType valueType)
    Status cusparseDestroySpVec(SpVecDescr spVecDescr)
    Status cusparseSpVecGet(SpVecDescr spVecDescr, int64_t* size, int64_t* nnz, void** indices, void** values, IndexType* idxType, IndexBase* idxBase, DataType* valueType)
    Status cusparseSpVecGetIndexBase(SpVecDescr spVecDescr, IndexBase* idxBase)
    Status cusparseSpVecGetValues(SpVecDescr spVecDescr, void** values)
    Status cusparseSpVecSetValues(SpVecDescr spVecDescr, void* values)

    # cuSPARSE Generic API - Sparse Matrix APIs
    Status cusparseCreateCoo(SpMatDescr* spMatDescr, int64_t rows, int64_t cols, int64_t nnz, void* cooRowInd, void* cooColInd, void* cooValues, IndexType cooIdxType, IndexBase idxBase, DataType valueType)
    Status cusparseCreateCooAoS(SpMatDescr* spMatDescr, int64_t rows, int64_t cols, int64_t nnz, void* cooInd, void* cooValues, IndexType cooIdxType, IndexBase idxBase, DataType valueType)
    Status cusparseCreateCsr(SpMatDescr* spMatDescr, int64_t rows, int64_t cols, int64_t nnz, void* csrRowOffsets, void* csrColInd, void* csrValues, IndexType csrRowOffsetsType, IndexType csrColIndType, IndexBase idxBase, DataType valueType)
    Status cusparseDestroySpMat(SpMatDescr spMatDescr)
    Status cusparseCooGet(SpMatDescr spMatDescr, int64_t* rows, int64_t* cols, int64_t* nnz, void** cooRowInd, void** cooColInd, void** cooValues, IndexType* idxType, IndexBase* idxBase, DataType* valueType)
    Status cusparseCooAoSGet(SpMatDescr spMatDescr, int64_t* rows, int64_t* cols, int64_t* nnz, void** cooInd, void** cooValues, IndexType* idxType, IndexBase* idxBase, DataType* valueType)
    Status cusparseCsrGet(SpMatDescr spMatDescr, int64_t* rows, int64_t* cols, int64_t* nnz, void** csrRowOffsets, void** csrColInd, void** csrValues, IndexType* csrRowOffsetsType, IndexType* csrColIndType, IndexBase* idxBase, DataType* valueType)
    Status cusparseSpMatGetFormat(SpMatDescr spMatDescr, Format* format)
    Status cusparseSpMatGetIndexBase(SpMatDescr spMatDescr, IndexBase* idxBase)
    Status cusparseSpMatGetValues(SpMatDescr spMatDescr, void** values)
    Status cusparseSpMatSetValues(SpMatDescr spMatDescr, void* values)
    Status cusparseSpMatGetStridedBatch(SpMatDescr spMatDescr, int* batchCount)
    Status cusparseSpMatSetStridedBatch(SpMatDescr spMatDescr, int batchCount)

    # cuSPARSE Generic API - Dense Vector APIs
    Status cusparseCreateDnVec(DnVecDescr* dnVecDescr, int64_t size, void* values, DataType valueType)
    Status cusparseDestroyDnVec(DnVecDescr dnVecDescr)
    Status cusparseDnVecGet(DnVecDescr dnVecDescr, int64_t* size, void** values, DataType* valueType)
    Status cusparseDnVecGetValues(DnVecDescr dnVecDescr, void** values)
    Status cusparseDnVecSetValues(DnVecDescr dnVecDescr, void* values)

    # cuSPARSE Generic API - Dense Matrix APIs
    Status cusparseCreateDnMat(DnMatDescr* dnMatDescr, int64_t rows, int64_t cols, int64_t ld, void* values, DataType valueType, Order order)
    Status cusparseDestroyDnMat(DnMatDescr dnMatDescr)
    Status cusparseDnMatGet(DnMatDescr dnMatDescr, int64_t* rows, int64_t* cols, int64_t* ld, void** values, DataType* type, Order* order)
    Status cusparseDnMatGetValues(DnMatDescr dnMatDescr, void** values)
    Status cusparseDnMatSetValues(DnMatDescr dnMatDescr, void* values)
    Status cusparseDnMatGetStridedBatch(DnMatDescr dnMatDescr, int* batchCount, int64_t* batchStride)
    Status cusparseDnMatSetStridedBatch(DnMatDescr dnMatDescr, int batchCount, int64_t batchStride)

    # cuSPARSE Generic API - Generic API Functions
    Status cusparseSpVV_bufferSize(Handle handle, Operation opX, SpVecDescr vecX, DnVecDescr vecY, const void* result, DataType computeType, size_t* bufferSize)
    Status cusparseSpVV(Handle handle, Operation opX, SpVecDescr vecX, DnVecDescr vecY, void* result, DataType computeType, void* externalBuffer)
    Status cusparseSpMV_bufferSize(Handle handle, Operation opA, const void* alpha, SpMatDescr matA, DnVecDescr vecX, const void* beta, DnVecDescr vecY, DataType computeType, SpMVAlg alg, size_t* bufferSize)
    Status cusparseSpMV(Handle handle, Operation opA, const void* alpha, SpMatDescr matA, DnVecDescr vecX, const void* beta, DnVecDescr vecY, DataType computeType, SpMVAlg alg, void* externalBuffer)
    Status cusparseSpMM_bufferSize(Handle handle, Operation opA, Operation opB, const void* alpha, SpMatDescr matA, DnMatDescr matB, const void* beta, DnMatDescr matC, DataType computeType, SpMMAlg alg, size_t* bufferSize)
    Status cusparseSpMM(Handle handle, Operation opA, Operation opB, const void* alpha, SpMatDescr matA, DnMatDescr matB, const void* beta, DnMatDescr matC, DataType computeType, SpMMAlg alg, void* externalBuffer)
    Status cusparseConstrainedGeMM_bufferSize(Handle handle, Operation opA, Operation opB, const void* alpha, DnMatDescr matA, DnMatDescr matB, const void* beta, SpMatDescr matC, DataType computeType, size_t* bufferSize)
    Status cusparseConstrainedGeMM(Handle handle, Operation opA, Operation opB, const void* alpha, DnMatDescr matA, DnMatDescr matB, const void* beta, SpMatDescr matC, DataType computeType, void* externalBuffer)

    # Build-time version
    int CUSPARSE_VERSION


########################################
# Status check

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


########################################
# Convert complex numbers

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
# Build-time version

def get_build_version():
    return CUSPARSE_VERSION


########################################
# Helper classes

cdef class SpVecAttributes:

    def __init__(self, int64_t size, int64_t nnz, intptr_t indices, intptr_t values, IndexType idxType, IndexBase idxBase, DataType valueType):
        self.size = size
        self.nnz = nnz
        self.indices = indices
        self.values = values
        self.idxType = idxType
        self.idxBase = idxBase
        self.valueType = valueType

cdef class CooAttributes:

    def __init__(self, int64_t rows, int64_t cols, int64_t nnz, intptr_t cooRowInd, intptr_t cooColInd, intptr_t cooValues, IndexType idxType, IndexBase idxBase, DataType valueType):
        self.rows = rows
        self.cols = cols
        self.nnz = nnz
        self.cooRowInd = cooRowInd
        self.cooColInd = cooColInd
        self.cooValues = cooValues
        self.idxType = idxType
        self.idxBase = idxBase
        self.valueType = valueType

cdef class CooAoSAttributes:

    def __init__(self, int64_t rows, int64_t cols, int64_t nnz, intptr_t cooInd, intptr_t cooValues, IndexType idxType, IndexBase idxBase, DataType valueType):
        self.rows = rows
        self.cols = cols
        self.nnz = nnz
        self.cooInd = cooInd
        self.cooValues = cooValues
        self.idxType = idxType
        self.idxBase = idxBase
        self.valueType = valueType

cdef class CsrAttributes:

    def __init__(self, int64_t rows, int64_t cols, int64_t nnz, intptr_t csrRowOffsets, intptr_t csrColInd, intptr_t csrValues, IndexType csrRowOffsetsType, IndexType csrColIndType, IndexBase idxBase, DataType valueType):
        self.rows = rows
        self.cols = cols
        self.nnz = nnz
        self.csrRowOffsets = csrRowOffsets
        self.csrColInd = csrColInd
        self.csrValues = csrValues
        self.csrRowOffsetsType = csrRowOffsetsType
        self.csrColIndType = csrColIndType
        self.idxBase = idxBase
        self.valueType = valueType

cdef class DnVecAttributes:

    def __init__(self, int64_t size, intptr_t values, DataType valueType):
        self.size = size
        self.values = values
        self.valueType = valueType

cdef class DnMatAttributes:

    def __init__(self, int64_t rows, int64_t cols, int64_t ld, intptr_t values, DataType type, Order order):
        self.rows = rows
        self.cols = cols
        self.ld = ld
        self.values = values
        self.type = type
        self.order = order

cdef class DnMatBatchAttributes:

    def __init__(self, int batchCount, int64_t batchStride):
        self.batchCount = batchCount
        self.batchStride = batchStride


########################################
# cuSPARSE Management Function

cpdef intptr_t create() except? 0:
    cdef Handle handle
    status = cusparseCreate(&handle)
    check_status(status)
    return <intptr_t>handle

cpdef destroy(intptr_t handle):
    status = cusparseDestroy(<Handle>handle)
    check_status(status)

cpdef int getVersion(intptr_t handle) except -1:
    cdef int version
    status = cusparseGetVersion(<Handle>handle, &version)
    check_status(status)
    return version

cpdef setPointerMode(intptr_t handle, int mode):
    status = cusparseSetPointerMode(<Handle>handle, <PointerMode>mode)
    check_status(status)

cpdef size_t getStream(intptr_t handle) except? 0:
    cdef driver.Stream streamId
    status = cusparseGetStream(<Handle>handle, &streamId)
    check_status(status)
    return <size_t>streamId

cpdef setStream(intptr_t handle, size_t streamId):
    status = cusparseSetStream(<Handle>handle, <driver.Stream>streamId)
    check_status(status)


########################################
# cuSPARSE Helper Function

cpdef size_t createMatDescr() except? 0:
    cdef MatDescr descrA
    status = cusparseCreateMatDescr(&descrA)
    check_status(status)
    return <size_t>descrA

cpdef destroyMatDescr(size_t descrA):
    status = cusparseDestroyMatDescr(<MatDescr>descrA)
    check_status(status)

cpdef setMatDiagType(size_t descrA, int diagType):
    status = cusparseSetMatDiagType(<MatDescr>descrA, <DiagType>diagType)
    check_status(status)

cpdef setMatFillMode(size_t descrA, int fillMode):
    status = cusparseSetMatFillMode(<MatDescr>descrA, <FillMode>fillMode)
    check_status(status)

cpdef setMatIndexBase(size_t descrA, int base):
    status = cusparseSetMatIndexBase(<MatDescr>descrA, <IndexBase>base)
    check_status(status)

cpdef setMatType(size_t descrA, int type):
    status = cusparseSetMatType(<MatDescr>descrA, <MatrixType>type)
    check_status(status)

cpdef size_t createCsrsv2Info() except? 0:
    cdef csrsv2Info_t info
    status = cusparseCreateCsrsv2Info(&info)
    check_status(status)
    return <size_t>info

cpdef destroyCsrsv2Info(size_t info):
    status = cusparseDestroyCsrsv2Info(<csrsv2Info_t>info)
    check_status(status)

cpdef size_t createCsrsm2Info() except? 0:
    cdef csrsm2Info_t info
    status = cusparseCreateCsrsm2Info(&info)
    check_status(status)
    return <size_t>info

cpdef destroyCsrsm2Info(size_t info):
    status = cusparseDestroyCsrsm2Info(<csrsm2Info_t>info)
    check_status(status)

cpdef size_t createCsric02Info() except? 0:
    cdef csric02Info_t info
    status = cusparseCreateCsric02Info(&info)
    check_status(status)
    return <size_t>info

cpdef destroyCsric02Info(size_t info):
    status = cusparseDestroyCsric02Info(<csric02Info_t>info)
    check_status(status)

cpdef size_t createCsrilu02Info() except? 0:
    cdef csrilu02Info_t info
    status = cusparseCreateCsrilu02Info(&info)
    check_status(status)
    return <size_t>info

cpdef destroyCsrilu02Info(size_t info):
    status = cusparseDestroyCsrilu02Info(<csrilu02Info_t>info)
    check_status(status)

cpdef size_t createBsric02Info() except? 0:
    cdef bsric02Info_t info
    status = cusparseCreateBsric02Info(&info)
    check_status(status)
    return <size_t>info

cpdef destroyBsric02Info(size_t info):
    status = cusparseDestroyBsric02Info(<bsric02Info_t>info)
    check_status(status)

cpdef size_t createBsrilu02Info() except? 0:
    cdef bsrilu02Info_t info
    status = cusparseCreateBsrilu02Info(&info)
    check_status(status)
    return <size_t>info

cpdef destroyBsrilu02Info(size_t info):
    status = cusparseDestroyBsrilu02Info(<bsrilu02Info_t>info)
    check_status(status)

cpdef size_t createCsrgemm2Info() except? 0:
    cdef csrgemm2Info_t info
    status = cusparseCreateCsrgemm2Info(&info)
    check_status(status)
    return <size_t>info

cpdef destroyCsrgemm2Info(size_t info):
    status = cusparseDestroyCsrgemm2Info(<csrgemm2Info_t>info)
    check_status(status)


########################################
# cuSPARSE Level 1 Function

cpdef sgthr(intptr_t handle, int nnz, intptr_t y, intptr_t xVal, intptr_t xInd, int idxBase):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseSgthr(<Handle>handle, nnz, <const float*>y, <float*>xVal, <const int*>xInd, <IndexBase>idxBase)
    check_status(status)

cpdef dgthr(intptr_t handle, int nnz, intptr_t y, intptr_t xVal, intptr_t xInd, int idxBase):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseDgthr(<Handle>handle, nnz, <const double*>y, <double*>xVal, <const int*>xInd, <IndexBase>idxBase)
    check_status(status)

cpdef cgthr(intptr_t handle, int nnz, intptr_t y, intptr_t xVal, intptr_t xInd, int idxBase):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseCgthr(<Handle>handle, nnz, <const cuComplex*>y, <cuComplex*>xVal, <const int*>xInd, <IndexBase>idxBase)
    check_status(status)

cpdef zgthr(intptr_t handle, int nnz, intptr_t y, intptr_t xVal, intptr_t xInd, int idxBase):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseZgthr(<Handle>handle, nnz, <const cuDoubleComplex*>y, <cuDoubleComplex*>xVal, <const int*>xInd, <IndexBase>idxBase)
    check_status(status)


########################################
# cuSPARSE Level 2 Function

# REMOVED
cpdef scsrmv(intptr_t handle, int transA, int m, int n, int nnz, intptr_t alpha, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t x, intptr_t beta, intptr_t y):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseScsrmv(<Handle>handle, <Operation>transA, m, n, nnz, <const float*>alpha, <const MatDescr>descrA, <const float*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <const float*>x, <const float*>beta, <float*>y)
    check_status(status)

# REMOVED
cpdef dcsrmv(intptr_t handle, int transA, int m, int n, int nnz, intptr_t alpha, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t x, intptr_t beta, intptr_t y):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseDcsrmv(<Handle>handle, <Operation>transA, m, n, nnz, <const double*>alpha, <const MatDescr>descrA, <const double*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <const double*>x, <const double*>beta, <double*>y)
    check_status(status)

# REMOVED
cpdef ccsrmv(intptr_t handle, int transA, int m, int n, int nnz, intptr_t alpha, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t x, intptr_t beta, intptr_t y):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseCcsrmv(<Handle>handle, <Operation>transA, m, n, nnz, <const cuComplex*>alpha, <const MatDescr>descrA, <const cuComplex*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <const cuComplex*>x, <const cuComplex*>beta, <cuComplex*>y)
    check_status(status)

# REMOVED
cpdef zcsrmv(intptr_t handle, int transA, int m, int n, int nnz, intptr_t alpha, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t x, intptr_t beta, intptr_t y):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseZcsrmv(<Handle>handle, <Operation>transA, m, n, nnz, <const cuDoubleComplex*>alpha, <const MatDescr>descrA, <const cuDoubleComplex*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <const cuDoubleComplex*>x, <const cuDoubleComplex*>beta, <cuDoubleComplex*>y)
    check_status(status)

cpdef size_t csrmvEx_bufferSize(intptr_t handle, int alg, int transA, int m, int n, int nnz, intptr_t alpha, size_t alphatype, size_t descrA, intptr_t csrValA, size_t csrValAtype, intptr_t csrRowPtrA, intptr_t csrColIndA, intptr_t x, size_t xtype, intptr_t beta, size_t betatype, intptr_t y, size_t ytype, size_t executiontype) except? 0:
    cdef size_t bufferSizeInBytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseCsrmvEx_bufferSize(<Handle>handle, <AlgMode>alg, <Operation>transA, m, n, nnz, <const void*>alpha, <DataType>alphatype, <const MatDescr>descrA, <const void*>csrValA, <DataType>csrValAtype, <const int*>csrRowPtrA, <const int*>csrColIndA, <const void*>x, <DataType>xtype, <const void*>beta, <DataType>betatype, <void*>y, <DataType>ytype, <DataType>executiontype, &bufferSizeInBytes)
    check_status(status)
    return bufferSizeInBytes

cpdef csrmvEx(intptr_t handle, int alg, int transA, int m, int n, int nnz, intptr_t alpha, size_t alphatype, size_t descrA, intptr_t csrValA, size_t csrValAtype, intptr_t csrRowPtrA, intptr_t csrColIndA, intptr_t x, size_t xtype, intptr_t beta, size_t betatype, intptr_t y, size_t ytype, size_t executiontype, intptr_t buffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseCsrmvEx(<Handle>handle, <AlgMode>alg, <Operation>transA, m, n, nnz, <const void*>alpha, <DataType>alphatype, <const MatDescr>descrA, <const void*>csrValA, <DataType>csrValAtype, <const int*>csrRowPtrA, <const int*>csrColIndA, <const void*>x, <DataType>xtype, <const void*>beta, <DataType>betatype, <void*>y, <DataType>ytype, <DataType>executiontype, <void*>buffer)
    check_status(status)

cpdef int scsrsv2_bufferSize(intptr_t handle, int transA, int m, int nnz, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info) except? 0:
    cdef int bufferSizeInBytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseScsrsv2_bufferSize(<Handle>handle, <Operation>transA, m, nnz, <const MatDescr>descrA, <float*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <csrsv2Info_t>info, &bufferSizeInBytes)
    check_status(status)
    return bufferSizeInBytes

cpdef int dcsrsv2_bufferSize(intptr_t handle, int transA, int m, int nnz, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info) except? 0:
    cdef int bufferSizeInBytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseDcsrsv2_bufferSize(<Handle>handle, <Operation>transA, m, nnz, <const MatDescr>descrA, <double*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <csrsv2Info_t>info, &bufferSizeInBytes)
    check_status(status)
    return bufferSizeInBytes

cpdef int ccsrsv2_bufferSize(intptr_t handle, int transA, int m, int nnz, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info) except? 0:
    cdef int bufferSizeInBytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseCcsrsv2_bufferSize(<Handle>handle, <Operation>transA, m, nnz, <const MatDescr>descrA, <cuComplex*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <csrsv2Info_t>info, &bufferSizeInBytes)
    check_status(status)
    return bufferSizeInBytes

cpdef int zcsrsv2_bufferSize(intptr_t handle, int transA, int m, int nnz, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info) except? 0:
    cdef int bufferSizeInBytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseZcsrsv2_bufferSize(<Handle>handle, <Operation>transA, m, nnz, <const MatDescr>descrA, <cuDoubleComplex*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <csrsv2Info_t>info, &bufferSizeInBytes)
    check_status(status)
    return bufferSizeInBytes

cpdef scsrsv2_analysis(intptr_t handle, int transA, int m, int nnz, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info, int policy, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseScsrsv2_analysis(<Handle>handle, <Operation>transA, m, nnz, <const MatDescr>descrA, <const float*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <csrsv2Info_t>info, <SolvePolicy>policy, <void*>pBuffer)
    check_status(status)

cpdef dcsrsv2_analysis(intptr_t handle, int transA, int m, int nnz, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info, int policy, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseDcsrsv2_analysis(<Handle>handle, <Operation>transA, m, nnz, <const MatDescr>descrA, <const double*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <csrsv2Info_t>info, <SolvePolicy>policy, <void*>pBuffer)
    check_status(status)

cpdef ccsrsv2_analysis(intptr_t handle, int transA, int m, int nnz, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info, int policy, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseCcsrsv2_analysis(<Handle>handle, <Operation>transA, m, nnz, <const MatDescr>descrA, <const cuComplex*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <csrsv2Info_t>info, <SolvePolicy>policy, <void*>pBuffer)
    check_status(status)

cpdef zcsrsv2_analysis(intptr_t handle, int transA, int m, int nnz, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info, int policy, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseZcsrsv2_analysis(<Handle>handle, <Operation>transA, m, nnz, <const MatDescr>descrA, <const cuDoubleComplex*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <csrsv2Info_t>info, <SolvePolicy>policy, <void*>pBuffer)
    check_status(status)

cpdef scsrsv2_solve(intptr_t handle, int transA, int m, int nnz, intptr_t alpha, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info, intptr_t f, intptr_t x, int policy, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseScsrsv2_solve(<Handle>handle, <Operation>transA, m, nnz, <const float*>alpha, <const MatDescr>descrA, <const float*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <csrsv2Info_t>info, <const float*>f, <float*>x, <SolvePolicy>policy, <void*>pBuffer)
    check_status(status)

cpdef dcsrsv2_solve(intptr_t handle, int transA, int m, int nnz, intptr_t alpha, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info, intptr_t f, intptr_t x, int policy, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseDcsrsv2_solve(<Handle>handle, <Operation>transA, m, nnz, <const double*>alpha, <const MatDescr>descrA, <const double*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <csrsv2Info_t>info, <const double*>f, <double*>x, <SolvePolicy>policy, <void*>pBuffer)
    check_status(status)

cpdef ccsrsv2_solve(intptr_t handle, int transA, int m, int nnz, intptr_t alpha, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info, intptr_t f, intptr_t x, int policy, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseCcsrsv2_solve(<Handle>handle, <Operation>transA, m, nnz, <const cuComplex*>alpha, <const MatDescr>descrA, <const cuComplex*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <csrsv2Info_t>info, <const cuComplex*>f, <cuComplex*>x, <SolvePolicy>policy, <void*>pBuffer)
    check_status(status)

cpdef zcsrsv2_solve(intptr_t handle, int transA, int m, int nnz, intptr_t alpha, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info, intptr_t f, intptr_t x, int policy, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseZcsrsv2_solve(<Handle>handle, <Operation>transA, m, nnz, <const cuDoubleComplex*>alpha, <const MatDescr>descrA, <const cuDoubleComplex*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <csrsv2Info_t>info, <const cuDoubleComplex*>f, <cuDoubleComplex*>x, <SolvePolicy>policy, <void*>pBuffer)
    check_status(status)

cpdef xcsrsv2_zeroPivot(intptr_t handle, size_t info, intptr_t position):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseXcsrsv2_zeroPivot(<Handle>handle, <csrsv2Info_t>info, <int*>position)
    check_status(status)


########################################
# cuSPARSE Level 3 Function

# REMOVED
cpdef scsrmm(intptr_t handle, int transA, int m, int n, int k, int nnz, intptr_t alpha, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseScsrmm(<Handle>handle, <Operation>transA, m, n, k, nnz, <const float*>alpha, <const MatDescr>descrA, <const float*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <const float*>B, ldb, <const float*>beta, <float*>C, ldc)
    check_status(status)

# REMOVED
cpdef dcsrmm(intptr_t handle, int transA, int m, int n, int k, int nnz, intptr_t alpha, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseDcsrmm(<Handle>handle, <Operation>transA, m, n, k, nnz, <const double*>alpha, <const MatDescr>descrA, <const double*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <const double*>B, ldb, <const double*>beta, <double*>C, ldc)
    check_status(status)

# REMOVED
cpdef ccsrmm(intptr_t handle, int transA, int m, int n, int k, int nnz, intptr_t alpha, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseCcsrmm(<Handle>handle, <Operation>transA, m, n, k, nnz, <const cuComplex*>alpha, <const MatDescr>descrA, <const cuComplex*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <const cuComplex*>B, ldb, <const cuComplex*>beta, <cuComplex*>C, ldc)
    check_status(status)

# REMOVED
cpdef zcsrmm(intptr_t handle, int transA, int m, int n, int k, int nnz, intptr_t alpha, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseZcsrmm(<Handle>handle, <Operation>transA, m, n, k, nnz, <const cuDoubleComplex*>alpha, <const MatDescr>descrA, <const cuDoubleComplex*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <const cuDoubleComplex*>B, ldb, <const cuDoubleComplex*>beta, <cuDoubleComplex*>C, ldc)
    check_status(status)

# REMOVED
cpdef scsrmm2(intptr_t handle, int transA, int transB, int m, int n, int k, int nnz, intptr_t alpha, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseScsrmm2(<Handle>handle, <Operation>transA, <Operation>transB, m, n, k, nnz, <const float*>alpha, <const MatDescr>descrA, <const float*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <const float*>B, ldb, <const float*>beta, <float*>C, ldc)
    check_status(status)

# REMOVED
cpdef dcsrmm2(intptr_t handle, int transA, int transB, int m, int n, int k, int nnz, intptr_t alpha, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseDcsrmm2(<Handle>handle, <Operation>transA, <Operation>transB, m, n, k, nnz, <const double*>alpha, <const MatDescr>descrA, <const double*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <const double*>B, ldb, <const double*>beta, <double*>C, ldc)
    check_status(status)

# REMOVED
cpdef ccsrmm2(intptr_t handle, int transA, int transB, int m, int n, int k, int nnz, intptr_t alpha, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseCcsrmm2(<Handle>handle, <Operation>transA, <Operation>transB, m, n, k, nnz, <const cuComplex*>alpha, <const MatDescr>descrA, <const cuComplex*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <const cuComplex*>B, ldb, <const cuComplex*>beta, <cuComplex*>C, ldc)
    check_status(status)

# REMOVED
cpdef zcsrmm2(intptr_t handle, int transA, int transB, int m, int n, int k, int nnz, intptr_t alpha, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseZcsrmm2(<Handle>handle, <Operation>transA, <Operation>transB, m, n, k, nnz, <const cuDoubleComplex*>alpha, <const MatDescr>descrA, <const cuDoubleComplex*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <const cuDoubleComplex*>B, ldb, <const cuDoubleComplex*>beta, <cuDoubleComplex*>C, ldc)
    check_status(status)

cpdef size_t scsrsm2_bufferSizeExt(intptr_t handle, int algo, int transA, int transB, int m, int nrhs, int nnz, intptr_t alpha, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t B, int ldb, size_t info, int policy) except? 0:
    cdef size_t bufferSize
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseScsrsm2_bufferSizeExt(<Handle>handle, algo, <Operation>transA, <Operation>transB, m, nrhs, nnz, <const float*>alpha, <const MatDescr>descrA, <const float*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <const float*>B, ldb, <csrsm2Info_t>info, <SolvePolicy>policy, &bufferSize)
    check_status(status)
    return bufferSize

cpdef size_t dcsrsm2_bufferSizeExt(intptr_t handle, int algo, int transA, int transB, int m, int nrhs, int nnz, intptr_t alpha, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t B, int ldb, size_t info, int policy) except? 0:
    cdef size_t bufferSize
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseDcsrsm2_bufferSizeExt(<Handle>handle, algo, <Operation>transA, <Operation>transB, m, nrhs, nnz, <const double*>alpha, <const MatDescr>descrA, <const double*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <const double*>B, ldb, <csrsm2Info_t>info, <SolvePolicy>policy, &bufferSize)
    check_status(status)
    return bufferSize

cpdef size_t ccsrsm2_bufferSizeExt(intptr_t handle, int algo, int transA, int transB, int m, int nrhs, int nnz, intptr_t alpha, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t B, int ldb, size_t info, int policy) except? 0:
    cdef size_t bufferSize
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseCcsrsm2_bufferSizeExt(<Handle>handle, algo, <Operation>transA, <Operation>transB, m, nrhs, nnz, <const cuComplex*>alpha, <const MatDescr>descrA, <const cuComplex*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <const cuComplex*>B, ldb, <csrsm2Info_t>info, <SolvePolicy>policy, &bufferSize)
    check_status(status)
    return bufferSize

cpdef size_t zcsrsm2_bufferSizeExt(intptr_t handle, int algo, int transA, int transB, int m, int nrhs, int nnz, intptr_t alpha, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t B, int ldb, size_t info, int policy) except? 0:
    cdef size_t bufferSize
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseZcsrsm2_bufferSizeExt(<Handle>handle, algo, <Operation>transA, <Operation>transB, m, nrhs, nnz, <const cuDoubleComplex*>alpha, <const MatDescr>descrA, <const cuDoubleComplex*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <const cuDoubleComplex*>B, ldb, <csrsm2Info_t>info, <SolvePolicy>policy, &bufferSize)
    check_status(status)
    return bufferSize

cpdef scsrsm2_analysis(intptr_t handle, int algo, int transA, int transB, int m, int nrhs, int nnz, intptr_t alpha, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t B, int ldb, size_t info, int policy, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseScsrsm2_analysis(<Handle>handle, algo, <Operation>transA, <Operation>transB, m, nrhs, nnz, <const float*>alpha, <const MatDescr>descrA, <const float*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <const float*>B, ldb, <csrsm2Info_t>info, <SolvePolicy>policy, <void*>pBuffer)
    check_status(status)

cpdef dcsrsm2_analysis(intptr_t handle, int algo, int transA, int transB, int m, int nrhs, int nnz, intptr_t alpha, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t B, int ldb, size_t info, int policy, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseDcsrsm2_analysis(<Handle>handle, algo, <Operation>transA, <Operation>transB, m, nrhs, nnz, <const double*>alpha, <const MatDescr>descrA, <const double*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <const double*>B, ldb, <csrsm2Info_t>info, <SolvePolicy>policy, <void*>pBuffer)
    check_status(status)

cpdef ccsrsm2_analysis(intptr_t handle, int algo, int transA, int transB, int m, int nrhs, int nnz, intptr_t alpha, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t B, int ldb, size_t info, int policy, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseCcsrsm2_analysis(<Handle>handle, algo, <Operation>transA, <Operation>transB, m, nrhs, nnz, <const cuComplex*>alpha, <const MatDescr>descrA, <const cuComplex*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <const cuComplex*>B, ldb, <csrsm2Info_t>info, <SolvePolicy>policy, <void*>pBuffer)
    check_status(status)

cpdef zcsrsm2_analysis(intptr_t handle, int algo, int transA, int transB, int m, int nrhs, int nnz, intptr_t alpha, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t B, int ldb, size_t info, int policy, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseZcsrsm2_analysis(<Handle>handle, algo, <Operation>transA, <Operation>transB, m, nrhs, nnz, <const cuDoubleComplex*>alpha, <const MatDescr>descrA, <const cuDoubleComplex*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <const cuDoubleComplex*>B, ldb, <csrsm2Info_t>info, <SolvePolicy>policy, <void*>pBuffer)
    check_status(status)

cpdef scsrsm2_solve(intptr_t handle, int algo, int transA, int transB, int m, int nrhs, int nnz, intptr_t alpha, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t B, int ldb, size_t info, int policy, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseScsrsm2_solve(<Handle>handle, algo, <Operation>transA, <Operation>transB, m, nrhs, nnz, <const float*>alpha, <const MatDescr>descrA, <const float*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <float*>B, ldb, <csrsm2Info_t>info, <SolvePolicy>policy, <void*>pBuffer)
    check_status(status)

cpdef dcsrsm2_solve(intptr_t handle, int algo, int transA, int transB, int m, int nrhs, int nnz, intptr_t alpha, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t B, int ldb, size_t info, int policy, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseDcsrsm2_solve(<Handle>handle, algo, <Operation>transA, <Operation>transB, m, nrhs, nnz, <const double*>alpha, <const MatDescr>descrA, <const double*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <double*>B, ldb, <csrsm2Info_t>info, <SolvePolicy>policy, <void*>pBuffer)
    check_status(status)

cpdef ccsrsm2_solve(intptr_t handle, int algo, int transA, int transB, int m, int nrhs, int nnz, intptr_t alpha, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t B, int ldb, size_t info, int policy, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseCcsrsm2_solve(<Handle>handle, algo, <Operation>transA, <Operation>transB, m, nrhs, nnz, <const cuComplex*>alpha, <const MatDescr>descrA, <const cuComplex*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <cuComplex*>B, ldb, <csrsm2Info_t>info, <SolvePolicy>policy, <void*>pBuffer)
    check_status(status)

cpdef zcsrsm2_solve(intptr_t handle, int algo, int transA, int transB, int m, int nrhs, int nnz, intptr_t alpha, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t B, int ldb, size_t info, int policy, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseZcsrsm2_solve(<Handle>handle, algo, <Operation>transA, <Operation>transB, m, nrhs, nnz, <const cuDoubleComplex*>alpha, <const MatDescr>descrA, <const cuDoubleComplex*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <cuDoubleComplex*>B, ldb, <csrsm2Info_t>info, <SolvePolicy>policy, <void*>pBuffer)
    check_status(status)

cpdef xcsrsm2_zeroPivot(intptr_t handle, size_t info, intptr_t position):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseXcsrsm2_zeroPivot(<Handle>handle, <csrsm2Info_t>info, <int*>position)
    check_status(status)


########################################
# cuSPARSE Extra Function

# REMOVED
cpdef xcsrgeamNnz(intptr_t handle, int m, int n, size_t descrA, int nnzA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t descrB, int nnzB, intptr_t csrSortedRowPtrB, intptr_t csrSortedColIndB, size_t descrC, intptr_t csrSortedRowPtrC, intptr_t nnzTotalDevHostPtr):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseXcsrgeamNnz(<Handle>handle, m, n, <const MatDescr>descrA, nnzA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <const MatDescr>descrB, nnzB, <const int*>csrSortedRowPtrB, <const int*>csrSortedColIndB, <const MatDescr>descrC, <int*>csrSortedRowPtrC, <int*>nnzTotalDevHostPtr)
    check_status(status)

# REMOVED
cpdef scsrgeam(intptr_t handle, int m, int n, intptr_t alpha, size_t descrA, int nnzA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t beta, size_t descrB, int nnzB, intptr_t csrSortedValB, intptr_t csrSortedRowPtrB, intptr_t csrSortedColIndB, size_t descrC, intptr_t csrSortedValC, intptr_t csrSortedRowPtrC, intptr_t csrSortedColIndC):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseScsrgeam(<Handle>handle, m, n, <const float*>alpha, <const MatDescr>descrA, nnzA, <const float*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <const float*>beta, <const MatDescr>descrB, nnzB, <const float*>csrSortedValB, <const int*>csrSortedRowPtrB, <const int*>csrSortedColIndB, <const MatDescr>descrC, <float*>csrSortedValC, <int*>csrSortedRowPtrC, <int*>csrSortedColIndC)
    check_status(status)

# REMOVED
cpdef dcsrgeam(intptr_t handle, int m, int n, intptr_t alpha, size_t descrA, int nnzA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t beta, size_t descrB, int nnzB, intptr_t csrSortedValB, intptr_t csrSortedRowPtrB, intptr_t csrSortedColIndB, size_t descrC, intptr_t csrSortedValC, intptr_t csrSortedRowPtrC, intptr_t csrSortedColIndC):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseDcsrgeam(<Handle>handle, m, n, <const double*>alpha, <const MatDescr>descrA, nnzA, <const double*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <const double*>beta, <const MatDescr>descrB, nnzB, <const double*>csrSortedValB, <const int*>csrSortedRowPtrB, <const int*>csrSortedColIndB, <const MatDescr>descrC, <double*>csrSortedValC, <int*>csrSortedRowPtrC, <int*>csrSortedColIndC)
    check_status(status)

# REMOVED
cpdef ccsrgeam(intptr_t handle, int m, int n, intptr_t alpha, size_t descrA, int nnzA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t beta, size_t descrB, int nnzB, intptr_t csrSortedValB, intptr_t csrSortedRowPtrB, intptr_t csrSortedColIndB, size_t descrC, intptr_t csrSortedValC, intptr_t csrSortedRowPtrC, intptr_t csrSortedColIndC):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseCcsrgeam(<Handle>handle, m, n, <const cuComplex*>alpha, <const MatDescr>descrA, nnzA, <const cuComplex*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <const cuComplex*>beta, <const MatDescr>descrB, nnzB, <const cuComplex*>csrSortedValB, <const int*>csrSortedRowPtrB, <const int*>csrSortedColIndB, <const MatDescr>descrC, <cuComplex*>csrSortedValC, <int*>csrSortedRowPtrC, <int*>csrSortedColIndC)
    check_status(status)

# REMOVED
cpdef zcsrgeam(intptr_t handle, int m, int n, intptr_t alpha, size_t descrA, int nnzA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t beta, size_t descrB, int nnzB, intptr_t csrSortedValB, intptr_t csrSortedRowPtrB, intptr_t csrSortedColIndB, size_t descrC, intptr_t csrSortedValC, intptr_t csrSortedRowPtrC, intptr_t csrSortedColIndC):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseZcsrgeam(<Handle>handle, m, n, <const cuDoubleComplex*>alpha, <const MatDescr>descrA, nnzA, <const cuDoubleComplex*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <const cuDoubleComplex*>beta, <const MatDescr>descrB, nnzB, <const cuDoubleComplex*>csrSortedValB, <const int*>csrSortedRowPtrB, <const int*>csrSortedColIndB, <const MatDescr>descrC, <cuDoubleComplex*>csrSortedValC, <int*>csrSortedRowPtrC, <int*>csrSortedColIndC)
    check_status(status)

cpdef size_t scsrgeam2_bufferSizeExt(intptr_t handle, int m, int n, intptr_t alpha, size_t descrA, int nnzA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t beta, size_t descrB, int nnzB, intptr_t csrSortedValB, intptr_t csrSortedRowPtrB, intptr_t csrSortedColIndB, size_t descrC, intptr_t csrSortedValC, intptr_t csrSortedRowPtrC, intptr_t csrSortedColIndC) except? 0:
    cdef size_t bufferSizeInBytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseScsrgeam2_bufferSizeExt(<Handle>handle, m, n, <const float*>alpha, <const MatDescr>descrA, nnzA, <const float*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <const float*>beta, <const MatDescr>descrB, nnzB, <const float*>csrSortedValB, <const int*>csrSortedRowPtrB, <const int*>csrSortedColIndB, <const MatDescr>descrC, <const float*>csrSortedValC, <const int*>csrSortedRowPtrC, <const int*>csrSortedColIndC, &bufferSizeInBytes)
    check_status(status)
    return bufferSizeInBytes

cpdef size_t dcsrgeam2_bufferSizeExt(intptr_t handle, int m, int n, intptr_t alpha, size_t descrA, int nnzA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t beta, size_t descrB, int nnzB, intptr_t csrSortedValB, intptr_t csrSortedRowPtrB, intptr_t csrSortedColIndB, size_t descrC, intptr_t csrSortedValC, intptr_t csrSortedRowPtrC, intptr_t csrSortedColIndC) except? 0:
    cdef size_t bufferSizeInBytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseDcsrgeam2_bufferSizeExt(<Handle>handle, m, n, <const double*>alpha, <const MatDescr>descrA, nnzA, <const double*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <const double*>beta, <const MatDescr>descrB, nnzB, <const double*>csrSortedValB, <const int*>csrSortedRowPtrB, <const int*>csrSortedColIndB, <const MatDescr>descrC, <const double*>csrSortedValC, <const int*>csrSortedRowPtrC, <const int*>csrSortedColIndC, &bufferSizeInBytes)
    check_status(status)
    return bufferSizeInBytes

cpdef size_t ccsrgeam2_bufferSizeExt(intptr_t handle, int m, int n, intptr_t alpha, size_t descrA, int nnzA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t beta, size_t descrB, int nnzB, intptr_t csrSortedValB, intptr_t csrSortedRowPtrB, intptr_t csrSortedColIndB, size_t descrC, intptr_t csrSortedValC, intptr_t csrSortedRowPtrC, intptr_t csrSortedColIndC) except? 0:
    cdef size_t bufferSizeInBytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseCcsrgeam2_bufferSizeExt(<Handle>handle, m, n, <const cuComplex*>alpha, <const MatDescr>descrA, nnzA, <const cuComplex*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <const cuComplex*>beta, <const MatDescr>descrB, nnzB, <const cuComplex*>csrSortedValB, <const int*>csrSortedRowPtrB, <const int*>csrSortedColIndB, <const MatDescr>descrC, <const cuComplex*>csrSortedValC, <const int*>csrSortedRowPtrC, <const int*>csrSortedColIndC, &bufferSizeInBytes)
    check_status(status)
    return bufferSizeInBytes

cpdef size_t zcsrgeam2_bufferSizeExt(intptr_t handle, int m, int n, intptr_t alpha, size_t descrA, int nnzA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t beta, size_t descrB, int nnzB, intptr_t csrSortedValB, intptr_t csrSortedRowPtrB, intptr_t csrSortedColIndB, size_t descrC, intptr_t csrSortedValC, intptr_t csrSortedRowPtrC, intptr_t csrSortedColIndC) except? 0:
    cdef size_t bufferSizeInBytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseZcsrgeam2_bufferSizeExt(<Handle>handle, m, n, <const cuDoubleComplex*>alpha, <const MatDescr>descrA, nnzA, <const cuDoubleComplex*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <const cuDoubleComplex*>beta, <const MatDescr>descrB, nnzB, <const cuDoubleComplex*>csrSortedValB, <const int*>csrSortedRowPtrB, <const int*>csrSortedColIndB, <const MatDescr>descrC, <const cuDoubleComplex*>csrSortedValC, <const int*>csrSortedRowPtrC, <const int*>csrSortedColIndC, &bufferSizeInBytes)
    check_status(status)
    return bufferSizeInBytes

cpdef xcsrgeam2Nnz(intptr_t handle, int m, int n, size_t descrA, int nnzA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t descrB, int nnzB, intptr_t csrSortedRowPtrB, intptr_t csrSortedColIndB, size_t descrC, intptr_t csrSortedRowPtrC, intptr_t nnzTotalDevHostPtr, intptr_t workspace):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseXcsrgeam2Nnz(<Handle>handle, m, n, <const MatDescr>descrA, nnzA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <const MatDescr>descrB, nnzB, <const int*>csrSortedRowPtrB, <const int*>csrSortedColIndB, <const MatDescr>descrC, <int*>csrSortedRowPtrC, <int*>nnzTotalDevHostPtr, <void*>workspace)
    check_status(status)

cpdef scsrgeam2(intptr_t handle, int m, int n, intptr_t alpha, size_t descrA, int nnzA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t beta, size_t descrB, int nnzB, intptr_t csrSortedValB, intptr_t csrSortedRowPtrB, intptr_t csrSortedColIndB, size_t descrC, intptr_t csrSortedValC, intptr_t csrSortedRowPtrC, intptr_t csrSortedColIndC, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseScsrgeam2(<Handle>handle, m, n, <const float*>alpha, <const MatDescr>descrA, nnzA, <const float*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <const float*>beta, <const MatDescr>descrB, nnzB, <const float*>csrSortedValB, <const int*>csrSortedRowPtrB, <const int*>csrSortedColIndB, <const MatDescr>descrC, <float*>csrSortedValC, <int*>csrSortedRowPtrC, <int*>csrSortedColIndC, <void*>pBuffer)
    check_status(status)

cpdef dcsrgeam2(intptr_t handle, int m, int n, intptr_t alpha, size_t descrA, int nnzA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t beta, size_t descrB, int nnzB, intptr_t csrSortedValB, intptr_t csrSortedRowPtrB, intptr_t csrSortedColIndB, size_t descrC, intptr_t csrSortedValC, intptr_t csrSortedRowPtrC, intptr_t csrSortedColIndC, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseDcsrgeam2(<Handle>handle, m, n, <const double*>alpha, <const MatDescr>descrA, nnzA, <const double*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <const double*>beta, <const MatDescr>descrB, nnzB, <const double*>csrSortedValB, <const int*>csrSortedRowPtrB, <const int*>csrSortedColIndB, <const MatDescr>descrC, <double*>csrSortedValC, <int*>csrSortedRowPtrC, <int*>csrSortedColIndC, <void*>pBuffer)
    check_status(status)

cpdef ccsrgeam2(intptr_t handle, int m, int n, intptr_t alpha, size_t descrA, int nnzA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t beta, size_t descrB, int nnzB, intptr_t csrSortedValB, intptr_t csrSortedRowPtrB, intptr_t csrSortedColIndB, size_t descrC, intptr_t csrSortedValC, intptr_t csrSortedRowPtrC, intptr_t csrSortedColIndC, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseCcsrgeam2(<Handle>handle, m, n, <const cuComplex*>alpha, <const MatDescr>descrA, nnzA, <const cuComplex*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <const cuComplex*>beta, <const MatDescr>descrB, nnzB, <const cuComplex*>csrSortedValB, <const int*>csrSortedRowPtrB, <const int*>csrSortedColIndB, <const MatDescr>descrC, <cuComplex*>csrSortedValC, <int*>csrSortedRowPtrC, <int*>csrSortedColIndC, <void*>pBuffer)
    check_status(status)

cpdef zcsrgeam2(intptr_t handle, int m, int n, intptr_t alpha, size_t descrA, int nnzA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t beta, size_t descrB, int nnzB, intptr_t csrSortedValB, intptr_t csrSortedRowPtrB, intptr_t csrSortedColIndB, size_t descrC, intptr_t csrSortedValC, intptr_t csrSortedRowPtrC, intptr_t csrSortedColIndC, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseZcsrgeam2(<Handle>handle, m, n, <const cuDoubleComplex*>alpha, <const MatDescr>descrA, nnzA, <const cuDoubleComplex*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <const cuDoubleComplex*>beta, <const MatDescr>descrB, nnzB, <const cuDoubleComplex*>csrSortedValB, <const int*>csrSortedRowPtrB, <const int*>csrSortedColIndB, <const MatDescr>descrC, <cuDoubleComplex*>csrSortedValC, <int*>csrSortedRowPtrC, <int*>csrSortedColIndC, <void*>pBuffer)
    check_status(status)

# REMOVED
cpdef xcsrgemmNnz(intptr_t handle, int transA, int transB, int m, int n, int k, size_t descrA, const int nnzA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t descrB, const int nnzB, intptr_t csrSortedRowPtrB, intptr_t csrSortedColIndB, size_t descrC, intptr_t csrSortedRowPtrC, intptr_t nnzTotalDevHostPtr):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseXcsrgemmNnz(<Handle>handle, <Operation>transA, <Operation>transB, m, n, k, <const MatDescr>descrA, nnzA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <const MatDescr>descrB, nnzB, <const int*>csrSortedRowPtrB, <const int*>csrSortedColIndB, <const MatDescr>descrC, <int*>csrSortedRowPtrC, <int*>nnzTotalDevHostPtr)
    check_status(status)

# REMOVED
cpdef scsrgemm(intptr_t handle, int transA, int transB, int m, int n, int k, size_t descrA, const int nnzA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t descrB, const int nnzB, intptr_t csrSortedValB, intptr_t csrSortedRowPtrB, intptr_t csrSortedColIndB, size_t descrC, intptr_t csrSortedValC, intptr_t csrSortedRowPtrC, intptr_t csrSortedColIndC):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseScsrgemm(<Handle>handle, <Operation>transA, <Operation>transB, m, n, k, <const MatDescr>descrA, nnzA, <const float*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <const MatDescr>descrB, nnzB, <const float*>csrSortedValB, <const int*>csrSortedRowPtrB, <const int*>csrSortedColIndB, <const MatDescr>descrC, <float*>csrSortedValC, <const int*>csrSortedRowPtrC, <int*>csrSortedColIndC)
    check_status(status)

# REMOVED
cpdef dcsrgemm(intptr_t handle, int transA, int transB, int m, int n, int k, size_t descrA, int nnzA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t descrB, int nnzB, intptr_t csrSortedValB, intptr_t csrSortedRowPtrB, intptr_t csrSortedColIndB, size_t descrC, intptr_t csrSortedValC, intptr_t csrSortedRowPtrC, intptr_t csrSortedColIndC):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseDcsrgemm(<Handle>handle, <Operation>transA, <Operation>transB, m, n, k, <const MatDescr>descrA, nnzA, <const double*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <const MatDescr>descrB, nnzB, <const double*>csrSortedValB, <const int*>csrSortedRowPtrB, <const int*>csrSortedColIndB, <const MatDescr>descrC, <double*>csrSortedValC, <const int*>csrSortedRowPtrC, <int*>csrSortedColIndC)
    check_status(status)

# REMOVED
cpdef ccsrgemm(intptr_t handle, int transA, int transB, int m, int n, int k, size_t descrA, int nnzA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t descrB, int nnzB, intptr_t csrSortedValB, intptr_t csrSortedRowPtrB, intptr_t csrSortedColIndB, size_t descrC, intptr_t csrSortedValC, intptr_t csrSortedRowPtrC, intptr_t csrSortedColIndC):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseCcsrgemm(<Handle>handle, <Operation>transA, <Operation>transB, m, n, k, <const MatDescr>descrA, nnzA, <const cuComplex*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <const MatDescr>descrB, nnzB, <const cuComplex*>csrSortedValB, <const int*>csrSortedRowPtrB, <const int*>csrSortedColIndB, <const MatDescr>descrC, <cuComplex*>csrSortedValC, <const int*>csrSortedRowPtrC, <int*>csrSortedColIndC)
    check_status(status)

# REMOVED
cpdef zcsrgemm(intptr_t handle, int transA, int transB, int m, int n, int k, size_t descrA, int nnzA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t descrB, int nnzB, intptr_t csrSortedValB, intptr_t csrSortedRowPtrB, intptr_t csrSortedColIndB, size_t descrC, intptr_t csrSortedValC, intptr_t csrSortedRowPtrC, intptr_t csrSortedColIndC):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseZcsrgemm(<Handle>handle, <Operation>transA, <Operation>transB, m, n, k, <const MatDescr>descrA, nnzA, <const cuDoubleComplex*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <const MatDescr>descrB, nnzB, <const cuDoubleComplex*>csrSortedValB, <const int*>csrSortedRowPtrB, <const int*>csrSortedColIndB, <const MatDescr>descrC, <cuDoubleComplex*>csrSortedValC, <const int*>csrSortedRowPtrC, <int*>csrSortedColIndC)
    check_status(status)

cpdef size_t scsrgemm2_bufferSizeExt(intptr_t handle, int m, int n, int k, intptr_t alpha, size_t descrA, int nnzA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t descrB, int nnzB, intptr_t csrSortedRowPtrB, intptr_t csrSortedColIndB, intptr_t beta, size_t descrD, int nnzD, intptr_t csrSortedRowPtrD, intptr_t csrSortedColIndD, size_t info) except? 0:
    cdef size_t bufferSizeInBytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseScsrgemm2_bufferSizeExt(<Handle>handle, m, n, k, <const float*>alpha, <const MatDescr>descrA, nnzA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <const MatDescr>descrB, nnzB, <const int*>csrSortedRowPtrB, <const int*>csrSortedColIndB, <const float*>beta, <const MatDescr>descrD, nnzD, <const int*>csrSortedRowPtrD, <const int*>csrSortedColIndD, <csrgemm2Info_t>info, &bufferSizeInBytes)
    check_status(status)
    return bufferSizeInBytes

cpdef size_t dcsrgemm2_bufferSizeExt(intptr_t handle, int m, int n, int k, intptr_t alpha, size_t descrA, int nnzA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t descrB, int nnzB, intptr_t csrSortedRowPtrB, intptr_t csrSortedColIndB, intptr_t beta, size_t descrD, int nnzD, intptr_t csrSortedRowPtrD, intptr_t csrSortedColIndD, size_t info) except? 0:
    cdef size_t bufferSizeInBytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseDcsrgemm2_bufferSizeExt(<Handle>handle, m, n, k, <const double*>alpha, <const MatDescr>descrA, nnzA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <const MatDescr>descrB, nnzB, <const int*>csrSortedRowPtrB, <const int*>csrSortedColIndB, <const double*>beta, <const MatDescr>descrD, nnzD, <const int*>csrSortedRowPtrD, <const int*>csrSortedColIndD, <csrgemm2Info_t>info, &bufferSizeInBytes)
    check_status(status)
    return bufferSizeInBytes

cpdef size_t ccsrgemm2_bufferSizeExt(intptr_t handle, int m, int n, int k, intptr_t alpha, size_t descrA, int nnzA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t descrB, int nnzB, intptr_t csrSortedRowPtrB, intptr_t csrSortedColIndB, intptr_t beta, size_t descrD, int nnzD, intptr_t csrSortedRowPtrD, intptr_t csrSortedColIndD, size_t info) except? 0:
    cdef size_t bufferSizeInBytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseCcsrgemm2_bufferSizeExt(<Handle>handle, m, n, k, <const cuComplex*>alpha, <const MatDescr>descrA, nnzA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <const MatDescr>descrB, nnzB, <const int*>csrSortedRowPtrB, <const int*>csrSortedColIndB, <const cuComplex*>beta, <const MatDescr>descrD, nnzD, <const int*>csrSortedRowPtrD, <const int*>csrSortedColIndD, <csrgemm2Info_t>info, &bufferSizeInBytes)
    check_status(status)
    return bufferSizeInBytes

cpdef size_t zcsrgemm2_bufferSizeExt(intptr_t handle, int m, int n, int k, intptr_t alpha, size_t descrA, int nnzA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t descrB, int nnzB, intptr_t csrSortedRowPtrB, intptr_t csrSortedColIndB, intptr_t beta, size_t descrD, int nnzD, intptr_t csrSortedRowPtrD, intptr_t csrSortedColIndD, size_t info) except? 0:
    cdef size_t bufferSizeInBytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseZcsrgemm2_bufferSizeExt(<Handle>handle, m, n, k, <const cuDoubleComplex*>alpha, <const MatDescr>descrA, nnzA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <const MatDescr>descrB, nnzB, <const int*>csrSortedRowPtrB, <const int*>csrSortedColIndB, <const cuDoubleComplex*>beta, <const MatDescr>descrD, nnzD, <const int*>csrSortedRowPtrD, <const int*>csrSortedColIndD, <csrgemm2Info_t>info, &bufferSizeInBytes)
    check_status(status)
    return bufferSizeInBytes

cpdef xcsrgemm2Nnz(intptr_t handle, int m, int n, int k, size_t descrA, int nnzA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t descrB, int nnzB, intptr_t csrSortedRowPtrB, intptr_t csrSortedColIndB, size_t descrD, int nnzD, intptr_t csrSortedRowPtrD, intptr_t csrSortedColIndD, size_t descrC, intptr_t csrSortedRowPtrC, intptr_t nnzTotalDevHostPtr, size_t info, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseXcsrgemm2Nnz(<Handle>handle, m, n, k, <const MatDescr>descrA, nnzA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <const MatDescr>descrB, nnzB, <const int*>csrSortedRowPtrB, <const int*>csrSortedColIndB, <const MatDescr>descrD, nnzD, <const int*>csrSortedRowPtrD, <const int*>csrSortedColIndD, <const MatDescr>descrC, <int*>csrSortedRowPtrC, <int*>nnzTotalDevHostPtr, <const csrgemm2Info_t>info, <void*>pBuffer)
    check_status(status)

cpdef scsrgemm2(intptr_t handle, int m, int n, int k, intptr_t alpha, size_t descrA, int nnzA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t descrB, int nnzB, intptr_t csrSortedValB, intptr_t csrSortedRowPtrB, intptr_t csrSortedColIndB, intptr_t beta, size_t descrD, int nnzD, intptr_t csrSortedValD, intptr_t csrSortedRowPtrD, intptr_t csrSortedColIndD, size_t descrC, intptr_t csrSortedValC, intptr_t csrSortedRowPtrC, intptr_t csrSortedColIndC, size_t info, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseScsrgemm2(<Handle>handle, m, n, k, <const float*>alpha, <const MatDescr>descrA, nnzA, <const float*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <const MatDescr>descrB, nnzB, <const float*>csrSortedValB, <const int*>csrSortedRowPtrB, <const int*>csrSortedColIndB, <const float*>beta, <const MatDescr>descrD, nnzD, <const float*>csrSortedValD, <const int*>csrSortedRowPtrD, <const int*>csrSortedColIndD, <const MatDescr>descrC, <float*>csrSortedValC, <const int*>csrSortedRowPtrC, <int*>csrSortedColIndC, <const csrgemm2Info_t>info, <void*>pBuffer)
    check_status(status)

cpdef dcsrgemm2(intptr_t handle, int m, int n, int k, intptr_t alpha, size_t descrA, int nnzA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t descrB, int nnzB, intptr_t csrSortedValB, intptr_t csrSortedRowPtrB, intptr_t csrSortedColIndB, intptr_t beta, size_t descrD, int nnzD, intptr_t csrSortedValD, intptr_t csrSortedRowPtrD, intptr_t csrSortedColIndD, size_t descrC, intptr_t csrSortedValC, intptr_t csrSortedRowPtrC, intptr_t csrSortedColIndC, size_t info, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseDcsrgemm2(<Handle>handle, m, n, k, <const double*>alpha, <const MatDescr>descrA, nnzA, <const double*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <const MatDescr>descrB, nnzB, <const double*>csrSortedValB, <const int*>csrSortedRowPtrB, <const int*>csrSortedColIndB, <const double*>beta, <const MatDescr>descrD, nnzD, <const double*>csrSortedValD, <const int*>csrSortedRowPtrD, <const int*>csrSortedColIndD, <const MatDescr>descrC, <double*>csrSortedValC, <const int*>csrSortedRowPtrC, <int*>csrSortedColIndC, <const csrgemm2Info_t>info, <void*>pBuffer)
    check_status(status)

cpdef ccsrgemm2(intptr_t handle, int m, int n, int k, intptr_t alpha, size_t descrA, int nnzA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t descrB, int nnzB, intptr_t csrSortedValB, intptr_t csrSortedRowPtrB, intptr_t csrSortedColIndB, intptr_t beta, size_t descrD, int nnzD, intptr_t csrSortedValD, intptr_t csrSortedRowPtrD, intptr_t csrSortedColIndD, size_t descrC, intptr_t csrSortedValC, intptr_t csrSortedRowPtrC, intptr_t csrSortedColIndC, size_t info, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseCcsrgemm2(<Handle>handle, m, n, k, <const cuComplex*>alpha, <const MatDescr>descrA, nnzA, <const cuComplex*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <const MatDescr>descrB, nnzB, <const cuComplex*>csrSortedValB, <const int*>csrSortedRowPtrB, <const int*>csrSortedColIndB, <const cuComplex*>beta, <const MatDescr>descrD, nnzD, <const cuComplex*>csrSortedValD, <const int*>csrSortedRowPtrD, <const int*>csrSortedColIndD, <const MatDescr>descrC, <cuComplex*>csrSortedValC, <const int*>csrSortedRowPtrC, <int*>csrSortedColIndC, <const csrgemm2Info_t>info, <void*>pBuffer)
    check_status(status)

cpdef zcsrgemm2(intptr_t handle, int m, int n, int k, intptr_t alpha, size_t descrA, int nnzA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t descrB, int nnzB, intptr_t csrSortedValB, intptr_t csrSortedRowPtrB, intptr_t csrSortedColIndB, intptr_t beta, size_t descrD, int nnzD, intptr_t csrSortedValD, intptr_t csrSortedRowPtrD, intptr_t csrSortedColIndD, size_t descrC, intptr_t csrSortedValC, intptr_t csrSortedRowPtrC, intptr_t csrSortedColIndC, size_t info, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseZcsrgemm2(<Handle>handle, m, n, k, <const cuDoubleComplex*>alpha, <const MatDescr>descrA, nnzA, <const cuDoubleComplex*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <const MatDescr>descrB, nnzB, <const cuDoubleComplex*>csrSortedValB, <const int*>csrSortedRowPtrB, <const int*>csrSortedColIndB, <const cuDoubleComplex*>beta, <const MatDescr>descrD, nnzD, <const cuDoubleComplex*>csrSortedValD, <const int*>csrSortedRowPtrD, <const int*>csrSortedColIndD, <const MatDescr>descrC, <cuDoubleComplex*>csrSortedValC, <const int*>csrSortedRowPtrC, <int*>csrSortedColIndC, <const csrgemm2Info_t>info, <void*>pBuffer)
    check_status(status)


#######################################################################
# cuSPARSE Preconditioners - Incomplete Cholesky Factorization: level 0

cpdef int scsric02_bufferSize(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info) except? 0:
    cdef int bufferSizeInBytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseScsric02_bufferSize(<Handle>handle, m, nnz, <const MatDescr>descrA, <float*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <csric02Info_t>info, &bufferSizeInBytes)
    check_status(status)
    return bufferSizeInBytes

cpdef int dcsric02_bufferSize(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info) except? 0:
    cdef int bufferSizeInBytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseDcsric02_bufferSize(<Handle>handle, m, nnz, <const MatDescr>descrA, <double*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <csric02Info_t>info, &bufferSizeInBytes)
    check_status(status)
    return bufferSizeInBytes

cpdef int ccsric02_bufferSize(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info) except? 0:
    cdef int bufferSizeInBytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseCcsric02_bufferSize(<Handle>handle, m, nnz, <const MatDescr>descrA, <cuComplex*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <csric02Info_t>info, &bufferSizeInBytes)
    check_status(status)
    return bufferSizeInBytes

cpdef int zcsric02_bufferSize(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info) except? 0:
    cdef int bufferSizeInBytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseZcsric02_bufferSize(<Handle>handle, m, nnz, <const MatDescr>descrA, <cuDoubleComplex*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <csric02Info_t>info, &bufferSizeInBytes)
    check_status(status)
    return bufferSizeInBytes

cpdef scsric02_analysis(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info, int policy, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseScsric02_analysis(<Handle>handle, m, nnz, <const MatDescr>descrA, <const float*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <csric02Info_t>info, <SolvePolicy>policy, <void*>pBuffer)
    check_status(status)

cpdef dcsric02_analysis(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info, int policy, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseDcsric02_analysis(<Handle>handle, m, nnz, <const MatDescr>descrA, <const double*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <csric02Info_t>info, <SolvePolicy>policy, <void*>pBuffer)
    check_status(status)

cpdef ccsric02_analysis(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info, int policy, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseCcsric02_analysis(<Handle>handle, m, nnz, <const MatDescr>descrA, <const cuComplex*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <csric02Info_t>info, <SolvePolicy>policy, <void*>pBuffer)
    check_status(status)

cpdef zcsric02_analysis(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info, int policy, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseZcsric02_analysis(<Handle>handle, m, nnz, <const MatDescr>descrA, <const cuDoubleComplex*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <csric02Info_t>info, <SolvePolicy>policy, <void*>pBuffer)
    check_status(status)

cpdef scsric02(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrSortedValA_valM, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info, int policy, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseScsric02(<Handle>handle, m, nnz, <const MatDescr>descrA, <float*>csrSortedValA_valM, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <csric02Info_t>info, <SolvePolicy>policy, <void*>pBuffer)
    check_status(status)

cpdef dcsric02(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrSortedValA_valM, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info, int policy, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseDcsric02(<Handle>handle, m, nnz, <const MatDescr>descrA, <double*>csrSortedValA_valM, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <csric02Info_t>info, <SolvePolicy>policy, <void*>pBuffer)
    check_status(status)

cpdef ccsric02(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrSortedValA_valM, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info, int policy, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseCcsric02(<Handle>handle, m, nnz, <const MatDescr>descrA, <cuComplex*>csrSortedValA_valM, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <csric02Info_t>info, <SolvePolicy>policy, <void*>pBuffer)
    check_status(status)

cpdef zcsric02(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrSortedValA_valM, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info, int policy, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseZcsric02(<Handle>handle, m, nnz, <const MatDescr>descrA, <cuDoubleComplex*>csrSortedValA_valM, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <csric02Info_t>info, <SolvePolicy>policy, <void*>pBuffer)
    check_status(status)

cpdef int xcsric02_zeroPivot(intptr_t handle, size_t info) except? 0:
    cdef int position
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseXcsric02_zeroPivot(<Handle>handle, <csric02Info_t>info, &position)
    check_status(status)
    return position

cpdef int sbsric02_bufferSize(intptr_t handle, int dirA, int mb, int nnzb, size_t descrA, intptr_t bsrSortedVal, intptr_t bsrSortedRowPtr, intptr_t bsrSortedColInd, int blockDim, size_t info) except? 0:
    cdef int bufferSizeInBytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseSbsric02_bufferSize(<Handle>handle, <Direction>dirA, mb, nnzb, <const MatDescr>descrA, <float*>bsrSortedVal, <const int*>bsrSortedRowPtr, <const int*>bsrSortedColInd, blockDim, <bsric02Info_t>info, &bufferSizeInBytes)
    check_status(status)
    return bufferSizeInBytes

cpdef int dbsric02_bufferSize(intptr_t handle, int dirA, int mb, int nnzb, size_t descrA, intptr_t bsrSortedVal, intptr_t bsrSortedRowPtr, intptr_t bsrSortedColInd, int blockDim, size_t info) except? 0:
    cdef int bufferSizeInBytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseDbsric02_bufferSize(<Handle>handle, <Direction>dirA, mb, nnzb, <const MatDescr>descrA, <double*>bsrSortedVal, <const int*>bsrSortedRowPtr, <const int*>bsrSortedColInd, blockDim, <bsric02Info_t>info, &bufferSizeInBytes)
    check_status(status)
    return bufferSizeInBytes

cpdef int cbsric02_bufferSize(intptr_t handle, int dirA, int mb, int nnzb, size_t descrA, intptr_t bsrSortedVal, intptr_t bsrSortedRowPtr, intptr_t bsrSortedColInd, int blockDim, size_t info) except? 0:
    cdef int bufferSizeInBytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseCbsric02_bufferSize(<Handle>handle, <Direction>dirA, mb, nnzb, <const MatDescr>descrA, <cuComplex*>bsrSortedVal, <const int*>bsrSortedRowPtr, <const int*>bsrSortedColInd, blockDim, <bsric02Info_t>info, &bufferSizeInBytes)
    check_status(status)
    return bufferSizeInBytes

cpdef int zbsric02_bufferSize(intptr_t handle, int dirA, int mb, int nnzb, size_t descrA, intptr_t bsrSortedVal, intptr_t bsrSortedRowPtr, intptr_t bsrSortedColInd, int blockDim, size_t info) except? 0:
    cdef int bufferSizeInBytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseZbsric02_bufferSize(<Handle>handle, <Direction>dirA, mb, nnzb, <const MatDescr>descrA, <cuDoubleComplex*>bsrSortedVal, <const int*>bsrSortedRowPtr, <const int*>bsrSortedColInd, blockDim, <bsric02Info_t>info, &bufferSizeInBytes)
    check_status(status)
    return bufferSizeInBytes

cpdef sbsric02_analysis(intptr_t handle, int dirA, int mb, int nnzb, size_t descrA, intptr_t bsrSortedVal, intptr_t bsrSortedRowPtr, intptr_t bsrSortedColInd, int blockDim, size_t info, int policy, intptr_t pInputBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseSbsric02_analysis(<Handle>handle, <Direction>dirA, mb, nnzb, <const MatDescr>descrA, <const float*>bsrSortedVal, <const int*>bsrSortedRowPtr, <const int*>bsrSortedColInd, blockDim, <bsric02Info_t>info, <SolvePolicy>policy, <void*>pInputBuffer)
    check_status(status)

cpdef dbsric02_analysis(intptr_t handle, int dirA, int mb, int nnzb, size_t descrA, intptr_t bsrSortedVal, intptr_t bsrSortedRowPtr, intptr_t bsrSortedColInd, int blockDim, size_t info, int policy, intptr_t pInputBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseDbsric02_analysis(<Handle>handle, <Direction>dirA, mb, nnzb, <const MatDescr>descrA, <const double*>bsrSortedVal, <const int*>bsrSortedRowPtr, <const int*>bsrSortedColInd, blockDim, <bsric02Info_t>info, <SolvePolicy>policy, <void*>pInputBuffer)
    check_status(status)

cpdef cbsric02_analysis(intptr_t handle, int dirA, int mb, int nnzb, size_t descrA, intptr_t bsrSortedVal, intptr_t bsrSortedRowPtr, intptr_t bsrSortedColInd, int blockDim, size_t info, int policy, intptr_t pInputBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseCbsric02_analysis(<Handle>handle, <Direction>dirA, mb, nnzb, <const MatDescr>descrA, <const cuComplex*>bsrSortedVal, <const int*>bsrSortedRowPtr, <const int*>bsrSortedColInd, blockDim, <bsric02Info_t>info, <SolvePolicy>policy, <void*>pInputBuffer)
    check_status(status)

cpdef zbsric02_analysis(intptr_t handle, int dirA, int mb, int nnzb, size_t descrA, intptr_t bsrSortedVal, intptr_t bsrSortedRowPtr, intptr_t bsrSortedColInd, int blockDim, size_t info, int policy, intptr_t pInputBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseZbsric02_analysis(<Handle>handle, <Direction>dirA, mb, nnzb, <const MatDescr>descrA, <const cuDoubleComplex*>bsrSortedVal, <const int*>bsrSortedRowPtr, <const int*>bsrSortedColInd, blockDim, <bsric02Info_t>info, <SolvePolicy>policy, <void*>pInputBuffer)
    check_status(status)

cpdef sbsric02(intptr_t handle, int dirA, int mb, int nnzb, size_t descrA, intptr_t bsrSortedVal, intptr_t bsrSortedRowPtr, intptr_t bsrSortedColInd, int blockDim, size_t info, int policy, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseSbsric02(<Handle>handle, <Direction>dirA, mb, nnzb, <const MatDescr>descrA, <float*>bsrSortedVal, <const int*>bsrSortedRowPtr, <const int*>bsrSortedColInd, blockDim, <bsric02Info_t>info, <SolvePolicy>policy, <void*>pBuffer)
    check_status(status)

cpdef dbsric02(intptr_t handle, int dirA, int mb, int nnzb, size_t descrA, intptr_t bsrSortedVal, intptr_t bsrSortedRowPtr, intptr_t bsrSortedColInd, int blockDim, size_t info, int policy, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseDbsric02(<Handle>handle, <Direction>dirA, mb, nnzb, <const MatDescr>descrA, <double*>bsrSortedVal, <const int*>bsrSortedRowPtr, <const int*>bsrSortedColInd, blockDim, <bsric02Info_t>info, <SolvePolicy>policy, <void*>pBuffer)
    check_status(status)

cpdef cbsric02(intptr_t handle, int dirA, int mb, int nnzb, size_t descrA, intptr_t bsrSortedVal, intptr_t bsrSortedRowPtr, intptr_t bsrSortedColInd, int blockDim, size_t info, int policy, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseCbsric02(<Handle>handle, <Direction>dirA, mb, nnzb, <const MatDescr>descrA, <cuComplex*>bsrSortedVal, <const int*>bsrSortedRowPtr, <const int*>bsrSortedColInd, blockDim, <bsric02Info_t>info, <SolvePolicy>policy, <void*>pBuffer)
    check_status(status)

cpdef zbsric02(intptr_t handle, int dirA, int mb, int nnzb, size_t descrA, intptr_t bsrSortedVal, intptr_t bsrSortedRowPtr, intptr_t bsrSortedColInd, int blockDim, size_t info, int policy, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseZbsric02(<Handle>handle, <Direction>dirA, mb, nnzb, <const MatDescr>descrA, <cuDoubleComplex*>bsrSortedVal, <const int*>bsrSortedRowPtr, <const int*>bsrSortedColInd, blockDim, <bsric02Info_t>info, <SolvePolicy>policy, <void*>pBuffer)
    check_status(status)

cpdef int xbsric02_zeroPivot(intptr_t handle, size_t info) except? 0:
    cdef int position
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseXbsric02_zeroPivot(<Handle>handle, <bsric02Info_t>info, &position)
    check_status(status)
    return position


#################################################################
# cuSPARSE Preconditioners - Incomplete LU Factorization: level 0

cpdef scsrilu02_numericBoost(intptr_t handle, size_t info, int enable_boost, intptr_t tol, intptr_t boost_val):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseScsrilu02_numericBoost(<Handle>handle, <csrilu02Info_t>info, enable_boost, <double*>tol, <float*>boost_val)
    check_status(status)

cpdef dcsrilu02_numericBoost(intptr_t handle, size_t info, int enable_boost, intptr_t tol, intptr_t boost_val):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseDcsrilu02_numericBoost(<Handle>handle, <csrilu02Info_t>info, enable_boost, <double*>tol, <double*>boost_val)
    check_status(status)

cpdef ccsrilu02_numericBoost(intptr_t handle, size_t info, int enable_boost, intptr_t tol, intptr_t boost_val):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseCcsrilu02_numericBoost(<Handle>handle, <csrilu02Info_t>info, enable_boost, <double*>tol, <cuComplex*>boost_val)
    check_status(status)

cpdef zcsrilu02_numericBoost(intptr_t handle, size_t info, int enable_boost, intptr_t tol, intptr_t boost_val):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseZcsrilu02_numericBoost(<Handle>handle, <csrilu02Info_t>info, enable_boost, <double*>tol, <cuDoubleComplex*>boost_val)
    check_status(status)

cpdef int scsrilu02_bufferSize(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info) except? 0:
    cdef int bufferSizeInBytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseScsrilu02_bufferSize(<Handle>handle, m, nnz, <const MatDescr>descrA, <float*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <csrilu02Info_t>info, &bufferSizeInBytes)
    check_status(status)
    return bufferSizeInBytes

cpdef int dcsrilu02_bufferSize(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info) except? 0:
    cdef int bufferSizeInBytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseDcsrilu02_bufferSize(<Handle>handle, m, nnz, <const MatDescr>descrA, <double*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <csrilu02Info_t>info, &bufferSizeInBytes)
    check_status(status)
    return bufferSizeInBytes

cpdef int ccsrilu02_bufferSize(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info) except? 0:
    cdef int bufferSizeInBytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseCcsrilu02_bufferSize(<Handle>handle, m, nnz, <const MatDescr>descrA, <cuComplex*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <csrilu02Info_t>info, &bufferSizeInBytes)
    check_status(status)
    return bufferSizeInBytes

cpdef int zcsrilu02_bufferSize(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info) except? 0:
    cdef int bufferSizeInBytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseZcsrilu02_bufferSize(<Handle>handle, m, nnz, <const MatDescr>descrA, <cuDoubleComplex*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <csrilu02Info_t>info, &bufferSizeInBytes)
    check_status(status)
    return bufferSizeInBytes

cpdef scsrilu02_analysis(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info, int policy, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseScsrilu02_analysis(<Handle>handle, m, nnz, <const MatDescr>descrA, <const float*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <csrilu02Info_t>info, <SolvePolicy>policy, <void*>pBuffer)
    check_status(status)

cpdef dcsrilu02_analysis(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info, int policy, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseDcsrilu02_analysis(<Handle>handle, m, nnz, <const MatDescr>descrA, <const double*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <csrilu02Info_t>info, <SolvePolicy>policy, <void*>pBuffer)
    check_status(status)

cpdef ccsrilu02_analysis(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info, int policy, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseCcsrilu02_analysis(<Handle>handle, m, nnz, <const MatDescr>descrA, <const cuComplex*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <csrilu02Info_t>info, <SolvePolicy>policy, <void*>pBuffer)
    check_status(status)

cpdef zcsrilu02_analysis(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info, int policy, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseZcsrilu02_analysis(<Handle>handle, m, nnz, <const MatDescr>descrA, <const cuDoubleComplex*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <csrilu02Info_t>info, <SolvePolicy>policy, <void*>pBuffer)
    check_status(status)

cpdef scsrilu02(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrSortedValA_valM, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info, int policy, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseScsrilu02(<Handle>handle, m, nnz, <const MatDescr>descrA, <float*>csrSortedValA_valM, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <csrilu02Info_t>info, <SolvePolicy>policy, <void*>pBuffer)
    check_status(status)

cpdef dcsrilu02(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrSortedValA_valM, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info, int policy, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseDcsrilu02(<Handle>handle, m, nnz, <const MatDescr>descrA, <double*>csrSortedValA_valM, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <csrilu02Info_t>info, <SolvePolicy>policy, <void*>pBuffer)
    check_status(status)

cpdef ccsrilu02(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrSortedValA_valM, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info, int policy, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseCcsrilu02(<Handle>handle, m, nnz, <const MatDescr>descrA, <cuComplex*>csrSortedValA_valM, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <csrilu02Info_t>info, <SolvePolicy>policy, <void*>pBuffer)
    check_status(status)

cpdef zcsrilu02(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrSortedValA_valM, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, size_t info, int policy, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseZcsrilu02(<Handle>handle, m, nnz, <const MatDescr>descrA, <cuDoubleComplex*>csrSortedValA_valM, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <csrilu02Info_t>info, <SolvePolicy>policy, <void*>pBuffer)
    check_status(status)

cpdef xcsrilu02_zeroPivot(intptr_t handle, size_t info, intptr_t position):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseXcsrilu02_zeroPivot(<Handle>handle, <csrilu02Info_t>info, <int*>position)
    check_status(status)

cpdef sbsrilu02_numericBoost(intptr_t handle, size_t info, int enable_boost, intptr_t tol, intptr_t boost_val):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseSbsrilu02_numericBoost(<Handle>handle, <bsrilu02Info_t>info, enable_boost, <double*>tol, <float*>boost_val)
    check_status(status)

cpdef dbsrilu02_numericBoost(intptr_t handle, size_t info, int enable_boost, intptr_t tol, intptr_t boost_val):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseDbsrilu02_numericBoost(<Handle>handle, <bsrilu02Info_t>info, enable_boost, <double*>tol, <double*>boost_val)
    check_status(status)

cpdef cbsrilu02_numericBoost(intptr_t handle, size_t info, int enable_boost, intptr_t tol, intptr_t boost_val):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseCbsrilu02_numericBoost(<Handle>handle, <bsrilu02Info_t>info, enable_boost, <double*>tol, <cuComplex*>boost_val)
    check_status(status)

cpdef zbsrilu02_numericBoost(intptr_t handle, size_t info, int enable_boost, intptr_t tol, intptr_t boost_val):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseZbsrilu02_numericBoost(<Handle>handle, <bsrilu02Info_t>info, enable_boost, <double*>tol, <cuDoubleComplex*>boost_val)
    check_status(status)

cpdef int sbsrilu02_bufferSize(intptr_t handle, int dirA, int mb, int nnzb, size_t descrA, intptr_t bsrSortedVal, intptr_t bsrSortedRowPtr, intptr_t bsrSortedColInd, int blockDim, size_t info) except? 0:
    cdef int bufferSizeInBytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseSbsrilu02_bufferSize(<Handle>handle, <Direction>dirA, mb, nnzb, <const MatDescr>descrA, <float*>bsrSortedVal, <const int*>bsrSortedRowPtr, <const int*>bsrSortedColInd, blockDim, <bsrilu02Info_t>info, &bufferSizeInBytes)
    check_status(status)
    return bufferSizeInBytes

cpdef int dbsrilu02_bufferSize(intptr_t handle, int dirA, int mb, int nnzb, size_t descrA, intptr_t bsrSortedVal, intptr_t bsrSortedRowPtr, intptr_t bsrSortedColInd, int blockDim, size_t info) except? 0:
    cdef int bufferSizeInBytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseDbsrilu02_bufferSize(<Handle>handle, <Direction>dirA, mb, nnzb, <const MatDescr>descrA, <double*>bsrSortedVal, <const int*>bsrSortedRowPtr, <const int*>bsrSortedColInd, blockDim, <bsrilu02Info_t>info, &bufferSizeInBytes)
    check_status(status)
    return bufferSizeInBytes

cpdef int cbsrilu02_bufferSize(intptr_t handle, int dirA, int mb, int nnzb, size_t descrA, intptr_t bsrSortedVal, intptr_t bsrSortedRowPtr, intptr_t bsrSortedColInd, int blockDim, size_t info) except? 0:
    cdef int bufferSizeInBytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseCbsrilu02_bufferSize(<Handle>handle, <Direction>dirA, mb, nnzb, <const MatDescr>descrA, <cuComplex*>bsrSortedVal, <const int*>bsrSortedRowPtr, <const int*>bsrSortedColInd, blockDim, <bsrilu02Info_t>info, &bufferSizeInBytes)
    check_status(status)
    return bufferSizeInBytes

cpdef int zbsrilu02_bufferSize(intptr_t handle, int dirA, int mb, int nnzb, size_t descrA, intptr_t bsrSortedVal, intptr_t bsrSortedRowPtr, intptr_t bsrSortedColInd, int blockDim, size_t info) except? 0:
    cdef int bufferSizeInBytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseZbsrilu02_bufferSize(<Handle>handle, <Direction>dirA, mb, nnzb, <const MatDescr>descrA, <cuDoubleComplex*>bsrSortedVal, <const int*>bsrSortedRowPtr, <const int*>bsrSortedColInd, blockDim, <bsrilu02Info_t>info, &bufferSizeInBytes)
    check_status(status)
    return bufferSizeInBytes

cpdef sbsrilu02_analysis(intptr_t handle, int dirA, int mb, int nnzb, size_t descrA, intptr_t bsrSortedVal, intptr_t bsrSortedRowPtr, intptr_t bsrSortedColInd, int blockDim, size_t info, int policy, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseSbsrilu02_analysis(<Handle>handle, <Direction>dirA, mb, nnzb, <const MatDescr>descrA, <float*>bsrSortedVal, <const int*>bsrSortedRowPtr, <const int*>bsrSortedColInd, blockDim, <bsrilu02Info_t>info, <SolvePolicy>policy, <void*>pBuffer)
    check_status(status)

cpdef dbsrilu02_analysis(intptr_t handle, int dirA, int mb, int nnzb, size_t descrA, intptr_t bsrSortedVal, intptr_t bsrSortedRowPtr, intptr_t bsrSortedColInd, int blockDim, size_t info, int policy, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseDbsrilu02_analysis(<Handle>handle, <Direction>dirA, mb, nnzb, <const MatDescr>descrA, <double*>bsrSortedVal, <const int*>bsrSortedRowPtr, <const int*>bsrSortedColInd, blockDim, <bsrilu02Info_t>info, <SolvePolicy>policy, <void*>pBuffer)
    check_status(status)

cpdef cbsrilu02_analysis(intptr_t handle, int dirA, int mb, int nnzb, size_t descrA, intptr_t bsrSortedVal, intptr_t bsrSortedRowPtr, intptr_t bsrSortedColInd, int blockDim, size_t info, int policy, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseCbsrilu02_analysis(<Handle>handle, <Direction>dirA, mb, nnzb, <const MatDescr>descrA, <cuComplex*>bsrSortedVal, <const int*>bsrSortedRowPtr, <const int*>bsrSortedColInd, blockDim, <bsrilu02Info_t>info, <SolvePolicy>policy, <void*>pBuffer)
    check_status(status)

cpdef zbsrilu02_analysis(intptr_t handle, int dirA, int mb, int nnzb, size_t descrA, intptr_t bsrSortedVal, intptr_t bsrSortedRowPtr, intptr_t bsrSortedColInd, int blockDim, size_t info, int policy, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseZbsrilu02_analysis(<Handle>handle, <Direction>dirA, mb, nnzb, <const MatDescr>descrA, <cuDoubleComplex*>bsrSortedVal, <const int*>bsrSortedRowPtr, <const int*>bsrSortedColInd, blockDim, <bsrilu02Info_t>info, <SolvePolicy>policy, <void*>pBuffer)
    check_status(status)

cpdef sbsrilu02(intptr_t handle, int dirA, int mb, int nnzb, size_t descrA, intptr_t bsrSortedVal, intptr_t bsrSortedRowPtr, intptr_t bsrSortedColInd, int blockDim, size_t info, int policy, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseSbsrilu02(<Handle>handle, <Direction>dirA, mb, nnzb, <const MatDescr>descrA, <float*>bsrSortedVal, <const int*>bsrSortedRowPtr, <const int*>bsrSortedColInd, blockDim, <bsrilu02Info_t>info, <SolvePolicy>policy, <void*>pBuffer)
    check_status(status)

cpdef dbsrilu02(intptr_t handle, int dirA, int mb, int nnzb, size_t descrA, intptr_t bsrSortedVal, intptr_t bsrSortedRowPtr, intptr_t bsrSortedColInd, int blockDim, size_t info, int policy, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseDbsrilu02(<Handle>handle, <Direction>dirA, mb, nnzb, <const MatDescr>descrA, <double*>bsrSortedVal, <const int*>bsrSortedRowPtr, <const int*>bsrSortedColInd, blockDim, <bsrilu02Info_t>info, <SolvePolicy>policy, <void*>pBuffer)
    check_status(status)

cpdef cbsrilu02(intptr_t handle, int dirA, int mb, int nnzb, size_t descrA, intptr_t bsrSortedVal, intptr_t bsrSortedRowPtr, intptr_t bsrSortedColInd, int blockDim, size_t info, int policy, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseCbsrilu02(<Handle>handle, <Direction>dirA, mb, nnzb, <const MatDescr>descrA, <cuComplex*>bsrSortedVal, <const int*>bsrSortedRowPtr, <const int*>bsrSortedColInd, blockDim, <bsrilu02Info_t>info, <SolvePolicy>policy, <void*>pBuffer)
    check_status(status)

cpdef zbsrilu02(intptr_t handle, int dirA, int mb, int nnzb, size_t descrA, intptr_t bsrSortedVal, intptr_t bsrSortedRowPtr, intptr_t bsrSortedColInd, int blockDim, size_t info, int policy, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseZbsrilu02(<Handle>handle, <Direction>dirA, mb, nnzb, <const MatDescr>descrA, <cuDoubleComplex*>bsrSortedVal, <const int*>bsrSortedRowPtr, <const int*>bsrSortedColInd, blockDim, <bsrilu02Info_t>info, <SolvePolicy>policy, <void*>pBuffer)
    check_status(status)

cpdef xbsrilu02_zeroPivot(intptr_t handle, size_t info, intptr_t position):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseXbsrilu02_zeroPivot(<Handle>handle, <bsrilu02Info_t>info, <int*>position)
    check_status(status)


##############################################
# cuSPARSE Preconditioners - Tridiagonal Solve

cpdef size_t sgtsv2_bufferSizeExt(intptr_t handle, int m, int n, intptr_t dl, intptr_t d, intptr_t du, intptr_t B, int ldb) except? 0:
    cdef size_t bufferSizeInBytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseSgtsv2_bufferSizeExt(<Handle>handle, m, n, <const float*>dl, <const float*>d, <const float*>du, <const float*>B, ldb, &bufferSizeInBytes)
    check_status(status)
    return bufferSizeInBytes

cpdef size_t dgtsv2_bufferSizeExt(intptr_t handle, int m, int n, intptr_t dl, intptr_t d, intptr_t du, intptr_t B, int ldb) except? 0:
    cdef size_t bufferSizeInBytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseDgtsv2_bufferSizeExt(<Handle>handle, m, n, <const double*>dl, <const double*>d, <const double*>du, <const double*>B, ldb, &bufferSizeInBytes)
    check_status(status)
    return bufferSizeInBytes

cpdef size_t cgtsv2_bufferSizeExt(intptr_t handle, int m, int n, intptr_t dl, intptr_t d, intptr_t du, intptr_t B, int ldb) except? 0:
    cdef size_t bufferSizeInBytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseCgtsv2_bufferSizeExt(<Handle>handle, m, n, <const cuComplex*>dl, <const cuComplex*>d, <const cuComplex*>du, <const cuComplex*>B, ldb, &bufferSizeInBytes)
    check_status(status)
    return bufferSizeInBytes

cpdef size_t zgtsv2_bufferSizeExt(intptr_t handle, int m, int n, intptr_t dl, intptr_t d, intptr_t du, intptr_t B, int ldb) except? 0:
    cdef size_t bufferSizeInBytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseZgtsv2_bufferSizeExt(<Handle>handle, m, n, <const cuDoubleComplex*>dl, <const cuDoubleComplex*>d, <const cuDoubleComplex*>du, <const cuDoubleComplex*>B, ldb, &bufferSizeInBytes)
    check_status(status)
    return bufferSizeInBytes

cpdef sgtsv2(intptr_t handle, int m, int n, intptr_t dl, intptr_t d, intptr_t du, intptr_t B, int ldb, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseSgtsv2(<Handle>handle, m, n, <const float*>dl, <const float*>d, <const float*>du, <float*>B, ldb, <void*>pBuffer)
    check_status(status)

cpdef dgtsv2(intptr_t handle, int m, int n, intptr_t dl, intptr_t d, intptr_t du, intptr_t B, int ldb, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseDgtsv2(<Handle>handle, m, n, <const double*>dl, <const double*>d, <const double*>du, <double*>B, ldb, <void*>pBuffer)
    check_status(status)

cpdef cgtsv2(intptr_t handle, int m, int n, intptr_t dl, intptr_t d, intptr_t du, intptr_t B, int ldb, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseCgtsv2(<Handle>handle, m, n, <const cuComplex*>dl, <const cuComplex*>d, <const cuComplex*>du, <cuComplex*>B, ldb, <void*>pBuffer)
    check_status(status)

cpdef zgtsv2(intptr_t handle, int m, int n, intptr_t dl, intptr_t d, intptr_t du, intptr_t B, int ldb, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseZgtsv2(<Handle>handle, m, n, <const cuDoubleComplex*>dl, <const cuDoubleComplex*>d, <const cuDoubleComplex*>du, <cuDoubleComplex*>B, ldb, <void*>pBuffer)
    check_status(status)

cpdef size_t sgtsv2_nopivot_bufferSizeExt(intptr_t handle, int m, int n, intptr_t dl, intptr_t d, intptr_t du, intptr_t B, int ldb) except? 0:
    cdef size_t bufferSizeInBytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseSgtsv2_nopivot_bufferSizeExt(<Handle>handle, m, n, <const float*>dl, <const float*>d, <const float*>du, <const float*>B, ldb, &bufferSizeInBytes)
    check_status(status)
    return bufferSizeInBytes

cpdef size_t dgtsv2_nopivot_bufferSizeExt(intptr_t handle, int m, int n, intptr_t dl, intptr_t d, intptr_t du, intptr_t B, int ldb) except? 0:
    cdef size_t bufferSizeInBytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseDgtsv2_nopivot_bufferSizeExt(<Handle>handle, m, n, <const double*>dl, <const double*>d, <const double*>du, <const double*>B, ldb, &bufferSizeInBytes)
    check_status(status)
    return bufferSizeInBytes

cpdef size_t cgtsv2_nopivot_bufferSizeExt(intptr_t handle, int m, int n, intptr_t dl, intptr_t d, intptr_t du, intptr_t B, int ldb) except? 0:
    cdef size_t bufferSizeInBytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseCgtsv2_nopivot_bufferSizeExt(<Handle>handle, m, n, <const cuComplex*>dl, <const cuComplex*>d, <const cuComplex*>du, <const cuComplex*>B, ldb, &bufferSizeInBytes)
    check_status(status)
    return bufferSizeInBytes

cpdef size_t zgtsv2_nopivot_bufferSizeExt(intptr_t handle, int m, int n, intptr_t dl, intptr_t d, intptr_t du, intptr_t B, int ldb) except? 0:
    cdef size_t bufferSizeInBytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseZgtsv2_nopivot_bufferSizeExt(<Handle>handle, m, n, <const cuDoubleComplex*>dl, <const cuDoubleComplex*>d, <const cuDoubleComplex*>du, <const cuDoubleComplex*>B, ldb, &bufferSizeInBytes)
    check_status(status)
    return bufferSizeInBytes

cpdef sgtsv2_nopivot(intptr_t handle, int m, int n, intptr_t dl, intptr_t d, intptr_t du, intptr_t B, int ldb, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseSgtsv2_nopivot(<Handle>handle, m, n, <const float*>dl, <const float*>d, <const float*>du, <float*>B, ldb, <void*>pBuffer)
    check_status(status)

cpdef dgtsv2_nopivot(intptr_t handle, int m, int n, intptr_t dl, intptr_t d, intptr_t du, intptr_t B, int ldb, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseDgtsv2_nopivot(<Handle>handle, m, n, <const double*>dl, <const double*>d, <const double*>du, <double*>B, ldb, <void*>pBuffer)
    check_status(status)

cpdef cgtsv2_nopivot(intptr_t handle, int m, int n, intptr_t dl, intptr_t d, intptr_t du, intptr_t B, int ldb, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseCgtsv2_nopivot(<Handle>handle, m, n, <const cuComplex*>dl, <const cuComplex*>d, <const cuComplex*>du, <cuComplex*>B, ldb, <void*>pBuffer)
    check_status(status)

cpdef zgtsv2_nopivot(intptr_t handle, int m, int n, intptr_t dl, intptr_t d, intptr_t du, intptr_t B, int ldb, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseZgtsv2_nopivot(<Handle>handle, m, n, <const cuDoubleComplex*>dl, <const cuDoubleComplex*>d, <const cuDoubleComplex*>du, <cuDoubleComplex*>B, ldb, <void*>pBuffer)
    check_status(status)


######################################################
# cuSPARSE Preconditioners - Batched Tridiagonal Solve

cpdef size_t sgtsv2StridedBatch_bufferSizeExt(intptr_t handle, int m, intptr_t dl, intptr_t d, intptr_t du, intptr_t x, int batchCount, int batchStride) except? 0:
    cdef size_t bufferSizeInBytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseSgtsv2StridedBatch_bufferSizeExt(<Handle>handle, m, <const float*>dl, <const float*>d, <const float*>du, <const float*>x, batchCount, batchStride, &bufferSizeInBytes)
    check_status(status)
    return bufferSizeInBytes

cpdef size_t dgtsv2StridedBatch_bufferSizeExt(intptr_t handle, int m, intptr_t dl, intptr_t d, intptr_t du, intptr_t x, int batchCount, int batchStride) except? 0:
    cdef size_t bufferSizeInBytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseDgtsv2StridedBatch_bufferSizeExt(<Handle>handle, m, <const double*>dl, <const double*>d, <const double*>du, <const double*>x, batchCount, batchStride, &bufferSizeInBytes)
    check_status(status)
    return bufferSizeInBytes

cpdef size_t cgtsv2StridedBatch_bufferSizeExt(intptr_t handle, int m, intptr_t dl, intptr_t d, intptr_t du, intptr_t x, int batchCount, int batchStride) except? 0:
    cdef size_t bufferSizeInBytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseCgtsv2StridedBatch_bufferSizeExt(<Handle>handle, m, <const cuComplex*>dl, <const cuComplex*>d, <const cuComplex*>du, <const cuComplex*>x, batchCount, batchStride, &bufferSizeInBytes)
    check_status(status)
    return bufferSizeInBytes

cpdef size_t zgtsv2StridedBatch_bufferSizeExt(intptr_t handle, int m, intptr_t dl, intptr_t d, intptr_t du, intptr_t x, int batchCount, int batchStride) except? 0:
    cdef size_t bufferSizeInBytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseZgtsv2StridedBatch_bufferSizeExt(<Handle>handle, m, <const cuDoubleComplex*>dl, <const cuDoubleComplex*>d, <const cuDoubleComplex*>du, <const cuDoubleComplex*>x, batchCount, batchStride, &bufferSizeInBytes)
    check_status(status)
    return bufferSizeInBytes

cpdef sgtsv2StridedBatch(intptr_t handle, int m, intptr_t dl, intptr_t d, intptr_t du, intptr_t x, int batchCount, int batchStride, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseSgtsv2StridedBatch(<Handle>handle, m, <const float*>dl, <const float*>d, <const float*>du, <float*>x, batchCount, batchStride, <void*>pBuffer)
    check_status(status)

cpdef dgtsv2StridedBatch(intptr_t handle, int m, intptr_t dl, intptr_t d, intptr_t du, intptr_t x, int batchCount, int batchStride, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseDgtsv2StridedBatch(<Handle>handle, m, <const double*>dl, <const double*>d, <const double*>du, <double*>x, batchCount, batchStride, <void*>pBuffer)
    check_status(status)

cpdef cgtsv2StridedBatch(intptr_t handle, int m, intptr_t dl, intptr_t d, intptr_t du, intptr_t x, int batchCount, int batchStride, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseCgtsv2StridedBatch(<Handle>handle, m, <const cuComplex*>dl, <const cuComplex*>d, <const cuComplex*>du, <cuComplex*>x, batchCount, batchStride, <void*>pBuffer)
    check_status(status)

cpdef zgtsv2StridedBatch(intptr_t handle, int m, intptr_t dl, intptr_t d, intptr_t du, intptr_t x, int batchCount, int batchStride, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseZgtsv2StridedBatch(<Handle>handle, m, <const cuDoubleComplex*>dl, <const cuDoubleComplex*>d, <const cuDoubleComplex*>du, <cuDoubleComplex*>x, batchCount, batchStride, <void*>pBuffer)
    check_status(status)

cpdef size_t sgtsvInterleavedBatch_bufferSizeExt(intptr_t handle, int algo, int m, intptr_t dl, intptr_t d, intptr_t du, intptr_t x, int batchCount) except? 0:
    cdef size_t bufferSizeInBytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseSgtsvInterleavedBatch_bufferSizeExt(<Handle>handle, algo, m, <const float*>dl, <const float*>d, <const float*>du, <const float*>x, batchCount, &bufferSizeInBytes)
    check_status(status)
    return bufferSizeInBytes

cpdef size_t dgtsvInterleavedBatch_bufferSizeExt(intptr_t handle, int algo, int m, intptr_t dl, intptr_t d, intptr_t du, intptr_t x, int batchCount) except? 0:
    cdef size_t bufferSizeInBytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseDgtsvInterleavedBatch_bufferSizeExt(<Handle>handle, algo, m, <const double*>dl, <const double*>d, <const double*>du, <const double*>x, batchCount, &bufferSizeInBytes)
    check_status(status)
    return bufferSizeInBytes

cpdef size_t cgtsvInterleavedBatch_bufferSizeExt(intptr_t handle, int algo, int m, intptr_t dl, intptr_t d, intptr_t du, intptr_t x, int batchCount) except? 0:
    cdef size_t bufferSizeInBytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseCgtsvInterleavedBatch_bufferSizeExt(<Handle>handle, algo, m, <const cuComplex*>dl, <const cuComplex*>d, <const cuComplex*>du, <const cuComplex*>x, batchCount, &bufferSizeInBytes)
    check_status(status)
    return bufferSizeInBytes

cpdef size_t zgtsvInterleavedBatch_bufferSizeExt(intptr_t handle, int algo, int m, intptr_t dl, intptr_t d, intptr_t du, intptr_t x, int batchCount) except? 0:
    cdef size_t bufferSizeInBytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseZgtsvInterleavedBatch_bufferSizeExt(<Handle>handle, algo, m, <const cuDoubleComplex*>dl, <const cuDoubleComplex*>d, <const cuDoubleComplex*>du, <const cuDoubleComplex*>x, batchCount, &bufferSizeInBytes)
    check_status(status)
    return bufferSizeInBytes

cpdef sgtsvInterleavedBatch(intptr_t handle, int algo, int m, intptr_t dl, intptr_t d, intptr_t du, intptr_t x, int batchCount, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseSgtsvInterleavedBatch(<Handle>handle, algo, m, <float*>dl, <float*>d, <float*>du, <float*>x, batchCount, <void*>pBuffer)
    check_status(status)

cpdef dgtsvInterleavedBatch(intptr_t handle, int algo, int m, intptr_t dl, intptr_t d, intptr_t du, intptr_t x, int batchCount, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseDgtsvInterleavedBatch(<Handle>handle, algo, m, <double*>dl, <double*>d, <double*>du, <double*>x, batchCount, <void*>pBuffer)
    check_status(status)

cpdef cgtsvInterleavedBatch(intptr_t handle, int algo, int m, intptr_t dl, intptr_t d, intptr_t du, intptr_t x, int batchCount, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseCgtsvInterleavedBatch(<Handle>handle, algo, m, <cuComplex*>dl, <cuComplex*>d, <cuComplex*>du, <cuComplex*>x, batchCount, <void*>pBuffer)
    check_status(status)

cpdef zgtsvInterleavedBatch(intptr_t handle, int algo, int m, intptr_t dl, intptr_t d, intptr_t du, intptr_t x, int batchCount, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseZgtsvInterleavedBatch(<Handle>handle, algo, m, <cuDoubleComplex*>dl, <cuDoubleComplex*>d, <cuDoubleComplex*>du, <cuDoubleComplex*>x, batchCount, <void*>pBuffer)
    check_status(status)


########################################################
# cuSPARSE Preconditioners - Batched Pentadiagonal Solve

cpdef size_t sgpsvInterleavedBatch_bufferSizeExt(intptr_t handle, int algo, int m, intptr_t ds, intptr_t dl, intptr_t d, intptr_t du, intptr_t dw, intptr_t x, int batchCount) except? 0:
    cdef size_t bufferSizeInBytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseSgpsvInterleavedBatch_bufferSizeExt(<Handle>handle, algo, m, <const float*>ds, <const float*>dl, <const float*>d, <const float*>du, <const float*>dw, <const float*>x, batchCount, &bufferSizeInBytes)
    check_status(status)
    return bufferSizeInBytes

cpdef size_t dgpsvInterleavedBatch_bufferSizeExt(intptr_t handle, int algo, int m, intptr_t ds, intptr_t dl, intptr_t d, intptr_t du, intptr_t dw, intptr_t x, int batchCount) except? 0:
    cdef size_t bufferSizeInBytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseDgpsvInterleavedBatch_bufferSizeExt(<Handle>handle, algo, m, <const double*>ds, <const double*>dl, <const double*>d, <const double*>du, <const double*>dw, <const double*>x, batchCount, &bufferSizeInBytes)
    check_status(status)
    return bufferSizeInBytes

cpdef size_t cgpsvInterleavedBatch_bufferSizeExt(intptr_t handle, int algo, int m, intptr_t ds, intptr_t dl, intptr_t d, intptr_t du, intptr_t dw, intptr_t x, int batchCount) except? 0:
    cdef size_t bufferSizeInBytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseCgpsvInterleavedBatch_bufferSizeExt(<Handle>handle, algo, m, <const cuComplex*>ds, <const cuComplex*>dl, <const cuComplex*>d, <const cuComplex*>du, <const cuComplex*>dw, <const cuComplex*>x, batchCount, &bufferSizeInBytes)
    check_status(status)
    return bufferSizeInBytes

cpdef size_t zgpsvInterleavedBatch_bufferSizeExt(intptr_t handle, int algo, int m, intptr_t ds, intptr_t dl, intptr_t d, intptr_t du, intptr_t dw, intptr_t x, int batchCount) except? 0:
    cdef size_t bufferSizeInBytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseZgpsvInterleavedBatch_bufferSizeExt(<Handle>handle, algo, m, <const cuDoubleComplex*>ds, <const cuDoubleComplex*>dl, <const cuDoubleComplex*>d, <const cuDoubleComplex*>du, <const cuDoubleComplex*>dw, <const cuDoubleComplex*>x, batchCount, &bufferSizeInBytes)
    check_status(status)
    return bufferSizeInBytes

cpdef sgpsvInterleavedBatch(intptr_t handle, int algo, int m, intptr_t ds, intptr_t dl, intptr_t d, intptr_t du, intptr_t dw, intptr_t x, int batchCount, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseSgpsvInterleavedBatch(<Handle>handle, algo, m, <float*>ds, <float*>dl, <float*>d, <float*>du, <float*>dw, <float*>x, batchCount, <void*>pBuffer)
    check_status(status)

cpdef dgpsvInterleavedBatch(intptr_t handle, int algo, int m, intptr_t ds, intptr_t dl, intptr_t d, intptr_t du, intptr_t dw, intptr_t x, int batchCount, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseDgpsvInterleavedBatch(<Handle>handle, algo, m, <double*>ds, <double*>dl, <double*>d, <double*>du, <double*>dw, <double*>x, batchCount, <void*>pBuffer)
    check_status(status)

cpdef cgpsvInterleavedBatch(intptr_t handle, int algo, int m, intptr_t ds, intptr_t dl, intptr_t d, intptr_t du, intptr_t dw, intptr_t x, int batchCount, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseCgpsvInterleavedBatch(<Handle>handle, algo, m, <cuComplex*>ds, <cuComplex*>dl, <cuComplex*>d, <cuComplex*>du, <cuComplex*>dw, <cuComplex*>x, batchCount, <void*>pBuffer)
    check_status(status)

cpdef zgpsvInterleavedBatch(intptr_t handle, int algo, int m, intptr_t ds, intptr_t dl, intptr_t d, intptr_t du, intptr_t dw, intptr_t x, int batchCount, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseZgpsvInterleavedBatch(<Handle>handle, algo, m, <cuDoubleComplex*>ds, <cuDoubleComplex*>dl, <cuDoubleComplex*>d, <cuDoubleComplex*>du, <cuDoubleComplex*>dw, <cuDoubleComplex*>x, batchCount, <void*>pBuffer)
    check_status(status)


########################################
# cuSPARSE Reorderings


########################################
# cuSPARSE Format Conversion

cpdef xcoo2csr(intptr_t handle, intptr_t cooRowInd, int nnz, int m, intptr_t csrSortedRowPtr, int idxBase):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseXcoo2csr(<Handle>handle, <const int*>cooRowInd, nnz, m, <int*>csrSortedRowPtr, <IndexBase>idxBase)
    check_status(status)

cpdef scsc2dense(intptr_t handle, int m, int n, size_t descrA, intptr_t cscSortedValA, intptr_t cscSortedRowIndA, intptr_t cscSortedColPtrA, intptr_t A, int lda):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseScsc2dense(<Handle>handle, m, n, <const MatDescr>descrA, <const float*>cscSortedValA, <const int*>cscSortedRowIndA, <const int*>cscSortedColPtrA, <float*>A, lda)
    check_status(status)

cpdef dcsc2dense(intptr_t handle, int m, int n, size_t descrA, intptr_t cscSortedValA, intptr_t cscSortedRowIndA, intptr_t cscSortedColPtrA, intptr_t A, int lda):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseDcsc2dense(<Handle>handle, m, n, <const MatDescr>descrA, <const double*>cscSortedValA, <const int*>cscSortedRowIndA, <const int*>cscSortedColPtrA, <double*>A, lda)
    check_status(status)

cpdef ccsc2dense(intptr_t handle, int m, int n, size_t descrA, intptr_t cscSortedValA, intptr_t cscSortedRowIndA, intptr_t cscSortedColPtrA, intptr_t A, int lda):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseCcsc2dense(<Handle>handle, m, n, <const MatDescr>descrA, <const cuComplex*>cscSortedValA, <const int*>cscSortedRowIndA, <const int*>cscSortedColPtrA, <cuComplex*>A, lda)
    check_status(status)

cpdef zcsc2dense(intptr_t handle, int m, int n, size_t descrA, intptr_t cscSortedValA, intptr_t cscSortedRowIndA, intptr_t cscSortedColPtrA, intptr_t A, int lda):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseZcsc2dense(<Handle>handle, m, n, <const MatDescr>descrA, <const cuDoubleComplex*>cscSortedValA, <const int*>cscSortedRowIndA, <const int*>cscSortedColPtrA, <cuDoubleComplex*>A, lda)
    check_status(status)

cpdef xcsr2coo(intptr_t handle, intptr_t csrSortedRowPtr, int nnz, int m, intptr_t cooRowInd, int idxBase):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseXcsr2coo(<Handle>handle, <const int*>csrSortedRowPtr, nnz, m, <int*>cooRowInd, <IndexBase>idxBase)
    check_status(status)

# REMOVED
cpdef scsr2csc(intptr_t handle, int m, int n, int nnz, intptr_t csrSortedVal, intptr_t csrSortedRowPtr, intptr_t csrSortedColInd, intptr_t cscSortedVal, intptr_t cscSortedRowInd, intptr_t cscSortedColPtr, int copyValues, int idxBase):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseScsr2csc(<Handle>handle, m, n, nnz, <const float*>csrSortedVal, <const int*>csrSortedRowPtr, <const int*>csrSortedColInd, <float*>cscSortedVal, <int*>cscSortedRowInd, <int*>cscSortedColPtr, <Action>copyValues, <IndexBase>idxBase)
    check_status(status)

# REMOVED
cpdef dcsr2csc(intptr_t handle, int m, int n, int nnz, intptr_t csrSortedVal, intptr_t csrSortedRowPtr, intptr_t csrSortedColInd, intptr_t cscSortedVal, intptr_t cscSortedRowInd, intptr_t cscSortedColPtr, int copyValues, int idxBase):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseDcsr2csc(<Handle>handle, m, n, nnz, <const double*>csrSortedVal, <const int*>csrSortedRowPtr, <const int*>csrSortedColInd, <double*>cscSortedVal, <int*>cscSortedRowInd, <int*>cscSortedColPtr, <Action>copyValues, <IndexBase>idxBase)
    check_status(status)

# REMOVED
cpdef ccsr2csc(intptr_t handle, int m, int n, int nnz, intptr_t csrSortedVal, intptr_t csrSortedRowPtr, intptr_t csrSortedColInd, intptr_t cscSortedVal, intptr_t cscSortedRowInd, intptr_t cscSortedColPtr, int copyValues, int idxBase):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseCcsr2csc(<Handle>handle, m, n, nnz, <const cuComplex*>csrSortedVal, <const int*>csrSortedRowPtr, <const int*>csrSortedColInd, <cuComplex*>cscSortedVal, <int*>cscSortedRowInd, <int*>cscSortedColPtr, <Action>copyValues, <IndexBase>idxBase)
    check_status(status)

# REMOVED
cpdef zcsr2csc(intptr_t handle, int m, int n, int nnz, intptr_t csrSortedVal, intptr_t csrSortedRowPtr, intptr_t csrSortedColInd, intptr_t cscSortedVal, intptr_t cscSortedRowInd, intptr_t cscSortedColPtr, int copyValues, int idxBase):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseZcsr2csc(<Handle>handle, m, n, nnz, <const cuDoubleComplex*>csrSortedVal, <const int*>csrSortedRowPtr, <const int*>csrSortedColInd, <cuDoubleComplex*>cscSortedVal, <int*>cscSortedRowInd, <int*>cscSortedColPtr, <Action>copyValues, <IndexBase>idxBase)
    check_status(status)

cpdef size_t csr2cscEx2_bufferSize(intptr_t handle, int m, int n, int nnz, intptr_t csrVal, intptr_t csrRowPtr, intptr_t csrColInd, intptr_t cscVal, intptr_t cscColPtr, intptr_t cscRowInd, size_t valType, int copyValues, int idxBase, int alg) except? 0:
    cdef size_t bufferSize
    status = cusparseCsr2cscEx2_bufferSize(<Handle>handle, m, n, nnz, <const void*>csrVal, <const int*>csrRowPtr, <const int*>csrColInd, <void*>cscVal, <int*>cscColPtr, <int*>cscRowInd, <DataType>valType, <Action>copyValues, <IndexBase>idxBase, <Csr2CscAlg>alg, &bufferSize)
    check_status(status)
    return bufferSize

cpdef csr2cscEx2(intptr_t handle, int m, int n, int nnz, intptr_t csrVal, intptr_t csrRowPtr, intptr_t csrColInd, intptr_t cscVal, intptr_t cscColPtr, intptr_t cscRowInd, size_t valType, int copyValues, int idxBase, int alg, intptr_t buffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseCsr2cscEx2(<Handle>handle, m, n, nnz, <const void*>csrVal, <const int*>csrRowPtr, <const int*>csrColInd, <void*>cscVal, <int*>cscColPtr, <int*>cscRowInd, <DataType>valType, <Action>copyValues, <IndexBase>idxBase, <Csr2CscAlg>alg, <void*>buffer)
    check_status(status)

cpdef scsr2dense(intptr_t handle, int m, int n, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t A, int lda):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseScsr2dense(<Handle>handle, m, n, <const MatDescr>descrA, <const float*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <float*>A, lda)
    check_status(status)

cpdef dcsr2dense(intptr_t handle, int m, int n, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t A, int lda):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseDcsr2dense(<Handle>handle, m, n, <const MatDescr>descrA, <const double*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <double*>A, lda)
    check_status(status)

cpdef ccsr2dense(intptr_t handle, int m, int n, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t A, int lda):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseCcsr2dense(<Handle>handle, m, n, <const MatDescr>descrA, <const cuComplex*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <cuComplex*>A, lda)
    check_status(status)

cpdef zcsr2dense(intptr_t handle, int m, int n, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA, intptr_t A, int lda):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseZcsr2dense(<Handle>handle, m, n, <const MatDescr>descrA, <const cuDoubleComplex*>csrSortedValA, <const int*>csrSortedRowPtrA, <const int*>csrSortedColIndA, <cuDoubleComplex*>A, lda)
    check_status(status)

cpdef int snnz_compress(intptr_t handle, int m, size_t descr, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t nnzPerRow, float tol) except? 0:
    cdef int nnzC
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseSnnz_compress(<Handle>handle, m, <const MatDescr>descr, <const float*>csrSortedValA, <const int*>csrSortedRowPtrA, <int*>nnzPerRow, &nnzC, tol)
    check_status(status)
    return nnzC

cpdef int dnnz_compress(intptr_t handle, int m, size_t descr, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t nnzPerRow, double tol) except? 0:
    cdef int nnzC
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseDnnz_compress(<Handle>handle, m, <const MatDescr>descr, <const double*>csrSortedValA, <const int*>csrSortedRowPtrA, <int*>nnzPerRow, &nnzC, tol)
    check_status(status)
    return nnzC

cpdef int cnnz_compress(intptr_t handle, int m, size_t descr, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t nnzPerRow, complex tol) except? 0:
    cdef int nnzC
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseCnnz_compress(<Handle>handle, m, <const MatDescr>descr, <const cuComplex*>csrSortedValA, <const int*>csrSortedRowPtrA, <int*>nnzPerRow, &nnzC, complex_to_cuda(tol))
    check_status(status)
    return nnzC

cpdef int znnz_compress(intptr_t handle, int m, size_t descr, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t nnzPerRow, double complex tol) except? 0:
    cdef int nnzC
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseZnnz_compress(<Handle>handle, m, <const MatDescr>descr, <const cuDoubleComplex*>csrSortedValA, <const int*>csrSortedRowPtrA, <int*>nnzPerRow, &nnzC, double_complex_to_cuda(tol))
    check_status(status)
    return nnzC

cpdef scsr2csr_compress(intptr_t handle, int m, int n, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedColIndA, intptr_t csrSortedRowPtrA, int nnzA, intptr_t nnzPerRow, intptr_t csrSortedValC, intptr_t csrSortedColIndC, intptr_t csrSortedRowPtrC, float tol):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseScsr2csr_compress(<Handle>handle, m, n, <const MatDescr>descrA, <const float*>csrSortedValA, <const int*>csrSortedColIndA, <const int*>csrSortedRowPtrA, nnzA, <const int*>nnzPerRow, <float*>csrSortedValC, <int*>csrSortedColIndC, <int*>csrSortedRowPtrC, tol)
    check_status(status)

cpdef dcsr2csr_compress(intptr_t handle, int m, int n, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedColIndA, intptr_t csrSortedRowPtrA, int nnzA, intptr_t nnzPerRow, intptr_t csrSortedValC, intptr_t csrSortedColIndC, intptr_t csrSortedRowPtrC, double tol):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseDcsr2csr_compress(<Handle>handle, m, n, <const MatDescr>descrA, <const double*>csrSortedValA, <const int*>csrSortedColIndA, <const int*>csrSortedRowPtrA, nnzA, <const int*>nnzPerRow, <double*>csrSortedValC, <int*>csrSortedColIndC, <int*>csrSortedRowPtrC, tol)
    check_status(status)

cpdef ccsr2csr_compress(intptr_t handle, int m, int n, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedColIndA, intptr_t csrSortedRowPtrA, int nnzA, intptr_t nnzPerRow, intptr_t csrSortedValC, intptr_t csrSortedColIndC, intptr_t csrSortedRowPtrC, complex tol):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseCcsr2csr_compress(<Handle>handle, m, n, <const MatDescr>descrA, <const cuComplex*>csrSortedValA, <const int*>csrSortedColIndA, <const int*>csrSortedRowPtrA, nnzA, <const int*>nnzPerRow, <cuComplex*>csrSortedValC, <int*>csrSortedColIndC, <int*>csrSortedRowPtrC, complex_to_cuda(tol))
    check_status(status)

cpdef zcsr2csr_compress(intptr_t handle, int m, int n, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedColIndA, intptr_t csrSortedRowPtrA, int nnzA, intptr_t nnzPerRow, intptr_t csrSortedValC, intptr_t csrSortedColIndC, intptr_t csrSortedRowPtrC, double complex tol):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseZcsr2csr_compress(<Handle>handle, m, n, <const MatDescr>descrA, <const cuDoubleComplex*>csrSortedValA, <const int*>csrSortedColIndA, <const int*>csrSortedRowPtrA, nnzA, <const int*>nnzPerRow, <cuDoubleComplex*>csrSortedValC, <int*>csrSortedColIndC, <int*>csrSortedRowPtrC, double_complex_to_cuda(tol))
    check_status(status)

cpdef sdense2csc(intptr_t handle, int m, int n, size_t descrA, intptr_t A, int lda, intptr_t nnzPerCol, intptr_t cscSortedValA, intptr_t cscSortedRowIndA, intptr_t cscSortedColPtrA):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseSdense2csc(<Handle>handle, m, n, <const MatDescr>descrA, <const float*>A, lda, <const int*>nnzPerCol, <float*>cscSortedValA, <int*>cscSortedRowIndA, <int*>cscSortedColPtrA)
    check_status(status)

cpdef ddense2csc(intptr_t handle, int m, int n, size_t descrA, intptr_t A, int lda, intptr_t nnzPerCol, intptr_t cscSortedValA, intptr_t cscSortedRowIndA, intptr_t cscSortedColPtrA):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseDdense2csc(<Handle>handle, m, n, <const MatDescr>descrA, <const double*>A, lda, <const int*>nnzPerCol, <double*>cscSortedValA, <int*>cscSortedRowIndA, <int*>cscSortedColPtrA)
    check_status(status)

cpdef cdense2csc(intptr_t handle, int m, int n, size_t descrA, intptr_t A, int lda, intptr_t nnzPerCol, intptr_t cscSortedValA, intptr_t cscSortedRowIndA, intptr_t cscSortedColPtrA):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseCdense2csc(<Handle>handle, m, n, <const MatDescr>descrA, <const cuComplex*>A, lda, <const int*>nnzPerCol, <cuComplex*>cscSortedValA, <int*>cscSortedRowIndA, <int*>cscSortedColPtrA)
    check_status(status)

cpdef zdense2csc(intptr_t handle, int m, int n, size_t descrA, intptr_t A, int lda, intptr_t nnzPerCol, intptr_t cscSortedValA, intptr_t cscSortedRowIndA, intptr_t cscSortedColPtrA):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseZdense2csc(<Handle>handle, m, n, <const MatDescr>descrA, <const cuDoubleComplex*>A, lda, <const int*>nnzPerCol, <cuDoubleComplex*>cscSortedValA, <int*>cscSortedRowIndA, <int*>cscSortedColPtrA)
    check_status(status)

cpdef sdense2csr(intptr_t handle, int m, int n, size_t descrA, intptr_t A, int lda, intptr_t nnzPerRow, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseSdense2csr(<Handle>handle, m, n, <const MatDescr>descrA, <const float*>A, lda, <const int*>nnzPerRow, <float*>csrSortedValA, <int*>csrSortedRowPtrA, <int*>csrSortedColIndA)
    check_status(status)

cpdef ddense2csr(intptr_t handle, int m, int n, size_t descrA, intptr_t A, int lda, intptr_t nnzPerRow, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseDdense2csr(<Handle>handle, m, n, <const MatDescr>descrA, <const double*>A, lda, <const int*>nnzPerRow, <double*>csrSortedValA, <int*>csrSortedRowPtrA, <int*>csrSortedColIndA)
    check_status(status)

cpdef cdense2csr(intptr_t handle, int m, int n, size_t descrA, intptr_t A, int lda, intptr_t nnzPerRow, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseCdense2csr(<Handle>handle, m, n, <const MatDescr>descrA, <const cuComplex*>A, lda, <const int*>nnzPerRow, <cuComplex*>csrSortedValA, <int*>csrSortedRowPtrA, <int*>csrSortedColIndA)
    check_status(status)

cpdef zdense2csr(intptr_t handle, int m, int n, size_t descrA, intptr_t A, int lda, intptr_t nnzPerRow, intptr_t csrSortedValA, intptr_t csrSortedRowPtrA, intptr_t csrSortedColIndA):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseZdense2csr(<Handle>handle, m, n, <const MatDescr>descrA, <const cuDoubleComplex*>A, lda, <const int*>nnzPerRow, <cuDoubleComplex*>csrSortedValA, <int*>csrSortedRowPtrA, <int*>csrSortedColIndA)
    check_status(status)

cpdef snnz(intptr_t handle, int dirA, int m, int n, size_t descrA, intptr_t A, int lda, intptr_t nnzPerRowCol, intptr_t nnzTotalDevHostPtr):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseSnnz(<Handle>handle, <Direction>dirA, m, n, <const MatDescr>descrA, <const float*>A, lda, <int*>nnzPerRowCol, <int*>nnzTotalDevHostPtr)
    check_status(status)

cpdef dnnz(intptr_t handle, int dirA, int m, int n, size_t descrA, intptr_t A, int lda, intptr_t nnzPerRowCol, intptr_t nnzTotalDevHostPtr):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseDnnz(<Handle>handle, <Direction>dirA, m, n, <const MatDescr>descrA, <const double*>A, lda, <int*>nnzPerRowCol, <int*>nnzTotalDevHostPtr)
    check_status(status)

cpdef cnnz(intptr_t handle, int dirA, int m, int n, size_t descrA, intptr_t A, int lda, intptr_t nnzPerRowCol, intptr_t nnzTotalDevHostPtr):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseCnnz(<Handle>handle, <Direction>dirA, m, n, <const MatDescr>descrA, <const cuComplex*>A, lda, <int*>nnzPerRowCol, <int*>nnzTotalDevHostPtr)
    check_status(status)

cpdef znnz(intptr_t handle, int dirA, int m, int n, size_t descrA, intptr_t A, int lda, intptr_t nnzPerRowCol, intptr_t nnzTotalDevHostPtr):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseZnnz(<Handle>handle, <Direction>dirA, m, n, <const MatDescr>descrA, <const cuDoubleComplex*>A, lda, <int*>nnzPerRowCol, <int*>nnzTotalDevHostPtr)
    check_status(status)

cpdef createIdentityPermutation(intptr_t handle, int n, intptr_t p):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseCreateIdentityPermutation(<Handle>handle, n, <int*>p)
    check_status(status)

cpdef size_t xcoosort_bufferSizeExt(intptr_t handle, int m, int n, int nnz, intptr_t cooRowsA, intptr_t cooColsA) except? 0:
    cdef size_t bufferSizeInBytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseXcoosort_bufferSizeExt(<Handle>handle, m, n, nnz, <const int*>cooRowsA, <const int*>cooColsA, &bufferSizeInBytes)
    check_status(status)
    return bufferSizeInBytes

cpdef xcoosortByRow(intptr_t handle, int m, int n, int nnz, intptr_t cooRowsA, intptr_t cooColsA, intptr_t P, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseXcoosortByRow(<Handle>handle, m, n, nnz, <int*>cooRowsA, <int*>cooColsA, <int*>P, <void*>pBuffer)
    check_status(status)

cpdef xcoosortByColumn(intptr_t handle, int m, int n, int nnz, intptr_t cooRowsA, intptr_t cooColsA, intptr_t P, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseXcoosortByColumn(<Handle>handle, m, n, nnz, <int*>cooRowsA, <int*>cooColsA, <int*>P, <void*>pBuffer)
    check_status(status)

cpdef size_t xcsrsort_bufferSizeExt(intptr_t handle, int m, int n, int nnz, intptr_t csrRowPtrA, intptr_t csrColIndA) except? 0:
    cdef size_t bufferSizeInBytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseXcsrsort_bufferSizeExt(<Handle>handle, m, n, nnz, <const int*>csrRowPtrA, <const int*>csrColIndA, &bufferSizeInBytes)
    check_status(status)
    return bufferSizeInBytes

cpdef xcsrsort(intptr_t handle, int m, int n, int nnz, size_t descrA, intptr_t csrRowPtrA, intptr_t csrColIndA, intptr_t P, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseXcsrsort(<Handle>handle, m, n, nnz, <const MatDescr>descrA, <const int*>csrRowPtrA, <int*>csrColIndA, <int*>P, <void*>pBuffer)
    check_status(status)

cpdef size_t xcscsort_bufferSizeExt(intptr_t handle, int m, int n, int nnz, intptr_t cscColPtrA, intptr_t cscRowIndA) except? 0:
    cdef size_t bufferSizeInBytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseXcscsort_bufferSizeExt(<Handle>handle, m, n, nnz, <const int*>cscColPtrA, <const int*>cscRowIndA, &bufferSizeInBytes)
    check_status(status)
    return bufferSizeInBytes

cpdef xcscsort(intptr_t handle, int m, int n, int nnz, size_t descrA, intptr_t cscColPtrA, intptr_t cscRowIndA, intptr_t P, intptr_t pBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseXcscsort(<Handle>handle, m, n, nnz, <const MatDescr>descrA, <const int*>cscColPtrA, <int*>cscRowIndA, <int*>P, <void*>pBuffer)
    check_status(status)


###########################################
# cuSPARSE Generic API - Sparse Vector APIs

cpdef size_t createSpVec(int64_t size, int64_t nnz, intptr_t indices, intptr_t values, int idxType, int idxBase, size_t valueType) except? 0:
    cdef SpVecDescr spVecDescr
    status = cusparseCreateSpVec(&spVecDescr, size, nnz, <void*>indices, <void*>values, <IndexType>idxType, <IndexBase>idxBase, <DataType>valueType)
    check_status(status)
    return <size_t>spVecDescr

cpdef destroySpVec(size_t spVecDescr):
    status = cusparseDestroySpVec(<SpVecDescr>spVecDescr)
    check_status(status)

cpdef SpVecAttributes spVecGet(size_t spVecDescr):
    cdef int64_t size
    cdef int64_t nnz
    cdef void* indices
    cdef void* values
    cdef IndexType idxType
    cdef IndexBase idxBase
    cdef DataType valueType
    status = cusparseSpVecGet(<SpVecDescr>spVecDescr, &size, &nnz, &indices, &values, &idxType, &idxBase, &valueType)
    check_status(status)
    return SpVecAttributes(size, nnz, <intptr_t>indices, <intptr_t>values, idxType, idxBase, valueType)

cpdef int spVecGetIndexBase(size_t spVecDescr) except? 0:
    cdef IndexBase idxBase
    status = cusparseSpVecGetIndexBase(<SpVecDescr>spVecDescr, &idxBase)
    check_status(status)
    return <int>idxBase

cpdef intptr_t spVecGetValues(size_t spVecDescr) except? 0:
    cdef void* values
    status = cusparseSpVecGetValues(<SpVecDescr>spVecDescr, &values)
    check_status(status)
    return <intptr_t>values

cpdef spVecSetValues(size_t spVecDescr, intptr_t values):
    status = cusparseSpVecSetValues(<SpVecDescr>spVecDescr, <void*>values)
    check_status(status)


###########################################
# cuSPARSE Generic API - Sparse Matrix APIs

cpdef size_t createCoo(int64_t rows, int64_t cols, int64_t nnz, intptr_t cooRowInd, intptr_t cooColInd, intptr_t cooValues, int cooIdxType, int idxBase, size_t valueType) except? 0:
    cdef SpMatDescr spMatDescr
    status = cusparseCreateCoo(&spMatDescr, rows, cols, nnz, <void*>cooRowInd, <void*>cooColInd, <void*>cooValues, <IndexType>cooIdxType, <IndexBase>idxBase, <DataType>valueType)
    check_status(status)
    return <size_t>spMatDescr

cpdef size_t createCooAoS(int64_t rows, int64_t cols, int64_t nnz, intptr_t cooInd, intptr_t cooValues, int cooIdxType, int idxBase, size_t valueType) except? 0:
    cdef SpMatDescr spMatDescr
    status = cusparseCreateCooAoS(&spMatDescr, rows, cols, nnz, <void*>cooInd, <void*>cooValues, <IndexType>cooIdxType, <IndexBase>idxBase, <DataType>valueType)
    check_status(status)
    return <size_t>spMatDescr

cpdef size_t createCsr(int64_t rows, int64_t cols, int64_t nnz, intptr_t csrRowOffsets, intptr_t csrColInd, intptr_t csrValues, int csrRowOffsetsType, int csrColIndType, int idxBase, size_t valueType) except? 0:
    cdef SpMatDescr spMatDescr
    status = cusparseCreateCsr(&spMatDescr, rows, cols, nnz, <void*>csrRowOffsets, <void*>csrColInd, <void*>csrValues, <IndexType>csrRowOffsetsType, <IndexType>csrColIndType, <IndexBase>idxBase, <DataType>valueType)
    check_status(status)
    return <size_t>spMatDescr

cpdef destroySpMat(size_t spMatDescr):
    status = cusparseDestroySpMat(<SpMatDescr>spMatDescr)
    check_status(status)

cpdef CooAttributes cooGet(size_t spMatDescr):
    cdef int64_t rows
    cdef int64_t cols
    cdef int64_t nnz
    cdef void* cooRowInd
    cdef void* cooColInd
    cdef void* cooValues
    cdef IndexType idxType
    cdef IndexBase idxBase
    cdef DataType valueType
    status = cusparseCooGet(<SpMatDescr>spMatDescr, &rows, &cols, &nnz, &cooRowInd, &cooColInd, &cooValues, &idxType, &idxBase, &valueType)
    check_status(status)
    return CooAttributes(rows, cols, nnz, <intptr_t>cooRowInd, <intptr_t>cooColInd, <intptr_t>cooValues, idxType, idxBase, valueType)

cpdef CooAoSAttributes cooAoSGet(size_t spMatDescr):
    cdef int64_t rows
    cdef int64_t cols
    cdef int64_t nnz
    cdef void* cooInd
    cdef void* cooValues
    cdef IndexType idxType
    cdef IndexBase idxBase
    cdef DataType valueType
    status = cusparseCooAoSGet(<SpMatDescr>spMatDescr, &rows, &cols, &nnz, &cooInd, &cooValues, &idxType, &idxBase, &valueType)
    check_status(status)
    return CooAoSAttributes(rows, cols, nnz, <intptr_t>cooInd, <intptr_t>cooValues, idxType, idxBase, valueType)

cpdef CsrAttributes csrGet(size_t spMatDescr):
    cdef int64_t rows
    cdef int64_t cols
    cdef int64_t nnz
    cdef void* csrRowOffsets
    cdef void* csrColInd
    cdef void* csrValues
    cdef IndexType csrRowOffsetsType
    cdef IndexType csrColIndType
    cdef IndexBase idxBase
    cdef DataType valueType
    status = cusparseCsrGet(<SpMatDescr>spMatDescr, &rows, &cols, &nnz, &csrRowOffsets, &csrColInd, &csrValues, &csrRowOffsetsType, &csrColIndType, &idxBase, &valueType)
    check_status(status)
    return CsrAttributes(rows, cols, nnz, <intptr_t>csrRowOffsets, <intptr_t>csrColInd, <intptr_t>csrValues, csrRowOffsetsType, csrColIndType, idxBase, valueType)

cpdef int spMatGetFormat(size_t spMatDescr) except? 0:
    cdef Format format
    status = cusparseSpMatGetFormat(<SpMatDescr>spMatDescr, &format)
    check_status(status)
    return <int>format

cpdef int spMatGetIndexBase(size_t spMatDescr) except? 0:
    cdef IndexBase idxBase
    status = cusparseSpMatGetIndexBase(<SpMatDescr>spMatDescr, &idxBase)
    check_status(status)
    return <int>idxBase

cpdef intptr_t spMatGetValues(size_t spMatDescr) except? 0:
    cdef void* values
    status = cusparseSpMatGetValues(<SpMatDescr>spMatDescr, &values)
    check_status(status)
    return <intptr_t>values

cpdef spMatSetValues(size_t spMatDescr, intptr_t values):
    status = cusparseSpMatSetValues(<SpMatDescr>spMatDescr, <void*>values)
    check_status(status)

cpdef int spMatGetStridedBatch(size_t spMatDescr) except? 0:
    cdef int batchCount
    status = cusparseSpMatGetStridedBatch(<SpMatDescr>spMatDescr, &batchCount)
    check_status(status)
    return batchCount

cpdef spMatSetStridedBatch(size_t spMatDescr, int batchCount):
    status = cusparseSpMatSetStridedBatch(<SpMatDescr>spMatDescr, batchCount)
    check_status(status)


##########################################
# cuSPARSE Generic API - Dense Vector APIs

cpdef size_t createDnVec(int64_t size, intptr_t values, size_t valueType) except? 0:
    cdef DnVecDescr dnVecDescr
    status = cusparseCreateDnVec(&dnVecDescr, size, <void*>values, <DataType>valueType)
    check_status(status)
    return <size_t>dnVecDescr

cpdef destroyDnVec(size_t dnVecDescr):
    status = cusparseDestroyDnVec(<DnVecDescr>dnVecDescr)
    check_status(status)

cpdef DnVecAttributes dnVecGet(size_t dnVecDescr):
    cdef int64_t size
    cdef void* values
    cdef DataType valueType
    status = cusparseDnVecGet(<DnVecDescr>dnVecDescr, &size, &values, &valueType)
    check_status(status)
    return DnVecAttributes(size, <intptr_t>values, valueType)

cpdef intptr_t dnVecGetValues(size_t dnVecDescr) except? 0:
    cdef void* values
    status = cusparseDnVecGetValues(<DnVecDescr>dnVecDescr, &values)
    check_status(status)
    return <intptr_t>values

cpdef dnVecSetValues(size_t dnVecDescr, intptr_t values):
    status = cusparseDnVecSetValues(<DnVecDescr>dnVecDescr, <void*>values)
    check_status(status)


##########################################
# cuSPARSE Generic API - Dense Matrix APIs

cpdef size_t createDnMat(int64_t rows, int64_t cols, int64_t ld, intptr_t values, size_t valueType, int order) except? 0:
    cdef DnMatDescr dnMatDescr
    status = cusparseCreateDnMat(&dnMatDescr, rows, cols, ld, <void*>values, <DataType>valueType, <Order>order)
    check_status(status)
    return <size_t>dnMatDescr

cpdef destroyDnMat(size_t dnMatDescr):
    status = cusparseDestroyDnMat(<DnMatDescr>dnMatDescr)
    check_status(status)

cpdef DnMatAttributes dnMatGet(size_t dnMatDescr):
    cdef int64_t rows
    cdef int64_t cols
    cdef int64_t ld
    cdef void* values
    cdef DataType type
    cdef Order order
    status = cusparseDnMatGet(<DnMatDescr>dnMatDescr, &rows, &cols, &ld, &values, &type, &order)
    check_status(status)
    return DnMatAttributes(rows, cols, ld, <intptr_t>values, type, order)

cpdef intptr_t dnMatGetValues(size_t dnMatDescr) except? 0:
    cdef void* values
    status = cusparseDnMatGetValues(<DnMatDescr>dnMatDescr, &values)
    check_status(status)
    return <intptr_t>values

cpdef dnMatSetValues(size_t dnMatDescr, intptr_t values):
    status = cusparseDnMatSetValues(<DnMatDescr>dnMatDescr, <void*>values)
    check_status(status)

cpdef DnMatBatchAttributes dnMatGetStridedBatch(size_t dnMatDescr):
    cdef int batchCount
    cdef int64_t batchStride
    status = cusparseDnMatGetStridedBatch(<DnMatDescr>dnMatDescr, &batchCount, &batchStride)
    check_status(status)
    return DnMatBatchAttributes(batchCount, batchStride)

cpdef dnMatSetStridedBatch(size_t dnMatDescr, int batchCount, int64_t batchStride):
    status = cusparseDnMatSetStridedBatch(<DnMatDescr>dnMatDescr, batchCount, batchStride)
    check_status(status)


##############################################
# cuSPARSE Generic API - Generic API Functions

cpdef size_t spVV_bufferSize(intptr_t handle, int opX, size_t vecX, size_t vecY, intptr_t result, size_t computeType) except? 0:
    cdef size_t bufferSize
    status = cusparseSpVV_bufferSize(<Handle>handle, <Operation>opX, <SpVecDescr>vecX, <DnVecDescr>vecY, <const void*>result, <DataType>computeType, &bufferSize)
    check_status(status)
    return bufferSize

cpdef spVV(intptr_t handle, int opX, size_t vecX, size_t vecY, intptr_t result, size_t computeType, intptr_t externalBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseSpVV(<Handle>handle, <Operation>opX, <SpVecDescr>vecX, <DnVecDescr>vecY, <void*>result, <DataType>computeType, <void*>externalBuffer)
    check_status(status)

cpdef size_t spMV_bufferSize(intptr_t handle, int opA, intptr_t alpha, size_t matA, size_t vecX, intptr_t beta, size_t vecY, size_t computeType, int alg) except? 0:
    cdef size_t bufferSize
    status = cusparseSpMV_bufferSize(<Handle>handle, <Operation>opA, <const void*>alpha, <SpMatDescr>matA, <DnVecDescr>vecX, <const void*>beta, <DnVecDescr>vecY, <DataType>computeType, <SpMVAlg>alg, &bufferSize)
    check_status(status)
    return bufferSize

cpdef spMV(intptr_t handle, int opA, intptr_t alpha, size_t matA, size_t vecX, intptr_t beta, size_t vecY, size_t computeType, int alg, intptr_t externalBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseSpMV(<Handle>handle, <Operation>opA, <const void*>alpha, <SpMatDescr>matA, <DnVecDescr>vecX, <const void*>beta, <DnVecDescr>vecY, <DataType>computeType, <SpMVAlg>alg, <void*>externalBuffer)
    check_status(status)

cpdef size_t spMM_bufferSize(intptr_t handle, int opA, int opB, intptr_t alpha, size_t matA, size_t matB, intptr_t beta, size_t matC, size_t computeType, int alg) except? 0:
    cdef size_t bufferSize
    status = cusparseSpMM_bufferSize(<Handle>handle, <Operation>opA, <Operation>opB, <const void*>alpha, <SpMatDescr>matA, <DnMatDescr>matB, <const void*>beta, <DnMatDescr>matC, <DataType>computeType, <SpMMAlg>alg, &bufferSize)
    check_status(status)
    return bufferSize

cpdef spMM(intptr_t handle, int opA, int opB, intptr_t alpha, size_t matA, size_t matB, intptr_t beta, size_t matC, size_t computeType, int alg, intptr_t externalBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseSpMM(<Handle>handle, <Operation>opA, <Operation>opB, <const void*>alpha, <SpMatDescr>matA, <DnMatDescr>matB, <const void*>beta, <DnMatDescr>matC, <DataType>computeType, <SpMMAlg>alg, <void*>externalBuffer)
    check_status(status)

cpdef size_t constrainedGeMM_bufferSize(intptr_t handle, int opA, int opB, intptr_t alpha, size_t matA, size_t matB, intptr_t beta, size_t matC, size_t computeType) except? 0:
    cdef size_t bufferSize
    status = cusparseConstrainedGeMM_bufferSize(<Handle>handle, <Operation>opA, <Operation>opB, <const void*>alpha, <DnMatDescr>matA, <DnMatDescr>matB, <const void*>beta, <SpMatDescr>matC, <DataType>computeType, &bufferSize)
    check_status(status)
    return bufferSize

cpdef constrainedGeMM(intptr_t handle, int opA, int opB, intptr_t alpha, size_t matA, size_t matB, intptr_t beta, size_t matC, size_t computeType, intptr_t externalBuffer):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusparseConstrainedGeMM(<Handle>handle, <Operation>opA, <Operation>opB, <const void*>alpha, <DnMatDescr>matA, <DnMatDescr>matB, <const void*>beta, <SpMatDescr>matC, <DataType>computeType, <void*>externalBuffer)
    check_status(status)
