// This file is a stub header file of cusparse for Read the Docs.

#ifndef INCLUDE_GUARD_STUB_CUPY_CUSPARSE_H
#include <hipsparse.h>

extern "C" {

typedef hipsparseIndexBase_t cusparseIndexBase_t;
typedef hipsparseStatus_t cusparseStatus_t;

typedef hipsparseHandle_t cusparseHandle_t;
typedef hipsparseMatDescr_t cusparseMatDescr_t;
typedef void* csrsv2Info_t;
typedef void* csrsm2Info_t;
typedef void* csric02Info_t;
typedef void* bsric02Info_t;
typedef void* csrilu02Info_t;
typedef void* bsrilu02Info_t;
typedef void* csrgemm2Info_t;

typedef hipsparseMatrixType_t cusparseMatrixType_t;
typedef hipsparseFillMode_t cusparseFillMode_t;
typedef hipsparseDiagType_t cusparseDiagType_t;
typedef enum {} cusparseOperation_t;
typedef hipsparsePointerMode_t cusparsePointerMode_t;
typedef hipsparseAction_t cusparseAction_t;
typedef hipsparseDirection_t cusparseDirection_t;
typedef enum {} cusparseAlgMode_t;
typedef enum {} cusparseSolvePolicy_t;

// Version
cusparseStatus_t cusparseGetVersion(cusparseHandle_t handle,
                                    int*             version) {
  return hipsparseGetVersion(handle, version);
}

// cuSPARSE Helper Function
cusparseStatus_t cusparseCreate(cusparseHandle_t* handle) {
  return hipsparseCreate(handle);
}

cusparseStatus_t cusparseCreateMatDescr(cusparseMatDescr_t* descrA) {
  return hipsparseCreateMatDescr(descrA);
}

cusparseStatus_t cusparseDestroy(cusparseHandle_t handle) {
  return hipsparseDestroy(handle);
}

cusparseStatus_t cusparseDestroyMatDescr(cusparseMatDescr_t descrA) {
  return hipsparseDestroyMatDescr(descrA);
}

cusparseStatus_t cusparseSetMatIndexBase(cusparseMatDescr_t  descrA,
                                         cusparseIndexBase_t base) {
  return hipsparseSetMatIndexBase(descrA, base);
}

cusparseStatus_t cusparseSetMatType(cusparseMatDescr_t   descrA,
                                    cusparseMatrixType_t type) {
  return hipsparseSetMatType(descrA, type);
}

cusparseStatus_t cusparseSetMatFillMode(cusparseMatDescr_t descrA,
                                        cusparseFillMode_t fillMode) {
  return hipsparseSetMatFillMode(descrA, fillMode);
}

cusparseStatus_t cusparseSetMatDiagType(cusparseMatDescr_t descrA,
                                        cusparseDiagType_t diagType) {
  return hipsparseSetMatDiagType(descrA, diagType);
}

cusparseStatus_t cusparseSetPointerMode(cusparseHandle_t      handle,
                                        cusparsePointerMode_t mode) {
  return hipsparseSetPointerMode(handle, mode);
}

// Stream
cusparseStatus_t cusparseSetStream(cusparseHandle_t handle,
                                   cudaStream_t     streamId) {
  return hipsparseSetStream(handle, streamId);
}

cusparseStatus_t cusparseGetStream(cusparseHandle_t handle,
                                   cudaStream_t*    streamId) {
  return hipsparseGetStream(handle, streamId);
}

// cuSPARSE Level1 Function
cusparseStatus_t cusparseSgthr(cusparseHandle_t    handle,
                               int                 nnz,
                               const float*        y,
                               float*              xVal,
                               const int*          xInd,
                               cusparseIndexBase_t idxBase) {
  return hipsparseSgthr(handle, nnz, y, xVal, xInd, idxBase);
}

cusparseStatus_t cusparseDgthr(cusparseHandle_t    handle,
                               int                 nnz,
                               const double*       y,
                               double*             xVal,
                               const int*          xInd,
                               cusparseIndexBase_t idxBase) {
  return hipsparseDgthr(handle, nnz, y, xVal, xInd, idxBase);
}

cusparseStatus_t cusparseCgthr(cusparseHandle_t    handle,
                               int                 nnz,
                               const cuComplex*    y,
                               cuComplex*          xVal,
                               const int*          xInd,
                               cusparseIndexBase_t idxBase) {
  return hipsparseCgthr(handle, nnz, y, xVal, xInd, idxBase);
}

cusparseStatus_t cusparseZgthr(cusparseHandle_t       handle,
                               int                    nnz,
                               const cuDoubleComplex* y,
                               cuDoubleComplex*       xVal,
                               const int*             xInd,
                               cusparseIndexBase_t    idxBase) {
  return hipsparseZgthr(handle, nnz, y, xVal, xInd, idxBase);
}

// cuSPARSE Level2 Function
cusparseStatus_t cusparseScsrmv(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDcsrmv(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCcsrmv(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseZcsrmv(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCsrmvEx_bufferSize(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCsrmvEx(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCreateCsrsv2Info(csrsv2Info_t* info) {
  return hipsparseCreateCsrsv2Info(info);
}

cusparseStatus_t cusparseDestroyCsrsv2Info(csrsv2Info_t info) {
  return hipsparseDestroyCsrsv2Info(info);
}

cusparseStatus_t cusparseScsrsv2_bufferSize(cusparseHandle_t         handle,
                                            cusparseOperation_t      transA,
                                            int                      m,
                                            int                      nnz,
                                            const cusparseMatDescr_t descrA,
                                            float*                   csrSortedValA,
                                            const int*               csrSortedRowPtrA,
                                            const int*               csrSortedColIndA,
                                            csrsv2Info_t             info,
                                            int*                     pBufferSizeInBytes) {
  return hipsparseScsrsv2_bufferSize(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, pBufferSizeInBytes);
}

cusparseStatus_t cusparseDcsrsv2_bufferSize(cusparseHandle_t         handle,
                                            cusparseOperation_t      transA,
                                            int                      m,
                                            int                      nnz,
                                            const cusparseMatDescr_t descrA,
                                            double*                  csrSortedValA,
                                            const int*               csrSortedRowPtrA,
                                            const int*               csrSortedColIndA,
                                            csrsv2Info_t             info,
                                            int*                     pBufferSizeInBytes) {
  return hipsparseDcsrsv2_bufferSize(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, pBufferSizeInBytes);
}

cusparseStatus_t cusparseCcsrsv2_bufferSize(cusparseHandle_t         handle,
                                            cusparseOperation_t      transA,
                                            int                      m,
                                            int                      nnz,
                                            const cusparseMatDescr_t descrA,
                                            cuComplex*               csrSortedValA,
                                            const int*               csrSortedRowPtrA,
                                            const int*               csrSortedColIndA,
                                            csrsv2Info_t             info,
                                            int*                     pBufferSizeInBytes) {
  return hipsparseCcsrsv2_bufferSize(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, pBufferSizeInBytes);
}

cusparseStatus_t cusparseZcsrsv2_bufferSize(cusparseHandle_t         handle,
                                            cusparseOperation_t      transA,
                                            int                      m,
                                            int                      nnz,
                                            const cusparseMatDescr_t descrA,
                                            cuDoubleComplex*         csrSortedValA,
                                            const int*               csrSortedRowPtrA,
                                            const int*               csrSortedColIndA,
                                            csrsv2Info_t             info,
                                            int*                     pBufferSizeInBytes) {
  return hipsparseZcsrsv2_bufferSize(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, pBufferSizeInBytes);
}

cusparseStatus_t cusparseScsrsv2_analysis(cusparseHandle_t         handle,
                                          cusparseOperation_t      transA,
                                          int                      m,
                                          int                      nnz,
                                          const cusparseMatDescr_t descrA,
                                          const float*             csrSortedValA,
                                          const int*               csrSortedRowPtrA,
                                          const int*               csrSortedColIndA,
                                          csrsv2Info_t             info,
                                          cusparseSolvePolicy_t    policy,
                                          void*                    pBuffer) {
  return hipsparseScsrsv2_analysis(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer);
}

cusparseStatus_t cusparseDcsrsv2_analysis(cusparseHandle_t         handle,
                                          cusparseOperation_t      transA,
                                          int                      m,
                                          int                      nnz,
                                          const cusparseMatDescr_t descrA,
                                          const double*            csrSortedValA,
                                          const int*               csrSortedRowPtrA,
                                          const int*               csrSortedColIndA,
                                          csrsv2Info_t             info,
                                          cusparseSolvePolicy_t    policy,
                                          void*                    pBuffer) {
  return hipsparseDcsrsv2_analysis(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer);
}

cusparseStatus_t cusparseCcsrsv2_analysis(cusparseHandle_t         handle,
                                          cusparseOperation_t      transA,
                                          int                      m,
                                          int                      nnz,
                                          const cusparseMatDescr_t descrA,
                                          const cuComplex*         csrSortedValA,
                                          const int*               csrSortedRowPtrA,
                                          const int*               csrSortedColIndA,
                                          csrsv2Info_t             info,
                                          cusparseSolvePolicy_t    policy,
                                          void*                    pBuffer) {
  return hipsparseCcsrsv2_analysis(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer);
}

cusparseStatus_t cusparseZcsrsv2_analysis(cusparseHandle_t         handle,
                                          cusparseOperation_t      transA,
                                          int                      m,
                                          int                      nnz,
                                          const cusparseMatDescr_t descrA,
                                          const cuDoubleComplex*   csrSortedValA,
                                          const int*               csrSortedRowPtrA,
                                          const int*               csrSortedColIndA,
                                          csrsv2Info_t             info,
                                          cusparseSolvePolicy_t    policy,
                                          void*                    pBuffer) {
  return hipsparseZcsrsv2_analysis(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer);
}

cusparseStatus_t cusparseScsrsv2_solve(cusparseHandle_t         handle,
                                       cusparseOperation_t      transA,
                                       int                      m,
                                       int                      nnz,
                                       const float*             alpha,
                                       const cusparseMatDescr_t descrA,
                                       const float*             csrSortedValA,
                                       const int*               csrSortedRowPtrA,
                                       const int*               csrSortedColIndA,
                                       csrsv2Info_t             info,
                                       const float*             f,
                                       float*                   x,
                                       cusparseSolvePolicy_t    policy,
                                       void*                    pBuffer) {
  return hipsparseScsrsv2_solve(handle, transA, m, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, f, x, policy, pBuffer);
}

cusparseStatus_t cusparseDcsrsv2_solve(cusparseHandle_t         handle,
                                       cusparseOperation_t      transA,
                                       int                      m,
                                       int                      nnz,
                                       const double*            alpha,
                                       const cusparseMatDescr_t descrA,
                                       const double*            csrSortedValA,
                                       const int*               csrSortedRowPtrA,
                                       const int*               csrSortedColIndA,
                                       csrsv2Info_t             info,
                                       const double*            f,
                                       double*                  x,
                                       cusparseSolvePolicy_t    policy,
                                       void*                    pBuffer) {
  return hipsparseDcsrsv2_solve(handle, transA, m, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, f, x, policy, pBuffer);
}

cusparseStatus_t cusparseCcsrsv2_solve(cusparseHandle_t         handle,
                                       cusparseOperation_t      transA,
                                       int                      m,
                                       int                      nnz,
                                       const cuComplex*         alpha,
                                       const cusparseMatDescr_t descrA,
                                       const cuComplex*         csrSortedValA,
                                       const int*               csrSortedRowPtrA,
                                       const int*               csrSortedColIndA,
                                       csrsv2Info_t             info,
                                       const cuComplex*         f,
                                       cuComplex*               x,
                                       cusparseSolvePolicy_t    policy,
                                       void*                    pBuffer) {
  return hipsparseCcsrsv2_solve(handle, transA, m, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, f, x, policy, pBuffer);
}

cusparseStatus_t cusparseZcsrsv2_solve(cusparseHandle_t         handle,
                                       cusparseOperation_t      transA,
                                       int                      m,
                                       int                      nnz,
                                       const cuDoubleComplex*   alpha,
                                       const cusparseMatDescr_t descrA,
                                       const cuDoubleComplex*   csrSortedValA,
                                       const int*               csrSortedRowPtrA,
                                       const int*               csrSortedColIndA,
                                       csrsv2Info_t             info,
                                       const cuDoubleComplex*   f,
                                       cuDoubleComplex*         x,
                                       cusparseSolvePolicy_t    policy,
                                       void*                    pBuffer) {
  return hipsparseZcsrsv2_solve(handle, transA, m, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, f, x, policy, pBuffer);
}

cusparseStatus_t cusparseXcsrsv2_zeroPivot(cusparseHandle_t handle,
                                           csrsv2Info_t     info,
                                           int*             position) {
  return hipsparseXcsrsv2_zeroPivot(handle, info, position);
}

// cuSPARSE Level3 Function
cusparseStatus_t cusparseScsrmm(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDcsrmm(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCcsrmm(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseZcsrmm(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseScsrmm2(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDcsrmm2(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCcsrmm2(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseZcsrmm2(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCreateCsrsm2Info(csrsm2Info_t* info) {
  return hipsparseCreateCsrsm2Info(info);
}
cusparseStatus_t cusparseDestroyCsrsm2Info(csrsm2Info_t info) {
  return hipsparseDestroyCsrsm2Info(info);
}

cusparseStatus_t cusparseScsrsm2_bufferSizeExt(cusparseHandle_t         handle,
                                               int                      algo,
                                               cusparseOperation_t      transA,
                                               cusparseOperation_t      transB,
                                               int                      m,
                                               int                      nrhs,
                                               int                      nnz,
                                               const float*             alpha,
                                               const cusparseMatDescr_t descrA,
                                               const float*             csrSortedValA,
                                               const int*               csrSortedRowPtrA,
                                               const int*               csrSortedColIndA,
                                               const float*             B,
                                               int                      ldb,
                                               csrsm2Info_t             info,
                                               cusparseSolvePolicy_t    policy,
                                               size_t*                  pBufferSize) {
  return hipsparseScsrsm2_bufferSizeExt(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBufferSize);
}
cusparseStatus_t cusparseDcsrsm2_bufferSizeExt(cusparseHandle_t         handle,
                                               int                      algo,
                                               cusparseOperation_t      transA,
                                               cusparseOperation_t      transB,
                                               int                      m,
                                               int                      nrhs,
                                               int                      nnz,
                                               const double*            alpha,
                                               const cusparseMatDescr_t descrA,
                                               const double*            csrSortedValA,
                                               const int*               csrSortedRowPtrA,
                                               const int*               csrSortedColIndA,
                                               const double*            B,
                                               int                      ldb,
                                               csrsm2Info_t             info,
                                               cusparseSolvePolicy_t    policy,
                                               size_t*                  pBufferSize) {
  return hipsparseDcsrsm2_bufferSizeExt(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBufferSize);
}
cusparseStatus_t cusparseCcsrsm2_bufferSizeExt(cusparseHandle_t         handle,
                                               int                      algo,
                                               cusparseOperation_t      transA,
                                               cusparseOperation_t      transB,
                                               int                      m,
                                               int                      nrhs,
                                               int                      nnz,
                                               const cuComplex*         alpha,
                                               const cusparseMatDescr_t descrA,
                                               const cuComplex*         csrSortedValA,
                                               const int*               csrSortedRowPtrA,
                                               const int*               csrSortedColIndA,
                                               const cuComplex*         B,
                                               int                      ldb,
                                               csrsm2Info_t             info,
                                               cusparseSolvePolicy_t    policy,
                                               size_t*                  pBufferSize) {
  return hipsparseCcsrsm2_bufferSizeExt(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBufferSize);
}
cusparseStatus_t cusparseZcsrsm2_bufferSizeExt(cusparseHandle_t         handle,
                                               int                      algo,
                                               cusparseOperation_t      transA,
                                               cusparseOperation_t      transB,
                                               int                      m,
                                               int                      nrhs,
                                               int                      nnz,
                                               const cuDoubleComplex*   alpha,
                                               const cusparseMatDescr_t descrA,
                                               const cuDoubleComplex*   csrSortedValA,
                                               const int*               csrSortedRowPtrA,
                                               const int*               csrSortedColIndA,
                                               const cuDoubleComplex*   B,
                                               int                      ldb,
                                               csrsm2Info_t             info,
                                               cusparseSolvePolicy_t    policy,
                                               size_t*                  pBufferSize) {
  return hipsparseZcsrsm2_bufferSizeExt(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBufferSize);
}

cusparseStatus_t cusparseScsrsm2_analysis(cusparseHandle_t         handle,
                                          int                      algo,
                                          cusparseOperation_t      transA,
                                          cusparseOperation_t      transB,
                                          int                      m,
                                          int                      nrhs,
                                          int                      nnz,
                                          const float*             alpha,
                                          const cusparseMatDescr_t descrA,
                                          const float*             csrSortedValA,
                                          const int*               csrSortedRowPtrA,
                                          const int*               csrSortedColIndA,
                                          const float*             B,
                                          int                      ldb,
                                          csrsm2Info_t             info,
                                          cusparseSolvePolicy_t    policy,
                                          void*                    pBuffer) {
  return hipsparseScsrsm2_analysis(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBuffer);
}
cusparseStatus_t cusparseDcsrsm2_analysis(cusparseHandle_t         handle,
                                          int                      algo,
                                          cusparseOperation_t      transA,
                                          cusparseOperation_t      transB,
                                          int                      m,
                                          int                      nrhs,
                                          int                      nnz,
                                          const double*            alpha,
                                          const cusparseMatDescr_t descrA,
                                          const double*            csrSortedValA,
                                          const int*               csrSortedRowPtrA,
                                          const int*               csrSortedColIndA,
                                          const double*            B,
                                          int                      ldb,
                                          csrsm2Info_t             info,
                                          cusparseSolvePolicy_t    policy,
                                          void*                    pBuffer) {
  return hipsparseDcsrsm2_analysis(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBuffer);
}
cusparseStatus_t cusparseCcsrsm2_analysis(cusparseHandle_t         handle,
                                          int                      algo,
                                          cusparseOperation_t      transA,
                                          cusparseOperation_t      transB,
                                          int                      m,
                                          int                      nrhs,
                                          int                      nnz,
                                          const cuComplex*         alpha,
                                          const cusparseMatDescr_t descrA,
                                          const cuComplex*         csrSortedValA,
                                          const int*               csrSortedRowPtrA,
                                          const int*               csrSortedColIndA,
                                          const cuComplex*         B,
                                          int                      ldb,
                                          csrsm2Info_t             info,
                                          cusparseSolvePolicy_t    policy,
                                          void*                    pBuffer) {
  return hipsparseCcsrsm2_analysis(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBuffer);
}
cusparseStatus_t cusparseZcsrsm2_analysis(cusparseHandle_t         handle,
                                          int                      algo,
                                          cusparseOperation_t      transA,
                                          cusparseOperation_t      transB,
                                          int                      m,
                                          int                      nrhs,
                                          int                      nnz,
                                          const cuDoubleComplex*   alpha,
                                          const cusparseMatDescr_t descrA,
                                          const cuDoubleComplex*   csrSortedValA,
                                          const int*               csrSortedRowPtrA,
                                          const int*               csrSortedColIndA,
                                          const cuDoubleComplex*   B,
                                          int                      ldb,
                                          csrsm2Info_t             info,
                                          cusparseSolvePolicy_t    policy,
                                          void*                    pBuffer) {
  return hipsparseZcsrsm2_analysis(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBuffer);
}

cusparseStatus_t cusparseScsrsm2_solve(cusparseHandle_t         handle,
                                       int                      algo,
                                       cusparseOperation_t      transA,
                                       cusparseOperation_t      transB,
                                       int                      m,
                                       int                      nrhs,
                                       int                      nnz,
                                       const float*             alpha,
                                       const cusparseMatDescr_t descrA,
                                       const float*             csrSortedValA,
                                       const int*               csrSortedRowPtrA,
                                       const int*               csrSortedColIndA,
                                       float*                   B,
                                       int                      ldb,
                                       csrsm2Info_t             info,
                                       cusparseSolvePolicy_t    policy,
                                       void*                    pBuffer) {
  return hipsparseScsrsm2_solve(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBuffer);
}
cusparseStatus_t cusparseDcsrsm2_solve(cusparseHandle_t         handle,
                                       int                      algo,
                                       cusparseOperation_t      transA,
                                       cusparseOperation_t      transB,
                                       int                      m,
                                       int                      nrhs,
                                       int                      nnz,
                                       const double*            alpha,
                                       const cusparseMatDescr_t descrA,
                                       const double*            csrSortedValA,
                                       const int*               csrSortedRowPtrA,
                                       const int*               csrSortedColIndA,
                                       double*                  B,
                                       int                      ldb,
                                       csrsm2Info_t             info,
                                       cusparseSolvePolicy_t    policy,
                                       void*                    pBuffer) {
  return hipsparseDcsrsm2_solve(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBuffer);
}
cusparseStatus_t cusparseCcsrsm2_solve(cusparseHandle_t         handle,
                                       int                      algo,
                                       cusparseOperation_t      transA,
                                       cusparseOperation_t      transB,
                                       int                      m,
                                       int                      nrhs,
                                       int                      nnz,
                                       const cuComplex*         alpha,
                                       const cusparseMatDescr_t descrA,
                                       const cuComplex*         csrSortedValA,
                                       const int*               csrSortedRowPtrA,
                                       const int*               csrSortedColIndA,
                                       cuComplex*               B,
                                       int                      ldb,
                                       csrsm2Info_t             info,
                                       cusparseSolvePolicy_t    policy,
                                       void*                    pBuffer) {
  return hipsparseCcsrsm2_solve(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBuffer);
}
cusparseStatus_t cusparseZcsrsm2_solve(cusparseHandle_t         handle,
                                       int                      algo,
                                       cusparseOperation_t      transA,
                                       cusparseOperation_t      transB,
                                       int                      m,
                                       int                      nrhs,
                                       int                      nnz,
                                       const cuDoubleComplex*   alpha,
                                       const cusparseMatDescr_t descrA,
                                       const cuDoubleComplex*   csrSortedValA,
                                       const int*               csrSortedRowPtrA,
                                       const int*               csrSortedColIndA,
                                       cuDoubleComplex*         B,
                                       int                      ldb,
                                       csrsm2Info_t             info,
                                       cusparseSolvePolicy_t    policy,
                                       void*                    pBuffer) {
  return hipsparseZcsrsm2_solve(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBuffer);
}

cusparseStatus_t cusparseXcsrsm2_zeroPivot(cusparseHandle_t handle,
                                           csrsm2Info_t     info,
                                           int* position) {
  return hipsparseXcsrsm2_zeroPivot(handle, info, position);
}

// cuSPARSE Extra Function
cusparseStatus_t cusparseXcsrgeamNnz(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseScsrgeam2_bufferSizeExt(cusparseHandle_t         handle,
                                                 int                      m,
                                                 int                      n,
                                                 const float*             alpha,
                                                 const cusparseMatDescr_t descrA,
                                                 int                      nnzA,
                                                 const float*             csrSortedValA,
                                                 const int*               csrSortedRowPtrA,
                                                 const int*               csrSortedColIndA,
                                                 const float*             beta,
                                                 const cusparseMatDescr_t descrB,
                                                 int                      nnzB,
                                                 const float*             csrSortedValB,
                                                 const int*               csrSortedRowPtrB,
                                                 const int*               csrSortedColIndB,
                                                 const cusparseMatDescr_t descrC,
                                                 const float*             csrSortedValC,
                                                 const int*               csrSortedRowPtrC,
                                                 const int*               csrSortedColIndC,
                                                 size_t*                  pBufferSizeInBytes) {
  return hipsparseScsrgeam(handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBufferSizeInBytes);
}

cusparseStatus_t cusparseDcsrgeam2_bufferSizeExt(cusparseHandle_t         handle,
                                                 int                      m,
                                                 int                      n,
                                                 const double*            alpha,
                                                 const cusparseMatDescr_t descrA,
                                                 int                      nnzA,
                                                 const double*            csrSortedValA,
                                                 const int*               csrSortedRowPtrA,
                                                 const int*               csrSortedColIndA,
                                                 const double*            beta,
                                                 const cusparseMatDescr_t descrB,
                                                 int                      nnzB,
                                                 const double*            csrSortedValB,
                                                 const int*               csrSortedRowPtrB,
                                                 const int*               csrSortedColIndB,
                                                 const cusparseMatDescr_t descrC,
                                                 const double*            csrSortedValC,
                                                 const int*               csrSortedRowPtrC,
                                                 const int*               csrSortedColIndC,
                                                 size_t*                  pBufferSizeInBytes) {
  return hipsparseDcsrgeam(handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBufferSizeInBytes);
}

cusparseStatus_t cusparseCcsrgeam2_bufferSizeExt(cusparseHandle_t         handle,
                                                 int                      m,
                                                 int                      n,
                                                 const cuComplex*         alpha,
                                                 const cusparseMatDescr_t descrA,
                                                 int                      nnzA,
                                                 const cuComplex*         csrSortedValA,
                                                 const int*               csrSortedRowPtrA,
                                                 const int*               csrSortedColIndA,
                                                 const cuComplex*         beta,
                                                 const cusparseMatDescr_t descrB,
                                                 int                      nnzB,
                                                 const cuComplex*         csrSortedValB,
                                                 const int*               csrSortedRowPtrB,
                                                 const int*               csrSortedColIndB,
                                                 const cusparseMatDescr_t descrC,
                                                 const cuComplex*         csrSortedValC,
                                                 const int*               csrSortedRowPtrC,
                                                 const int*               csrSortedColIndC,
                                                 size_t*                  pBufferSizeInBytes) {
  return hipsparseCcsrgeam(handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBufferSizeInBytes);
}

cusparseStatus_t cusparseZcsrgeam2_bufferSizeExt(cusparseHandle_t         handle,
                                                 int                      m,
                                                 int                      n,
                                                 const cuDoubleComplex*   alpha,
                                                 const cusparseMatDescr_t descrA,
                                                 int                      nnzA,
                                                 const cuDoubleComplex*   csrSortedValA,
                                                 const int*               csrSortedRowPtrA,
                                                 const int*               csrSortedColIndA,
                                                 const cuDoubleComplex*   beta,
                                                 const cusparseMatDescr_t descrB,
                                                 int                      nnzB,
                                                 const cuDoubleComplex*   csrSortedValB,
                                                 const int*               csrSortedRowPtrB,
                                                 const int*               csrSortedColIndB,
                                                 const cusparseMatDescr_t descrC,
                                                 const cuDoubleComplex*   csrSortedValC,
                                                 const int*               csrSortedRowPtrC,
                                                 const int*               csrSortedColIndC,
                                                 size_t*                  pBufferSizeInBytes) {
  return hipsparseZcsrgeam(handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBufferSizeInBytes);
}

cusparseStatus_t cusparseScsrgeam2_bufferSizeExt(cusparseHandle_t         handle,
                                                 int                      m,
                                                 int                      n,
                                                 const float*             alpha,
                                                 const cusparseMatDescr_t descrA,
                                                 int                      nnzA,
                                                 const float*             csrSortedValA,
                                                 const int*               csrSortedRowPtrA,
                                                 const int*               csrSortedColIndA,
                                                 const float*             beta,
                                                 const cusparseMatDescr_t descrB,
                                                 int                      nnzB,
                                                 const float*             csrSortedValB,
                                                 const int*               csrSortedRowPtrB,
                                                 const int*               csrSortedColIndB,
                                                 const cusparseMatDescr_t descrC,
                                                 const float*             csrSortedValC,
                                                 const int*               csrSortedRowPtrC,
                                                 const int*               csrSortedColIndC,
                                                 size_t*                  pBufferSizeInBytes) {
  return hipsparseScsrgeam2_bufferSizeExt(handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBufferSizeInBytes);
}

cusparseStatus_t cusparseDcsrgeam2_bufferSizeExt(cusparseHandle_t         handle,
                                                 int                      m,
                                                 int                      n,
                                                 const double*            alpha,
                                                 const cusparseMatDescr_t descrA,
                                                 int                      nnzA,
                                                 const double*            csrSortedValA,
                                                 const int*               csrSortedRowPtrA,
                                                 const int*               csrSortedColIndA,
                                                 const double*            beta,
                                                 const cusparseMatDescr_t descrB,
                                                 int                      nnzB,
                                                 const double*            csrSortedValB,
                                                 const int*               csrSortedRowPtrB,
                                                 const int*               csrSortedColIndB,
                                                 const cusparseMatDescr_t descrC,
                                                 const double*            csrSortedValC,
                                                 const int*               csrSortedRowPtrC,
                                                 const int*               csrSortedColIndC,
                                                 size_t*                  pBufferSizeInBytes) {
  return hipsparseDcsrgeam2_bufferSizeExt(handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBufferSizeInBytes);
}

cusparseStatus_t cusparseCcsrgeam2_bufferSizeExt(cusparseHandle_t         handle,
                                                 int                      m,
                                                 int                      n,
                                                 const cuComplex*         alpha,
                                                 const cusparseMatDescr_t descrA,
                                                 int                      nnzA,
                                                 const cuComplex*         csrSortedValA,
                                                 const int*               csrSortedRowPtrA,
                                                 const int*               csrSortedColIndA,
                                                 const cuComplex*         beta,
                                                 const cusparseMatDescr_t descrB,
                                                 int                      nnzB,
                                                 const cuComplex*         csrSortedValB,
                                                 const int*               csrSortedRowPtrB,
                                                 const int*               csrSortedColIndB,
                                                 const cusparseMatDescr_t descrC,
                                                 const cuComplex*         csrSortedValC,
                                                 const int*               csrSortedRowPtrC,
                                                 const int*               csrSortedColIndC,
                                                 size_t*                  pBufferSizeInBytes) {
  return hipsparseCcsrgeam2_bufferSizeExt(handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBufferSizeInBytes);
}

cusparseStatus_t cusparseZcsrgeam2_bufferSizeExt(cusparseHandle_t         handle,
                                                 int                      m,
                                                 int                      n,
                                                 const cuDoubleComplex*   alpha,
                                                 const cusparseMatDescr_t descrA,
                                                 int                      nnzA,
                                                 const cuDoubleComplex*   csrSortedValA,
                                                 const int*               csrSortedRowPtrA,
                                                 const int*               csrSortedColIndA,
                                                 const cuDoubleComplex*   beta,
                                                 const cusparseMatDescr_t descrB,
                                                 int                      nnzB,
                                                 const cuDoubleComplex*   csrSortedValB,
                                                 const int*               csrSortedRowPtrB,
                                                 const int*               csrSortedColIndB,
                                                 const cusparseMatDescr_t descrC,
                                                 const cuDoubleComplex*   csrSortedValC,
                                                 const int*               csrSortedRowPtrC,
                                                 const int*               csrSortedColIndC,
                                                 size_t*                  pBufferSizeInBytes) {
  return hipsparseZcsrgeam2_bufferSizeExt(handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBufferSizeInBytes);
}

cusparseStatus_t cusparseXcsrgeam2Nnz(cusparseHandle_t         handle,
                                      int                      m,
                                      int                      n,
                                      const cusparseMatDescr_t descrA,
                                      int                      nnzA,
                                      const int*               csrSortedRowPtrA,
                                      const int*               csrSortedColIndA,
                                      const cusparseMatDescr_t descrB,
                                      int                      nnzB,
                                      const int*               csrSortedRowPtrB,
                                      const int*               csrSortedColIndB,
                                      const cusparseMatDescr_t descrC,
                                      int*                     csrSortedRowPtrC,
                                      int*                     nnzTotalDevHostPtr,
                                      void*                    workspace) {
  return hipsparseXcsrgeam2Nnz(handle, m, n, descrA, nnzA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedRowPtrC, nnzTotalDevHostPtr, workspace);
}

cusparseStatus_t cusparseScsrgeam2_bufferSizeExt(cusparseHandle_t         handle,
                                                 int                      m,
                                                 int                      n,
                                                 const float*             alpha,
                                                 const cusparseMatDescr_t descrA,
                                                 int                      nnzA,
                                                 const float*             csrSortedValA,
                                                 const int*               csrSortedRowPtrA,
                                                 const int*               csrSortedColIndA,
                                                 const float*             beta,
                                                 const cusparseMatDescr_t descrB,
                                                 int                      nnzB,
                                                 const float*             csrSortedValB,
                                                 const int*               csrSortedRowPtrB,
                                                 const int*               csrSortedColIndB,
                                                 const cusparseMatDescr_t descrC,
                                                 const float*             csrSortedValC,
                                                 const int*               csrSortedRowPtrC,
                                                 const int*               csrSortedColIndC,
                                                 size_t*                  pBufferSizeInBytes) {
  return hipsparseScsrgeam2(handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBufferSizeInBytes);
}

cusparseStatus_t cusparseDcsrgeam2_bufferSizeExt(cusparseHandle_t         handle,
                                                 int                      m,
                                                 int                      n,
                                                 const double*            alpha,
                                                 const cusparseMatDescr_t descrA,
                                                 int                      nnzA,
                                                 const double*            csrSortedValA,
                                                 const int*               csrSortedRowPtrA,
                                                 const int*               csrSortedColIndA,
                                                 const double*            beta,
                                                 const cusparseMatDescr_t descrB,
                                                 int                      nnzB,
                                                 const double*            csrSortedValB,
                                                 const int*               csrSortedRowPtrB,
                                                 const int*               csrSortedColIndB,
                                                 const cusparseMatDescr_t descrC,
                                                 const double*            csrSortedValC,
                                                 const int*               csrSortedRowPtrC,
                                                 const int*               csrSortedColIndC,
                                                 size_t*                  pBufferSizeInBytes) {
  return hipsparseDcsrgeam2(handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBufferSizeInBytes);
}

cusparseStatus_t cusparseCcsrgeam2_bufferSizeExt(cusparseHandle_t         handle,
                                                 int                      m,
                                                 int                      n,
                                                 const cuComplex*         alpha,
                                                 const cusparseMatDescr_t descrA,
                                                 int                      nnzA,
                                                 const cuComplex*         csrSortedValA,
                                                 const int*               csrSortedRowPtrA,
                                                 const int*               csrSortedColIndA,
                                                 const cuComplex*         beta,
                                                 const cusparseMatDescr_t descrB,
                                                 int                      nnzB,
                                                 const cuComplex*         csrSortedValB,
                                                 const int*               csrSortedRowPtrB,
                                                 const int*               csrSortedColIndB,
                                                 const cusparseMatDescr_t descrC,
                                                 const cuComplex*         csrSortedValC,
                                                 const int*               csrSortedRowPtrC,
                                                 const int*               csrSortedColIndC,
                                                 size_t*                  pBufferSizeInBytes) {
  return hipsparseCcsrgeam2(handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBufferSizeInBytes);
}

cusparseStatus_t cusparseZcsrgeam2_bufferSizeExt(cusparseHandle_t         handle,
                                                 int                      m,
                                                 int                      n,
                                                 const cuDoubleComplex*   alpha,
                                                 const cusparseMatDescr_t descrA,
                                                 int                      nnzA,
                                                 const cuDoubleComplex*   csrSortedValA,
                                                 const int*               csrSortedRowPtrA,
                                                 const int*               csrSortedColIndA,
                                                 const cuDoubleComplex*   beta,
                                                 const cusparseMatDescr_t descrB,
                                                 int                      nnzB,
                                                 const cuDoubleComplex*   csrSortedValB,
                                                 const int*               csrSortedRowPtrB,
                                                 const int*               csrSortedColIndB,
                                                 const cusparseMatDescr_t descrC,
                                                 const cuDoubleComplex*   csrSortedValC,
                                                 const int*               csrSortedRowPtrC,
                                                 const int*               csrSortedColIndC,
                                                 size_t*                  pBufferSizeInBytes) {
  return hipsparseZcsrgeam2(handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBufferSizeInBytes);
}

cusparseStatus_t cusparseXcsrgemmNnz(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseScsrgemm2_bufferSizeExt(cusparseHandle_t         handle,
                                                 int                      m,
                                                 int                      n,
                                                 int                      k,
                                                 const float*             alpha,
                                                 const cusparseMatDescr_t descrA,
                                                 int                      nnzA,
                                                 const int*               csrSortedRowPtrA,
                                                 const int*               csrSortedColIndA,
                                                 const cusparseMatDescr_t descrB,
                                                 int                      nnzB,
                                                 const int*               csrSortedRowPtrB,
                                                 const int*               csrSortedColIndB,
                                                 const float*             beta,
                                                 const cusparseMatDescr_t descrD,
                                                 int                      nnzD,
                                                 const int*               csrSortedRowPtrD,
                                                 const int*               csrSortedColIndD,
                                                 csrgemm2Info_t           info,
                                                 size_t*                  pBufferSizeInBytes) {
  return hipsparseScsrgemm(handle, m, n, k, alpha, descrA, nnzA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedRowPtrB, csrSortedColIndB, beta, descrD, nnzD, csrSortedRowPtrD, csrSortedColIndD, info, pBufferSizeInBytes);
}

cusparseStatus_t cusparseDcsrgemm2_bufferSizeExt(cusparseHandle_t         handle,
                                                 int                      m,
                                                 int                      n,
                                                 int                      k,
                                                 const double*            alpha,
                                                 const cusparseMatDescr_t descrA,
                                                 int                      nnzA,
                                                 const int*               csrSortedRowPtrA,
                                                 const int*               csrSortedColIndA,
                                                 const cusparseMatDescr_t descrB,
                                                 int                      nnzB,
                                                 const int*               csrSortedRowPtrB,
                                                 const int*               csrSortedColIndB,
                                                 const double*            beta,
                                                 const cusparseMatDescr_t descrD,
                                                 int                      nnzD,
                                                 const int*               csrSortedRowPtrD,
                                                 const int*               csrSortedColIndD,
                                                 csrgemm2Info_t           info,
                                                 size_t*                  pBufferSizeInBytes) {
  return hipsparseDcsrgemm(handle, m, n, k, alpha, descrA, nnzA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedRowPtrB, csrSortedColIndB, beta, descrD, nnzD, csrSortedRowPtrD, csrSortedColIndD, info, pBufferSizeInBytes);
}

cusparseStatus_t cusparseCcsrgemm2_bufferSizeExt(cusparseHandle_t         handle,
                                                 int                      m,
                                                 int                      n,
                                                 int                      k,
                                                 const cuComplex*         alpha,
                                                 const cusparseMatDescr_t descrA,
                                                 int                      nnzA,
                                                 const int*               csrSortedRowPtrA,
                                                 const int*               csrSortedColIndA,
                                                 const cusparseMatDescr_t descrB,
                                                 int                      nnzB,
                                                 const int*               csrSortedRowPtrB,
                                                 const int*               csrSortedColIndB,
                                                 const cuComplex*         beta,
                                                 const cusparseMatDescr_t descrD,
                                                 int                      nnzD,
                                                 const int*               csrSortedRowPtrD,
                                                 const int*               csrSortedColIndD,
                                                 csrgemm2Info_t           info,
                                                 size_t*                  pBufferSizeInBytes) {
  return hipsparseCcsrgemm(handle, m, n, k, alpha, descrA, nnzA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedRowPtrB, csrSortedColIndB, beta, descrD, nnzD, csrSortedRowPtrD, csrSortedColIndD, info, pBufferSizeInBytes);
}

cusparseStatus_t cusparseZcsrgemm2_bufferSizeExt(cusparseHandle_t         handle,
                                                 int                      m,
                                                 int                      n,
                                                 int                      k,
                                                 const cuDoubleComplex*   alpha,
                                                 const cusparseMatDescr_t descrA,
                                                 int                      nnzA,
                                                 const int*               csrSortedRowPtrA,
                                                 const int*               csrSortedColIndA,
                                                 const cusparseMatDescr_t descrB,
                                                 int                      nnzB,
                                                 const int*               csrSortedRowPtrB,
                                                 const int*               csrSortedColIndB,
                                                 const cuDoubleComplex*   beta,
                                                 const cusparseMatDescr_t descrD,
                                                 int                      nnzD,
                                                 const int*               csrSortedRowPtrD,
                                                 const int*               csrSortedColIndD,
                                                 csrgemm2Info_t           info,
                                                 size_t*                  pBufferSizeInBytes) {
  return hipsparseZcsrgemm(handle, m, n, k, alpha, descrA, nnzA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedRowPtrB, csrSortedColIndB, beta, descrD, nnzD, csrSortedRowPtrD, csrSortedColIndD, info, pBufferSizeInBytes);
}

cusparseStatus_t cusparseCreateCsrgemm2Info(csrgemm2Info_t* info) {
  return hipsparseCreateCsrgemm2Info(info);
}

cusparseStatus_t cusparseDestroyCsrgemm2Info(csrgemm2Info_t info) {
  return hipsparseDestroyCsrgemm2Info(info);
}

cusparseStatus_t cusparseScsrgemm2_bufferSizeExt(cusparseHandle_t         handle,
                                                 int                      m,
                                                 int                      n,
                                                 int                      k,
                                                 const float*             alpha,
                                                 const cusparseMatDescr_t descrA,
                                                 int                      nnzA,
                                                 const int*               csrSortedRowPtrA,
                                                 const int*               csrSortedColIndA,
                                                 const cusparseMatDescr_t descrB,
                                                 int                      nnzB,
                                                 const int*               csrSortedRowPtrB,
                                                 const int*               csrSortedColIndB,
                                                 const float*             beta,
                                                 const cusparseMatDescr_t descrD,
                                                 int                      nnzD,
                                                 const int*               csrSortedRowPtrD,
                                                 const int*               csrSortedColIndD,
                                                 csrgemm2Info_t           info,
                                                 size_t*                  pBufferSizeInBytes) {
  return hipsparseScsrgemm2_bufferSizeExt(handle, m, n, k, alpha, descrA, nnzA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedRowPtrB, csrSortedColIndB, beta, descrD, nnzD, csrSortedRowPtrD, csrSortedColIndD, info, pBufferSizeInBytes);
}

cusparseStatus_t cusparseDcsrgemm2_bufferSizeExt(cusparseHandle_t         handle,
                                                 int                      m,
                                                 int                      n,
                                                 int                      k,
                                                 const double*            alpha,
                                                 const cusparseMatDescr_t descrA,
                                                 int                      nnzA,
                                                 const int*               csrSortedRowPtrA,
                                                 const int*               csrSortedColIndA,
                                                 const cusparseMatDescr_t descrB,
                                                 int                      nnzB,
                                                 const int*               csrSortedRowPtrB,
                                                 const int*               csrSortedColIndB,
                                                 const double*            beta,
                                                 const cusparseMatDescr_t descrD,
                                                 int                      nnzD,
                                                 const int*               csrSortedRowPtrD,
                                                 const int*               csrSortedColIndD,
                                                 csrgemm2Info_t           info,
                                                 size_t*                  pBufferSizeInBytes) {
  return hipsparseDcsrgemm2_bufferSizeExt(handle, m, n, k, alpha, descrA, nnzA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedRowPtrB, csrSortedColIndB, beta, descrD, nnzD, csrSortedRowPtrD, csrSortedColIndD, info, pBufferSizeInBytes);
}

cusparseStatus_t cusparseCcsrgemm2_bufferSizeExt(cusparseHandle_t         handle,
                                                 int                      m,
                                                 int                      n,
                                                 int                      k,
                                                 const cuComplex*         alpha,
                                                 const cusparseMatDescr_t descrA,
                                                 int                      nnzA,
                                                 const int*               csrSortedRowPtrA,
                                                 const int*               csrSortedColIndA,
                                                 const cusparseMatDescr_t descrB,
                                                 int                      nnzB,
                                                 const int*               csrSortedRowPtrB,
                                                 const int*               csrSortedColIndB,
                                                 const cuComplex*         beta,
                                                 const cusparseMatDescr_t descrD,
                                                 int                      nnzD,
                                                 const int*               csrSortedRowPtrD,
                                                 const int*               csrSortedColIndD,
                                                 csrgemm2Info_t           info,
                                                 size_t*                  pBufferSizeInBytes) {
  return hipsparseCcsrgemm2_bufferSizeExt(handle, m, n, k, alpha, descrA, nnzA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedRowPtrB, csrSortedColIndB, beta, descrD, nnzD, csrSortedRowPtrD, csrSortedColIndD, info, pBufferSizeInBytes);
}

cusparseStatus_t cusparseZcsrgemm2_bufferSizeExt(cusparseHandle_t         handle,
                                                 int                      m,
                                                 int                      n,
                                                 int                      k,
                                                 const cuDoubleComplex*   alpha,
                                                 const cusparseMatDescr_t descrA,
                                                 int                      nnzA,
                                                 const int*               csrSortedRowPtrA,
                                                 const int*               csrSortedColIndA,
                                                 const cusparseMatDescr_t descrB,
                                                 int                      nnzB,
                                                 const int*               csrSortedRowPtrB,
                                                 const int*               csrSortedColIndB,
                                                 const cuDoubleComplex*   beta,
                                                 const cusparseMatDescr_t descrD,
                                                 int                      nnzD,
                                                 const int*               csrSortedRowPtrD,
                                                 const int*               csrSortedColIndD,
                                                 csrgemm2Info_t           info,
                                                 size_t*                  pBufferSizeInBytes) {
  return hipsparseZcsrgemm2_bufferSizeExt(handle, m, n, k, alpha, descrA, nnzA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedRowPtrB, csrSortedColIndB, beta, descrD, nnzD, csrSortedRowPtrD, csrSortedColIndD, info, pBufferSizeInBytes);
}

cusparseStatus_t cusparseXcsrgemm2Nnz(cusparseHandle_t         handle,
                                      int                      m,
                                      int                      n,
                                      int                      k,
                                      const cusparseMatDescr_t descrA,
                                      int                      nnzA,
                                      const int*               csrSortedRowPtrA,
                                      const int*               csrSortedColIndA,
                                      const cusparseMatDescr_t descrB,
                                      int                      nnzB,
                                      const int*               csrSortedRowPtrB,
                                      const int*               csrSortedColIndB,
                                      const cusparseMatDescr_t descrD,
                                      int                      nnzD,
                                      const int*               csrSortedRowPtrD,
                                      const int*               csrSortedColIndD,
                                      const cusparseMatDescr_t descrC,
                                      int*                     csrSortedRowPtrC,
                                      int*                     nnzTotalDevHostPtr,
                                      const csrgemm2Info_t     info,
                                      void*                    pBuffer) {
  return hipsparseXcsrgemm2Nnz(handle, m, n, k, descrA, nnzA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedRowPtrB, csrSortedColIndB, descrD, nnzD, csrSortedRowPtrD, csrSortedColIndD, descrC, csrSortedRowPtrC, nnzTotalDevHostPtr, info, pBuffer);
}

cusparseStatus_t cusparseScsrgemm2_bufferSizeExt(cusparseHandle_t         handle,
                                                 int                      m,
                                                 int                      n,
                                                 int                      k,
                                                 const float*             alpha,
                                                 const cusparseMatDescr_t descrA,
                                                 int                      nnzA,
                                                 const int*               csrSortedRowPtrA,
                                                 const int*               csrSortedColIndA,
                                                 const cusparseMatDescr_t descrB,
                                                 int                      nnzB,
                                                 const int*               csrSortedRowPtrB,
                                                 const int*               csrSortedColIndB,
                                                 const float*             beta,
                                                 const cusparseMatDescr_t descrD,
                                                 int                      nnzD,
                                                 const int*               csrSortedRowPtrD,
                                                 const int*               csrSortedColIndD,
                                                 csrgemm2Info_t           info,
                                                 size_t*                  pBufferSizeInBytes) {
  return hipsparseScsrgemm2(handle, m, n, k, alpha, descrA, nnzA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedRowPtrB, csrSortedColIndB, beta, descrD, nnzD, csrSortedRowPtrD, csrSortedColIndD, info, pBufferSizeInBytes);
}

cusparseStatus_t cusparseDcsrgemm2_bufferSizeExt(cusparseHandle_t         handle,
                                                 int                      m,
                                                 int                      n,
                                                 int                      k,
                                                 const double*            alpha,
                                                 const cusparseMatDescr_t descrA,
                                                 int                      nnzA,
                                                 const int*               csrSortedRowPtrA,
                                                 const int*               csrSortedColIndA,
                                                 const cusparseMatDescr_t descrB,
                                                 int                      nnzB,
                                                 const int*               csrSortedRowPtrB,
                                                 const int*               csrSortedColIndB,
                                                 const double*            beta,
                                                 const cusparseMatDescr_t descrD,
                                                 int                      nnzD,
                                                 const int*               csrSortedRowPtrD,
                                                 const int*               csrSortedColIndD,
                                                 csrgemm2Info_t           info,
                                                 size_t*                  pBufferSizeInBytes) {
  return hipsparseDcsrgemm2(handle, m, n, k, alpha, descrA, nnzA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedRowPtrB, csrSortedColIndB, beta, descrD, nnzD, csrSortedRowPtrD, csrSortedColIndD, info, pBufferSizeInBytes);
}

cusparseStatus_t cusparseCcsrgemm2_bufferSizeExt(cusparseHandle_t         handle,
                                                 int                      m,
                                                 int                      n,
                                                 int                      k,
                                                 const cuComplex*         alpha,
                                                 const cusparseMatDescr_t descrA,
                                                 int                      nnzA,
                                                 const int*               csrSortedRowPtrA,
                                                 const int*               csrSortedColIndA,
                                                 const cusparseMatDescr_t descrB,
                                                 int                      nnzB,
                                                 const int*               csrSortedRowPtrB,
                                                 const int*               csrSortedColIndB,
                                                 const cuComplex*         beta,
                                                 const cusparseMatDescr_t descrD,
                                                 int                      nnzD,
                                                 const int*               csrSortedRowPtrD,
                                                 const int*               csrSortedColIndD,
                                                 csrgemm2Info_t           info,
                                                 size_t*                  pBufferSizeInBytes) {
  return hipsparseCcsrgemm2(handle, m, n, k, alpha, descrA, nnzA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedRowPtrB, csrSortedColIndB, beta, descrD, nnzD, csrSortedRowPtrD, csrSortedColIndD, info, pBufferSizeInBytes);
}

cusparseStatus_t cusparseZcsrgemm2_bufferSizeExt(cusparseHandle_t         handle,
                                                 int                      m,
                                                 int                      n,
                                                 int                      k,
                                                 const cuDoubleComplex*   alpha,
                                                 const cusparseMatDescr_t descrA,
                                                 int                      nnzA,
                                                 const int*               csrSortedRowPtrA,
                                                 const int*               csrSortedColIndA,
                                                 const cusparseMatDescr_t descrB,
                                                 int                      nnzB,
                                                 const int*               csrSortedRowPtrB,
                                                 const int*               csrSortedColIndB,
                                                 const cuDoubleComplex*   beta,
                                                 const cusparseMatDescr_t descrD,
                                                 int                      nnzD,
                                                 const int*               csrSortedRowPtrD,
                                                 const int*               csrSortedColIndD,
                                                 csrgemm2Info_t           info,
                                                 size_t*                  pBufferSizeInBytes) {
  return hipsparseZcsrgemm2(handle, m, n, k, alpha, descrA, nnzA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedRowPtrB, csrSortedColIndB, beta, descrD, nnzD, csrSortedRowPtrD, csrSortedColIndD, info, pBufferSizeInBytes);
}

// cuSPARSE Format Convrsion
cusparseStatus_t cusparseXcoo2csr(cusparseHandle_t    handle,
                                  const int*          cooRowInd,
                                  int                 nnz,
                                  int                 m,
                                  int*                csrSortedRowPtr,
                                  cusparseIndexBase_t idxBase) {
  return hipsparseXcoo2csr(handle, cooRowInd, nnz, m, csrSortedRowPtr, idxBase);
}

cusparseStatus_t cusparseScsc2dense(cusparseHandle_t         handle,
                                    int                      m,
                                    int                      n,
                                    const cusparseMatDescr_t descrA,
                                    const float*             cscSortedValA,
                                    const int*               cscSortedRowIndA,
                                    const int*               cscSortedColPtrA,
                                    float*                   A,
                                    int                      lda) {
  return hipsparseScsc2dense(handle, m, n, descrA, cscSortedValA, cscSortedRowIndA, cscSortedColPtrA, A, lda);
}

cusparseStatus_t cusparseDcsc2dense(cusparseHandle_t         handle,
                                    int                      m,
                                    int                      n,
                                    const cusparseMatDescr_t descrA,
                                    const double*            cscSortedValA,
                                    const int*               cscSortedRowIndA,
                                    const int*               cscSortedColPtrA,
                                    double*                  A,
                                    int                      lda) {
  return hipsparseDcsc2dense(handle, m, n, descrA, cscSortedValA, cscSortedRowIndA, cscSortedColPtrA, A, lda);
}

cusparseStatus_t cusparseCcsc2dense(cusparseHandle_t         handle,
                                    int                      m,
                                    int                      n,
                                    const cusparseMatDescr_t descrA,
                                    const cuComplex*         cscSortedValA,
                                    const int*               cscSortedRowIndA,
                                    const int*               cscSortedColPtrA,
                                    cuComplex*               A,
                                    int                      lda) {
  return hipsparseCcsc2dense(handle, m, n, descrA, cscSortedValA, cscSortedRowIndA, cscSortedColPtrA, A, lda);
}

cusparseStatus_t cusparseZcsc2dense(cusparseHandle_t         handle,
                                    int                      m,
                                    int                      n,
                                    const cusparseMatDescr_t descrA,
                                    const cuDoubleComplex*   cscSortedValA,
                                    const int*               cscSortedRowIndA,
                                    const int*               cscSortedColPtrA,
                                    cuDoubleComplex*         A,
                                    int                      lda) {
  return hipsparseZcsc2dense(handle, m, n, descrA, cscSortedValA, cscSortedRowIndA, cscSortedColPtrA, A, lda);
}

cusparseStatus_t cusparseXcsr2coo(cusparseHandle_t    handle,
                                  const int*          csrSortedRowPtr,
                                  int                 nnz,
                                  int                 m,
                                  int*                cooRowInd,
                                  cusparseIndexBase_t idxBase) {
  return hipsparseXcsr2coo(handle, csrSortedRowPtr, nnz, m, cooRowInd, idxBase);
}

cusparseStatus_t cusparseScsr2csc(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDcsr2csc(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCcsr2csc(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseZcsr2csc(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}


cusparseStatus_t cusparseScsr2dense(cusparseHandle_t         handle,
                                    int                      m,
                                    int                      n,
                                    const cusparseMatDescr_t descrA,
                                    const float*             csrSortedValA,
                                    const int*               csrSortedRowPtrA,
                                    const int*               csrSortedColIndA,
                                    float*                   A,
                                    int                      lda) {
  return hipsparseScsr2dense(handle, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, A, lda);
}

cusparseStatus_t cusparseDcsr2dense(cusparseHandle_t         handle,
                                    int                      m,
                                    int                      n,
                                    const cusparseMatDescr_t descrA,
                                    const double*            csrSortedValA,
                                    const int*               csrSortedRowPtrA,
                                    const int*               csrSortedColIndA,
                                    double*                  A,
                                    int                      lda) {
  return hipsparseDcsr2dense(handle, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, A, lda);
}

cusparseStatus_t cusparseCcsr2dense(cusparseHandle_t         handle,
                                    int                      m,
                                    int                      n,
                                    const cusparseMatDescr_t descrA,
                                    const cuComplex*         csrSortedValA,
                                    const int*               csrSortedRowPtrA,
                                    const int*               csrSortedColIndA,
                                    cuComplex*               A,
                                    int                      lda) {
  return hipsparseCcsr2dense(handle, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, A, lda);
}

cusparseStatus_t cusparseZcsr2dense(cusparseHandle_t         handle,
                                 int                      m,
                                 int                      n,
                                 const cusparseMatDescr_t descrA,
                                 const cuDoubleComplex*   csrSortedValA,
                                 const int*               csrSortedRowPtrA,
                                 const int*               csrSortedColIndA,
                                 cuDoubleComplex*         A,
                                 int                      lda) {
  return hipsparseZcsr2dense(handle, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, A, lda);
}

cusparseStatus_t cusparseSdense2csc(cusparseHandle_t         handle,
                                    int                      m,
                                    int                      n,
                                    const cusparseMatDescr_t descrA,
                                    const float*             A,
                                    int                      lda,
                                    const int*               nnzPerCol,
                                    float*                   cscSortedValA,
                                    int*                     cscSortedRowIndA,
                                    int*                     cscSortedColPtrA) {
  return hipsparseSdense2csc(handle, m, n, descrA, A, lda, nnzPerCol, cscSortedValA, cscSortedRowIndA, cscSortedColPtrA);
}

cusparseStatus_t cusparseDdense2csc(cusparseHandle_t         handle,
                                    int                      m,
                                    int                      n,
                                    const cusparseMatDescr_t descrA,
                                    const double*            A,
                                    int                      lda,
                                    const int*               nnzPerCol,
                                    double*                  cscSortedValA,
                                    int*                     cscSortedRowIndA,
                                    int*                     cscSortedColPtrA) {
  return hipsparseDdense2csc(handle, m, n, descrA, A, lda, nnzPerCol, cscSortedValA, cscSortedRowIndA, cscSortedColPtrA);
}

cusparseStatus_t cusparseCdense2csc(cusparseHandle_t         handle,
                                    int                      m,
                                    int                      n,
                                    const cusparseMatDescr_t descrA,
                                    const cuComplex*         A,
                                    int                      lda,
                                    const int*               nnzPerCol,
                                    cuComplex*               cscSortedValA,
                                    int*                     cscSortedRowIndA,
                                    int*                     cscSortedColPtrA) {
  return hipsparseCdense2csc(handle, m, n, descrA, A, lda, nnzPerCol, cscSortedValA, cscSortedRowIndA, cscSortedColPtrA);
}

cusparseStatus_t cusparseZdense2csc(cusparseHandle_t         handle,
                                    int                      m,
                                    int                      n,
                                    const cusparseMatDescr_t descrA,
                                    const cuDoubleComplex*   A,
                                    int                      lda,
                                    const int*               nnzPerCol,
                                    cuDoubleComplex*         cscSortedValA,
                                    int*                     cscSortedRowIndA,
                                    int*                     cscSortedColPtrA) {
  return hipsparseZdense2csc(handle, m, n, descrA, A, lda, nnzPerCol, cscSortedValA, cscSortedRowIndA, cscSortedColPtrA);
}

cusparseStatus_t cusparseSdense2csr(cusparseHandle_t         handle,
                                    int                      m,
                                    int                      n,
                                    const cusparseMatDescr_t descrA,
                                    const float*             A,
                                    int                      lda,
                                    const int*               nnzPerRow,
                                    float*                   csrSortedValA,
                                    int*                     csrSortedRowPtrA,
                                    int*                     csrSortedColIndA) {
  return hipsparseSdense2csr(handle, m, n, descrA, A, lda, nnzPerRow, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA);
}

cusparseStatus_t cusparseDdense2csr(cusparseHandle_t         handle,
                                    int                      m,
                                    int                      n,
                                    const cusparseMatDescr_t descrA,
                                    const double*            A,
                                    int                      lda,
                                    const int*               nnzPerRow,
                                    double*                  csrSortedValA,
                                    int*                     csrSortedRowPtrA,
                                    int*                     csrSortedColIndA) {
  return hipsparseDdense2csr(handle, m, n, descrA, A, lda, nnzPerRow, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA);
}

cusparseStatus_t cusparseCdense2csr(cusparseHandle_t           handle,
                                      int                      m,
                                      int                      n,
                                      const cusparseMatDescr_t descrA,
                                      const cuComplex*         A,
                                      int                      lda,
                                      const int*               nnzPerRow,
                                      cuComplex*               csrSortedValA,
                                      int*                     csrSortedRowPtrA,
                                      int*                     csrSortedColIndA) {
  return hipsparseCdense2csr(handle, m, n, descrA, A, lda, nnzPerRow, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA);
}

cusparseStatus_t cusparseZdense2csr(cusparseHandle_t         handle,
                                    int                      m,
                                    int                      n,
                                    const cusparseMatDescr_t descrA,
                                    const cuDoubleComplex*   A,
                                    int                      lda,
                                    const int*               nnzPerRow,
                                    cuDoubleComplex*         csrSortedValA,
                                    int*                     csrSortedRowPtrA,
                                    int*                     csrSortedColIndA) {
  return hipsparseZdense2csr(handle, m, n, descrA, A, lda, nnzPerRow, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA);
}

cusparseStatus_t cusparseSnnz(cusparseHandle_t         handle,
                              cusparseDirection_t      dirA,
                              int                      m,
                              int                      n,
                              const cusparseMatDescr_t descrA,
                              const float*             A,
                              int                      lda,
                              int*                     nnzPerRowCol,
                              int*                     nnzTotalDevHostPtr) {
  return hipsparseSnnz(handle, dirA, m, n, descrA, A, lda, nnzPerRowCol, nnzTotalDevHostPtr);
}

cusparseStatus_t cusparseDnnz(cusparseHandle_t         handle,
                              cusparseDirection_t      dirA,
                              int                      m,
                              int                      n,
                              const cusparseMatDescr_t descrA,
                              const double*            A,
                              int                      lda,
                              int*                     nnzPerRowCol,
                              int*                     nnzTotalDevHostPtr) {
  return hipsparseDnnz(handle, dirA, m, n, descrA, A, lda, nnzPerRowCol, nnzTotalDevHostPtr);
}

cusparseStatus_t cusparseCnnz(cusparseHandle_t         handle,
                              cusparseDirection_t      dirA,
                              int                      m,
                              int                      n,
                              const cusparseMatDescr_t descrA,
                              const cuComplex*         A,
                              int                      lda,
                              int*                     nnzPerRowCol,
                              int*                     nnzTotalDevHostPtr) {
  return hipsparseCnnz(handle, dirA, m, n, descrA, A, lda, nnzPerRowCol, nnzTotalDevHostPtr);
}

cusparseStatus_t cusparseZnnz(cusparseHandle_t         handle,
                              cusparseDirection_t      dirA,
                              int                      m,
                              int                      n,
                              const cusparseMatDescr_t descrA,
                              const cuDoubleComplex*   A,
                              int                      lda,
                              int*                     nnzPerRowCol,
                              int*                     nnzTotalDevHostPtr) {
  return hipsparseZnnz(handle, dirA, m, n, descrA, A, lda, nnzPerRowCol, nnzTotalDevHostPtr);
}

cusparseStatus_t cusparseCreateIdentityPermutation(cusparseHandle_t handle,
                                                   int              n,
                                                   int*             p) {
  return hipsparseCreateIdentityPermutation(handle, n, p);
}

cusparseStatus_t cusparseXcoosort_bufferSizeExt(cusparseHandle_t handle,
                                                int              m,
                                                int              n,
                                                int              nnz,
                                                const int*       cooRowsA,
                                                const int*       cooColsA,
                                                size_t*          pBufferSizeInBytes) {
  return hipsparseXcoosort_bufferSizeExt(handle, m, n, nnz, cooRowsA, cooColsA, pBufferSizeInBytes);
}

cusparseStatus_t cusparseXcoosortByRow(cusparseHandle_t handle,
                                       int              m,
                                       int              n,
                                       int              nnz,
                                       int*             cooRowsA,
                                       int*             cooColsA,
                                       int*             P,
                                       void*            pBuffer) {
  return hipsparseXcoosortByRow(handle, m, n, nnz, cooRowsA, cooColsA, P, pBuffer);
}

cusparseStatus_t cusparseXcoosortByColumn(cusparseHandle_t handle,
                                          int              m,
                                          int              n,
                                          int              nnz,
                                          int*             cooRowsA,
                                          int*             cooColsA,
                                          int*             P,
                                          void*            pBuffer) {
  return hipsparseXcoosortByColumn(handle, m, n, nnz, cooRowsA, cooColsA, P, pBuffer);
}

cusparseStatus_t cusparseXcsrsort_bufferSizeExt(cusparseHandle_t handle,
                                                int              m,
                                                int              n,
                                                int              nnz,
                                                const int*       csrRowPtrA,
                                                const int*       csrColIndA,
                                                size_t*          pBufferSizeInBytes) {
  return hipsparseXcsrsort_bufferSizeExt(handle, m, n, nnz, csrRowPtrA, csrColIndA, pBufferSizeInBytes);
}

cusparseStatus_t cusparseXcsrsort_bufferSizeExt(cusparseHandle_t handle,
                                                int              m,
                                                int              n,
                                                int              nnz,
                                                const int*       csrRowPtrA,
                                                const int*       csrColIndA,
                                                size_t*          pBufferSizeInBytes) {
  return hipsparseXcsrsort(handle, m, n, nnz, csrRowPtrA, csrColIndA, pBufferSizeInBytes);
}

cusparseStatus_t cusparseXcscsort_bufferSizeExt(cusparseHandle_t handle,
                                                int              m,
                                                int              n,
                                                int              nnz,
                                                const int*       cscColPtrA,
                                                const int*       cscRowIndA,
                                                size_t*          pBufferSizeInBytes) {
  return hipsparseXcscsort_bufferSizeExt(handle, m, n, nnz, cscColPtrA, cscRowIndA, pBufferSizeInBytes);
}

cusparseStatus_t cusparseXcscsort_bufferSizeExt(cusparseHandle_t handle,
                                                int              m,
                                                int              n,
                                                int              nnz,
                                                const int*       cscColPtrA,
                                                const int*       cscRowIndA,
                                                size_t*          pBufferSizeInBytes) {
  return hipsparseXcscsort(handle, m, n, nnz, cscColPtrA, cscRowIndA, pBufferSizeInBytes);
}

// cuSPARSE PRECONDITIONERS

cusparseStatus_t cusparseCreateCsrilu02Info(csrilu02Info_t* info) {
  return hipsparseCreateCsrilu02Info(info);
}

cusparseStatus_t cusparseDestroyCsrilu02Info(csrilu02Info_t info) {
  return hipsparseDestroyCsrilu02Info(info);
}

cusparseStatus_t cusparseCreateBsrilu02Info(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDestroyBsrilu02Info(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCreateCsric02Info(csric02Info_t* info) {
  return hipsparseCreateCsric02Info(info);
}

cusparseStatus_t cusparseDestroyCsric02Info(csric02Info_t info) {
  return hipsparseDestroyCsric02Info(info);
}

cusparseStatus_t cusparseCreateBsric02Info(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDestroyBsric02Info(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseScsrilu02_numericBoost(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDcsrilu02_numericBoost(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCcsrilu02_numericBoost(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseZcsrilu02_numericBoost(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseXcsrilu02_zeroPivot(cusparseHandle_t handle,
                                             csrilu02Info_t   info,
                                             int*             position) {
  return hipsparseXcsrilu02_zeroPivot(handle, info, position);
}

cusparseStatus_t cusparseScsrilu02_bufferSize(cusparseHandle_t         handle,
                                              int                      m,
                                              int                      nnz,
                                              const cusparseMatDescr_t descrA,
                                              float*                   csrSortedValA,
                                              const int*               csrSortedRowPtrA,
                                              const int*               csrSortedColIndA,
                                              csrilu02Info_t           info,
                                              int*                     pBufferSizeInBytes) {
  return hipsparseScsrilu02_bufferSize(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, pBufferSizeInBytes);
}

cusparseStatus_t cusparseDcsrilu02_bufferSize(cusparseHandle_t         handle,
                                              int                      m,
                                              int                      nnz,
                                              const cusparseMatDescr_t descrA,
                                              double*                  csrSortedValA,
                                              const int*               csrSortedRowPtrA,
                                              const int*               csrSortedColIndA,
                                              csrilu02Info_t           info,
                                              int*                     pBufferSizeInBytes) {
  return hipsparseDcsrilu02_bufferSize(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, pBufferSizeInBytes);
}

cusparseStatus_t cusparseCcsrilu02_bufferSize(cusparseHandle_t         handle,
                                              int                      m,
                                              int                      nnz,
                                              const cusparseMatDescr_t descrA,
                                              cuComplex*               csrSortedValA,
                                              const int*               csrSortedRowPtrA,
                                              const int*               csrSortedColIndA,
                                              csrilu02Info_t           info,
                                              int*                     pBufferSizeInBytes) {
  return hipsparseCcsrilu02_bufferSize(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, pBufferSizeInBytes);
}

cusparseStatus_t cusparseZcsrilu02_bufferSize(cusparseHandle_t         handle,
                                              int                      m,
                                              int                      nnz,
                                              const cusparseMatDescr_t descrA,
                                              cuDoubleComplex*         csrSortedValA,
                                              const int*               csrSortedRowPtrA,
                                              const int*               csrSortedColIndA,
                                              csrilu02Info_t           info,
                                              int*                     pBufferSizeInBytes) {
  return hipsparseZcsrilu02_bufferSize(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, pBufferSizeInBytes);
}

cusparseStatus_t cusparseScsrilu02_analysis(cusparseHandle_t         handle,
                                            int                      m,
                                            int                      nnz,
                                            const cusparseMatDescr_t descrA,
                                            const float*             csrSortedValA,
                                            const int*               csrSortedRowPtrA,
                                            const int*               csrSortedColIndA,
                                            csrilu02Info_t           info,
                                            cusparseSolvePolicy_t    policy,
                                            void*                    pBuffer) {
  return hipsparseScsrilu02_analysis(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer);
}

cusparseStatus_t cusparseDcsrilu02_analysis(cusparseHandle_t         handle,
                                            int                      m,
                                            int                      nnz,
                                            const cusparseMatDescr_t descrA,
                                            const double*            csrSortedValA,
                                            const int*               csrSortedRowPtrA,
                                            const int*               csrSortedColIndA,
                                            csrilu02Info_t           info,
                                            cusparseSolvePolicy_t    policy,
                                            void*                    pBuffer) {
  return hipsparseDcsrilu02_analysis(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer);
}

cusparseStatus_t cusparseCcsrilu02_analysis(cusparseHandle_t         handle,
                                            int                      m,
                                            int                      nnz,
                                            const cusparseMatDescr_t descrA,
                                            const cuComplex*         csrSortedValA,
                                            const int*               csrSortedRowPtrA,
                                            const int*               csrSortedColIndA,
                                            csrilu02Info_t           info,
                                            cusparseSolvePolicy_t    policy,
                                            void*                    pBuffer) {
  return hipsparseCcsrilu02_analysis(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer);
}

cusparseStatus_t cusparseZcsrilu02_analysis(cusparseHandle_t         handle,
                                            int                      m,
                                            int                      nnz,
                                            const cusparseMatDescr_t descrA,
                                            const cuDoubleComplex*   csrSortedValA,
                                            const int*               csrSortedRowPtrA,
                                            const int*               csrSortedColIndA,
                                            csrilu02Info_t           info,
                                            cusparseSolvePolicy_t    policy,
                                            void*                    pBuffer) {
  return hipsparseZcsrilu02_analysis(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer);
}

cusparseStatus_t cusparseScsrilu02_numericBoost(cusparseHandle_t handle,
                                                csrilu02Info_t   info,
                                                int              enable_boost,
                                                double*          tol,
                                                float*           boost_val) {
  return hipsparseScsrilu02(handle, info, enable_boost, tol, boost_val);
}

cusparseStatus_t cusparseDcsrilu02_numericBoost(cusparseHandle_t handle,
                                                csrilu02Info_t   info,
                                                int              enable_boost,
                                                double*          tol,
                                                double*          boost_val) {
  return hipsparseDcsrilu02(handle, info, enable_boost, tol, boost_val);
}

cusparseStatus_t cusparseCcsrilu02_numericBoost(cusparseHandle_t handle,
                                                csrilu02Info_t   info,
                                                int              enable_boost,
                                                double*          tol,
                                                cuComplex*       boost_val) {
  return hipsparseCcsrilu02(handle, info, enable_boost, tol, boost_val);
}

cusparseStatus_t cusparseZcsrilu02_numericBoost(cusparseHandle_t handle,
                                                csrilu02Info_t   info,
                                                int              enable_boost,
                                                double*          tol,
                                                cuDoubleComplex* boost_val) {
  return hipsparseZcsrilu02(handle, info, enable_boost, tol, boost_val);
}

cusparseStatus_t cusparseSbsrilu02_numericBoost(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDbsrilu02_numericBoost(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCbsrilu02_numericBoost(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseZbsrilu02_numericBoost(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseXbsrilu02_zeroPivot(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSbsrilu02_bufferSize(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDbsrilu02_bufferSize(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCbsrilu02_bufferSize(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseZbsrilu02_bufferSize(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSbsrilu02_analysis(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDbsrilu02_analysis(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCbsrilu02_analysis(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseZbsrilu02_analysis(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSbsrilu02(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDbsrilu02(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCbsrilu02(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseZbsrilu02(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseXcsric02_zeroPivot(cusparseHandle_t handle,
                                            csric02Info_t    info,
                                            int*             position) {
  return hipsparseXcsric02_zeroPivot(handle, info, position);
}

cusparseStatus_t cusparseScsric02_bufferSize(cusparseHandle_t         handle,
                                             int                      m,
                                             int                      nnz,
                                             const cusparseMatDescr_t descrA,
                                             float*                   csrSortedValA,
                                             const int*               csrSortedRowPtrA,
                                             const int*               csrSortedColIndA,
                                             csric02Info_t            info,
                                             int*                     pBufferSizeInBytes) {
  return hipsparseScsric02_bufferSize(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, pBufferSizeInBytes);
}

cusparseStatus_t cusparseDcsric02_bufferSize(cusparseHandle_t         handle,
                                             int                      m,
                                             int                      nnz,
                                             const cusparseMatDescr_t descrA,
                                             double*                  csrSortedValA,
                                             const int*               csrSortedRowPtrA,
                                             const int*               csrSortedColIndA,
                                             csric02Info_t            info,
                                             int*                     pBufferSizeInBytes) {
  return hipsparseDcsric02_bufferSize(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, pBufferSizeInBytes);
}

cusparseStatus_t cusparseCcsric02_bufferSize(cusparseHandle_t         handle,
                                             int                      m,
                                             int                      nnz,
                                             const cusparseMatDescr_t descrA,
                                             cuComplex*               csrSortedValA,
                                             const int*               csrSortedRowPtrA,
                                             const int*               csrSortedColIndA,
                                             csric02Info_t            info,
                                             int*                     pBufferSizeInBytes) {
  return hipsparseCcsric02_bufferSize(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, pBufferSizeInBytes);
}

cusparseStatus_t cusparseZcsric02_bufferSize(cusparseHandle_t         handle,
                                             int                      m,
                                             int                      nnz,
                                             const cusparseMatDescr_t descrA,
                                             cuDoubleComplex*         csrSortedValA,
                                             const int*               csrSortedRowPtrA,
                                             const int*               csrSortedColIndA,
                                             csric02Info_t            info,
                                             int*                     pBufferSizeInBytes) {
  return hipsparseZcsric02_bufferSize(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, pBufferSizeInBytes);
}

cusparseStatus_t cusparseScsric02_analysis(cusparseHandle_t         handle,
                                           int                      m,
                                           int                      nnz,
                                           const cusparseMatDescr_t descrA,
                                           const float*             csrSortedValA,
                                           const int*               csrSortedRowPtrA,
                                           const int*               csrSortedColIndA,
                                           csric02Info_t            info,
                                           cusparseSolvePolicy_t    policy,
                                           void*                    pBuffer) {
  return hipsparseScsric02_analysis(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer);
}

cusparseStatus_t cusparseDcsric02_analysis(cusparseHandle_t         handle,
                                           int                      m,
                                           int                      nnz,
                                           const cusparseMatDescr_t descrA,
                                           const double*            csrSortedValA,
                                           const int*               csrSortedRowPtrA,
                                           const int*               csrSortedColIndA,
                                           csric02Info_t            info,
                                           cusparseSolvePolicy_t    policy,
                                           void*                    pBuffer) {
  return hipsparseDcsric02_analysis(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer);
}

cusparseStatus_t cusparseCcsric02_analysis(cusparseHandle_t         handle,
                                           int                      m,
                                           int                      nnz,
                                           const cusparseMatDescr_t descrA,
                                           const cuComplex*         csrSortedValA,
                                           const int*               csrSortedRowPtrA,
                                           const int*               csrSortedColIndA,
                                           csric02Info_t            info,
                                           cusparseSolvePolicy_t    policy,
                                           void*                    pBuffer) {
  return hipsparseCcsric02_analysis(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer);
}

cusparseStatus_t cusparseZcsric02_analysis(cusparseHandle_t         handle,
                                           int                      m,
                                           int                      nnz,
                                           const cusparseMatDescr_t descrA,
                                           const cuDoubleComplex*   csrSortedValA,
                                           const int*               csrSortedRowPtrA,
                                           const int*               csrSortedColIndA,
                                           csric02Info_t            info,
                                           cusparseSolvePolicy_t    policy,
                                           void*                    pBuffer) {
  return hipsparseZcsric02_analysis(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer);
}

cusparseStatus_t cusparseScsric02_bufferSize(cusparseHandle_t         handle,
                                             int                      m,
                                             int                      nnz,
                                             const cusparseMatDescr_t descrA,
                                             float*                   csrSortedValA,
                                             const int*               csrSortedRowPtrA,
                                             const int*               csrSortedColIndA,
                                             csric02Info_t            info,
                                             int*                     pBufferSizeInBytes) {
  return hipsparseScsric02(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, pBufferSizeInBytes);
}

cusparseStatus_t cusparseDcsric02_bufferSize(cusparseHandle_t         handle,
                                             int                      m,
                                             int                      nnz,
                                             const cusparseMatDescr_t descrA,
                                             double*                  csrSortedValA,
                                             const int*               csrSortedRowPtrA,
                                             const int*               csrSortedColIndA,
                                             csric02Info_t            info,
                                             int*                     pBufferSizeInBytes) {
  return hipsparseDcsric02(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, pBufferSizeInBytes);
}

cusparseStatus_t cusparseCcsric02_bufferSize(cusparseHandle_t         handle,
                                             int                      m,
                                             int                      nnz,
                                             const cusparseMatDescr_t descrA,
                                             cuComplex*               csrSortedValA,
                                             const int*               csrSortedRowPtrA,
                                             const int*               csrSortedColIndA,
                                             csric02Info_t            info,
                                             int*                     pBufferSizeInBytes) {
  return hipsparseCcsric02(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, pBufferSizeInBytes);
}

cusparseStatus_t cusparseZcsric02_bufferSize(cusparseHandle_t         handle,
                                             int                      m,
                                             int                      nnz,
                                             const cusparseMatDescr_t descrA,
                                             cuDoubleComplex*         csrSortedValA,
                                             const int*               csrSortedRowPtrA,
                                             const int*               csrSortedColIndA,
                                             csric02Info_t            info,
                                             int*                     pBufferSizeInBytes) {
  return hipsparseZcsric02(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, pBufferSizeInBytes);
}

cusparseStatus_t cusparseXbsric02_zeroPivot(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSbsric02_bufferSize(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDbsric02_bufferSize(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCbsric02_bufferSize(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseZbsric02_bufferSize(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSbsric02_analysis(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDbsric02_analysis(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCbsric02_analysis(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseZbsric02_analysis(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSbsric02(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDbsric02(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCbsric02(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseZbsric02(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSgtsv2_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDgtsv2_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCgtsv2_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseZgtsv2_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSgtsv2(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDgtsv2(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCgtsv2(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseZgtsv2(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSgtsv2_nopivot_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDgtsv2_nopivot_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCgtsv2_nopivot_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseZgtsv2_nopivot_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSgtsv2_nopivot(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDgtsv2_nopivot(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCgtsv2_nopivot(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseZgtsv2_nopivot(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSgtsv2StridedBatch_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDgtsv2StridedBatch_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCgtsv2StridedBatch_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseZgtsv2StridedBatch_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSgtsv2StridedBatch(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDgtsv2StridedBatch(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCgtsv2StridedBatch(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseZgtsv2StridedBatch(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSgtsvInterleavedBatch_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDgtsvInterleavedBatch_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCgtsvInterleavedBatch_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseZgtsvInterleavedBatch_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSgtsvInterleavedBatch(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDgtsvInterleavedBatch(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCgtsvInterleavedBatch(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseZgtsvInterleavedBatch(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSgpsvInterleavedBatch_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDgpsvInterleavedBatch_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCgpsvInterleavedBatch_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseZgpsvInterleavedBatch_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSgpsvInterleavedBatch(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDgpsvInterleavedBatch(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCgpsvInterleavedBatch(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseZgpsvInterleavedBatch(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

#define CUSPARSE_VERSION -1

// cuSPARSE generic API
typedef void* cusparseSpVecDescr_t;
typedef void* cusparseDnVecDescr_t;
typedef void* cusparseSpMatDescr_t;
typedef void* cusparseDnMatDescr_t;

typedef enum {} cusparseIndexType_t;
typedef enum {} cusparseFormat_t;
typedef enum {} cusparseOrder_t;
typedef enum {} cusparseSpMVAlg_t;
typedef enum {} cusparseSpMMAlg_t;
typedef enum {} cusparseSparseToDenseAlg_t;
typedef enum {} cusparseDenseToSparseAlg_t;

cusparseStatus_t cusparseCreateSpVec(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDestroySpVec(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSpVecGet(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSpVecGetIndexBase(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSpVecGetValues(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSpVecSetValues(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCreateCoo(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCreateCooAoS(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCreateCsrsv2Info(csrsv2Info_t* info) {
  return hipsparseCreateCsr(info);
}

cusparseStatus_t cusparseCreateCsc(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDestroySpMat(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCooGet(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCooAoSGet(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCsrGet(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCsrSetPointers(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSpMatGetSize(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSpMatGetFormat(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSpMatGetIndexBase(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSpMatGetValues(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSpMatSetValues(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSpMatGetStridedBatch(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSpMatSetStridedBatch(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCreateDnVec(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDestroyDnVec(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDnVecGet(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDnVecGetValues(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDnVecSetValues(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCreateDnMat(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDestroyDnMat(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDnMatGet(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDnMatGetValues(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDnMatSetValues(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDnMatGetStridedBatch(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDnMatSetStridedBatch(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSpVV_bufferSize(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSpVV(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSpMV_bufferSize(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSpMV(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSpMM_bufferSize(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSpMM(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseConstrainedGeMM_bufferSize(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseConstrainedGeMM(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSparseToDense_bufferSize(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseSparseToDense(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDenseToSparse_bufferSize(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDenseToSparse_analysis(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseDenseToSparse_convert(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

typedef enum {} cusparseCsr2CscAlg_t;

cusparseStatus_t cusparseCsr2cscEx2_bufferSize(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t cusparseCsr2cscEx2(...) {
  return HIPSPARSE_STATUS_INTERNAL_ERROR;
}


///////////////////////////////////////////////////////////////////////////////
// Definitions are for compatibility
///////////////////////////////////////////////////////////////////////////////

cusparseStatus_t cusparseSnnz_compress(cusparseHandle_t         handle,
                                       int                      m,
                                       const cusparseMatDescr_t descr,
                                       const float*             csrSortedValA,
                                       const int*               csrSortedRowPtrA,
                                       int*                     nnzPerRow,
                                       int*                     nnzC,
                                       float                    tol) {
  return hipsparseSnnz_compress(handle, m, descr, csrSortedValA, csrSortedRowPtrA, nnzPerRow, nnzC, tol);
}

cusparseStatus_t cusparseDnnz_compress(cusparseHandle_t         handle,
                                       int                      m,
                                       const cusparseMatDescr_t descr,
                                       const double*            csrSortedValA,
                                       const int*               csrSortedRowPtrA,
                                       int*                     nnzPerRow,
                                       int*                     nnzC,
                                       double                   tol) {
  return hipsparseDnnz_compress(handle, m, descr, csrSortedValA, csrSortedRowPtrA, nnzPerRow, nnzC, tol);
}

cusparseStatus_t cusparseCnnz_compress(cusparseHandle_t         handle,
                                       int                      m,
                                       const cusparseMatDescr_t descr,
                                       const cuComplex*         csrSortedValA,
                                       const int*               csrSortedRowPtrA,
                                       int*                     nnzPerRow,
                                       int*                     nnzC,
                                       cuComplex                tol) {
  return hipsparseCnnz_compress(handle, m, descr, csrSortedValA, csrSortedRowPtrA, nnzPerRow, nnzC, tol);
}

cusparseStatus_t cusparseZnnz_compress(cusparseHandle_t         handle,
                                       int                      m,
                                       const cusparseMatDescr_t descr,
                                       const cuDoubleComplex*   csrSortedValA,
                                       const int*               csrSortedRowPtrA,
                                       int*                     nnzPerRow,
                                       int*                     nnzC,
                                       cuDoubleComplex          tol) {
  return hipsparseZnnz_compress(handle, m, descr, csrSortedValA, csrSortedRowPtrA, nnzPerRow, nnzC, tol);
}

cusparseStatus_t cusparseScsr2csr_compress(cusparseHandle_t         handle,
                                           int                      m,
                                           int                      n,
                                           const cusparseMatDescr_t descrA,
                                           const float*             csrSortedValA,
                                           const int*               csrSortedColIndA,
                                           const int*               csrSortedRowPtrA,
                                           int                      nnzA,
                                           const int*               nnzPerRow,
                                           float*                   csrSortedValC,
                                           int*                     csrSortedColIndC,
                                           int*                     csrSortedRowPtrC,
                                           float                    tol) {
  return hipsparseScsr2csr_compress(handle, m, n, descrA, csrSortedValA, csrSortedColIndA, csrSortedRowPtrA, nnzA, nnzPerRow, csrSortedValC, csrSortedColIndC, csrSortedRowPtrC, tol);
}

cusparseStatus_t cusparseDcsr2csr_compress(cusparseHandle_t         handle,
                                           int                      m,
                                           int                      n,
                                           const cusparseMatDescr_t descrA,
                                           const double*            csrSortedValA,
                                           const int*               csrSortedColIndA,
                                           const int*               csrSortedRowPtrA,
                                           int                      nnzA,
                                           const int*               nnzPerRow,
                                           double*                  csrSortedValC,
                                           int*                     csrSortedColIndC,
                                           int*                     csrSortedRowPtrC,
                                           double                   tol) {
  return hipsparseDcsr2csr_compress(handle, m, n, descrA, csrSortedValA, csrSortedColIndA, csrSortedRowPtrA, nnzA, nnzPerRow, csrSortedValC, csrSortedColIndC, csrSortedRowPtrC, tol);
}

cusparseStatus_t cusparseCcsr2csr_compress(cusparseHandle_t         handle,
                                           int                      m,
                                           int                      n,
                                           const cusparseMatDescr_t descrA,
                                           const cuComplex*         csrSortedValA,
                                           const int*               csrSortedColIndA,
                                           const int*               csrSortedRowPtrA,
                                           int                      nnzA,
                                           const int*               nnzPerRow,
                                           cuComplex*               csrSortedValC,
                                           int*                     csrSortedColIndC,
                                           int*                     csrSortedRowPtrC,
                                           cuComplex                tol) {
  return hipsparseCcsr2csr_compress(handle, m, n, descrA, csrSortedValA, csrSortedColIndA, csrSortedRowPtrA, nnzA, nnzPerRow, csrSortedValC, csrSortedColIndC, csrSortedRowPtrC, tol);
}

cusparseStatus_t cusparseZcsr2csr_compress(cusparseHandle_t         handle,
                                           int                      m,
                                           int                      n,
                                           const cusparseMatDescr_t descrA,
                                           const cuDoubleComplex*   csrSortedValA,
                                           const int*               csrSortedColIndA,
                                           const int*               csrSortedRowPtrA,
                                           int                      nnzA,
                                           const int*               nnzPerRow,
                                           cuDoubleComplex*         csrSortedValC,
                                           int*                     csrSortedColIndC,
                                           int*                     csrSortedRowPtrC,
                                           cuDoubleComplex          tol) {
  return hipsparseZcsr2csr_compress(handle, m, n, descrA, csrSortedValA, csrSortedColIndA, csrSortedRowPtrA, nnzA, nnzPerRow, csrSortedValC, csrSortedColIndC, csrSortedRowPtrC, tol);
}

}  // extern "C"


#endif  // INCLUDE_GUARD_STUB_CUPY_CUSPARSE_H
