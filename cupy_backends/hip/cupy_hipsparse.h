

#ifndef INCLUDE_GUARD_HIP_CUPY_HIPSPARSE_H
#define INCLUDE_GUARD_HIP_CUPY_HIPSPARSE_H
#include <hipsparse.h>
#include <hip/hip_version.h>    // for HIP_VERSION
#include <hip/library_types.h>  // for hipDataType
#include <stdexcept>  // for gcc 10.0

#if HIP_VERSION >= 402
static hipDataType convert_hipDatatype(cudaDataType type) {
    switch(static_cast<int>(type)) {
        case 2 /* CUDA_R_16F */: return HIP_R_16F;
        case 0 /* CUDA_R_32F */: return HIP_R_32F;
        case 1 /* CUDA_R_64F */: return HIP_R_64F;
        case 6 /* CUDA_C_16F */: return HIP_C_16F;
        case 4 /* CUDA_C_32F */: return HIP_C_32F;
        case 5 /* CUDA_C_64F */: return HIP_C_64F;
        default: throw std::runtime_error("unrecognized type");
    }
}
#endif


#if HIP_VERSION < 401
#define HIPSPARSE_STATUS_NOT_SUPPORTED (hipsparseStatus_t)10
#endif


extern "C" {

typedef hipsparseIndexBase_t cusparseIndexBase_t;
typedef hipsparseStatus_t cusparseStatus_t;

typedef hipsparseHandle_t cusparseHandle_t;
typedef hipsparseMatDescr_t cusparseMatDescr_t;
#if HIP_VERSION < 308
typedef void* bsric02Info_t;
#endif

#if HIP_VERSION < 309
typedef void* bsrilu02Info_t;
#endif


typedef hipsparseMatrixType_t cusparseMatrixType_t;
typedef hipsparseFillMode_t cusparseFillMode_t;
typedef hipsparseDiagType_t cusparseDiagType_t;
typedef hipsparseOperation_t cusparseOperation_t;
typedef hipsparsePointerMode_t cusparsePointerMode_t;
typedef hipsparseAction_t cusparseAction_t;
typedef hipsparseDirection_t cusparseDirection_t;
typedef enum {} cusparseAlgMode_t;
typedef hipsparseSolvePolicy_t cusparseSolvePolicy_t;

// Version
cusparseStatus_t cusparseGetVersion(cusparseHandle_t handle,
                                    int*             version) {
  return hipsparseGetVersion(handle, version);
}

// Error handling
const char* cusparseGetErrorName(...) {
    // Unavailable in hipSparse; this should not be called
    return "CUPY_HIPSPARSE_BINDING_UNEXPECTED_ERROR";
}

const char* cusparseGetErrorString(...) {
    // Unavailable in hipSparse; this should not be called
    return "unexpected error in CuPy hipSparse binding";
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
  return hipsparseCgthr(handle, nnz, reinterpret_cast<const hipComplex*>(y), reinterpret_cast<hipComplex*>(xVal), xInd, idxBase);
}

cusparseStatus_t cusparseZgthr(cusparseHandle_t       handle,
                               int                    nnz,
                               const cuDoubleComplex* y,
                               cuDoubleComplex*       xVal,
                               const int*             xInd,
                               cusparseIndexBase_t    idxBase) {
  return hipsparseZgthr(handle, nnz, reinterpret_cast<const hipDoubleComplex*>(y), reinterpret_cast<hipDoubleComplex*>(xVal), xInd, idxBase);
}

// cuSPARSE Level2 Function
cusparseStatus_t cusparseScsrmv(cusparseHandle_t         handle,
                                cusparseOperation_t      transA,
                                int                      m,
                                int                      n,
                                int                      nnz,
                                const float*             alpha,
                                const cusparseMatDescr_t descrA,
                                const float*             csrSortedValA,
                                const int*               csrSortedRowPtrA,
                                const int*               csrSortedColIndA,
                                const float*             x,
                                const float*             beta,
                                float*                   y) {
  return hipsparseScsrmv(handle, transA, m, n, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, x, beta, y);
}

cusparseStatus_t cusparseDcsrmv(cusparseHandle_t         handle,
                                cusparseOperation_t      transA,
                                int                      m,
                                int                      n,
                                int                      nnz,
                                const double*            alpha,
                                const cusparseMatDescr_t descrA,
                                const double*            csrSortedValA,
                                const int*               csrSortedRowPtrA,
                                const int*               csrSortedColIndA,
                                const double*            x,
                                const double*            beta,
                                double*                  y) {
  return hipsparseDcsrmv(handle, transA, m, n, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, x, beta, y);
}

cusparseStatus_t cusparseCcsrmv(cusparseHandle_t         handle,
                                cusparseOperation_t      transA,
                                int                      m,
                                int                      n,
                                int                      nnz,
                                const cuComplex*         alpha,
                                const cusparseMatDescr_t descrA,
                                const cuComplex*         csrSortedValA,
                                const int*               csrSortedRowPtrA,
                                const int*               csrSortedColIndA,
                                const cuComplex*         x,
                                const cuComplex*         beta,
                                cuComplex*               y) {
  return hipsparseCcsrmv(handle, transA, m, n, nnz, reinterpret_cast<const hipComplex*>(alpha), descrA, reinterpret_cast<const hipComplex*>(csrSortedValA), csrSortedRowPtrA, csrSortedColIndA, reinterpret_cast<const hipComplex*>(x), reinterpret_cast<const hipComplex*>(beta), reinterpret_cast<hipComplex*>(y));
}

cusparseStatus_t cusparseZcsrmv(cusparseHandle_t         handle,
                                cusparseOperation_t      transA,
                                int                      m,
                                int                      n,
                                int                      nnz,
                                const cuDoubleComplex*   alpha,
                                const cusparseMatDescr_t descrA,
                                const cuDoubleComplex*   csrSortedValA,
                                const int*               csrSortedRowPtrA,
                                const int*               csrSortedColIndA,
                                const cuDoubleComplex*   x,
                                const cuDoubleComplex*   beta,
                                cuDoubleComplex*         y) {
  return hipsparseZcsrmv(handle, transA, m, n, nnz, reinterpret_cast<const hipDoubleComplex*>(alpha), descrA, reinterpret_cast<const hipDoubleComplex*>(csrSortedValA), csrSortedRowPtrA, csrSortedColIndA, reinterpret_cast<const hipDoubleComplex*>(x), reinterpret_cast<const hipDoubleComplex*>(beta), reinterpret_cast<hipDoubleComplex*>(y));
}

cusparseStatus_t cusparseCsrmvEx_bufferSize(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

cusparseStatus_t cusparseCsrmvEx(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
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
  return hipsparseCcsrsv2_bufferSize(handle, transA, m, nnz, descrA, reinterpret_cast<hipComplex*>(csrSortedValA), csrSortedRowPtrA, csrSortedColIndA, info, pBufferSizeInBytes);
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
  return hipsparseZcsrsv2_bufferSize(handle, transA, m, nnz, descrA, reinterpret_cast<hipDoubleComplex*>(csrSortedValA), csrSortedRowPtrA, csrSortedColIndA, info, pBufferSizeInBytes);
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
  return hipsparseCcsrsv2_analysis(handle, transA, m, nnz, descrA, reinterpret_cast<const hipComplex*>(csrSortedValA), csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer);
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
  return hipsparseZcsrsv2_analysis(handle, transA, m, nnz, descrA, reinterpret_cast<const hipDoubleComplex*>(csrSortedValA), csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer);
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
  return hipsparseCcsrsv2_solve(handle, transA, m, nnz, reinterpret_cast<const hipComplex*>(alpha), descrA, reinterpret_cast<const hipComplex*>(csrSortedValA), csrSortedRowPtrA, csrSortedColIndA, info, reinterpret_cast<const hipComplex*>(f), reinterpret_cast<hipComplex*>(x), policy, pBuffer);
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
  return hipsparseZcsrsv2_solve(handle, transA, m, nnz, reinterpret_cast<const hipDoubleComplex*>(alpha), descrA, reinterpret_cast<const hipDoubleComplex*>(csrSortedValA), csrSortedRowPtrA, csrSortedColIndA, info, reinterpret_cast<const hipDoubleComplex*>(f), reinterpret_cast<hipDoubleComplex*>(x), policy, pBuffer);
}

cusparseStatus_t cusparseXcsrsv2_zeroPivot(cusparseHandle_t handle,
                                           csrsv2Info_t     info,
                                           int*             position) {
  return hipsparseXcsrsv2_zeroPivot(handle, info, position);
}

// cuSPARSE Level3 Function
cusparseStatus_t cusparseScsrmm(cusparseHandle_t         handle,
                                cusparseOperation_t      transA,
                                int                      m,
                                int                      n,
                                int                      k,
                                int                      nnz,
                                const float*             alpha,
                                const cusparseMatDescr_t descrA,
                                const float*             csrSortedValA,
                                const int*               csrSortedRowPtrA,
                                const int*               csrSortedColIndA,
                                const float*             B,
                                int                      ldb,
                                const float*             beta,
                                float*                   C,
                                int                      ldc) {
  return hipsparseScsrmm(handle, transA, m, n, k, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb, beta, C, ldc);
}

cusparseStatus_t cusparseDcsrmm(cusparseHandle_t         handle,
                                cusparseOperation_t      transA,
                                int                      m,
                                int                      n,
                                int                      k,
                                int                      nnz,
                                const double*            alpha,
                                const cusparseMatDescr_t descrA,
                                const double*            csrSortedValA,
                                const int*               csrSortedRowPtrA,
                                const int*               csrSortedColIndA,
                                const double*            B,
                                int                      ldb,
                                const double*            beta,
                                double*                  C,
                                int                      ldc) {
  return hipsparseDcsrmm(handle, transA, m, n, k, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb, beta, C, ldc);
}

cusparseStatus_t cusparseCcsrmm(cusparseHandle_t         handle,
                                cusparseOperation_t      transA,
                                int                      m,
                                int                      n,
                                int                      k,
                                int                      nnz,
                                const cuComplex*         alpha,
                                const cusparseMatDescr_t descrA,
                                const cuComplex*         csrSortedValA,
                                const int*               csrSortedRowPtrA,
                                const int*               csrSortedColIndA,
                                const cuComplex*         B,
                                int                      ldb,
                                const cuComplex*         beta,
                                cuComplex*               C,
                                int                      ldc) {
  return hipsparseCcsrmm(handle, transA, m, n, k, nnz, reinterpret_cast<const hipComplex*>(alpha), descrA, reinterpret_cast<const hipComplex*>(csrSortedValA), csrSortedRowPtrA, csrSortedColIndA, reinterpret_cast<const hipComplex*>(B), ldb, reinterpret_cast<const hipComplex*>(beta), reinterpret_cast<hipComplex*>(C), ldc);
}

cusparseStatus_t cusparseZcsrmm(cusparseHandle_t         handle,
                                cusparseOperation_t      transA,
                                int                      m,
                                int                      n,
                                int                      k,
                                int                      nnz,
                                const cuDoubleComplex*   alpha,
                                const cusparseMatDescr_t descrA,
                                const cuDoubleComplex*   csrSortedValA,
                                const int*               csrSortedRowPtrA,
                                const int*               csrSortedColIndA,
                                const cuDoubleComplex*   B,
                                int                      ldb,
                                const cuDoubleComplex*   beta,
                                cuDoubleComplex*         C,
                                int                      ldc) {
  return hipsparseZcsrmm(handle, transA, m, n, k, nnz, reinterpret_cast<const hipDoubleComplex*>(alpha), descrA, reinterpret_cast<const hipDoubleComplex*>(csrSortedValA), csrSortedRowPtrA, csrSortedColIndA, reinterpret_cast<const hipDoubleComplex*>(B), ldb, reinterpret_cast<const hipDoubleComplex*>(beta), reinterpret_cast<hipDoubleComplex*>(C), ldc);
}

cusparseStatus_t cusparseScsrmm2(cusparseHandle_t         handle,
                                 cusparseOperation_t      transA,
                                 cusparseOperation_t      transB,
                                 int                      m,
                                 int                      n,
                                 int                      k,
                                 int                      nnz,
                                 const float*             alpha,
                                 const cusparseMatDescr_t descrA,
                                 const float*             csrSortedValA,
                                 const int*               csrSortedRowPtrA,
                                 const int*               csrSortedColIndA,
                                 const float*             B,
                                 int                      ldb,
                                 const float*             beta,
                                 float*                   C,
                                 int                      ldc) {
  return hipsparseScsrmm2(handle, transA, transB, m, n, k, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb, beta, C, ldc);
}

cusparseStatus_t cusparseDcsrmm2(cusparseHandle_t         handle,
                                 cusparseOperation_t      transA,
                                 cusparseOperation_t      transB,
                                 int                      m,
                                 int                      n,
                                 int                      k,
                                 int                      nnz,
                                 const double*            alpha,
                                 const cusparseMatDescr_t descrA,
                                 const double* csrSortedValA,
                                 const int*    csrSortedRowPtrA,
                                 const int*    csrSortedColIndA,
                                 const double* B,
                                 int           ldb,
                                 const double* beta,
                                 double*       C,
                                 int           ldc) {
  return hipsparseDcsrmm2(handle, transA, transB, m, n, k, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb, beta, C, ldc);
}

cusparseStatus_t cusparseCcsrmm2(cusparseHandle_t         handle,
                                 cusparseOperation_t      transA,
                                 cusparseOperation_t      transB,
                                 int                      m,
                                 int                      n,
                                 int                      k,
                                 int                      nnz,
                                 const cuComplex*         alpha,
                                 const cusparseMatDescr_t descrA,
                                 const cuComplex* csrSortedValA,
                                 const int*       csrSortedRowPtrA,
                                 const int*       csrSortedColIndA,
                                 const cuComplex* B,
                                 int              ldb,
                                 const cuComplex* beta,
                                 cuComplex*       C,
                                 int              ldc) {
  return hipsparseCcsrmm2(handle, transA, transB, m, n, k, nnz, reinterpret_cast<const hipComplex*>(alpha), descrA, reinterpret_cast<const hipComplex*>(csrSortedValA), csrSortedRowPtrA, csrSortedColIndA, reinterpret_cast<const hipComplex*>(B), ldb, reinterpret_cast<const hipComplex*>(beta), reinterpret_cast<hipComplex*>(C), ldc);
}

cusparseStatus_t cusparseZcsrmm2(cusparseHandle_t         handle,
                                 cusparseOperation_t      transA,
                                 cusparseOperation_t      transB,
                                 int                      m,
                                 int                      n,
                                 int                      k,
                                 int                      nnz,
                                 const cuDoubleComplex*   alpha,
                                 const cusparseMatDescr_t descrA,
                                 const cuDoubleComplex*   csrSortedValA,
                                 const int*               csrSortedRowPtrA,
                                 const int*               csrSortedColIndA,
                                 const cuDoubleComplex*   B,
                                 int                      ldb,
                                 const cuDoubleComplex*   beta,
                                 cuDoubleComplex*         C,
                                 int                      ldc) {
  return hipsparseZcsrmm2(handle, transA, transB, m, n, k, nnz, reinterpret_cast<const hipDoubleComplex*>(alpha), descrA, reinterpret_cast<const hipDoubleComplex*>(csrSortedValA), csrSortedRowPtrA, csrSortedColIndA, reinterpret_cast<const hipDoubleComplex*>(B), ldb, reinterpret_cast<const hipDoubleComplex*>(beta), reinterpret_cast<hipDoubleComplex*>(C), ldc);
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
  return hipsparseCcsrsm2_bufferSizeExt(handle, algo, transA, transB, m, nrhs, nnz, reinterpret_cast<const hipComplex*>(alpha), descrA, reinterpret_cast<const hipComplex*>(csrSortedValA), csrSortedRowPtrA, csrSortedColIndA, reinterpret_cast<const hipComplex*>(B), ldb, info, policy, pBufferSize);
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
  return hipsparseZcsrsm2_bufferSizeExt(handle, algo, transA, transB, m, nrhs, nnz, reinterpret_cast<const hipDoubleComplex*>(alpha), descrA, reinterpret_cast<const hipDoubleComplex*>(csrSortedValA), csrSortedRowPtrA, csrSortedColIndA, reinterpret_cast<const hipDoubleComplex*>(B), ldb, info, policy, pBufferSize);
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
  return hipsparseCcsrsm2_analysis(handle, algo, transA, transB, m, nrhs, nnz, reinterpret_cast<const hipComplex*>(alpha), descrA, reinterpret_cast<const hipComplex*>(csrSortedValA), csrSortedRowPtrA, csrSortedColIndA, reinterpret_cast<const hipComplex*>(B), ldb, info, policy, pBuffer);
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
  return hipsparseZcsrsm2_analysis(handle, algo, transA, transB, m, nrhs, nnz, reinterpret_cast<const hipDoubleComplex*>(alpha), descrA, reinterpret_cast<const hipDoubleComplex*>(csrSortedValA), csrSortedRowPtrA, csrSortedColIndA, reinterpret_cast<const hipDoubleComplex*>(B), ldb, info, policy, pBuffer);
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
  return hipsparseCcsrsm2_solve(handle, algo, transA, transB, m, nrhs, nnz, reinterpret_cast<const hipComplex*>(alpha), descrA, reinterpret_cast<const hipComplex*>(csrSortedValA), csrSortedRowPtrA, csrSortedColIndA, reinterpret_cast<hipComplex*>(B), ldb, info, policy, pBuffer);
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
  return hipsparseZcsrsm2_solve(handle, algo, transA, transB, m, nrhs, nnz, reinterpret_cast<const hipDoubleComplex*>(alpha), descrA, reinterpret_cast<const hipDoubleComplex*>(csrSortedValA), csrSortedRowPtrA, csrSortedColIndA, reinterpret_cast<hipDoubleComplex*>(B), ldb, info, policy, pBuffer);
}

cusparseStatus_t cusparseXcsrsm2_zeroPivot(cusparseHandle_t handle,
                                           csrsm2Info_t     info,
                                           int* position) {
  return hipsparseXcsrsm2_zeroPivot(handle, info, position);
}

// cuSPARSE Extra Function
cusparseStatus_t cusparseXcsrgeamNnz(cusparseHandle_t         handle,
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
                                     int*                     nnzTotalDevHostPtr) {
  return hipsparseXcsrgeamNnz(handle, m, n, descrA, nnzA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedRowPtrC, nnzTotalDevHostPtr);
}

cusparseStatus_t cusparseScsrgeam(cusparseHandle_t         handle,
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
                                  float*                   csrSortedValC,
                                  int*                     csrSortedRowPtrC,
                                  int*                     csrSortedColIndC) {
  return hipsparseScsrgeam(handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC);
}

cusparseStatus_t cusparseDcsrgeam(cusparseHandle_t         handle,
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
                                  double*                  csrSortedValC,
                                  int*                     csrSortedRowPtrC,
                                  int*                     csrSortedColIndC) {
  return hipsparseDcsrgeam(handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC);
}

cusparseStatus_t cusparseCcsrgeam(cusparseHandle_t         handle,
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
                                  cuComplex*               csrSortedValC,
                                  int*                     csrSortedRowPtrC,
                                  int*                     csrSortedColIndC) {
  return hipsparseCcsrgeam(handle, m, n, reinterpret_cast<const hipComplex*>(alpha), descrA, nnzA, reinterpret_cast<const hipComplex*>(csrSortedValA), csrSortedRowPtrA, csrSortedColIndA, reinterpret_cast<const hipComplex*>(beta), descrB, nnzB, reinterpret_cast<const hipComplex*>(csrSortedValB), csrSortedRowPtrB, csrSortedColIndB, descrC, reinterpret_cast<hipComplex*>(csrSortedValC), csrSortedRowPtrC, csrSortedColIndC);
}

cusparseStatus_t cusparseZcsrgeam(cusparseHandle_t         handle,
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
                                  cuDoubleComplex*         csrSortedValC,
                                  int*                     csrSortedRowPtrC,
                                  int*                     csrSortedColIndC) {
  return hipsparseZcsrgeam(handle, m, n, reinterpret_cast<const hipDoubleComplex*>(alpha), descrA, nnzA, reinterpret_cast<const hipDoubleComplex*>(csrSortedValA), csrSortedRowPtrA, csrSortedColIndA, reinterpret_cast<const hipDoubleComplex*>(beta), descrB, nnzB, reinterpret_cast<const hipDoubleComplex*>(csrSortedValB), csrSortedRowPtrB, csrSortedColIndB, descrC, reinterpret_cast<hipDoubleComplex*>(csrSortedValC), csrSortedRowPtrC, csrSortedColIndC);
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
  return hipsparseCcsrgeam2_bufferSizeExt(handle, m, n, reinterpret_cast<const hipComplex*>(alpha), descrA, nnzA, reinterpret_cast<const hipComplex*>(csrSortedValA), csrSortedRowPtrA, csrSortedColIndA, reinterpret_cast<const hipComplex*>(beta), descrB, nnzB, reinterpret_cast<const hipComplex*>(csrSortedValB), csrSortedRowPtrB, csrSortedColIndB, descrC, reinterpret_cast<const hipComplex*>(csrSortedValC), csrSortedRowPtrC, csrSortedColIndC, pBufferSizeInBytes);
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
  return hipsparseZcsrgeam2_bufferSizeExt(handle, m, n, reinterpret_cast<const hipDoubleComplex*>(alpha), descrA, nnzA, reinterpret_cast<const hipDoubleComplex*>(csrSortedValA), csrSortedRowPtrA, csrSortedColIndA, reinterpret_cast<const hipDoubleComplex*>(beta), descrB, nnzB, reinterpret_cast<const hipDoubleComplex*>(csrSortedValB), csrSortedRowPtrB, csrSortedColIndB, descrC, reinterpret_cast<const hipDoubleComplex*>(csrSortedValC), csrSortedRowPtrC, csrSortedColIndC, pBufferSizeInBytes);
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

cusparseStatus_t cusparseScsrgeam2(cusparseHandle_t         handle,
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
                                   float*                   csrSortedValC,
                                   int*                     csrSortedRowPtrC,
                                   int*                     csrSortedColIndC,
                                   void*                    pBuffer) {
  return hipsparseScsrgeam2(handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBuffer);
}

cusparseStatus_t cusparseDcsrgeam2(cusparseHandle_t         handle,
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
                                   double*                  csrSortedValC,
                                   int*                     csrSortedRowPtrC,
                                   int*                     csrSortedColIndC,
                                   void*                    pBuffer) {
  return hipsparseDcsrgeam2(handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBuffer);
}

cusparseStatus_t cusparseCcsrgeam2(cusparseHandle_t         handle,
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
                                   cuComplex*               csrSortedValC,
                                   int*                     csrSortedRowPtrC,
                                   int*                     csrSortedColIndC,
                                   void*                    pBuffer) {
  return hipsparseCcsrgeam2(handle, m, n, reinterpret_cast<const hipComplex*>(alpha), descrA, nnzA, reinterpret_cast<const hipComplex*>(csrSortedValA), csrSortedRowPtrA, csrSortedColIndA, reinterpret_cast<const hipComplex*>(beta), descrB, nnzB, reinterpret_cast<const hipComplex*>(csrSortedValB), csrSortedRowPtrB, csrSortedColIndB, descrC, reinterpret_cast<hipComplex*>(csrSortedValC), csrSortedRowPtrC, csrSortedColIndC, pBuffer);
}

cusparseStatus_t cusparseZcsrgeam2(cusparseHandle_t         handle,
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
                                   cuDoubleComplex*         csrSortedValC,
                                   int*                     csrSortedRowPtrC,
                                   int*                     csrSortedColIndC,
                                   void*                    pBuffer) {
  return hipsparseZcsrgeam2(handle, m, n, reinterpret_cast<const hipDoubleComplex*>(alpha), descrA, nnzA, reinterpret_cast<const hipDoubleComplex*>(csrSortedValA), csrSortedRowPtrA, csrSortedColIndA, reinterpret_cast<const hipDoubleComplex*>(beta), descrB, nnzB, reinterpret_cast<const hipDoubleComplex*>(csrSortedValB), csrSortedRowPtrB, csrSortedColIndB, descrC, reinterpret_cast<hipDoubleComplex*>(csrSortedValC), csrSortedRowPtrC, csrSortedColIndC, pBuffer);
}

cusparseStatus_t cusparseXcsrgemmNnz(cusparseHandle_t         handle,
                                     cusparseOperation_t      transA,
                                     cusparseOperation_t      transB,
                                     int                      m,
                                     int                      n,
                                     int                      k,
                                     const cusparseMatDescr_t descrA,
                                     const int                nnzA,
                                     const int*               csrSortedRowPtrA,
                                     const int*               csrSortedColIndA,
                                     const cusparseMatDescr_t descrB,
                                     const int                nnzB,
                                     const int*               csrSortedRowPtrB,
                                     const int*               csrSortedColIndB,
                                     const cusparseMatDescr_t descrC,
                                     int*                     csrSortedRowPtrC,
                                     int*                     nnzTotalDevHostPtr) {
  return hipsparseXcsrgemmNnz(handle, transA, transB, m, n, k, descrA, nnzA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedRowPtrC, nnzTotalDevHostPtr);
}

cusparseStatus_t cusparseScsrgemm(cusparseHandle_t         handle,
                                  cusparseOperation_t      transA,
                                  cusparseOperation_t      transB,
                                  int                      m,
                                  int                      n,
                                  int                      k,
                                  const cusparseMatDescr_t descrA,
                                  const int                nnzA,
                                  const float*             csrSortedValA,
                                  const int*               csrSortedRowPtrA,
                                  const int*               csrSortedColIndA,
                                  const cusparseMatDescr_t descrB,
                                  const int                nnzB,
                                  const float*             csrSortedValB,
                                  const int*               csrSortedRowPtrB,
                                  const int*               csrSortedColIndB,
                                  const cusparseMatDescr_t descrC,
                                  float*                   csrSortedValC,
                                  const int*               csrSortedRowPtrC,
                                  int*                     csrSortedColIndC) {
  return hipsparseScsrgemm(handle, transA, transB, m, n, k, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC);
}

cusparseStatus_t cusparseDcsrgemm(cusparseHandle_t         handle,
                                  cusparseOperation_t      transA,
                                  cusparseOperation_t      transB,
                                  int                      m,
                                  int                      n,
                                  int                      k,
                                  const cusparseMatDescr_t descrA,
                                  int                      nnzA,
                                  const double*            csrSortedValA,
                                  const int*               csrSortedRowPtrA,
                                  const int*               csrSortedColIndA,
                                  const cusparseMatDescr_t descrB,
                                  int                      nnzB,
                                  const double*            csrSortedValB,
                                  const int*               csrSortedRowPtrB,
                                  const int*               csrSortedColIndB,
                                  const cusparseMatDescr_t descrC,
                                  double*                  csrSortedValC,
                                  const int*               csrSortedRowPtrC,
                                  int*                     csrSortedColIndC) {
  return hipsparseDcsrgemm(handle, transA, transB, m, n, k, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC);
}

cusparseStatus_t cusparseCcsrgemm(cusparseHandle_t         handle,
                                  cusparseOperation_t      transA,
                                  cusparseOperation_t      transB,
                                  int                      m,
                                  int                      n,
                                  int                      k,
                                  const cusparseMatDescr_t descrA,
                                  int                      nnzA,
                                  const cuComplex*         csrSortedValA,
                                  const int*               csrSortedRowPtrA,
                                  const int*               csrSortedColIndA,
                                  const cusparseMatDescr_t descrB,
                                  int                      nnzB,
                                  const cuComplex*         csrSortedValB,
                                  const int*               csrSortedRowPtrB,
                                  const int*               csrSortedColIndB,
                                  const cusparseMatDescr_t descrC,
                                  cuComplex*               csrSortedValC,
                                  const int*               csrSortedRowPtrC,
                                  int*                     csrSortedColIndC) {
  return hipsparseCcsrgemm(handle, transA, transB, m, n, k, descrA, nnzA, reinterpret_cast<const hipComplex*>(csrSortedValA), csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, reinterpret_cast<const hipComplex*>(csrSortedValB), csrSortedRowPtrB, csrSortedColIndB, descrC, reinterpret_cast<hipComplex*>(csrSortedValC), csrSortedRowPtrC, csrSortedColIndC);
}

cusparseStatus_t cusparseZcsrgemm(cusparseHandle_t         handle,
                                  cusparseOperation_t      transA,
                                  cusparseOperation_t      transB,
                                  int                      m,
                                  int                      n,
                                  int                      k,
                                  const cusparseMatDescr_t descrA,
                                  int                      nnzA,
                                  const cuDoubleComplex*   csrSortedValA,
                                  const int*               csrSortedRowPtrA,
                                  const int*               csrSortedColIndA,
                                  const cusparseMatDescr_t descrB,
                                  int                      nnzB,
                                  const cuDoubleComplex*   csrSortedValB,
                                  const int*               csrSortedRowPtrB,
                                  const int*               csrSortedColIndB,
                                  const cusparseMatDescr_t descrC,
                                  cuDoubleComplex*         csrSortedValC,
                                  const int*               csrSortedRowPtrC,
                                  int*                     csrSortedColIndC) {
  return hipsparseZcsrgemm(handle, transA, transB, m, n, k, descrA, nnzA, reinterpret_cast<const hipDoubleComplex*>(csrSortedValA), csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, reinterpret_cast<const hipDoubleComplex*>(csrSortedValB), csrSortedRowPtrB, csrSortedColIndB, descrC, reinterpret_cast<hipDoubleComplex*>(csrSortedValC), csrSortedRowPtrC, csrSortedColIndC);
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
  return hipsparseCcsrgemm2_bufferSizeExt(handle, m, n, k, reinterpret_cast<const hipComplex*>(alpha), descrA, nnzA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedRowPtrB, csrSortedColIndB, reinterpret_cast<const hipComplex*>(beta), descrD, nnzD, csrSortedRowPtrD, csrSortedColIndD, info, pBufferSizeInBytes);
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
  return hipsparseZcsrgemm2_bufferSizeExt(handle, m, n, k, reinterpret_cast<const hipDoubleComplex*>(alpha), descrA, nnzA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedRowPtrB, csrSortedColIndB, reinterpret_cast<const hipDoubleComplex*>(beta), descrD, nnzD, csrSortedRowPtrD, csrSortedColIndD, info, pBufferSizeInBytes);
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

cusparseStatus_t cusparseScsrgemm2(cusparseHandle_t         handle,
                                   int                      m,
                                   int                      n,
                                   int                      k,
                                   const float*             alpha,
                                   const cusparseMatDescr_t descrA,
                                   int                      nnzA,
                                   const float*             csrSortedValA,
                                   const int*               csrSortedRowPtrA,
                                   const int*               csrSortedColIndA,
                                   const cusparseMatDescr_t descrB,
                                   int                      nnzB,
                                   const float*             csrSortedValB,
                                   const int*               csrSortedRowPtrB,
                                   const int*               csrSortedColIndB,
                                   const float*             beta,
                                   const cusparseMatDescr_t descrD,
                                   int                      nnzD,
                                   const float*             csrSortedValD,
                                   const int*               csrSortedRowPtrD,
                                   const int*               csrSortedColIndD,
                                   const cusparseMatDescr_t descrC,
                                   float*                   csrSortedValC,
                                   const int*               csrSortedRowPtrC,
                                   int*                     csrSortedColIndC,
                                   const csrgemm2Info_t     info,
                                   void*                    pBuffer) {
  return hipsparseScsrgemm2(handle, m, n, k, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, beta, descrD, nnzD, csrSortedValD, csrSortedRowPtrD, csrSortedColIndD, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, info, pBuffer);
}

cusparseStatus_t cusparseDcsrgemm2(cusparseHandle_t         handle,
                                   int                      m,
                                   int                      n,
                                   int                      k,
                                   const double*            alpha,
                                   const cusparseMatDescr_t descrA,
                                   int                      nnzA,
                                   const double*            csrSortedValA,
                                   const int*               csrSortedRowPtrA,
                                   const int*               csrSortedColIndA,
                                   const cusparseMatDescr_t descrB,
                                   int                      nnzB,
                                   const double*            csrSortedValB,
                                   const int*               csrSortedRowPtrB,
                                   const int*               csrSortedColIndB,
                                   const double*            beta,
                                   const cusparseMatDescr_t descrD,
                                   int                      nnzD,
                                   const double*            csrSortedValD,
                                   const int*               csrSortedRowPtrD,
                                   const int*               csrSortedColIndD,
                                   const cusparseMatDescr_t descrC,
                                   double*                  csrSortedValC,
                                   const int*               csrSortedRowPtrC,
                                   int*                     csrSortedColIndC,
                                   const csrgemm2Info_t     info,
                                   void*                    pBuffer) {
  return hipsparseDcsrgemm2(handle, m, n, k, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, beta, descrD, nnzD, csrSortedValD, csrSortedRowPtrD, csrSortedColIndD, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, info, pBuffer);
}

cusparseStatus_t cusparseCcsrgemm2(cusparseHandle_t         handle,
                                  int                      m,
                                  int                      n,
                                  int                      k,
                                  const cuComplex*         alpha,
                                  const cusparseMatDescr_t descrA,
                                  int                      nnzA,
                                  const cuComplex*         csrSortedValA,
                                  const int*               csrSortedRowPtrA,
                                  const int*               csrSortedColIndA,
                                  const cusparseMatDescr_t descrB,
                                  int                      nnzB,
                                  const cuComplex*         csrSortedValB,
                                  const int*               csrSortedRowPtrB,
                                  const int*               csrSortedColIndB,
                                  const cuComplex*         beta,
                                  const cusparseMatDescr_t descrD,
                                  int                      nnzD,
                                  const cuComplex*         csrSortedValD,
                                  const int*               csrSortedRowPtrD,
                                  const int*               csrSortedColIndD,
                                  const cusparseMatDescr_t descrC,
                                  cuComplex*               csrSortedValC,
                                  const int*               csrSortedRowPtrC,
                                  int*                     csrSortedColIndC,
                                  const csrgemm2Info_t     info,
                                  void*                    pBuffer) {
  return hipsparseCcsrgemm2(handle, m, n, k, reinterpret_cast<const hipComplex*>(alpha), descrA, nnzA, reinterpret_cast<const hipComplex*>(csrSortedValA), csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, reinterpret_cast<const hipComplex*>(csrSortedValB), csrSortedRowPtrB, csrSortedColIndB, reinterpret_cast<const hipComplex*>(beta), descrD, nnzD, reinterpret_cast<const hipComplex*>(csrSortedValD), csrSortedRowPtrD, csrSortedColIndD, descrC, reinterpret_cast<hipComplex*>(csrSortedValC), csrSortedRowPtrC, csrSortedColIndC, info, pBuffer);
}

cusparseStatus_t cusparseZcsrgemm2(cusparseHandle_t         handle,
                                   int                      m,
                                   int                      n,
                                   int                      k,
                                   const cuDoubleComplex*   alpha,
                                   const cusparseMatDescr_t descrA,
                                   int                      nnzA,
                                   const cuDoubleComplex*   csrSortedValA,
                                   const int*               csrSortedRowPtrA,
                                   const int*               csrSortedColIndA,
                                   const cusparseMatDescr_t descrB,
                                   int                      nnzB,
                                   const cuDoubleComplex*   csrSortedValB,
                                   const int*               csrSortedRowPtrB,
                                   const int*               csrSortedColIndB,
                                   const cuDoubleComplex*   beta,
                                   const cusparseMatDescr_t descrD,
                                   int                      nnzD,
                                   const cuDoubleComplex*   csrSortedValD,
                                   const int*               csrSortedRowPtrD,
                                   const int*               csrSortedColIndD,
                                   const cusparseMatDescr_t descrC,
                                   cuDoubleComplex*         csrSortedValC,
                                   const int*               csrSortedRowPtrC,
                                   int*                     csrSortedColIndC,
                                   const csrgemm2Info_t     info,
                                   void*                    pBuffer) {
  return hipsparseZcsrgemm2(handle, m, n, k, reinterpret_cast<const hipDoubleComplex*>(alpha), descrA, nnzA, reinterpret_cast<const hipDoubleComplex*>(csrSortedValA), csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, reinterpret_cast<const hipDoubleComplex*>(csrSortedValB), csrSortedRowPtrB, csrSortedColIndB, reinterpret_cast<const hipDoubleComplex*>(beta), descrD, nnzD, reinterpret_cast<const hipDoubleComplex*>(csrSortedValD), csrSortedRowPtrD, csrSortedColIndD, descrC, reinterpret_cast<hipDoubleComplex*>(csrSortedValC), csrSortedRowPtrC, csrSortedColIndC, info, pBuffer);
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
  return hipsparseCcsc2dense(handle, m, n, descrA, reinterpret_cast<const hipComplex*>(cscSortedValA), cscSortedRowIndA, cscSortedColPtrA, reinterpret_cast<hipComplex*>(A), lda);
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
  return hipsparseZcsc2dense(handle, m, n, descrA, reinterpret_cast<const hipDoubleComplex*>(cscSortedValA), cscSortedRowIndA, cscSortedColPtrA, reinterpret_cast<hipDoubleComplex*>(A), lda);
}

cusparseStatus_t cusparseXcsr2coo(cusparseHandle_t    handle,
                                  const int*          csrSortedRowPtr,
                                  int                 nnz,
                                  int                 m,
                                  int*                cooRowInd,
                                  cusparseIndexBase_t idxBase) {
  return hipsparseXcsr2coo(handle, csrSortedRowPtr, nnz, m, cooRowInd, idxBase);
}

cusparseStatus_t cusparseScsr2csc(cusparseHandle_t    handle,
                                  int                 m,
                                  int                 n,
                                  int                 nnz,
                                  const float*        csrSortedVal,
                                  const int*          csrSortedRowPtr,
                                  const int*          csrSortedColInd,
                                  float*              cscSortedVal,
                                  int*                cscSortedRowInd,
                                  int*                cscSortedColPtr,
                                  cusparseAction_t    copyValues,
                                  cusparseIndexBase_t idxBase) {
  return hipsparseScsr2csc(handle, m, n, nnz, csrSortedVal, csrSortedRowPtr, csrSortedColInd, cscSortedVal, cscSortedRowInd, cscSortedColPtr, copyValues, idxBase);
}

cusparseStatus_t cusparseDcsr2csc(cusparseHandle_t    handle,
                                  int                 m,
                                  int                 n,
                                  int                 nnz,
                                  const double*       csrSortedVal,
                                  const int*          csrSortedRowPtr,
                                  const int*          csrSortedColInd,
                                  double*             cscSortedVal,
                                  int*                cscSortedRowInd,
                                  int*                cscSortedColPtr,
                                  cusparseAction_t    copyValues,
                                  cusparseIndexBase_t idxBase) {
  return hipsparseDcsr2csc(handle, m, n, nnz, csrSortedVal, csrSortedRowPtr, csrSortedColInd, cscSortedVal, cscSortedRowInd, cscSortedColPtr, copyValues, idxBase);
}

cusparseStatus_t cusparseCcsr2csc(cusparseHandle_t    handle,
                                  int                 m,
                                  int                 n,
                                  int                 nnz,
                                  const cuComplex*    csrSortedVal,
                                  const int*          csrSortedRowPtr,
                                  const int*          csrSortedColInd,
                                  cuComplex*          cscSortedVal,
                                  int*                cscSortedRowInd,
                                  int*                cscSortedColPtr,
                                  cusparseAction_t    copyValues,
                                  cusparseIndexBase_t idxBase) {
  return hipsparseCcsr2csc(handle, m, n, nnz, reinterpret_cast<const hipComplex*>(csrSortedVal), csrSortedRowPtr, csrSortedColInd, reinterpret_cast<hipComplex*>(cscSortedVal), cscSortedRowInd, cscSortedColPtr, copyValues, idxBase);
}

cusparseStatus_t cusparseZcsr2csc(cusparseHandle_t       handle,
                                  int                    m,
                                  int                    n,
                                  int                    nnz,
                                  const cuDoubleComplex* csrSortedVal,
                                  const int*             csrSortedRowPtr,
                                  const int*             csrSortedColInd,
                                  cuDoubleComplex*       cscSortedVal,
                                  int*                   cscSortedRowInd,
                                  int*                   cscSortedColPtr,
                                  cusparseAction_t       copyValues,
                                  cusparseIndexBase_t    idxBase) {
  return hipsparseZcsr2csc(handle, m, n, nnz, reinterpret_cast<const hipDoubleComplex*>(csrSortedVal), csrSortedRowPtr, csrSortedColInd, reinterpret_cast<hipDoubleComplex*>(cscSortedVal), cscSortedRowInd, cscSortedColPtr, copyValues, idxBase);
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
  return hipsparseCcsr2dense(handle, m, n, descrA, reinterpret_cast<const hipComplex*>(csrSortedValA), csrSortedRowPtrA, csrSortedColIndA, reinterpret_cast<hipComplex*>(A), lda);
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
  return hipsparseZcsr2dense(handle, m, n, descrA, reinterpret_cast<const hipDoubleComplex*>(csrSortedValA), csrSortedRowPtrA, csrSortedColIndA, reinterpret_cast<hipDoubleComplex*>(A), lda);
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
  return hipsparseCdense2csc(handle, m, n, descrA, reinterpret_cast<const hipComplex*>(A), lda, nnzPerCol, reinterpret_cast<hipComplex*>(cscSortedValA), cscSortedRowIndA, cscSortedColPtrA);
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
  return hipsparseZdense2csc(handle, m, n, descrA, reinterpret_cast<const hipDoubleComplex*>(A), lda, nnzPerCol, reinterpret_cast<hipDoubleComplex*>(cscSortedValA), cscSortedRowIndA, cscSortedColPtrA);
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
  return hipsparseCdense2csr(handle, m, n, descrA, reinterpret_cast<const hipComplex*>(A), lda, nnzPerRow, reinterpret_cast<hipComplex*>(csrSortedValA), csrSortedRowPtrA, csrSortedColIndA);
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
  return hipsparseZdense2csr(handle, m, n, descrA, reinterpret_cast<const hipDoubleComplex*>(A), lda, nnzPerRow, reinterpret_cast<hipDoubleComplex*>(csrSortedValA), csrSortedRowPtrA, csrSortedColIndA);
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
  return hipsparseCnnz(handle, dirA, m, n, descrA, reinterpret_cast<const hipComplex*>(A), lda, nnzPerRowCol, nnzTotalDevHostPtr);
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
  return hipsparseZnnz(handle, dirA, m, n, descrA, reinterpret_cast<const hipDoubleComplex*>(A), lda, nnzPerRowCol, nnzTotalDevHostPtr);
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

cusparseStatus_t cusparseXcsrsort(cusparseHandle_t         handle,
                                  int                      m,
                                  int                      n,
                                  int                      nnz,
                                  const cusparseMatDescr_t descrA,
                                  const int*               csrRowPtrA,
                                  int*                     csrColIndA,
                                  int*                     P,
                                  void*                    pBuffer) {
  return hipsparseXcsrsort(handle, m, n, nnz, descrA, csrRowPtrA, csrColIndA, P, pBuffer);
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

cusparseStatus_t cusparseXcscsort(cusparseHandle_t         handle,
                                  int                      m,
                                  int                      n,
                                  int                      nnz,
                                  const cusparseMatDescr_t descrA,
                                  const int*               cscColPtrA,
                                  int*                     cscRowIndA,
                                  int*                     P,
                                  void*                    pBuffer) {
  return hipsparseXcscsort(handle, m, n, nnz, descrA, cscColPtrA, cscRowIndA, P, pBuffer);
}

// cuSPARSE PRECONDITIONERS

cusparseStatus_t cusparseCreateCsrilu02Info(csrilu02Info_t* info) {
  return hipsparseCreateCsrilu02Info(info);
}

cusparseStatus_t cusparseDestroyCsrilu02Info(csrilu02Info_t info) {
  return hipsparseDestroyCsrilu02Info(info);
}

cusparseStatus_t cusparseCreateBsrilu02Info(bsrilu02Info_t* info) {
#if HIP_VERSION >= 309
  return hipsparseCreateBsrilu02Info(info);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseDestroyBsrilu02Info(bsrilu02Info_t info) {
#if HIP_VERSION >= 309
  return hipsparseDestroyBsrilu02Info(info);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseCreateCsric02Info(csric02Info_t* info) {
  return hipsparseCreateCsric02Info(info);
}

cusparseStatus_t cusparseDestroyCsric02Info(csric02Info_t info) {
  return hipsparseDestroyCsric02Info(info);
}

cusparseStatus_t cusparseCreateBsric02Info(bsric02Info_t* info) {
#if HIP_VERSION >= 308
  return hipsparseCreateBsric02Info(info);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseDestroyBsric02Info(bsric02Info_t info) {
#if HIP_VERSION >= 308
  return hipsparseDestroyBsric02Info(info);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseScsrilu02_numericBoost(cusparseHandle_t handle,
                                                csrilu02Info_t   info,
                                                int              enable_boost,
                                                double*          tol,
                                                float*           boost_val) {
#if HIP_VERSION >= 400
  return hipsparseScsrilu02_numericBoost(handle, info, enable_boost, tol, boost_val);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseDcsrilu02_numericBoost(cusparseHandle_t handle,
                                                csrilu02Info_t   info,
                                                int              enable_boost,
                                                double*          tol,
                                                double*          boost_val) {
#if HIP_VERSION >= 400
  return hipsparseDcsrilu02_numericBoost(handle, info, enable_boost, tol, boost_val);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseCcsrilu02_numericBoost(cusparseHandle_t handle,
                                                csrilu02Info_t   info,
                                                int              enable_boost,
                                                double*          tol,
                                                cuComplex*       boost_val) {
#if HIP_VERSION >= 400
  return hipsparseCcsrilu02_numericBoost(handle, info, enable_boost, tol, reinterpret_cast<hipComplex*>(boost_val));
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseZcsrilu02_numericBoost(cusparseHandle_t handle,
                                                csrilu02Info_t   info,
                                                int              enable_boost,
                                                double*          tol,
                                                cuDoubleComplex* boost_val) {
#if HIP_VERSION >= 400
  return hipsparseZcsrilu02_numericBoost(handle, info, enable_boost, tol, reinterpret_cast<hipDoubleComplex*>(boost_val));
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
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
  return hipsparseCcsrilu02_bufferSize(handle, m, nnz, descrA, reinterpret_cast<hipComplex*>(csrSortedValA), csrSortedRowPtrA, csrSortedColIndA, info, pBufferSizeInBytes);
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
  return hipsparseZcsrilu02_bufferSize(handle, m, nnz, descrA, reinterpret_cast<hipDoubleComplex*>(csrSortedValA), csrSortedRowPtrA, csrSortedColIndA, info, pBufferSizeInBytes);
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
  return hipsparseCcsrilu02_analysis(handle, m, nnz, descrA, reinterpret_cast<const hipComplex*>(csrSortedValA), csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer);
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
  return hipsparseZcsrilu02_analysis(handle, m, nnz, descrA, reinterpret_cast<const hipDoubleComplex*>(csrSortedValA), csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer);
}

cusparseStatus_t cusparseScsrilu02(cusparseHandle_t         handle,
                                   int                      m,
                                   int                      nnz,
                                   const cusparseMatDescr_t descrA,
                                   float*                   csrSortedValA_valM,
                                   const int*            csrSortedRowPtrA,
                                   const int*            csrSortedColIndA,
                                   csrilu02Info_t        info,
                                   cusparseSolvePolicy_t policy,
                                   void*                 pBuffer) {
  return hipsparseScsrilu02(handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer);
}

cusparseStatus_t cusparseDcsrilu02(cusparseHandle_t         handle,
                                   int                      m,
                                   int                      nnz,
                                   const cusparseMatDescr_t descrA,
                                   double*                  csrSortedValA_valM,
                                   const int*            csrSortedRowPtrA,
                                   const int*            csrSortedColIndA,
                                   csrilu02Info_t        info,
                                   cusparseSolvePolicy_t policy,
                                   void*                 pBuffer) {
  return hipsparseDcsrilu02(handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer);
}

cusparseStatus_t cusparseCcsrilu02(cusparseHandle_t         handle,
                                   int                      m,
                                   int                      nnz,
                                   const cusparseMatDescr_t descrA,
                                   cuComplex*               csrSortedValA_valM,
                                   const int*            csrSortedRowPtrA,
                                   const int*            csrSortedColIndA,
                                   csrilu02Info_t        info,
                                   cusparseSolvePolicy_t policy,
                                   void*                 pBuffer) {
  return hipsparseCcsrilu02(handle, m, nnz, descrA, reinterpret_cast<hipComplex*>(csrSortedValA_valM), csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer);
}

cusparseStatus_t cusparseZcsrilu02(cusparseHandle_t         handle,
                                   int                      m,
                                   int                      nnz,
                                   const cusparseMatDescr_t descrA,
                                   cuDoubleComplex*         csrSortedValA_valM,
                                   const int*            csrSortedRowPtrA,
                                   const int*            csrSortedColIndA,
                                   csrilu02Info_t        info,
                                   cusparseSolvePolicy_t policy,
                                   void*                 pBuffer) {
  return hipsparseZcsrilu02(handle, m, nnz, descrA, reinterpret_cast<hipDoubleComplex*>(csrSortedValA_valM), csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer);
}

cusparseStatus_t cusparseSbsrilu02_numericBoost(cusparseHandle_t handle,
                                                bsrilu02Info_t   info,
                                                int              enable_boost,
                                                double*          tol,
                                                float*           boost_val) {
#if HIP_VERSION >= 309
  return hipsparseSbsrilu02_numericBoost(handle, info, enable_boost, tol, boost_val);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseDbsrilu02_numericBoost(cusparseHandle_t handle,
                                                bsrilu02Info_t   info,
                                                int              enable_boost,
                                                double*          tol,
                                                double*          boost_val) {
#if HIP_VERSION >= 309
  return hipsparseDbsrilu02_numericBoost(handle, info, enable_boost, tol, boost_val);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseCbsrilu02_numericBoost(cusparseHandle_t handle,
                                                bsrilu02Info_t   info,
                                                int              enable_boost,
                                                double*          tol,
                                                cuComplex*       boost_val) {
#if HIP_VERSION >= 309
  return hipsparseCbsrilu02_numericBoost(handle, info, enable_boost, tol, reinterpret_cast<hipComplex*>(boost_val));
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseZbsrilu02_numericBoost(cusparseHandle_t handle,
                                                bsrilu02Info_t   info,
                                                int              enable_boost,
                                                double*          tol,
                                                cuDoubleComplex* boost_val) {
#if HIP_VERSION >= 309
  return hipsparseZbsrilu02_numericBoost(handle, info, enable_boost, tol, reinterpret_cast<hipDoubleComplex*>(boost_val));
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseXbsrilu02_zeroPivot(cusparseHandle_t handle,
                                             bsrilu02Info_t   info,
                                             int*             position) {
#if HIP_VERSION >= 309
  return hipsparseXbsrilu02_zeroPivot(handle, info, position);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseSbsrilu02_bufferSize(cusparseHandle_t         handle,
                                              cusparseDirection_t      dirA,
                                              int                      mb,
                                              int                      nnzb,
                                              const cusparseMatDescr_t descrA,
                                              float*                   bsrSortedVal,
                                              const int*               bsrSortedRowPtr,
                                              const int*               bsrSortedColInd,
                                              int                      blockDim,
                                              bsrilu02Info_t           info,
                                              int*                     pBufferSizeInBytes) {
#if HIP_VERSION >= 309
  return hipsparseSbsrilu02_bufferSize(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, pBufferSizeInBytes);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseDbsrilu02_bufferSize(cusparseHandle_t         handle,
                                              cusparseDirection_t      dirA,
                                              int                      mb,
                                              int                      nnzb,
                                              const cusparseMatDescr_t descrA,
                                              double*                  bsrSortedVal,
                                              const int*               bsrSortedRowPtr,
                                              const int*               bsrSortedColInd,
                                              int                      blockDim,
                                              bsrilu02Info_t           info,
                                              int*                     pBufferSizeInBytes) {
#if HIP_VERSION >= 309
  return hipsparseDbsrilu02_bufferSize(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, pBufferSizeInBytes);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseCbsrilu02_bufferSize(cusparseHandle_t         handle,
                                              cusparseDirection_t      dirA,
                                              int                      mb,
                                              int                      nnzb,
                                              const cusparseMatDescr_t descrA,
                                              cuComplex*               bsrSortedVal,
                                              const int*               bsrSortedRowPtr,
                                              const int*               bsrSortedColInd,
                                              int                      blockDim,
                                              bsrilu02Info_t           info,
                                              int*                     pBufferSizeInBytes) {
#if HIP_VERSION >= 309
  return hipsparseCbsrilu02_bufferSize(handle, dirA, mb, nnzb, descrA, reinterpret_cast<hipComplex*>(bsrSortedVal), bsrSortedRowPtr, bsrSortedColInd, blockDim, info, pBufferSizeInBytes);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseZbsrilu02_bufferSize(cusparseHandle_t         handle,
                                              cusparseDirection_t      dirA,
                                              int                      mb,
                                              int                      nnzb,
                                              const cusparseMatDescr_t descrA,
                                              cuDoubleComplex*         bsrSortedVal,
                                              const int*               bsrSortedRowPtr,
                                              const int*               bsrSortedColInd,
                                              int                      blockDim,
                                              bsrilu02Info_t           info,
                                              int*                     pBufferSizeInBytes) {
#if HIP_VERSION >= 309
  return hipsparseZbsrilu02_bufferSize(handle, dirA, mb, nnzb, descrA, reinterpret_cast<hipDoubleComplex*>(bsrSortedVal), bsrSortedRowPtr, bsrSortedColInd, blockDim, info, pBufferSizeInBytes);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseSbsrilu02_analysis(cusparseHandle_t         handle,
                                            cusparseDirection_t      dirA,
                                            int                      mb,
                                            int                      nnzb,
                                            const cusparseMatDescr_t descrA,
                                            float*                   bsrSortedVal,
                                            const int*               bsrSortedRowPtr,
                                            const int*               bsrSortedColInd,
                                            int                      blockDim,
                                            bsrilu02Info_t           info,
                                            cusparseSolvePolicy_t    policy,
                                            void*                    pBuffer) {
#if HIP_VERSION >= 309
  return hipsparseSbsrilu02_analysis(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, policy, pBuffer);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseDbsrilu02_analysis(cusparseHandle_t         handle,
                                            cusparseDirection_t      dirA,
                                            int                      mb,
                                            int                      nnzb,
                                            const cusparseMatDescr_t descrA,
                                            double*                  bsrSortedVal,
                                            const int*               bsrSortedRowPtr,
                                            const int*               bsrSortedColInd,
                                            int                      blockDim,
                                            bsrilu02Info_t           info,
                                            cusparseSolvePolicy_t    policy,
                                            void*                    pBuffer) {
#if HIP_VERSION >= 309
  return hipsparseDbsrilu02_analysis(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, policy, pBuffer);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseCbsrilu02_analysis(cusparseHandle_t         handle,
                                            cusparseDirection_t      dirA,
                                            int                      mb,
                                            int                      nnzb,
                                            const cusparseMatDescr_t descrA,
                                            cuComplex*               bsrSortedVal,
                                            const int*               bsrSortedRowPtr,
                                            const int*               bsrSortedColInd,
                                            int                      blockDim,
                                            bsrilu02Info_t           info,
                                            cusparseSolvePolicy_t    policy,
                                            void*                    pBuffer) {
#if HIP_VERSION >= 309
  return hipsparseCbsrilu02_analysis(handle, dirA, mb, nnzb, descrA, reinterpret_cast<hipComplex*>(bsrSortedVal), bsrSortedRowPtr, bsrSortedColInd, blockDim, info, policy, pBuffer);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseZbsrilu02_analysis(cusparseHandle_t         handle,
                                            cusparseDirection_t      dirA,
                                            int                      mb,
                                            int                      nnzb,
                                            const cusparseMatDescr_t descrA,
                                            cuDoubleComplex*         bsrSortedVal,
                                            const int*               bsrSortedRowPtr,
                                            const int*               bsrSortedColInd,
                                            int                      blockDim,
                                            bsrilu02Info_t           info,
                                            cusparseSolvePolicy_t    policy,
                                            void*                    pBuffer) {
#if HIP_VERSION >= 309
  return hipsparseZbsrilu02_analysis(handle, dirA, mb, nnzb, descrA, reinterpret_cast<hipDoubleComplex*>(bsrSortedVal), bsrSortedRowPtr, bsrSortedColInd, blockDim, info, policy, pBuffer);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseSbsrilu02(cusparseHandle_t         handle,
                                   cusparseDirection_t      dirA,
                                   int                      mb,
                                   int                      nnzb,
                                   const cusparseMatDescr_t descrA,
                                   float*                   bsrSortedVal,
                                   const int*               bsrSortedRowPtr,
                                   const int*               bsrSortedColInd,
                                   int                      blockDim,
                                   bsrilu02Info_t           info,
                                   cusparseSolvePolicy_t    policy,
                                   void*                    pBuffer) {
#if HIP_VERSION >= 309
  return hipsparseSbsrilu02(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, policy, pBuffer);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseDbsrilu02(cusparseHandle_t         handle,
                                   cusparseDirection_t      dirA,
                                   int                      mb,
                                   int                      nnzb,
                                   const cusparseMatDescr_t descrA,
                                   double*                  bsrSortedVal,
                                   const int*               bsrSortedRowPtr,
                                   const int*               bsrSortedColInd,
                                   int                      blockDim,
                                   bsrilu02Info_t           info,
                                   cusparseSolvePolicy_t    policy,
                                   void*                    pBuffer) {
#if HIP_VERSION >= 309
  return hipsparseDbsrilu02(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, policy, pBuffer);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseCbsrilu02(cusparseHandle_t         handle,
                                   cusparseDirection_t      dirA,
                                   int                      mb,
                                   int                      nnzb,
                                   const cusparseMatDescr_t descrA,
                                   cuComplex*               bsrSortedVal,
                                   const int*               bsrSortedRowPtr,
                                   const int*               bsrSortedColInd,
                                   int                      blockDim,
                                   bsrilu02Info_t           info,
                                   cusparseSolvePolicy_t    policy,
                                   void*                    pBuffer) {
#if HIP_VERSION >= 309
  return hipsparseCbsrilu02(handle, dirA, mb, nnzb, descrA, reinterpret_cast<hipComplex*>(bsrSortedVal), bsrSortedRowPtr, bsrSortedColInd, blockDim, info, policy, pBuffer);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseZbsrilu02(cusparseHandle_t         handle,
                                   cusparseDirection_t      dirA,
                                   int                      mb,
                                   int                      nnzb,
                                   const cusparseMatDescr_t descrA,
                                   cuDoubleComplex*         bsrSortedVal,
                                   const int*               bsrSortedRowPtr,
                                   const int*               bsrSortedColInd,
                                   int                      blockDim,
                                   bsrilu02Info_t           info,
                                   cusparseSolvePolicy_t    policy,
                                   void*                    pBuffer) {
#if HIP_VERSION >= 309
  return hipsparseZbsrilu02(handle, dirA, mb, nnzb, descrA, reinterpret_cast<hipDoubleComplex*>(bsrSortedVal), bsrSortedRowPtr, bsrSortedColInd, blockDim, info, policy, pBuffer);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
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
  return hipsparseCcsric02_bufferSize(handle, m, nnz, descrA, reinterpret_cast<hipComplex*>(csrSortedValA), csrSortedRowPtrA, csrSortedColIndA, info, pBufferSizeInBytes);
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
  return hipsparseZcsric02_bufferSize(handle, m, nnz, descrA, reinterpret_cast<hipDoubleComplex*>(csrSortedValA), csrSortedRowPtrA, csrSortedColIndA, info, pBufferSizeInBytes);
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
  return hipsparseCcsric02_analysis(handle, m, nnz, descrA, reinterpret_cast<const hipComplex*>(csrSortedValA), csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer);
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
  return hipsparseZcsric02_analysis(handle, m, nnz, descrA, reinterpret_cast<const hipDoubleComplex*>(csrSortedValA), csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer);
}

cusparseStatus_t cusparseScsric02(cusparseHandle_t         handle,
                                  int                      m,
                                  int                      nnz,
                                  const cusparseMatDescr_t descrA,
                                  float*                   csrSortedValA_valM,
                                  const int*               csrSortedRowPtrA,
                                  const int*               csrSortedColIndA,
                                  csric02Info_t            info,
                                  cusparseSolvePolicy_t    policy,
                                  void*                    pBuffer) {
  return hipsparseScsric02(handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer);
}

cusparseStatus_t cusparseDcsric02(cusparseHandle_t         handle,
                                  int                      m,
                                  int                      nnz,
                                  const cusparseMatDescr_t descrA,
                                  double*                  csrSortedValA_valM,
                                  const int*               csrSortedRowPtrA,
                                  const int*               csrSortedColIndA,
                                  csric02Info_t            info,
                                  cusparseSolvePolicy_t    policy,
                                  void*                    pBuffer) {
  return hipsparseDcsric02(handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer);
}

cusparseStatus_t cusparseCcsric02(cusparseHandle_t         handle,
                                  int                      m,
                                  int                      nnz,
                                  const cusparseMatDescr_t descrA,
                                  cuComplex*               csrSortedValA_valM,
                                  const int*               csrSortedRowPtrA,
                                  const int*               csrSortedColIndA,
                                  csric02Info_t            info,
                                  cusparseSolvePolicy_t    policy,
                                  void*                    pBuffer) {
  return hipsparseCcsric02(handle, m, nnz, descrA, reinterpret_cast<hipComplex*>(csrSortedValA_valM), csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer);
}

cusparseStatus_t cusparseZcsric02(cusparseHandle_t         handle,
                                  int                      m,
                                  int                      nnz,
                                  const cusparseMatDescr_t descrA,
                                  cuDoubleComplex*         csrSortedValA_valM,
                                  const int*               csrSortedRowPtrA,
                                  const int*               csrSortedColIndA,
                                  csric02Info_t            info,
                                  cusparseSolvePolicy_t    policy,
                                  void*                    pBuffer) {
  return hipsparseZcsric02(handle, m, nnz, descrA, reinterpret_cast<hipDoubleComplex*>(csrSortedValA_valM), csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer);
}

cusparseStatus_t cusparseXbsric02_zeroPivot(cusparseHandle_t handle,
                                            bsric02Info_t    info,
                                            int*             position) {
#if HIP_VERSION >= 308
  return hipsparseXbsric02_zeroPivot(handle, info, position);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseSbsric02_bufferSize(cusparseHandle_t         handle,
                                             cusparseDirection_t      dirA,
                                             int                      mb,
                                             int                      nnzb,
                                             const cusparseMatDescr_t descrA,
                                             float*                   bsrSortedVal,
                                             const int*               bsrSortedRowPtr,
                                             const int*               bsrSortedColInd,
                                             int                      blockDim,
                                             bsric02Info_t            info,
                                             int*                     pBufferSizeInBytes) {
#if HIP_VERSION >= 308
  return hipsparseSbsric02_bufferSize(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, pBufferSizeInBytes);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseDbsric02_bufferSize(cusparseHandle_t         handle,
                                             cusparseDirection_t      dirA,
                                             int                      mb,
                                             int                      nnzb,
                                             const cusparseMatDescr_t descrA,
                                             double*                  bsrSortedVal,
                                             const int*               bsrSortedRowPtr,
                                             const int*               bsrSortedColInd,
                                             int                      blockDim,
                                             bsric02Info_t            info,
                                             int*                     pBufferSizeInBytes) {
#if HIP_VERSION >= 308
  return hipsparseDbsric02_bufferSize(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, pBufferSizeInBytes);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseCbsric02_bufferSize(cusparseHandle_t         handle,
                                             cusparseDirection_t      dirA,
                                             int                      mb,
                                             int                      nnzb,
                                             const cusparseMatDescr_t descrA,
                                             cuComplex*               bsrSortedVal,
                                             const int*               bsrSortedRowPtr,
                                             const int*               bsrSortedColInd,
                                             int                      blockDim,
                                             bsric02Info_t            info,
                                             int*                     pBufferSizeInBytes) {
#if HIP_VERSION >= 308
  return hipsparseCbsric02_bufferSize(handle, dirA, mb, nnzb, descrA, reinterpret_cast<hipComplex*>(bsrSortedVal), bsrSortedRowPtr, bsrSortedColInd, blockDim, info, pBufferSizeInBytes);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseZbsric02_bufferSize(cusparseHandle_t         handle,
                                             cusparseDirection_t      dirA,
                                             int                      mb,
                                             int                      nnzb,
                                             const cusparseMatDescr_t descrA,
                                             cuDoubleComplex*         bsrSortedVal,
                                             const int*               bsrSortedRowPtr,
                                             const int*               bsrSortedColInd,
                                             int                      blockDim,
                                             bsric02Info_t            info,
                                             int*                     pBufferSizeInBytes) {
#if HIP_VERSION >= 308
  return hipsparseZbsric02_bufferSize(handle, dirA, mb, nnzb, descrA, reinterpret_cast<hipDoubleComplex*>(bsrSortedVal), bsrSortedRowPtr, bsrSortedColInd, blockDim, info, pBufferSizeInBytes);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseSbsric02_analysis(cusparseHandle_t         handle,
                                           cusparseDirection_t      dirA,
                                           int                      mb,
                                           int                      nnzb,
                                           const cusparseMatDescr_t descrA,
                                           const float*             bsrSortedVal,
                                           const int*               bsrSortedRowPtr,
                                           const int*               bsrSortedColInd,
                                           int                      blockDim,
                                           bsric02Info_t            info,
                                           cusparseSolvePolicy_t    policy,
                                           void*                    pInputBuffer) {
#if HIP_VERSION >= 308
  return hipsparseSbsric02_analysis(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, policy, pInputBuffer);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseDbsric02_analysis(cusparseHandle_t         handle,
                                           cusparseDirection_t      dirA,
                                           int                      mb,
                                           int                      nnzb,
                                           const cusparseMatDescr_t descrA,
                                           const double*            bsrSortedVal,
                                           const int*               bsrSortedRowPtr,
                                           const int*               bsrSortedColInd,
                                           int                      blockDim,
                                           bsric02Info_t            info,
                                           cusparseSolvePolicy_t    policy,
                                           void*                    pInputBuffer) {
#if HIP_VERSION >= 308
  return hipsparseDbsric02_analysis(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, policy, pInputBuffer);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseCbsric02_analysis(cusparseHandle_t         handle,
                                           cusparseDirection_t      dirA,
                                           int                      mb,
                                           int                      nnzb,
                                           const cusparseMatDescr_t descrA,
                                           const cuComplex*         bsrSortedVal,
                                           const int*               bsrSortedRowPtr,
                                           const int*               bsrSortedColInd,
                                           int                      blockDim,
                                           bsric02Info_t            info,
                                           cusparseSolvePolicy_t    policy,
                                           void*                    pInputBuffer) {
#if HIP_VERSION >= 308
  return hipsparseCbsric02_analysis(handle, dirA, mb, nnzb, descrA, reinterpret_cast<const hipComplex*>(bsrSortedVal), bsrSortedRowPtr, bsrSortedColInd, blockDim, info, policy, pInputBuffer);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseZbsric02_analysis(cusparseHandle_t         handle,
                                           cusparseDirection_t      dirA,
                                           int                      mb,
                                           int                      nnzb,
                                           const cusparseMatDescr_t descrA,
                                           const cuDoubleComplex*   bsrSortedVal,
                                           const int*               bsrSortedRowPtr,
                                           const int*               bsrSortedColInd,
                                           int                      blockDim,
                                           bsric02Info_t            info,
                                           cusparseSolvePolicy_t    policy,
                                           void*                    pInputBuffer) {
#if HIP_VERSION >= 308
  return hipsparseZbsric02_analysis(handle, dirA, mb, nnzb, descrA, reinterpret_cast<const hipDoubleComplex*>(bsrSortedVal), bsrSortedRowPtr, bsrSortedColInd, blockDim, info, policy, pInputBuffer);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseSbsric02(cusparseHandle_t         handle,
                                  cusparseDirection_t      dirA,
                                  int                      mb,
                                  int                      nnzb,
                                  const cusparseMatDescr_t descrA,
                                  float*                   bsrSortedVal,
                                  const int*               bsrSortedRowPtr,
                                  const int*               bsrSortedColInd,
                                  int                      blockDim,
                                  bsric02Info_t            info,
                                  cusparseSolvePolicy_t    policy,
                                  void*                    pBuffer) {
#if HIP_VERSION >= 308
  return hipsparseSbsric02(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, policy, pBuffer);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseDbsric02(cusparseHandle_t         handle,
                                  cusparseDirection_t      dirA,
                                  int                      mb,
                                  int                      nnzb,
                                  const cusparseMatDescr_t descrA,
                                  double*                  bsrSortedVal,
                                  const int*               bsrSortedRowPtr,
                                  const int*               bsrSortedColInd,
                                  int                      blockDim,
                                  bsric02Info_t            info,
                                  cusparseSolvePolicy_t    policy,
                                  void*                    pBuffer) {
#if HIP_VERSION >= 308
  return hipsparseDbsric02(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, policy, pBuffer);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseCbsric02(cusparseHandle_t         handle,
                                  cusparseDirection_t      dirA,
                                  int                      mb,
                                  int                      nnzb,
                                  const cusparseMatDescr_t descrA,
                                  cuComplex*               bsrSortedVal,
                                  const int*               bsrSortedRowPtr,
                                  const int*
                                       bsrSortedColInd,
                                  int                      blockDim,
                                  bsric02Info_t            info,
                                  cusparseSolvePolicy_t    policy,
                                  void*                    pBuffer) {
#if HIP_VERSION >= 308
  return hipsparseCbsric02(handle, dirA, mb, nnzb, descrA, reinterpret_cast<hipComplex*>(bsrSortedVal), bsrSortedRowPtr, bsrSortedColInd, blockDim, info, policy, pBuffer);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseZbsric02(cusparseHandle_t         handle,
                                  cusparseDirection_t      dirA,
                                  int                      mb,
                                  int                      nnzb,
                                  const cusparseMatDescr_t descrA,
                                  cuDoubleComplex*         bsrSortedVal,
                                  const int*               bsrSortedRowPtr,
                                  const int*               bsrSortedColInd,
                                  int                      blockDim,
                                  bsric02Info_t            info,
                                  cusparseSolvePolicy_t    policy,
                                  void*                    pBuffer) {
#if HIP_VERSION >= 308
  return hipsparseZbsric02(handle, dirA, mb, nnzb, descrA, reinterpret_cast<hipDoubleComplex*>(bsrSortedVal), bsrSortedRowPtr, bsrSortedColInd, blockDim, info, policy, pBuffer);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseSgtsv2_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

cusparseStatus_t cusparseDgtsv2_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

cusparseStatus_t cusparseCgtsv2_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

cusparseStatus_t cusparseZgtsv2_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

cusparseStatus_t cusparseSgtsv2(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

cusparseStatus_t cusparseDgtsv2(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

cusparseStatus_t cusparseCgtsv2(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

cusparseStatus_t cusparseZgtsv2(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

cusparseStatus_t cusparseSgtsv2_nopivot_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

cusparseStatus_t cusparseDgtsv2_nopivot_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

cusparseStatus_t cusparseCgtsv2_nopivot_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

cusparseStatus_t cusparseZgtsv2_nopivot_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

cusparseStatus_t cusparseSgtsv2_nopivot(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

cusparseStatus_t cusparseDgtsv2_nopivot(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

cusparseStatus_t cusparseCgtsv2_nopivot(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

cusparseStatus_t cusparseZgtsv2_nopivot(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

cusparseStatus_t cusparseSgtsv2StridedBatch_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

cusparseStatus_t cusparseDgtsv2StridedBatch_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

cusparseStatus_t cusparseCgtsv2StridedBatch_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

cusparseStatus_t cusparseZgtsv2StridedBatch_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

cusparseStatus_t cusparseSgtsv2StridedBatch(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

cusparseStatus_t cusparseDgtsv2StridedBatch(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

cusparseStatus_t cusparseCgtsv2StridedBatch(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

cusparseStatus_t cusparseZgtsv2StridedBatch(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

cusparseStatus_t cusparseSgtsvInterleavedBatch_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

cusparseStatus_t cusparseDgtsvInterleavedBatch_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

cusparseStatus_t cusparseCgtsvInterleavedBatch_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

cusparseStatus_t cusparseZgtsvInterleavedBatch_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

cusparseStatus_t cusparseSgtsvInterleavedBatch(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

cusparseStatus_t cusparseDgtsvInterleavedBatch(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

cusparseStatus_t cusparseCgtsvInterleavedBatch(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

cusparseStatus_t cusparseZgtsvInterleavedBatch(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

cusparseStatus_t cusparseSgpsvInterleavedBatch_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

cusparseStatus_t cusparseDgpsvInterleavedBatch_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

cusparseStatus_t cusparseCgpsvInterleavedBatch_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

cusparseStatus_t cusparseZgpsvInterleavedBatch_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

cusparseStatus_t cusparseSgpsvInterleavedBatch(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

cusparseStatus_t cusparseDgpsvInterleavedBatch(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

cusparseStatus_t cusparseCgpsvInterleavedBatch(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

cusparseStatus_t cusparseZgpsvInterleavedBatch(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

#define CUSPARSE_VERSION (hipsparseVersionMajor*100000+hipsparseVersionMinor*100+hipsparseVersionPatch)

// cuSPARSE generic API
#if HIP_VERSION >= 402
typedef hipsparseSpVecDescr_t cusparseSpVecDescr_t;
#else
typedef void* cusparseSpVecDescr_t;
#endif

#if HIP_VERSION >= 402
typedef hipsparseDnVecDescr_t cusparseDnVecDescr_t;
#else
typedef void* cusparseDnVecDescr_t;
#endif

#if HIP_VERSION >= 402
typedef hipsparseSpMatDescr_t cusparseSpMatDescr_t;
#else
typedef void* cusparseSpMatDescr_t;
#endif

#if HIP_VERSION >= 402
typedef hipsparseDnMatDescr_t cusparseDnMatDescr_t;
#else
typedef void* cusparseDnMatDescr_t;
#endif


#if HIP_VERSION >= 402
typedef hipsparseIndexType_t cusparseIndexType_t;
#else
typedef enum {} cusparseIndexType_t;
#endif

#if HIP_VERSION >= 402
typedef hipsparseFormat_t cusparseFormat_t;
#else
typedef enum {} cusparseFormat_t;
#endif


#if HIP_VERSION >= 402
typedef enum {} cusparseOrder_t;
static hipsparseOrder_t convert_hipsparseOrder_t(cusparseOrder_t type) {
    switch(static_cast<int>(type)) {
        case 1 /* CUSPARSE_ORDER_COL */: return HIPSPARSE_ORDER_COLUMN;
        case 2 /* CUSPARSE_ORDER_ROW */: return HIPSPARSE_ORDER_ROW;
        default: throw std::runtime_error("unrecognized type");
    }
}

#else
typedef enum {} cusparseOrder_t;
#endif

#if HIP_VERSION >= 402
typedef hipsparseSpMVAlg_t cusparseSpMVAlg_t;
#else
typedef enum {} cusparseSpMVAlg_t;
#endif

#if HIP_VERSION >= 50000000
typedef hipsparseSpMatAttribute_t cusparseSpMatAttribute_t;
typedef hipsparseSpSMAlg_t cusparseSpSMAlg_t;
typedef hipsparseSpSMDescr_t cusparseSpSMDescr_t;
#else
typedef enum {} cusparseSpMatAttribute_t;
typedef enum {} cusparseSpSMAlg_t;
typedef void * cusparseSpSMDescr_t;
#endif

#if HIP_VERSION >= 402
typedef hipsparseSpMMAlg_t cusparseSpMMAlg_t;
#else
typedef enum {} cusparseSpMMAlg_t;
#endif

#if HIP_VERSION >= 402
typedef hipsparseSparseToDenseAlg_t cusparseSparseToDenseAlg_t;
#else
typedef enum {} cusparseSparseToDenseAlg_t;
#endif

#if HIP_VERSION >= 402
typedef hipsparseDenseToSparseAlg_t cusparseDenseToSparseAlg_t;
#else
typedef enum {} cusparseDenseToSparseAlg_t;
#endif


cusparseStatus_t cusparseCreateSpVec(cusparseSpVecDescr_t* spVecDescr,
                                     int64_t               size,
                                     int64_t               nnz,
                                     void*                 indices,
                                     void*                 values,
                                     cusparseIndexType_t   idxType,
                                     cusparseIndexBase_t   idxBase,
                                     cudaDataType          valueType) {
#if HIP_VERSION >= 402
  hipDataType blah = convert_hipDatatype(valueType);
  return hipsparseCreateSpVec(spVecDescr, size, nnz, indices, values, idxType, idxBase, blah);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseDestroySpVec(cusparseSpVecDescr_t spVecDescr) {
#if HIP_VERSION >= 402
  return hipsparseDestroySpVec(spVecDescr);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseSpVecGet(cusparseSpVecDescr_t spVecDescr,
                                  int64_t*             size,
                                  int64_t*             nnz,
                                  void**               indices,
                                  void**               values,
                                  cusparseIndexType_t* idxType,
                                  cusparseIndexBase_t* idxBase,
                                  cudaDataType*        valueType) {
#if HIP_VERSION >= 402
  return hipsparseSpVecGet(spVecDescr, size, nnz, indices, values, idxType, idxBase, reinterpret_cast<hipDataType*>(valueType));
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseSpVecGetIndexBase(cusparseSpVecDescr_t spVecDescr,
                                           cusparseIndexBase_t* idxBase) {
#if HIP_VERSION >= 402
  return hipsparseSpVecGetIndexBase(spVecDescr, idxBase);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseSpVecGetValues(cusparseSpVecDescr_t spVecDescr,
                                        void**               values) {
#if HIP_VERSION >= 402
  return hipsparseSpVecGetValues(spVecDescr, values);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseSpVecSetValues(cusparseSpVecDescr_t spVecDescr,
                                        void*                values) {
#if HIP_VERSION >= 402
  return hipsparseSpVecSetValues(spVecDescr, values);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseCreateCoo(cusparseSpMatDescr_t* spMatDescr,
                                   int64_t               rows,
                                   int64_t               cols,
                                   int64_t               nnz,
                                   void*                 cooRowInd,
                                   void*                 cooColInd,
                                   void*                 cooValues,
                                   cusparseIndexType_t   cooIdxType,
                                   cusparseIndexBase_t   idxBase,
                                   cudaDataType          valueType) {
#if HIP_VERSION >= 402
  hipDataType blah = convert_hipDatatype(valueType);
  return hipsparseCreateCoo(spMatDescr, rows, cols, nnz, cooRowInd, cooColInd, cooValues, cooIdxType, idxBase, blah);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseCreateCooAoS(cusparseSpMatDescr_t* spMatDescr,
                                      int64_t               rows,
                                      int64_t               cols,
                                      int64_t               nnz,
                                      void*                 cooInd,
                                      void*                 cooValues,
                                      cusparseIndexType_t   cooIdxType,
                                      cusparseIndexBase_t   idxBase,
                                      cudaDataType          valueType) {
#if HIP_VERSION >= 402
  hipDataType blah = convert_hipDatatype(valueType);
  return hipsparseCreateCooAoS(spMatDescr, rows, cols, nnz, cooInd, cooValues, cooIdxType, idxBase, blah);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseCreateCsr(cusparseSpMatDescr_t* spMatDescr,
                                   int64_t               rows,
                                   int64_t               cols,
                                   int64_t               nnz,
                                   void*                 csrRowOffsets,
                                   void*                 csrColInd,
                                   void*                 csrValues,
                                   cusparseIndexType_t   csrRowOffsetsType,
                                   cusparseIndexType_t   csrColIndType,
                                   cusparseIndexBase_t   idxBase,
                                   cudaDataType          valueType) {
#if HIP_VERSION >= 402
  hipDataType blah = convert_hipDatatype(valueType);
  return hipsparseCreateCsr(spMatDescr, rows, cols, nnz, csrRowOffsets, csrColInd, csrValues, csrRowOffsetsType, csrColIndType, idxBase, blah);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseCreateCsc(cusparseSpMatDescr_t* spMatDescr,
                                   int64_t               rows,
                                   int64_t               cols,
                                   int64_t               nnz,
                                   void*                 cscColOffsets,
                                   void*                 cscRowInd,
                                   void*                 cscValues,
                                   cusparseIndexType_t   cscColOffsetsType,
                                   cusparseIndexType_t   cscRowIndType,
                                   cusparseIndexBase_t   idxBase,
                                   cudaDataType          valueType) {
#if HIP_VERSION >= 402
  hipDataType blah = convert_hipDatatype(valueType);
  return hipsparseCreateCsc(spMatDescr, rows, cols, nnz, cscColOffsets, cscRowInd, cscValues, cscColOffsetsType, cscRowIndType, idxBase, blah);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseDestroySpMat(cusparseSpMatDescr_t spMatDescr) {
#if HIP_VERSION >= 402
  return hipsparseDestroySpMat(spMatDescr);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseCooGet(cusparseSpMatDescr_t spMatDescr,
                                int64_t*             rows,
                                int64_t*             cols,
                                int64_t*             nnz,
                                void**               cooRowInd,  // COO row indices
                                void**               cooColInd,  // COO column indices
                                void**               cooValues,  // COO values
                                cusparseIndexType_t* idxType,
                                cusparseIndexBase_t* idxBase,
                                cudaDataType*        valueType) {
#if HIP_VERSION >= 402
  return hipsparseCooGet(spMatDescr, rows, cols, nnz, cooRowInd, cooColInd, cooValues, idxType, idxBase, reinterpret_cast<hipDataType*>(valueType));
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseCooAoSGet(cusparseSpMatDescr_t spMatDescr,
                                   int64_t*             rows,
                                   int64_t*             cols,
                                   int64_t*             nnz,
                                   void**               cooInd,     // COO indices
                                   void**               cooValues,  // COO values
                                   cusparseIndexType_t* idxType,
                                   cusparseIndexBase_t* idxBase,
                                   cudaDataType*        valueType) {
#if HIP_VERSION >= 402
  return hipsparseCooAoSGet(spMatDescr, rows, cols, nnz, cooInd, cooValues, idxType, idxBase, reinterpret_cast<hipDataType*>(valueType));
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseCsrGet(cusparseSpMatDescr_t spMatDescr,
                                int64_t*             rows,
                                int64_t*             cols,
                                int64_t*             nnz,
                                void**               csrRowOffsets,
                                void**               csrColInd,
                                void**               csrValues,
                                cusparseIndexType_t* csrRowOffsetsType,
                                cusparseIndexType_t* csrColIndType,
                                cusparseIndexBase_t* idxBase,
                                cudaDataType*        valueType) {
#if HIP_VERSION >= 402
  return hipsparseCsrGet(spMatDescr, rows, cols, nnz, csrRowOffsets, csrColInd, csrValues, csrRowOffsetsType, csrColIndType, idxBase, reinterpret_cast<hipDataType*>(valueType));
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseCsrSetPointers(cusparseSpMatDescr_t spMatDescr,
                                        void*                csrRowOffsets,
                                        void*                csrColInd,
                                        void*                csrValues) {
#if HIP_VERSION >= 402
  return hipsparseCsrSetPointers(spMatDescr, csrRowOffsets, csrColInd, csrValues);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseSpMatGetSize(cusparseSpMatDescr_t spMatDescr,
                                      int64_t*             rows,
                                      int64_t*             cols,
                                      int64_t*             nnz) {
#if HIP_VERSION >= 402
  return hipsparseSpMatGetSize(spMatDescr, rows, cols, nnz);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseSpMatGetFormat(cusparseSpMatDescr_t spMatDescr,
                                        cusparseFormat_t*    format) {
#if HIP_VERSION >= 402
  return hipsparseSpMatGetFormat(spMatDescr, format);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseSpMatGetIndexBase(cusparseSpMatDescr_t spMatDescr,
                                           cusparseIndexBase_t* idxBase) {
#if HIP_VERSION >= 402
  return hipsparseSpMatGetIndexBase(spMatDescr, idxBase);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseSpMatGetValues(cusparseSpMatDescr_t spMatDescr,
                                        void**               values) {
#if HIP_VERSION >= 402
  return hipsparseSpMatGetValues(spMatDescr, values);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseSpMatSetValues(cusparseSpMatDescr_t spMatDescr,
                                        void*                values) {
#if HIP_VERSION >= 402
  return hipsparseSpMatSetValues(spMatDescr, values);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseSpMatGetStridedBatch(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

cusparseStatus_t cusparseSpMatSetStridedBatch(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

cusparseStatus_t cusparseSpMatSetAttribute(cusparseSpMatDescr_t     spMatDescr,
                                           cusparseSpMatAttribute_t attribute,
                                           void*                    data,
                                           size_t                   dataSize) {
#if HIP_VERSION >= 50000000
  return hipsparseSpMatSetAttribute(spMatDescr, attribute, data, dataSize);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseCreateDnVec(cusparseDnVecDescr_t* dnVecDescr,
                                     int64_t               size,
                                     void*                 values,
                                     cudaDataType          valueType) {
#if HIP_VERSION >= 402
  hipDataType blah = convert_hipDatatype(valueType);
  return hipsparseCreateDnVec(dnVecDescr, size, values, blah);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseDestroyDnVec(cusparseDnVecDescr_t dnVecDescr) {
#if HIP_VERSION >= 402
  return hipsparseDestroyDnVec(dnVecDescr);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseDnVecGet(cusparseDnVecDescr_t dnVecDescr,
                                  int64_t*             size,
                                  void**               values,
                                  cudaDataType*        valueType) {
#if HIP_VERSION >= 402
  return hipsparseDnVecGet(dnVecDescr, size, values, reinterpret_cast<hipDataType*>(valueType));
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseDnVecGetValues(cusparseDnVecDescr_t dnVecDescr,
                                        void**               values) {
#if HIP_VERSION >= 402
  return hipsparseDnVecGetValues(dnVecDescr, values);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseDnVecSetValues(cusparseDnVecDescr_t dnVecDescr,
                                        void*                values) {
#if HIP_VERSION >= 402
  return hipsparseDnVecSetValues(dnVecDescr, values);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseCreateDnMat(cusparseDnMatDescr_t* dnMatDescr,
                                     int64_t               rows,
                                     int64_t               cols,
                                     int64_t               ld,
                                     void*                 values,
                                     cudaDataType          valueType,
                                     cusparseOrder_t       order) {
#if HIP_VERSION >= 402
  hipDataType blah = convert_hipDatatype(valueType);
  hipsparseOrder_t blah2 = convert_hipsparseOrder_t(order);
  return hipsparseCreateDnMat(dnMatDescr, rows, cols, ld, values, blah, blah2);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseDestroyDnMat(cusparseDnMatDescr_t dnMatDescr) {
#if HIP_VERSION >= 402
  return hipsparseDestroyDnMat(dnMatDescr);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseDnMatGet(cusparseDnMatDescr_t dnMatDescr,
                                  int64_t*             rows,
                                  int64_t*             cols,
                                  int64_t*             ld,
                                  void**               values,
                                  cudaDataType*        type,
                                  cusparseOrder_t*     order) {
#if HIP_VERSION >= 402
  hipsparseOrder_t blah2 = convert_hipsparseOrder_t(*order);
  return hipsparseDnMatGet(dnMatDescr, rows, cols, ld, values, reinterpret_cast<hipDataType*>(type), &blah2);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseDnMatGetValues(cusparseDnMatDescr_t dnMatDescr,
                                        void**               values) {
#if HIP_VERSION >= 402
  return hipsparseDnMatGetValues(dnMatDescr, values);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseDnMatSetValues(cusparseDnMatDescr_t dnMatDescr,
                                        void*                values) {
#if HIP_VERSION >= 402
  return hipsparseDnMatSetValues(dnMatDescr, values);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseDnMatGetStridedBatch(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

cusparseStatus_t cusparseDnMatSetStridedBatch(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

cusparseStatus_t cusparseSpVV_bufferSize(cusparseHandle_t     handle,
                                         cusparseOperation_t  opX,
                                         cusparseSpVecDescr_t vecX,
                                         cusparseDnVecDescr_t vecY,
                                         const void*          result,
                                         cudaDataType         computeType,
                                         size_t*              bufferSize) {
#if HIP_VERSION >= 402
  hipDataType blah = convert_hipDatatype(computeType);
  return hipsparseSpVV_bufferSize(handle, opX, vecX, vecY, const_cast<void*>(result), blah, bufferSize);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseSpVV(cusparseHandle_t     handle,
                              cusparseOperation_t  opX,
                              cusparseSpVecDescr_t vecX,
                              cusparseDnVecDescr_t vecY,
                              void*                result,
                              cudaDataType         computeType,
                              void*                externalBuffer) {
#if HIP_VERSION >= 402
  hipDataType blah = convert_hipDatatype(computeType);
  return hipsparseSpVV(handle, opX, vecX, vecY, result, blah, externalBuffer);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseSpMV_bufferSize(cusparseHandle_t    handle,
                                         cusparseOperation_t opA,
                                         const void*         alpha,
                                         cusparseSpMatDescr_t matA,
                                         cusparseDnVecDescr_t vecX,
                                         const void*          beta,
                                         cusparseDnVecDescr_t vecY,
                                         cudaDataType         computeType,
                                         cusparseSpMVAlg_t    alg,
                                         size_t*              bufferSize) {
#if HIP_VERSION >= 402
  hipDataType blah = convert_hipDatatype(computeType);
  return hipsparseSpMV_bufferSize(handle, opA, alpha, matA, vecX, beta, vecY, blah, alg, bufferSize);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseSpMV(cusparseHandle_t     handle,
                              cusparseOperation_t  opA,
                              const void*          alpha,
                              cusparseSpMatDescr_t matA,
                              cusparseDnVecDescr_t vecX,
                              const void*          beta,
                              cusparseDnVecDescr_t vecY,
                              cudaDataType         computeType,
                              cusparseSpMVAlg_t    alg,
                              void*                externalBuffer) {
#if HIP_VERSION >= 402
  hipDataType blah = convert_hipDatatype(computeType);
  return hipsparseSpMV(handle, opA, alpha, matA, vecX, beta, vecY, blah, alg, externalBuffer);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseSpSM_createDescr(cusparseSpSMDescr_t* descr) {
#if HIP_VERSION >= 50000000
  return hipsparseSpSM_createDescr(descr);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseSpSM_destroyDescr(cusparseSpSMDescr_t descr) {
#if HIP_VERSION >= 50000000
  return hipsparseSpSM_destroyDescr(descr);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseSpSM_bufferSize(cusparseHandle_t     handle,
                                         cusparseOperation_t  opA,
                                         cusparseOperation_t  opB,
                                         const void*          alpha,
                                         cusparseSpMatDescr_t matA,
                                         cusparseDnMatDescr_t matB,
                                         cusparseDnMatDescr_t matC,
                                         cudaDataType         computeType,
                                         cusparseSpSMAlg_t    alg,
                                         cusparseSpSMDescr_t  spsmDescr,
                                         size_t*              bufferSize) {
#if HIP_VERSION >= 50000000
  hipDataType computeType1 = convert_hipDatatype(computeType);
  return hipsparseSpSM_bufferSize(handle, opA, opB, alpha, matA, matB, matC, computeType1, alg, spsmDescr, bufferSize);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseSpSM_analysis(cusparseHandle_t     handle,
                                       cusparseOperation_t  opA,
                                       cusparseOperation_t  opB,
                                       const void*          alpha,
                                       cusparseSpMatDescr_t matA,
                                       cusparseDnMatDescr_t matB,
                                       cusparseDnMatDescr_t matC,
                                       cudaDataType         computeType,
                                       cusparseSpSMAlg_t    alg,
                                       cusparseSpSMDescr_t  spsmDescr,
                                       void*                externalBuffer) {
#if HIP_VERSION >= 50000000
  hipDataType computeType1 = convert_hipDatatype(computeType);
  return hipsparseSpSM_analysis(handle, opA, opB, alpha, matA, matB, matC, computeType1, alg, spsmDescr, externalBuffer);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

// See cusparse.pyx for a comment
cusparseStatus_t cusparseSpSM_solve(cusparseHandle_t     handle,
                                    cusparseOperation_t  opA,
                                    cusparseOperation_t  opB,
                                    const void*          alpha,
                                    cusparseSpMatDescr_t matA,
                                    cusparseDnMatDescr_t matB,
                                    cusparseDnMatDescr_t matC,
                                    cudaDataType         computeType,
                                    cusparseSpSMAlg_t    alg,
                                    cusparseSpSMDescr_t  spsmDescr,
                                    void*                externalBuffer) {
#if HIP_VERSION >= 50000000
  hipDataType computeType1 = convert_hipDatatype(computeType);
  return hipsparseSpSM_solve(handle, opA, opB, alpha, matA, matB, matC, computeType1, alg, spsmDescr, externalBuffer);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseSpMM_bufferSize(cusparseHandle_t     handle,
                                         cusparseOperation_t  opA,
                                         cusparseOperation_t  opB,
                                         const void*          alpha,
                                         cusparseSpMatDescr_t matA,
                                         cusparseDnMatDescr_t matB,
                                         const void*          beta,
                                         cusparseDnMatDescr_t matC,
                                         cudaDataType         computeType,
                                         cusparseSpMMAlg_t    alg,
                                         size_t*              bufferSize) {
#if HIP_VERSION >= 402
  hipDataType blah = convert_hipDatatype(computeType);
  return hipsparseSpMM_bufferSize(handle, opA, opB, alpha, matA, matB, beta, matC, blah, alg, bufferSize);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseSpMM(cusparseHandle_t     handle,
                              cusparseOperation_t  opA,
                              cusparseOperation_t  opB,
                              const void*          alpha,
                              cusparseSpMatDescr_t matA,
                              cusparseDnMatDescr_t matB,
                              const void*          beta,
                              cusparseDnMatDescr_t matC,
                              cudaDataType         computeType,
                              cusparseSpMMAlg_t    alg,
                              void*                externalBuffer) {
#if HIP_VERSION >= 402
  hipDataType blah = convert_hipDatatype(computeType);
  return hipsparseSpMM(handle, opA, opB, alpha, matA, matB, beta, matC, blah, alg, externalBuffer);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseConstrainedGeMM_bufferSize(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

cusparseStatus_t cusparseConstrainedGeMM(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

cusparseStatus_t cusparseSparseToDense_bufferSize(cusparseHandle_t           handle,
                                                  cusparseSpMatDescr_t       matA,
                                                  cusparseDnMatDescr_t       matB,
                                                  cusparseSparseToDenseAlg_t alg,
                                                  size_t*                    bufferSize) {
#if HIP_VERSION >= 402
  return hipsparseSparseToDense_bufferSize(handle, matA, matB, alg, bufferSize);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseSparseToDense(cusparseHandle_t           handle,
                                       cusparseSpMatDescr_t       matA,
                                       cusparseDnMatDescr_t       matB,
                                       cusparseSparseToDenseAlg_t alg,
                                       void*                      buffer) {
#if HIP_VERSION >= 402
  return hipsparseSparseToDense(handle, matA, matB, alg, buffer);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseDenseToSparse_bufferSize(cusparseHandle_t           handle,
                                                  cusparseDnMatDescr_t       matA,
                                                  cusparseSpMatDescr_t       matB,
                                                  cusparseDenseToSparseAlg_t alg,
                                                  size_t*                    bufferSize) {
#if HIP_VERSION >= 402
  return hipsparseDenseToSparse_bufferSize(handle, matA, matB, alg, bufferSize);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseDenseToSparse_analysis(cusparseHandle_t           handle,
                                                cusparseDnMatDescr_t       matA,
                                                cusparseSpMatDescr_t       matB,
                                                cusparseDenseToSparseAlg_t alg,
                                                void*                      buffer) {
#if HIP_VERSION >= 402
  return hipsparseDenseToSparse_analysis(handle, matA, matB, alg, buffer);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

cusparseStatus_t cusparseDenseToSparse_convert(cusparseHandle_t           handle,
                                               cusparseDnMatDescr_t       matA,
                                               cusparseSpMatDescr_t       matB,
                                               cusparseDenseToSparseAlg_t alg,
                                               void*                      buffer) {
#if HIP_VERSION >= 402
  return hipsparseDenseToSparse_convert(handle, matA, matB, alg, buffer);
#else
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

typedef enum {} cusparseCsr2CscAlg_t;

cusparseStatus_t cusparseCsr2cscEx2_bufferSize(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

cusparseStatus_t cusparseCsr2cscEx2(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

typedef void* cusparseSpGEMMDescr_t;
typedef enum {} cusparseSpGEMMAlg_t;

cusparseStatus_t cusparseSpGEMM_createDescr(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

cusparseStatus_t cusparseSpGEMM_destroyDescr(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

cusparseStatus_t cusparseSpGEMM_workEstimation(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

cusparseStatus_t cusparseSpGEMM_compute(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

cusparseStatus_t cusparseSpGEMM_copy(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

cusparseStatus_t cusparseGather(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
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
  hipComplex blah;
  blah.x=tol.x;
  blah.y=tol.y;
  return hipsparseCnnz_compress(handle, m, descr, reinterpret_cast<const hipComplex*>(csrSortedValA), csrSortedRowPtrA, nnzPerRow, nnzC, blah);
}

cusparseStatus_t cusparseZnnz_compress(cusparseHandle_t         handle,
                                       int                      m,
                                       const cusparseMatDescr_t descr,
                                       const cuDoubleComplex*   csrSortedValA,
                                       const int*               csrSortedRowPtrA,
                                       int*                     nnzPerRow,
                                       int*                     nnzC,
                                       cuDoubleComplex          tol) {
  hipDoubleComplex blah;
  blah.x=tol.x;
  blah.y=tol.y;
  return hipsparseZnnz_compress(handle, m, descr, reinterpret_cast<const hipDoubleComplex*>(csrSortedValA), csrSortedRowPtrA, nnzPerRow, nnzC, blah);
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
  hipComplex blah;
  blah.x=tol.x;
  blah.y=tol.y;
  return hipsparseCcsr2csr_compress(handle, m, n, descrA, reinterpret_cast<const hipComplex*>(csrSortedValA), csrSortedColIndA, csrSortedRowPtrA, nnzA, nnzPerRow, reinterpret_cast<hipComplex*>(csrSortedValC), csrSortedColIndC, csrSortedRowPtrC, blah);
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
  hipDoubleComplex blah;
  blah.x=tol.x;
  blah.y=tol.y;
  return hipsparseZcsr2csr_compress(handle, m, n, descrA, reinterpret_cast<const hipDoubleComplex*>(csrSortedValA), csrSortedColIndA, csrSortedRowPtrA, nnzA, nnzPerRow, reinterpret_cast<hipDoubleComplex*>(csrSortedValC), csrSortedColIndC, csrSortedRowPtrC, blah);
}

}  // extern "C"


#endif  // INCLUDE_GUARD_HIP_CUPY_HIPSPARSE_H
