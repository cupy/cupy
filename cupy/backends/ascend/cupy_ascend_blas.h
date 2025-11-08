#ifndef INCLUDE_GUARD_ASCEND_CUPY_asdBlas_H
#define INCLUDE_GUARD_ASCEND_CUPY_asdBlas_H

#include <blas_api.h>
#include <acl_blas.h> // GEMM matrix multiplication in this header

// for ASCEND_VERSION
#include <stdexcept>  // for gcc 10

#include "cupy_ascend_common.h"
#include "cupy_ascend_complex.h"
using namespace AsdSip;
extern "C" {

///////////////////////////////////////////////////////////////////////////////
// blas & lapack ()
///////////////////////////////////////////////////////////////////////////////

/* BLAS API are put in AsdSip package with asdBlas prefix
 */

/* ---------- helpers ---------- */
aclTensor* convert_complex_tensor(const cuComplex* dptr, std::vector<int> shape)
{
    return nullptr;
}

static asdBlasOperation_t convert_asdBlasOperation_t(cublasOperation_t op) {
    return static_cast<asdBlasOperation_t>(static_cast<int>(op) + 111);
}

static asdBlasFillMode_t convert_asdBlasFillMode_t(cublasFillMode_t mode) {
    switch(static_cast<int>(mode)) {
        case 0 /* CUBLAS_FILL_MODE_LOWER */: return ASDBLAS_FILL_MODE_LOWER;
        case 1 /* CUBLAS_FILL_MODE_UPPER */: return ASDBLAS_FILL_MODE_UPPER;
        default: throw std::runtime_error("unrecognized mode");
    }
}

static asdBlasDiagType_t convert_asdBlasDiagType_t(cublasDiagType_t type) {
    return static_cast<asdBlasDiagType_t>(static_cast<int>(type) + 131);
}

static asdBlasSideMode_t convert_asdBlasSideMode_t(cublasSideMode_t mode) {
    return static_cast<asdBlasSideMode_t>(static_cast<int>(mode) + 141);
}

// aclDataType
static asdBlasDatatype_t convert_asdBlasDatatype_t(cudaDataType_t type) {
    switch(static_cast<int>(type)) {
        case 0 /* CUDA_R_32F */: return asdBlas_R_32F;
        case 1 /* CUDA_R_64F */: return asdBlas_R_64F;
        case 2 /* CUDA_R_16F */: return asdBlas_R_16F;
        case 3 /* CUDA_R_8I */ : return asdBlas_R_8I;
        case 4 /* CUDA_C_32F */: return asdBlas_C_32F;
        case 5 /* CUDA_C_64F */: return asdBlas_C_64F;
        case 6 /* CUDA_C_16F */: return asdBlas_C_16F;
        case 7 /* CUDA_C_8I */ : return asdBlas_C_8I;
        case 8 /* CUDA_R_8U */ : return asdBlas_R_8U;
        case 9 /* CUDA_C_8U */ : return asdBlas_C_8U;
        default: throw std::runtime_error("unrecognized type");
    }
}

// Context
cublasStatus_t cublasCreate(cublasHandle_t* handle) {
    return asdBlasCreate(handle);
}

cublasStatus_t cublasDestroy(cublasHandle_t handle) {
    return asdBlasDestroy(handle);
}

cublasStatus_t cublasGetVersion(cublasHandle_t handle, int *version) {
    int32_t major = 0;
    int32_t minor = 0;
    int32_t patch = 0;
    aclError ret = aclrtGetVersion(&major, &minor, &patch);
    *version = major * 0x01000000 + minor * 0x0001000 * patch * 0x0010;
    return ret;
}

cublasStatus_t cublasSetPointerMode(cublasHandle_t handle, cublasPointerMode_t mode) {
    return ACL_ERROR_FEATURE_UNSUPPORTED;
    //return asdBlasSetPointerMode(handle, mode);
}

cublasStatus_t cublasGetPointerMode(cublasHandle_t handle, cublasPointerMode_t *mode) {
    return ACL_ERROR_FEATURE_UNSUPPORTED;
    //return asdBlasGetPointerMode(handle, mode);
}

// Stream
cublasStatus_t cublasSetStream(cublasHandle_t handle, cudaStream_t streamId) {
    return asdBlasSetStream(handle, streamId);
}

cublasStatus_t cublasGetStream(cublasHandle_t handle, cudaStream_t *streamId) {
    return ACL_ERROR_FEATURE_UNSUPPORTED;
    //return asdBlasGetStream(handle, streamId);
}

// Math Mode
cublasStatus_t cublasSetMathMode(...) {
    return ACL_ERROR_FEATURE_UNSUPPORTED;;
}

cublasStatus_t cublasGetMathMode(...) {
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

// BLAS Level 1
cublasStatus_t cublasIsamax(cublasHandle_t handle, int n, const float *x, int incx, int *result) {
    return asdBlasIsamax(handle, n, x, incx, result);
}

cublasStatus_t cublasIdamax(cublasHandle_t handle, int n, const double *x, int incx, int *result) {
    // return asdBlasIdamax(handle, n, x, incx, result);  // double
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

cublasStatus_t cublasIcamax(cublasHandle_t handle, int n, const cuComplex *x, int incx, int *result) {
    return asdBlasIcamax(handle, n, x, incx, result);
}

cublasStatus_t cublasIzamax(cublasHandle_t handle, int n, const cuDoubleComplex *x, int incx, int *result) {
    // return asdBlasIzamax(handle, n, reinterpret_cast<const asdBlasDoubleComplex*>(x), incx, result);
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

cublasStatus_t cublasIsamin(cublasHandle_t handle, int n, float* x, int incx, int* result) {
    return asdBlasIsamin(handle, n, x, incx, result);
}

cublasStatus_t cublasIdamin(cublasHandle_t handle, int n, const double *x, int incx, int *result) {
    // return asdBlasIdamin(handle, n, x, incx, result);
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

cublasStatus_t cublasIcamin(cublasHandle_t handle, int n, const cuComplex *x, int incx, int *result) {
    return asdBlasIcamin(handle, n, reinterpret_cast<const asdBlasComplex*>(x), incx, result);
}

cublasStatus_t cublasIzamin(cublasHandle_t handle, int n, const cuDoubleComplex *x, int incx, int *result) {
    return asdBlasIzamin(handle, n, reinterpret_cast<const asdBlasDoubleComplex*>(x), incx, result);
}

cublasStatus_t cublasSasum(cublasHandle_t handle, int n, float* x, int incx, float* result) {
    return asdBlasSasum(handle, n, x, incx, result);
}

cublasStatus_t cublasDasum(cublasHandle_t handle, int n, double* x, int incx, double* result) {
    return asdBlasDasum(handle, n, x, incx, result);
}

cublasStatus_t cublasScasum(cublasHandle_t handle, int n, cuComplex* x, int incx, float* result) {
    return asdBlasScasum(handle, n, reinterpret_cast<const asdBlasComplex*>(x), incx, result);
}

cublasStatus_t cublasDzasum(cublasHandle_t handle, int n, cuDoubleComplex* x, int incx, double* result) {
    return asdBlasDzasum(handle, n, reinterpret_cast<const asdBlasDoubleComplex*>(x), incx, result);
}

cublasStatus_t cublasSaxpy(cublasHandle_t handle, int n, float* alpha, float* x, int incx, float* y, int incy) {
    return asdBlasSaxpy(handle, n, alpha, x, incx, y, incy);
}

cublasStatus_t cublasDaxpy(cublasHandle_t handle, int n, double* alpha, double* x, int incx, double* y, int incy) {
    return asdBlasDaxpy(handle, n, alpha, x, incx, y, incy);
}

cublasStatus_t cublasCaxpy(cublasHandle_t handle, int n, cuComplex* alpha, cuComplex* x, int incx, cuComplex* y, int incy) {
    return asdBlasCaxpy(handle, n,
                        reinterpret_cast<const asdBlasComplex*>(alpha),
                        reinterpret_cast<const asdBlasComplex*>(x), incx,
                        reinterpret_cast<asdBlasComplex*>(y), incy);
}

cublasStatus_t cublasZaxpy(cublasHandle_t handle, int n, cuDoubleComplex* alpha, cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy) {
    return asdBlasZaxpy(handle, n,
                        reinterpret_cast<const asdBlasDoubleComplex*>(alpha),
                        reinterpret_cast<const asdBlasDoubleComplex*>(x), incx,
                        reinterpret_cast<asdBlasDoubleComplex*>(y), incy);
}

cublasStatus_t cublasSdot(cublasHandle_t handle, int n, float* x, int incx, float* y, int incy, float* result) {
    return asdBlasSdot(handle, n, x, incx, y, incy, result);
}

cublasStatus_t cublasDdot(cublasHandle_t handle, int n, double* x, int incx, double* y, int incy, double* result) {
    return asdBlasDdot(handle, n, x, incx, y, incy, result);
}

cublasStatus_t cublasCdotu(cublasHandle_t handle, int n, cuComplex* x, int incx, cuComplex* y,
                           int incy, cuComplex* result) {
    return asdBlasCdotu(handle, n,
                        reinterpret_cast<const asdBlasComplex*>(x), incx,
                        reinterpret_cast<const asdBlasComplex*>(y), incy,
                        reinterpret_cast<asdBlasComplex*>(result));
}

cublasStatus_t cublasCdotc(cublasHandle_t handle, int n, cuComplex* x, int incx, cuComplex* y,
                           int incy, cuComplex* result) {
    return asdBlasCdotc(handle, n,
                        reinterpret_cast<const asdBlasComplex*>(x), incx,
                        reinterpret_cast<const asdBlasComplex*>(y), incy,
                        reinterpret_cast<asdBlasComplex*>(result));
}

cublasStatus_t cublasZdotu(cublasHandle_t handle, int n, cuDoubleComplex* x, int incx, cuDoubleComplex* y,
                           int incy, cuDoubleComplex* result) {
    return asdBlasZdotu(handle, n,
                        reinterpret_cast<const asdBlasDoubleComplex*>(x), incx,
                        reinterpret_cast<const asdBlasDoubleComplex*>(y), incy,
                        reinterpret_cast<asdBlasDoubleComplex*>(result));
}

cublasStatus_t cublasZdotc(cublasHandle_t handle, int n, cuDoubleComplex* x, int incx, cuDoubleComplex* y,
                           int incy, cuDoubleComplex* result) {
    return asdBlasZdotc(handle, n,
                        reinterpret_cast<const asdBlasDoubleComplex*>(x), incx,
                        reinterpret_cast<const asdBlasDoubleComplex*>(y), incy,
                        reinterpret_cast<asdBlasDoubleComplex*>(result));
}

cublasStatus_t cublasSnrm2(cublasHandle_t handle, int n, float* x, int incx, float* result) {
    return asdBlasSnrm2(handle, n, x, incx, result);
}

cublasStatus_t cublasDnrm2(cublasHandle_t handle, int n, double* x, int incx, double* result) {
    return asdBlasDnrm2(handle, n, x, incx, result);
}

cublasStatus_t cublasScnrm2(cublasHandle_t handle, int n, cuComplex* x, int incx, float* result) {
    return asdBlasScnrm2(handle, n, reinterpret_cast<const asdBlasComplex*>(x), incx, result);
}

cublasStatus_t cublasDznrm2(cublasHandle_t handle, int n, cuDoubleComplex* x, int incx, double* result) {
    return asdBlasDznrm2(handle, n, reinterpret_cast<const asdBlasDoubleComplex*>(x), incx, result);
}

cublasStatus_t cublasSscal(cublasHandle_t handle, int n, float* alpha, float* x, int incx) {
    return asdBlasSscal(handle, n, alpha, x, incx);
}

cublasStatus_t cublasDscal(cublasHandle_t handle, int n, double* alpha, double* x, int incx) {
    return asdBlasDscal(handle, n, alpha, x, incx);
}

cublasStatus_t cublasCscal(cublasHandle_t handle, int n, cuComplex* alpha, cuComplex* x, int incx) {
    return asdBlasCscal(handle, n, reinterpret_cast<const asdBlasComplex*>(alpha), reinterpret_cast<asdBlasComplex*>(x), incx);
}

cublasStatus_t cublasCsscal(cublasHandle_t handle, int n, float* alpha, cuComplex* x, int incx) {
    return asdBlasCsscal(handle, n, alpha, reinterpret_cast<asdBlasComplex*>(x), incx);
}

cublasStatus_t cublasZscal(cublasHandle_t handle, int n, cuDoubleComplex* alpha, cuDoubleComplex* x, int incx) {
    return asdBlasZscal(handle, n, reinterpret_cast<const asdBlasDoubleComplex*>(alpha), reinterpret_cast<asdBlasDoubleComplex*>(x), incx);
}

cublasStatus_t cublasZdscal(cublasHandle_t handle, int n, double* alpha, cuDoubleComplex* x, int incx) {
    return asdBlasZdscal(handle, n, alpha, reinterpret_cast<asdBlasDoubleComplex*>(x), incx);
}


// BLAS Level 2
cublasStatus_t cublasSgemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, float* alpha,
                           float* A, int lda, float* x, int incx, float* beta,
                           float* y, int incy) {
    return asdBlasSgemv(handle, convert_asdBlasOperation_t(trans), m, n,
                        alpha, A, lda, x, incx, beta, y, incy);
}

cublasStatus_t cublasDgemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, double* alpha,
                           double* A, int lda, double* x, int incx, double* beta,
                           double* y, int incy) {
    return asdBlasDgemv(handle, convert_asdBlasOperation_t(trans), m, n,
                        alpha, A, lda, x, incx, beta, y, incy);
}

cublasStatus_t cublasCgemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, cuComplex* alpha,
                           cuComplex* A, int lda, cuComplex* x, int incx, cuComplex* beta,
                           cuComplex* y, int incy) {
    return asdBlasCgemv(handle, convert_asdBlasOperation_t(trans), m, n,
                        reinterpret_cast<const asdBlasComplex*>(alpha),
                        reinterpret_cast<const asdBlasComplex*>(A), lda,
                        reinterpret_cast<const asdBlasComplex*>(x), incx,
                        reinterpret_cast<const asdBlasComplex*>(beta),
                        reinterpret_cast<asdBlasComplex*>(y), incy);
}

cublasStatus_t cublasZgemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, cuDoubleComplex* alpha,
                           cuDoubleComplex* A, int lda, cuDoubleComplex* x, int incx,
                           cuDoubleComplex* beta, cuDoubleComplex* y, int incy) {
    return asdBlasZgemv(handle, convert_asdBlasOperation_t(trans), m, n,
                        reinterpret_cast<const asdBlasDoubleComplex*>(alpha),
                        reinterpret_cast<const asdBlasDoubleComplex*>(A), lda,
                        reinterpret_cast<const asdBlasDoubleComplex*>(x), incx,
                        reinterpret_cast<const asdBlasDoubleComplex*>(beta),
                        reinterpret_cast<asdBlasDoubleComplex*>(y), incy);
}

cublasStatus_t cublasSger(cublasHandle_t handle, int m, int n, float* alpha, float* x, int incx,
                          float* y, int incy, float* A, int lda) {
    return asdBlasSger(handle, m, n, alpha, x, incx, y, incy, A, lda);
}

cublasStatus_t cublasDger(cublasHandle_t handle, int m, int n, double* alpha, double* x,
                          int incx, double* y, int incy, double* A, int lda) {
    return asdBlasDger(handle, m, n, alpha, x, incx, y, incy, A, lda);
}

cublasStatus_t cublasCgeru(cublasHandle_t handle, int m, int n, cuComplex* alpha, cuComplex* x,
                           int incx, cuComplex* y, int incy, cuComplex* A, int lda) {
    return asdBlasCgeru(handle, m, n,
                        reinterpret_cast<const asdBlasComplex*>(alpha),
                        reinterpret_cast<const asdBlasComplex*>(x), incx,
                        reinterpret_cast<const asdBlasComplex*>(y), incy,
                        reinterpret_cast<asdBlasComplex*>(A), lda);
}

cublasStatus_t cublasCgerc(cublasHandle_t handle, int m, int n, cuComplex* alpha, cuComplex* x,
                           int incx, cuComplex* y, int incy, cuComplex* A, int lda) {
    return asdBlasCgerc(handle, m, n,
                        reinterpret_cast<const asdBlasComplex*>(alpha),
                        reinterpret_cast<const asdBlasComplex*>(x), incx,
                        reinterpret_cast<const asdBlasComplex*>(y), incy,
                        reinterpret_cast<asdBlasComplex*>(A), lda);
}

cublasStatus_t cublasZgeru(cublasHandle_t handle, int m, int n, cuDoubleComplex* alpha,
                           cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy,
                           cuDoubleComplex* A, int lda) {
    return asdBlasZgeru(handle, m, n,
                        reinterpret_cast<const asdBlasDoubleComplex*>(alpha),
                        reinterpret_cast<const asdBlasDoubleComplex*>(x), incx,
                        reinterpret_cast<const asdBlasDoubleComplex*>(y), incy,
                        reinterpret_cast<asdBlasDoubleComplex*>(A), lda);
}

cublasStatus_t cublasZgerc(cublasHandle_t handle, int m, int n, cuDoubleComplex* alpha,
                           cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy,
                           cuDoubleComplex* A, int lda) {
    return asdBlasZgerc(handle, m, n,
                        reinterpret_cast<const asdBlasDoubleComplex*>(alpha),
                        reinterpret_cast<const asdBlasDoubleComplex*>(x), incx,
                        reinterpret_cast<const asdBlasDoubleComplex*>(y), incy,
                        reinterpret_cast<asdBlasDoubleComplex*>(A), lda);
}

cublasStatus_t cublasSsbmv(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k,
                           const float* alpha, const float* A, int lda,
                           const float* x, int incx,
                           const float* beta, float* y, int incy) {
    return asdBlasSsbmv(handle, convert_asdBlasFillMode_t(uplo), n, k,
                        alpha, A, lda, x, incx, beta, y, incy);
}

cublasStatus_t cublasDsbmv(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k,
                           const double* alpha, const double* A, int lda,
                           const double* x, int incx,
                           const double* beta, double* y, int incy) {
    return asdBlasDsbmv(handle, convert_asdBlasFillMode_t(uplo), n, k,
        alpha, A, lda, x, incx, beta, y, incy);
}


// BLAS Level 3
cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                                          int m, int n, int k, const void* alpha,
                                          const void* A, cudaDataType Atype, int lda, long long int strideA,
                                          const void* B, cudaDataType Btype, int ldb, long long int strideB,
                                          const void* beta,
                                          void* C, cudaDataType Ctype, int ldc, long long int strideC,
                                          int batchCount, cudaDataType_t computeType, cublasGemmAlgo_t algo) {
    if (algo != -1) { // must be CUBLAS_GEMM_DEFAULT
        return ACL_ERROR_FEATURE_UNSUPPORTED;
    }
    return asdBlasGemmStridedBatchedEx(handle, convert_asdBlasOperation_t(transa), convert_asdBlasOperation_t(transb),
                                       m, n, k, alpha,
                                       A, convert_asdBlasDatatype_t(Atype), lda, strideA,
                                       B, convert_asdBlasDatatype_t(Btype), ldb, strideB,
                                       beta,
                                       C, convert_asdBlasDatatype_t(Ctype), ldc, strideC,
                                       batchCount, convert_asdBlasDatatype_t(computeType),
                                       static_cast<asdBlasGemmAlgo_t>(160));  // asdBlas_GEMM_DEFAULT
}

cublasStatus_t cublasSgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                            int m, int n, int k, const float *alpha,
                            const float *A, int lda,
                            const float *B, int ldb,
                            const float *beta,
                            float *C, int ldc) {
    return asdBlasSgemm(handle, convert_asdBlasOperation_t(transa), convert_asdBlasOperation_t(transb), m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

cublasStatus_t cublasDgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k, const double *alpha,
                           const double *A, int lda,
                           const double *B, int ldb,
                           const double *beta, double *C, int ldc) {
    return asdBlasDgemm(handle, convert_asdBlasOperation_t(transa), convert_asdBlasOperation_t(transb), m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}


cublasStatus_t cublasCgemm(
        cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, int k, const cuComplex *alpha,
        const cuComplex *A, int lda,
        const cuComplex *B, int ldb,
        const cuComplex *beta, cuComplex *C, int ldc)
{
    return asdBlasCgemm(
        handle, convert_asdBlasOperation_t(transa), convert_asdBlasOperation_t(transb),
        m, n, k,
        *alpha,
        convert_complex_tensor(A, {}), lda,
        convert_complex_tensor(B, {}), ldb,
        *beta,
        convert_complex_tensor(C, {}), ldc);
}

cublasStatus_t cublasZgemm(
        cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, int k, const cuDoubleComplex *alpha,
        const cuDoubleComplex *A, int lda,
        const cuDoubleComplex *B, int ldb,
        const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc)
{
    return asdBlasZgemm(
        handle, convert_asdBlasOperation_t(transa), convert_asdBlasOperation_t(transb),
        m, n, k,
        reinterpret_cast<const asdBlasDoubleComplex *>(alpha),
        reinterpret_cast<const asdBlasDoubleComplex *>(A), lda,
        reinterpret_cast<const asdBlasDoubleComplex *>(B), ldb,
        reinterpret_cast<const asdBlasDoubleComplex *>(beta),
        reinterpret_cast<asdBlasDoubleComplex *>(C), ldc);
}

cublasStatus_t cublasSgemmBatched(
        cublasHandle_t handle,
        cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, int k,  const float *alpha,
        const float *A[], int lda,
        const float *B[], int ldb,
        const float *beta,
        float *C[], int ldc, int batchCount) {
    return asdBlasSgemmBatched(handle, convert_asdBlasOperation_t(transa), convert_asdBlasOperation_t(transb), m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
}

cublasStatus_t cublasDgemmBatched(
        cublasHandle_t handle,
        cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, int k,  const double *alpha,
        const double *A[], int lda,
        const double *B[], int ldb,
        const double *beta,
        double *C[], int ldc, int batchCount) {
    return asdBlasDgemmBatched(handle, convert_asdBlasOperation_t(transa), convert_asdBlasOperation_t(transb), m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
}

cublasStatus_t cublasCgemmBatched(
        cublasHandle_t handle,
        cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, int k,  const cuComplex *alpha,
        const cuComplex *A[], int lda,
        const cuComplex *B[], int ldb,
        const cuComplex *beta,
        cuComplex *C[], int ldc, int batchCount) {
    return asdBlasCgemmBatched(
        handle, convert_asdBlasOperation_t(transa), convert_asdBlasOperation_t(transb),
        m, n, k,
        reinterpret_cast<const asdBlasComplex*>(alpha),
        reinterpret_cast<const asdBlasComplex**>(A), lda,
        reinterpret_cast<const asdBlasComplex**>(B), ldb,
        reinterpret_cast<const asdBlasComplex*>(beta),
        reinterpret_cast<asdBlasComplex**>(C), ldc, batchCount);
}

cublasStatus_t cublasZgemmBatched(
        cublasHandle_t handle,
        cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, int k,  const cuDoubleComplex *alpha,
        const cuDoubleComplex *A[], int lda,
        const cuDoubleComplex *B[], int ldb,
        const cuDoubleComplex *beta,
        cuDoubleComplex *C[], int ldc, int batchCount) {
    return asdBlasZgemmBatched(
        handle, convert_asdBlasOperation_t(transa), convert_asdBlasOperation_t(transb),
        m, n, k,
        reinterpret_cast<const asdBlasDoubleComplex*>(alpha),
        reinterpret_cast<const asdBlasDoubleComplex**>(A), lda,
        reinterpret_cast<const asdBlasDoubleComplex**>(B), ldb,
        reinterpret_cast<const asdBlasDoubleComplex*>(beta),
        reinterpret_cast<asdBlasDoubleComplex**>(C), ldc, batchCount);
}

cublasStatus_t cublasSgemmEx(
        cublasHandle_t handle,
        cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, int k,
        const float *alpha, const void *A, cudaDataType_t Atype, int lda,
        const void *B, cudaDataType_t Btype, int ldb, const float *beta,
        void *C, cudaDataType_t Ctype, int ldc) {
    if (Atype != 0 || Btype != 0 || Ctype != 0) {
        return ACL_ERROR_FEATURE_UNSUPPORTED;
    }
    return asdBlasSgemm(
        handle,
        convert_asdBlasOperation_t(transa), convert_asdBlasOperation_t(transb),
        m, n, k,
        alpha, static_cast<const float*>(A), lda,
        static_cast<const float*>(B), ldb, beta,
        static_cast<float*>(C), ldc);
}

cublasStatus_t cublasGemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                            int m, int n, int k, const void *alpha,
                            const void *A, cudaDataType_t Atype, int lda,
                            const void *B, cudaDataType_t Btype, int ldb,
                            const void *beta,
                            void *C, cudaDataType_t Ctype, int ldc,
                            cudaDataType_t computetype, cublasGemmAlgo_t algo) {
    if (algo != -1) { // must be CUBLAS_GEMM_DEFAULT
        return ACL_ERROR_FEATURE_UNSUPPORTED;
    }
    return asdBlasGemmEx(handle, convert_asdBlasOperation_t(transa), convert_asdBlasOperation_t(transb),
                         m, n, k, alpha,
                         A, convert_asdBlasDatatype_t(Atype), lda,
                         B, convert_asdBlasDatatype_t(Btype), ldb,
                         beta,
                         C, convert_asdBlasDatatype_t(Ctype), ldc,
                         convert_asdBlasDatatype_t(computetype),
                         static_cast<asdBlasGemmAlgo_t>(160));  // asdBlas_GEMM_DEFAULT
}

cublasStatus_t cublasGemmEx_v11(...) {
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}
cublasStatus_t cublasGemmStridedBatchedEx_v11(...) {
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

cublasStatus_t cublasStrsm(cublasHandle_t handle, cublasSideMode_t size, cublasFillMode_t uplo, cublasOperation_t trans,
                           cublasDiagType_t diag, int m, int n, const float* alpha,
                           const float* A, int lda, float* B, int ldb) {
    return asdBlasStrsm(handle,
                        convert_asdBlasSideMode_t(size),
                        convert_asdBlasFillMode_t(uplo),
                        convert_asdBlasOperation_t(trans),
                        convert_asdBlasDiagType_t(diag),
                        m, n, alpha, const_cast<float*>(A), lda, B, ldb);
}

cublasStatus_t cublasDtrsm(cublasHandle_t handle, cublasSideMode_t size, cublasFillMode_t uplo, cublasOperation_t trans,
                           cublasDiagType_t diag, int m, int n, const double* alpha,
                           const double* A, int lda, double* B, int ldb) {
    return asdBlasDtrsm(handle,
                        convert_asdBlasSideMode_t(size),
                        convert_asdBlasFillMode_t(uplo),
                        convert_asdBlasOperation_t(trans),
                        convert_asdBlasDiagType_t(diag),
                        m, n, alpha, const_cast<double*>(A), lda, B, ldb);
}

cublasStatus_t cublasCtrsm(cublasHandle_t handle, cublasSideMode_t size, cublasFillMode_t uplo, cublasOperation_t trans,
                           cublasDiagType_t diag, int m, int n, const cuComplex* alpha,
                           const cuComplex* A, int lda, cuComplex* B, int ldb) {
    return asdBlasCtrsm(handle,
                        convert_asdBlasSideMode_t(size),
                        convert_asdBlasFillMode_t(uplo),
                        convert_asdBlasOperation_t(trans),
                        convert_asdBlasDiagType_t(diag),
                        m, n,
                        reinterpret_cast<const asdBlasComplex*>(alpha),
                        reinterpret_cast<asdBlasComplex*>(const_cast<cuComplex*>(A)), lda,
                        reinterpret_cast<asdBlasComplex*>(B), ldb);
}

cublasStatus_t cublasZtrsm(cublasHandle_t handle, cublasSideMode_t size, cublasFillMode_t uplo, cublasOperation_t trans,
                           cublasDiagType_t diag, int m, int n, const cuDoubleComplex* alpha,
                           const cuDoubleComplex* A, int lda, cuDoubleComplex* B, int ldb) {
    return asdBlasZtrsm(handle,
                        convert_asdBlasSideMode_t(size),
                        convert_asdBlasFillMode_t(uplo),
                        convert_asdBlasOperation_t(trans),
                        convert_asdBlasDiagType_t(diag),
                        m, n,
                        reinterpret_cast<const asdBlasDoubleComplex*>(alpha),
                        reinterpret_cast<asdBlasDoubleComplex*>(const_cast<cuDoubleComplex*>(A)), lda,
                        reinterpret_cast<asdBlasDoubleComplex*>(B), ldb);
}

cublasStatus_t cublasStrsmBatched(cublasHandle_t handle, cublasSideMode_t size, cublasFillMode_t uplo, cublasOperation_t trans,
                                  cublasDiagType_t diag, int m, int n, const float* alpha,
                                  const float* const A[], int lda, float* const B[], int ldb, int batchCount) {
    return asdBlasStrsmBatched(handle,
                               convert_asdBlasSideMode_t(size),
                               convert_asdBlasFillMode_t(uplo),
                               convert_asdBlasOperation_t(trans),
                               convert_asdBlasDiagType_t(diag),
                               m, n, alpha, const_cast<float* const*>(A), lda, const_cast<float**>(B), ldb, batchCount);
}

cublasStatus_t cublasDtrsmBatched(cublasHandle_t handle, cublasSideMode_t size, cublasFillMode_t uplo, cublasOperation_t trans,
                                  cublasDiagType_t diag, int m, int n, const double* alpha,
                                  const double* const A[], int lda, double* const B[], int ldb, int batchCount) {
    return asdBlasDtrsmBatched(handle,
                               convert_asdBlasSideMode_t(size),
                               convert_asdBlasFillMode_t(uplo),
                               convert_asdBlasOperation_t(trans),
                               convert_asdBlasDiagType_t(diag),
                               m, n, alpha, const_cast<double* const*>(A), lda, const_cast<double**>(B), ldb, batchCount);
}

cublasStatus_t cublasCtrsmBatched(cublasHandle_t handle, cublasSideMode_t size, cublasFillMode_t uplo, cublasOperation_t trans,
                                  cublasDiagType_t diag, int m, int n, const cuComplex* alpha,
                                  const cuComplex* const A[], int lda, cuComplex* const B[], int ldb, int batchCount) {
    return asdBlasCtrsmBatched(handle,
                               convert_asdBlasSideMode_t(size),
                               convert_asdBlasFillMode_t(uplo),
                               convert_asdBlasOperation_t(trans),
                               convert_asdBlasDiagType_t(diag),
                               m, n,
                               reinterpret_cast<const asdBlasComplex*>(alpha),
                               reinterpret_cast<asdBlasComplex* const*>(const_cast<cuComplex* const*>(A)), lda,
                               reinterpret_cast<asdBlasComplex**>(const_cast<cuComplex**>(B)), ldb, batchCount);
}

cublasStatus_t cublasZtrsmBatched(cublasHandle_t handle, cublasSideMode_t size, cublasFillMode_t uplo, cublasOperation_t trans,
                                  cublasDiagType_t diag, int m, int n, const cuDoubleComplex* alpha,
                                  const cuDoubleComplex* const A[], int lda, cuDoubleComplex* const B[], int ldb, int batchCount) {
    return asdBlasZtrsmBatched(handle,
                               convert_asdBlasSideMode_t(size),
                               convert_asdBlasFillMode_t(uplo),
                               convert_asdBlasOperation_t(trans),
                               convert_asdBlasDiagType_t(diag),
                               m, n,
                               reinterpret_cast<const asdBlasDoubleComplex*>(alpha),
                               reinterpret_cast<asdBlasDoubleComplex* const*>(const_cast<cuDoubleComplex* const*>(A)), lda,
                               reinterpret_cast<asdBlasDoubleComplex**>(const_cast<cuDoubleComplex**>(B)), ldb, batchCount);
}

cublasStatus_t cublasSsyrk(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k,
                           const float* alpha, const float* A, int lda,
                           const float* beta, float* C, int ldc) {
    return asdBlasSsyrk(handle, convert_asdBlasFillMode_t(uplo), convert_asdBlasOperation_t(trans), n, k,
                        alpha, A, lda, beta, C, ldc);
}

cublasStatus_t cublasDsyrk(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, 
                           const double* alpha, const double* A, int lda,
                           const double* beta, double* C, int ldc) {
    return asdBlasDsyrk(handle, convert_asdBlasFillMode_t(uplo), convert_asdBlasOperation_t(trans),  n, k, 
                        alpha, A, lda, beta, C, ldc);
}

cublasStatus_t cublasCsyrk(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k,
                           const cuComplex* alpha, const cuComplex* A,int lda,
                           const cuComplex* beta, cuComplex* C, int ldc)
{
    return asdBlasCsyrk(handle, convert_asdBlasFillMode_t(uplo), convert_asdBlasOperation_t(trans), n, k,
                        reinterpret_cast<const asdBlasComplex*>(alpha),
                        reinterpret_cast<const asdBlasComplex*>(A), lda,
                        reinterpret_cast<const asdBlasComplex*>(beta),
                        reinterpret_cast<asdBlasComplex*>(C), ldc);
}

cublasStatus_t cublasZsyrk(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k,
                           const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda,
                           const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc)
{
    return asdBlasZsyrk(handle, convert_asdBlasFillMode_t(uplo), convert_asdBlasOperation_t(trans), n, k,
                        reinterpret_cast<const asdBlasDoubleComplex*>(alpha),
                        reinterpret_cast<const asdBlasDoubleComplex*>(A), lda,
                        reinterpret_cast<const asdBlasDoubleComplex*>(beta),
                        reinterpret_cast<asdBlasDoubleComplex*>(C), ldc);
}

// BLAS extension
cublasStatus_t cublasSgeam(
        cublasHandle_t handle,
        cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, const float *alpha,
        const float *A, int lda, const float *beta, const float *B, int ldb,
        float *C, int ldc) {
    return asdBlasSgeam(handle, convert_asdBlasOperation_t(transa), convert_asdBlasOperation_t(transb), m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}

cublasStatus_t cublasDgeam(
        cublasHandle_t handle,
        cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, const double *alpha,
        const double *A, int lda, const double *beta, const double *B, int ldb,
        double *C, int ldc) {
    return asdBlasDgeam(handle, convert_asdBlasOperation_t(transa), convert_asdBlasOperation_t(transb), m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}

cublasStatus_t cublasCgeam(
        cublasHandle_t handle,
        cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, const cuComplex *alpha,
        const cuComplex *A, int lda, const cuComplex *beta, const cuComplex *B, int ldb,
        cuComplex *C, int ldc) {
    #if HIP_VERSION < 307
    return ACL_ERROR_FEATURE_UNSUPPORTED;
    #else
    return asdBlasCgeam(handle, convert_asdBlasOperation_t(transa), convert_asdBlasOperation_t(transb), m, n,
                        reinterpret_cast<const asdBlasComplex*>(alpha),
                        reinterpret_cast<const asdBlasComplex*>(A),
                        lda,
                        reinterpret_cast<const asdBlasComplex*>(beta),
                        reinterpret_cast<const asdBlasComplex*>(B),
                        ldb,
                        reinterpret_cast<asdBlasComplex*>(C),
                        ldc);
    #endif
}

cublasStatus_t cublasZgeam(
        cublasHandle_t handle,
        cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, const cuDoubleComplex *alpha,
        const cuDoubleComplex *A, int lda, const cuDoubleComplex *beta, const cuDoubleComplex *B, int ldb,
	    cuDoubleComplex *C, int ldc) {
    #if HIP_VERSION < 307
    return ACL_ERROR_FEATURE_UNSUPPORTED;
    #else
    return asdBlasZgeam(handle, convert_asdBlasOperation_t(transa), convert_asdBlasOperation_t(transb), m, n,
                        reinterpret_cast<const asdBlasDoubleComplex*>(alpha),
                        reinterpret_cast<const asdBlasDoubleComplex*>(A),
                        lda,
                        reinterpret_cast<const asdBlasDoubleComplex*>(beta),
                        reinterpret_cast<const asdBlasDoubleComplex*>(B),
                        ldb,
                        reinterpret_cast<asdBlasDoubleComplex*>(C),
                        ldc);
    #endif
}

cublasStatus_t cublasSdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n,
                           const float *A, int lda,
                           const float *x, int incx,
                           float *C, int ldc) {
    #if HIP_VERSION < 306
    return ACL_ERROR_FEATURE_UNSUPPORTED;
    #else
    return asdBlasSdgmm(handle, convert_asdBlasSideMode_t(mode), m, n, A, lda, x, incx, C, ldc);
    #endif
}

cublasStatus_t cublasDdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n,
                           const double *A, int lda,
                           const double *x, int incx,
                           double *C, int ldc) {
    #if HIP_VERSION < 306
    return ACL_ERROR_FEATURE_UNSUPPORTED;
    #else
    return asdBlasDdgmm(handle, convert_asdBlasSideMode_t(mode), m, n, A, lda, x, incx, C, ldc);
    #endif
}

cublasStatus_t cublasCdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n,
                           const cuComplex *A, int lda,
                           const cuComplex *x, int incx,
                           cuComplex *C, int ldc) {
    #if HIP_VERSION < 306
    return ACL_ERROR_FEATURE_UNSUPPORTED;
    #else
    return asdBlasCdgmm(handle, convert_asdBlasSideMode_t(mode), m, n,
                        reinterpret_cast<const asdBlasComplex*>(A), lda,
                        reinterpret_cast<const asdBlasComplex*>(x), incx,
                        reinterpret_cast<asdBlasComplex*>(C), ldc);
    #endif
}

cublasStatus_t cublasZdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n,
                           const cuDoubleComplex *A, int lda,
                           const cuDoubleComplex *x, int incx,
                           cuDoubleComplex *C, int ldc) {
    #if HIP_VERSION < 306
    return ACL_ERROR_FEATURE_UNSUPPORTED;
    #else
    return asdBlasZdgmm(handle, convert_asdBlasSideMode_t(mode), m, n,
                        reinterpret_cast<const asdBlasDoubleComplex*>(A), lda,
                        reinterpret_cast<const asdBlasDoubleComplex*>(x), incx,
                        reinterpret_cast<asdBlasDoubleComplex*>(C), ldc);
    #endif
}

cublasStatus_t cublasSgetriBatched(cublasHandle_t handle,
                                   int n,
                                   const float *const A[],
                                   int lda,
                                   const int *P,
                                   float *const C[],
                                   int ldc,
                                   int *info,
                                   int batchSize) {
    #if HIP_VERSION < 308
    return ACL_ERROR_FEATURE_UNSUPPORTED;
    #else
    return asdBlasSgetriBatched(handle, n, const_cast<float* const*>(A), lda, const_cast<int*>(P), C, ldc, info, batchSize);
    #endif
}

cublasStatus_t cublasDgetriBatched(cublasHandle_t handle,
                                   int n,
                                   const double *const A[],
                                   int lda,
                                   const int *P,
                                   double *const C[],
                                   int ldc,
                                   int *info,
                                   int batchSize) {
    #if HIP_VERSION < 308
    return ACL_ERROR_FEATURE_UNSUPPORTED;
    #else
    return asdBlasDgetriBatched(handle, n, const_cast<double* const*>(A), lda, const_cast<int*>(P), C, ldc, info, batchSize);
    #endif
}

cublasStatus_t cublasCgetriBatched(cublasHandle_t handle,
                                   int n,
                                   const cuComplex *const A[],
                                   int lda,
                                   const int *P,
                                   cuComplex *const C[],
                                   int ldc,
                                   int *info,
                                   int batchSize) {
    #if HIP_VERSION < 308
    return ACL_ERROR_FEATURE_UNSUPPORTED;
    #else
    return asdBlasCgetriBatched(handle, n,
                                reinterpret_cast<asdBlasComplex* const*>(const_cast<cuComplex* const*>(A)),
                                lda, const_cast<int*>(P),
                                reinterpret_cast<asdBlasComplex* const*>(C),
                                ldc, info, batchSize);
    #endif
}

cublasStatus_t cublasZgetriBatched(cublasHandle_t handle,
                                   int n,
                                   const cuDoubleComplex *const A[],
                                   int lda,
                                   const int *P,
                                   cuDoubleComplex *const C[],
                                   int ldc,
                                   int *info,
                                   int batchSize) {
    #if HIP_VERSION < 308
    return ACL_ERROR_FEATURE_UNSUPPORTED;
    #else
    return asdBlasZgetriBatched(handle, n,
                                reinterpret_cast<asdBlasDoubleComplex* const*>(const_cast<cuDoubleComplex* const*>(A)),
                                lda, const_cast<int*>(P),
                                reinterpret_cast<asdBlasDoubleComplex* const*>(C),
                                ldc, info, batchSize);
    #endif
}

cublasStatus_t cublasSgemmStridedBatched(
        cublasHandle_t handle,
        cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, int k, const float *alpha,
        const float *A, int lda, long long bsa,
        const float *B, int ldb, long long bsb, const float *beta,
        float *C, int ldc, long long bsc, int batchCount) {
    return asdBlasSgemmStridedBatched(
        handle,
        convert_asdBlasOperation_t(transa), convert_asdBlasOperation_t(transb),
        m, n, k, alpha,  A, lda, bsa, B, ldb, bsb, beta, C, ldc, bsc,
        batchCount);
}

cublasStatus_t cublasDgemmStridedBatched(
        cublasHandle_t handle,
        cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, int k, const double *alpha,
        const double *A, int lda, long long bsa,
        const double *B, int ldb, long long bsb, const double *beta,
        double *C, int ldc, long long bsc, int batchCount) {
    return asdBlasDgemmStridedBatched(
        handle,
        convert_asdBlasOperation_t(transa), convert_asdBlasOperation_t(transb),
        m, n, k, alpha,  A, lda, bsa, B, ldb, bsb, beta, C, ldc, bsc,
        batchCount);
}

cublasStatus_t cublasCgemmStridedBatched(
        cublasHandle_t handle,
        cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, int k, const cuComplex *alpha,
        const cuComplex *A, int lda, long long bsa,
        const cuComplex *B, int ldb, long long bsb,
        const cuComplex *beta,
        cuComplex *C, int ldc, long long bsc, int batchCount) {
    return asdBlasCgemmStridedBatched(
        handle,
        convert_asdBlasOperation_t(transa), convert_asdBlasOperation_t(transb),
        m, n, k,
        reinterpret_cast<const asdBlasComplex *>(alpha),
        reinterpret_cast<const asdBlasComplex *>(A), lda, bsa,
        reinterpret_cast<const asdBlasComplex *>(B), ldb, bsb,
        reinterpret_cast<const asdBlasComplex *>(beta),
        reinterpret_cast<asdBlasComplex *>(C), ldc, bsc,
        batchCount);
}

cublasStatus_t cublasZgemmStridedBatched(
        cublasHandle_t handle,
        cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, int k, const cuDoubleComplex *alpha,
        const cuDoubleComplex *A, int lda, long long bsa,
        const cuDoubleComplex *B, int ldb, long long bsb,
        const cuDoubleComplex *beta,
        cuDoubleComplex *C, int ldc, long long bsc, int batchCount) {
    return asdBlasZgemmStridedBatched(
        handle,
        convert_asdBlasOperation_t(transa), convert_asdBlasOperation_t(transb),
        m, n, k,
        reinterpret_cast<const asdBlasDoubleComplex *>(alpha),
        reinterpret_cast<const asdBlasDoubleComplex *>(A), lda, bsa,
        reinterpret_cast<const asdBlasDoubleComplex *>(B), ldb, bsb,
        reinterpret_cast<const asdBlasDoubleComplex *>(beta),
        reinterpret_cast<asdBlasDoubleComplex *>(C), ldc, bsc,
        batchCount);
}

cublasStatus_t cublasStrttp(...) {
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

cublasStatus_t cublasDtrttp(...) {
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

cublasStatus_t cublasStpttr(...) {
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

cublasStatus_t cublasDtpttr(...) {
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

cublasStatus_t cublasSgetrfBatched(cublasHandle_t handle, int n, float **Aarray, int lda,
                                   int *PivotArray, int *infoArray, int batchSize) {
    return asdBlasSgetrfBatched(handle, n, Aarray, lda, PivotArray, infoArray, batchSize);
}

cublasStatus_t cublasDgetrfBatched(cublasHandle_t handle, int n, double **Aarray, int lda,
                                   int *PivotArray, int *infoArray, int batchSize) {
    return asdBlasDgetrfBatched(handle, n, Aarray, lda, PivotArray, infoArray, batchSize);
}

cublasStatus_t cublasCgetrfBatched(cublasHandle_t handle, int n, cuComplex **Aarray, int lda,
                                   int *PivotArray, int *infoArray, int batchSize) {
    return asdBlasCgetrfBatched(handle, n,
                                reinterpret_cast<asdBlasComplex** const>(Aarray), lda,
                                PivotArray, infoArray, batchSize);
}

cublasStatus_t cublasZgetrfBatched(cublasHandle_t handle, int n, cuDoubleComplex **Aarray, int lda,
                                   int *PivotArray, int *infoArray, int batchSize) {
    return asdBlasZgetrfBatched(handle, n,
                                reinterpret_cast<asdBlasDoubleComplex** const>(Aarray), lda,
                                PivotArray, infoArray, batchSize);
}

cublasStatus_t cublasSgetrsBatched(cublasHandle_t handle,
                                   cublasOperation_t trans,
                                   int n,
                                   int nrhs,
                                   const float *const Aarray[],
                                   int lda,
                                   const int *devIpiv,
                                   float *const Barray[],
                                   int ldb,
                                   int *info,
                                   int batchSize) {
    return asdBlasSgetrsBatched(handle,
                                convert_asdBlasOperation_t(trans),
                                n, nrhs,
                                const_cast<float* const*>(Aarray), lda,
                                devIpiv,
                                Barray, ldb,
                                info, batchSize);
}

cublasStatus_t cublasDgetrsBatched(cublasHandle_t handle,
                                   cublasOperation_t trans,
                                   int n,
                                   int nrhs,
                                   const double *const Aarray[],
                                   int lda,
                                   const int *devIpiv,
                                   double *const Barray[],
                                   int ldb,
                                   int *info,
                                   int batchSize) {
    return asdBlasDgetrsBatched(handle,
                                convert_asdBlasOperation_t(trans),
                                n, nrhs,
                                const_cast<double* const*>(Aarray), lda,
                                devIpiv,
                                Barray, ldb,
                                info, batchSize);
}

cublasStatus_t cublasCgetrsBatched(cublasHandle_t handle,
                                   cublasOperation_t trans,
                                   int n,
                                   int nrhs,
                                   const cuComplex *const Aarray[],
                                   int lda,
                                   const int *devIpiv,
                                   cuComplex *const Barray[],
                                   int ldb,
                                   int *info,
                                   int batchSize) {
    return asdBlasCgetrsBatched(handle,
                                convert_asdBlasOperation_t(trans),
                                n, nrhs,
                                reinterpret_cast<asdBlasComplex* const*>(const_cast<cuComplex* const*>(Aarray)), lda,
                                devIpiv,
                                reinterpret_cast<asdBlasComplex* const*>(Barray), ldb,
                                info, batchSize);
}

cublasStatus_t cublasZgetrsBatched(cublasHandle_t handle,
                                   cublasOperation_t trans,
                                   int n,
                                   int nrhs,
                                   const cuDoubleComplex** Aarray,
                                   int lda,
                                   const int *devIpiv,
                                   cuDoubleComplex** Barray,
                                   int ldb,
                                   int *info,
                                   int batchSize) {
    return asdBlasZgetrsBatched(handle,
                                convert_asdBlasOperation_t(trans),
                                n, nrhs,
                                reinterpret_cast<asdBlasDoubleComplex* const*>(const_cast<cuDoubleComplex* const*>(Aarray)), lda,
                                devIpiv,
                                reinterpret_cast<asdBlasDoubleComplex* const*>(Barray), ldb,
                                info, batchSize);
}

} // extern "C"

#endif // #ifndef INCLUDE_GUARD_HIP_CUPY_asdBlas_H
