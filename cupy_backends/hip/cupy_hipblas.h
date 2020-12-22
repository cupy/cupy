#ifndef INCLUDE_GUARD_HIP_CUPY_HIPBLAS_H
#define INCLUDE_GUARD_HIP_CUPY_HIPBLAS_H

#include "cupy_hip_common.h"
#include <hipblas.h>
#include <hip/hip_version.h>  // for HIP_VERSION


extern "C" {

///////////////////////////////////////////////////////////////////////////////
// blas & lapack (hipBLAS/rocBLAS & rocSOLVER)
///////////////////////////////////////////////////////////////////////////////

/* As of ROCm 3.5.0 (this may have started earlier) many rocSOLVER helper functions
 * are deprecated and using their counterparts from rocBLAS is recommended. In
 * particular, rocSOLVER simply uses rocBLAS's handle for its API calls. This means
 * they are much more integrated than cuBLAS and cuSOLVER are, so it is better to
 * put all of the relevant function in one place.
 */

// TODO(leofang): investigate if we should just remove the hipBLAS layer and use
// rocBLAS directly, since we need to expose its handle anyway


/* ---------- helpers ---------- */
static hipblasOperation_t convert_hipblasOperation_t(cublasOperation_t op) {
    return static_cast<hipblasOperation_t>(static_cast<int>(op) + 111);
}

static hipblasFillMode_t convert_hipblasFillMode_t(cublasFillMode_t mode) {
    switch(static_cast<int>(mode)) {
        case 0 /* CUBLAS_FILL_MODE_LOWER */: return HIPBLAS_FILL_MODE_LOWER;
        case 1 /* CUBLAS_FILL_MODE_UPPER */: return HIPBLAS_FILL_MODE_UPPER;
        default: throw std::runtime_error("unrecognized mode");
    }
}

static hipblasDiagType_t convert_hipblasDiagType_t(cublasDiagType_t type) {
    return static_cast<hipblasDiagType_t>(static_cast<int>(type) + 131);
}

static hipblasSideMode_t convert_hipblasSideMode_t(cublasSideMode_t mode) {
    return static_cast<hipblasSideMode_t>(static_cast<int>(mode) + 141);
}

static hipblasDatatype_t convert_hipblasDatatype_t(cudaDataType_t type) {
    switch(static_cast<int>(type)) {
        case 0 /* CUDA_R_32F */: return HIPBLAS_R_32F;
        case 1 /* CUDA_R_64F */: return HIPBLAS_R_64F;
        case 2 /* CUDA_R_16F */: return HIPBLAS_R_16F;
        case 3 /* CUDA_R_8I */ : return HIPBLAS_R_8I;
        case 4 /* CUDA_C_32F */: return HIPBLAS_C_32F;
        case 5 /* CUDA_C_64F */: return HIPBLAS_C_64F;
        case 6 /* CUDA_C_16F */: return HIPBLAS_C_16F;
        case 7 /* CUDA_C_8I */ : return HIPBLAS_C_8I;
        case 8 /* CUDA_R_8U */ : return HIPBLAS_R_8U;
        case 9 /* CUDA_C_8U */ : return HIPBLAS_C_8U;
        default: throw std::runtime_error("unrecognized type");
    }
}


// Context
cublasStatus_t cublasCreate(cublasHandle_t* handle) {
    return hipblasCreate(handle);
}

cublasStatus_t cublasDestroy(cublasHandle_t handle) {
    return hipblasDestroy(handle);
}

cublasStatus_t cublasGetVersion(cublasHandle_t handle, int *version) {
    // We use the rocBLAS version here because 1. it is the underlying workhorse,
    // and 2. we might get rid of the hipBLAS layer at some point (see TODO above).
    // ex: the rocBLAS version string is 2.22.0.2367-b2cceba in ROCm 3.5.0
    *version = 10000 * ROCBLAS_VERSION_MAJOR + 100 * ROCBLAS_VERSION_MINOR + ROCBLAS_VERSION_PATCH;
    return HIPBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetPointerMode(cublasHandle_t handle, cublasPointerMode_t mode) {
    return hipblasSetPointerMode(handle, mode);
}

cublasStatus_t cublasGetPointerMode(cublasHandle_t handle, cublasPointerMode_t *mode) {
    return hipblasGetPointerMode(handle, mode);
}

// Stream
cublasStatus_t cublasSetStream(cublasHandle_t handle, cudaStream_t streamId) {
    return hipblasSetStream(handle, streamId);
}

cublasStatus_t cublasGetStream(cublasHandle_t handle, cudaStream_t *streamId) {
    return hipblasGetStream(handle, streamId);
}

// Math Mode
cublasStatus_t cublasSetMathMode(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasGetMathMode(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// BLAS Level 1
cublasStatus_t cublasIsamax(cublasHandle_t handle, int n, const float *x, int incx, int *result) {
    return hipblasIsamax(handle, n, x, incx, result);
}

cublasStatus_t cublasIdamax(cublasHandle_t handle, int n, const double *x, int incx, int *result) {
    return hipblasIdamax(handle, n, x, incx, result);
}

cublasStatus_t cublasIcamax(cublasHandle_t handle, int n, const cuComplex *x, int incx, int *result) {
    return hipblasIcamax(handle, n, reinterpret_cast<const hipblasComplex*>(x), incx, result);
}

cublasStatus_t cublasIzamax(cublasHandle_t handle, int n, const cuDoubleComplex *x, int incx, int *result) {
    return hipblasIzamax(handle, n, reinterpret_cast<const hipblasDoubleComplex*>(x), incx, result);
}

cublasStatus_t cublasIsamin(cublasHandle_t handle, int n, float* x, int incx, int* result) {
    return hipblasIsamin(handle, n, x, incx, result);
}

cublasStatus_t cublasIdamin(cublasHandle_t handle, int n, const double *x, int incx, int *result) {
    return hipblasIdamin(handle, n, x, incx, result);
}

cublasStatus_t cublasIcamin(cublasHandle_t handle, int n, const cuComplex *x, int incx, int *result) {
    return hipblasIcamin(handle, n, reinterpret_cast<const hipblasComplex*>(x), incx, result);
}

cublasStatus_t cublasIzamin(cublasHandle_t handle, int n, const cuDoubleComplex *x, int incx, int *result) {
    return hipblasIzamin(handle, n, reinterpret_cast<const hipblasDoubleComplex*>(x), incx, result);
}

cublasStatus_t cublasSasum(cublasHandle_t handle, int n, float* x, int incx, float* result) {
    return hipblasSasum(handle, n, x, incx, result);
}

cublasStatus_t cublasDasum(cublasHandle_t handle, int n, double* x, int incx, double* result) {
    return hipblasDasum(handle, n, x, incx, result);
}

cublasStatus_t cublasScasum(cublasHandle_t handle, int n, cuComplex* x, int incx, float* result) {
    return hipblasScasum(handle, n, reinterpret_cast<const hipblasComplex*>(x), incx, result);
}

cublasStatus_t cublasDzasum(cublasHandle_t handle, int n, cuDoubleComplex* x, int incx, double* result) {
    return hipblasDzasum(handle, n, reinterpret_cast<const hipblasDoubleComplex*>(x), incx, result);
}

cublasStatus_t cublasSaxpy(cublasHandle_t handle, int n, float* alpha, float* x, int incx, float* y, int incy) {
    return hipblasSaxpy(handle, n, alpha, x, incx, y, incy);
}

cublasStatus_t cublasDaxpy(cublasHandle_t handle, int n, double* alpha, double* x, int incx, double* y, int incy) {
    return hipblasDaxpy(handle, n, alpha, x, incx, y, incy);
}

cublasStatus_t cublasCaxpy(cublasHandle_t handle, int n, cuComplex* alpha, cuComplex* x, int incx, cuComplex* y, int incy) {
    return hipblasCaxpy(handle, n,
                        reinterpret_cast<const hipblasComplex*>(alpha),
                        reinterpret_cast<const hipblasComplex*>(x), incx,
                        reinterpret_cast<hipblasComplex*>(y), incy);
}

cublasStatus_t cublasZaxpy(cublasHandle_t handle, int n, cuDoubleComplex* alpha, cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy) {
    return hipblasZaxpy(handle, n,
                        reinterpret_cast<const hipblasDoubleComplex*>(alpha),
                        reinterpret_cast<const hipblasDoubleComplex*>(x), incx,
                        reinterpret_cast<hipblasDoubleComplex*>(y), incy);
}

cublasStatus_t cublasSdot(cublasHandle_t handle, int n, float* x, int incx, float* y, int incy, float* result) {
    return hipblasSdot(handle, n, x, incx, y, incy, result);
}

cublasStatus_t cublasDdot(cublasHandle_t handle, int n, double* x, int incx, double* y, int incy, double* result) {
    return hipblasDdot(handle, n, x, incx, y, incy, result);
}

cublasStatus_t cublasCdotu(cublasHandle_t handle, int n, cuComplex* x, int incx, cuComplex* y,
                           int incy, cuComplex* result) {
    return hipblasCdotu(handle, n,
                        reinterpret_cast<const hipblasComplex*>(x), incx,
                        reinterpret_cast<const hipblasComplex*>(y), incy,
                        reinterpret_cast<hipblasComplex*>(result));
}

cublasStatus_t cublasCdotc(cublasHandle_t handle, int n, cuComplex* x, int incx, cuComplex* y,
                           int incy, cuComplex* result) {
    return hipblasCdotc(handle, n,
                        reinterpret_cast<const hipblasComplex*>(x), incx,
                        reinterpret_cast<const hipblasComplex*>(y), incy,
                        reinterpret_cast<hipblasComplex*>(result));
}

cublasStatus_t cublasZdotu(cublasHandle_t handle, int n, cuDoubleComplex* x, int incx, cuDoubleComplex* y,
                           int incy, cuDoubleComplex* result) {
    return hipblasZdotu(handle, n,
                        reinterpret_cast<const hipblasDoubleComplex*>(x), incx,
                        reinterpret_cast<const hipblasDoubleComplex*>(y), incy,
                        reinterpret_cast<hipblasDoubleComplex*>(result));
}

cublasStatus_t cublasZdotc(cublasHandle_t handle, int n, cuDoubleComplex* x, int incx, cuDoubleComplex* y,
                           int incy, cuDoubleComplex* result) {
    return hipblasZdotc(handle, n,
                        reinterpret_cast<const hipblasDoubleComplex*>(x), incx,
                        reinterpret_cast<const hipblasDoubleComplex*>(y), incy,
                        reinterpret_cast<hipblasDoubleComplex*>(result));
}

cublasStatus_t cublasSnrm2(cublasHandle_t handle, int n, float* x, int incx, float* result) {
    return hipblasSnrm2(handle, n, x, incx, result);
}

cublasStatus_t cublasDnrm2(cublasHandle_t handle, int n, double* x, int incx, double* result) {
    return hipblasDnrm2(handle, n, x, incx, result);
}

cublasStatus_t cublasScnrm2(cublasHandle_t handle, int n, cuComplex* x, int incx, float* result) {
    return hipblasScnrm2(handle, n, reinterpret_cast<const hipblasComplex*>(x), incx, result);
}

cublasStatus_t cublasDznrm2(cublasHandle_t handle, int n, cuDoubleComplex* x, int incx, double* result) {
    return hipblasDznrm2(handle, n, reinterpret_cast<const hipblasDoubleComplex*>(x), incx, result);
}

cublasStatus_t cublasSscal(cublasHandle_t handle, int n, float* alpha, float* x, int incx) {
    return hipblasSscal(handle, n, alpha, x, incx);
}

cublasStatus_t cublasDscal(cublasHandle_t handle, int n, double* alpha, double* x, int incx) {
    return hipblasDscal(handle, n, alpha, x, incx);
}

cublasStatus_t cublasCscal(cublasHandle_t handle, int n, cuComplex* alpha, cuComplex* x, int incx) {
    return hipblasCscal(handle, n, reinterpret_cast<const hipblasComplex*>(alpha), reinterpret_cast<hipblasComplex*>(x), incx);
}

cublasStatus_t cublasCsscal(cublasHandle_t handle, int n, float* alpha, cuComplex* x, int incx) {
    return hipblasCsscal(handle, n, alpha, reinterpret_cast<hipblasComplex*>(x), incx);
}

cublasStatus_t cublasZscal(cublasHandle_t handle, int n, cuDoubleComplex* alpha, cuDoubleComplex* x, int incx) {
    return hipblasZscal(handle, n, reinterpret_cast<const hipblasDoubleComplex*>(alpha), reinterpret_cast<hipblasDoubleComplex*>(x), incx);
}

cublasStatus_t cublasZdscal(cublasHandle_t handle, int n, double* alpha, cuDoubleComplex* x, int incx) {
    return hipblasZdscal(handle, n, alpha, reinterpret_cast<hipblasDoubleComplex*>(x), incx);
}


// BLAS Level 2
cublasStatus_t cublasSgemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, float* alpha,
                           float* A, int lda, float* x, int incx, float* beta,
                           float* y, int incy) {
    return hipblasSgemv(handle, convert_hipblasOperation_t(trans), m, n,
                        alpha, A, lda, x, incx, beta, y, incy);
}

cublasStatus_t cublasDgemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, double* alpha,
                           double* A, int lda, double* x, int incx, double* beta,
                           double* y, int incy) {
    return hipblasDgemv(handle, convert_hipblasOperation_t(trans), m, n,
                        alpha, A, lda, x, incx, beta, y, incy);
}

cublasStatus_t cublasCgemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, cuComplex* alpha,
                           cuComplex* A, int lda, cuComplex* x, int incx, cuComplex* beta,
                           cuComplex* y, int incy) {
    return hipblasCgemv(handle, convert_hipblasOperation_t(trans), m, n,
                        reinterpret_cast<const hipblasComplex*>(alpha),
                        reinterpret_cast<const hipblasComplex*>(A), lda,
                        reinterpret_cast<const hipblasComplex*>(x), incx,
                        reinterpret_cast<const hipblasComplex*>(beta),
                        reinterpret_cast<hipblasComplex*>(y), incy);
}

cublasStatus_t cublasZgemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, cuDoubleComplex* alpha,
                           cuDoubleComplex* A, int lda, cuDoubleComplex* x, int incx,
                           cuDoubleComplex* beta, cuDoubleComplex* y, int incy) {
    return hipblasZgemv(handle, convert_hipblasOperation_t(trans), m, n,
                        reinterpret_cast<const hipblasDoubleComplex*>(alpha),
                        reinterpret_cast<const hipblasDoubleComplex*>(A), lda,
                        reinterpret_cast<const hipblasDoubleComplex*>(x), incx,
                        reinterpret_cast<const hipblasDoubleComplex*>(beta),
                        reinterpret_cast<hipblasDoubleComplex*>(y), incy);
}

cublasStatus_t cublasSger(cublasHandle_t handle, int m, int n, float* alpha, float* x, int incx,
                          float* y, int incy, float* A, int lda) {
    return hipblasSger(handle, m, n, alpha, x, incx, y, incy, A, lda);
}

cublasStatus_t cublasDger(cublasHandle_t handle, int m, int n, double* alpha, double* x,
                          int incx, double* y, int incy, double* A, int lda) {
    return hipblasDger(handle, m, n, alpha, x, incx, y, incy, A, lda);
}

cublasStatus_t cublasCgeru(cublasHandle_t handle, int m, int n, cuComplex* alpha, cuComplex* x,
                          int incx, cuComplex* y, int incy, cuComplex* A, int lda) {
    return hipblasCgeru(handle, m, n,
                        reinterpret_cast<const hipblasComplex*>(alpha),
                        reinterpret_cast<const hipblasComplex*>(x), incx,
                        reinterpret_cast<const hipblasComplex*>(y), incy,
                        reinterpret_cast<hipblasComplex*>(A), lda);
}

cublasStatus_t cublasCgerc(cublasHandle_t handle, int m, int n, cuComplex* alpha, cuComplex* x,
                          int incx, cuComplex* y, int incy, cuComplex* A, int lda) {
    return hipblasCgerc(handle, m, n,
                        reinterpret_cast<const hipblasComplex*>(alpha),
                        reinterpret_cast<const hipblasComplex*>(x), incx,
                        reinterpret_cast<const hipblasComplex*>(y), incy,
                        reinterpret_cast<hipblasComplex*>(A), lda);
}

cublasStatus_t cublasZgeru(cublasHandle_t handle, int m, int n, cuDoubleComplex* alpha,
                           cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy,
                           cuDoubleComplex* A, int lda) {
    return hipblasZgeru(handle, m, n,
                        reinterpret_cast<const hipblasDoubleComplex*>(alpha),
                        reinterpret_cast<const hipblasDoubleComplex*>(x), incx,
                        reinterpret_cast<const hipblasDoubleComplex*>(y), incy,
                        reinterpret_cast<hipblasDoubleComplex*>(A), lda);
}

cublasStatus_t cublasZgerc(cublasHandle_t handle, int m, int n, cuDoubleComplex* alpha,
                           cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy,
                           cuDoubleComplex* A, int lda) {
    return hipblasZgerc(handle, m, n,
                        reinterpret_cast<const hipblasDoubleComplex*>(alpha),
                        reinterpret_cast<const hipblasDoubleComplex*>(x), incx,
                        reinterpret_cast<const hipblasDoubleComplex*>(y), incy,
                        reinterpret_cast<hipblasDoubleComplex*>(A), lda);
}


// BLAS Level 3
cublasStatus_t cublasSgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                            int m, int n, int k, const float *alpha,
                            const float *A, int lda,
                            const float *B, int ldb,
                            const float *beta,
                            float *C, int ldc) {
    return hipblasSgemm(handle, convert_hipblasOperation_t(transa), convert_hipblasOperation_t(transb), m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

cublasStatus_t cublasDgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k, const double *alpha,
                           const double *A, int lda,
                           const double *B, int ldb,
                           const double *beta, double *C, int ldc) {
    return hipblasDgemm(handle, convert_hipblasOperation_t(transa), convert_hipblasOperation_t(transb), m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}


cublasStatus_t cublasCgemm(
        cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, int k, const cuComplex *alpha,
        const cuComplex *A, int lda,
        const cuComplex *B, int ldb,
        const cuComplex *beta, cuComplex *C, int ldc)
{
    return hipblasCgemm(
        handle, convert_hipblasOperation_t(transa), convert_hipblasOperation_t(transb),
        m, n, k,
        reinterpret_cast<const hipblasComplex *>(alpha),
        reinterpret_cast<const hipblasComplex *>(A), lda,
        reinterpret_cast<const hipblasComplex *>(B), ldb,
        reinterpret_cast<const hipblasComplex *>(beta),
        reinterpret_cast<hipblasComplex *>(C), ldc);
}

cublasStatus_t cublasZgemm(
        cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, int k, const cuDoubleComplex *alpha,
        const cuDoubleComplex *A, int lda,
        const cuDoubleComplex *B, int ldb,
        const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc)
{
    return hipblasZgemm(
        handle, convert_hipblasOperation_t(transa), convert_hipblasOperation_t(transb),
        m, n, k,
        reinterpret_cast<const hipblasDoubleComplex *>(alpha),
        reinterpret_cast<const hipblasDoubleComplex *>(A), lda,
        reinterpret_cast<const hipblasDoubleComplex *>(B), ldb,
        reinterpret_cast<const hipblasDoubleComplex *>(beta),
        reinterpret_cast<hipblasDoubleComplex *>(C), ldc);
}

cublasStatus_t cublasSgemmBatched(
        cublasHandle_t handle,
        cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, int k,  const float *alpha,
        const float *A[], int lda,
        const float *B[], int ldb,
        const float *beta,
        float *C[], int ldc, int batchCount) {
    return hipblasSgemmBatched(handle, convert_hipblasOperation_t(transa), convert_hipblasOperation_t(transb), m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
}

cublasStatus_t cublasDgemmBatched(
        cublasHandle_t handle,
        cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, int k,  const double *alpha,
        const double *A[], int lda,
        const double *B[], int ldb,
        const double *beta,
        double *C[], int ldc, int batchCount) {
    return hipblasDgemmBatched(handle, convert_hipblasOperation_t(transa), convert_hipblasOperation_t(transb), m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
}

cublasStatus_t cublasCgemmBatched(
        cublasHandle_t handle,
        cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, int k,  const cuComplex *alpha,
        const cuComplex *A[], int lda,
        const cuComplex *B[], int ldb,
        const cuComplex *beta,
        cuComplex *C[], int ldc, int batchCount) {
    return hipblasCgemmBatched(
        handle, convert_hipblasOperation_t(transa), convert_hipblasOperation_t(transb),
        m, n, k,
        reinterpret_cast<const hipblasComplex*>(alpha),
        reinterpret_cast<const hipblasComplex**>(A), lda,
        reinterpret_cast<const hipblasComplex**>(B), ldb,
        reinterpret_cast<const hipblasComplex*>(beta),
        reinterpret_cast<hipblasComplex**>(C), ldc, batchCount);
}

cublasStatus_t cublasZgemmBatched(
        cublasHandle_t handle,
        cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, int k,  const cuDoubleComplex *alpha,
        const cuDoubleComplex *A[], int lda,
        const cuDoubleComplex *B[], int ldb,
        const cuDoubleComplex *beta,
        cuDoubleComplex *C[], int ldc, int batchCount) {
    return hipblasZgemmBatched(
        handle, convert_hipblasOperation_t(transa), convert_hipblasOperation_t(transb),
        m, n, k,
        reinterpret_cast<const hipblasDoubleComplex*>(alpha),
        reinterpret_cast<const hipblasDoubleComplex**>(A), lda,
        reinterpret_cast<const hipblasDoubleComplex**>(B), ldb,
        reinterpret_cast<const hipblasDoubleComplex*>(beta),
        reinterpret_cast<hipblasDoubleComplex**>(C), ldc, batchCount);
}

cublasStatus_t cublasSgemmEx(
        cublasHandle_t handle,
        cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, int k,
        const float *alpha, const void *A, cudaDataType_t Atype, int lda,
        const void *B, cudaDataType_t Btype, int ldb, const float *beta,
        void *C, cudaDataType_t Ctype, int ldc) {
    if (Atype != 0 || Btype != 0 || Ctype != 0) {
        return HIPBLAS_STATUS_NOT_SUPPORTED;
    }
    return hipblasSgemm(
        handle,
        convert_hipblasOperation_t(transa), convert_hipblasOperation_t(transb),
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
        return HIPBLAS_STATUS_NOT_SUPPORTED;
    }
    return hipblasGemmEx(handle, convert_hipblasOperation_t(transa), convert_hipblasOperation_t(transb),
                         m, n, k, alpha,
                         A, convert_hipblasDatatype_t(Atype), lda,
                         B, convert_hipblasDatatype_t(Btype), ldb,
                         beta,
                         C, convert_hipblasDatatype_t(Ctype), ldc,
                         convert_hipblasDatatype_t(computetype),
                         static_cast<hipblasGemmAlgo_t>(160));  // HIPBLAS_GEMM_DEFAULT
}

cublasStatus_t cublasGemmEx_v11(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasStrsm(cublasHandle_t handle, cublasSideMode_t size, cublasFillMode_t uplo, cublasOperation_t trans,
                           cublasDiagType_t diag, int m, int n, const float* alpha,
                           const float* A, int lda, float* B, int ldb) {
    return hipblasStrsm(handle,
                        convert_hipblasSideMode_t(size),
                        convert_hipblasFillMode_t(uplo),
                        convert_hipblasOperation_t(trans),
                        convert_hipblasDiagType_t(diag),
                        m, n, alpha, const_cast<float*>(A), lda, B, ldb);
}

cublasStatus_t cublasDtrsm(cublasHandle_t handle, cublasSideMode_t size, cublasFillMode_t uplo, cublasOperation_t trans,
                           cublasDiagType_t diag, int m, int n, const double* alpha,
                           const double* A, int lda, double* B, int ldb) {
    return hipblasDtrsm(handle,
                        convert_hipblasSideMode_t(size),
                        convert_hipblasFillMode_t(uplo),
                        convert_hipblasOperation_t(trans),
                        convert_hipblasDiagType_t(diag),
                        m, n, alpha, const_cast<double*>(A), lda, B, ldb);
}

cublasStatus_t cublasCtrsm(cublasHandle_t handle, cublasSideMode_t size, cublasFillMode_t uplo, cublasOperation_t trans,
                           cublasDiagType_t diag, int m, int n, const cuComplex* alpha,
                           const cuComplex* A, int lda, cuComplex* B, int ldb) {
    return hipblasCtrsm(handle,
                        convert_hipblasSideMode_t(size),
                        convert_hipblasFillMode_t(uplo),
                        convert_hipblasOperation_t(trans),
                        convert_hipblasDiagType_t(diag),
                        m, n,
                        reinterpret_cast<const hipblasComplex*>(alpha),
                        reinterpret_cast<hipblasComplex*>(const_cast<cuComplex*>(A)), lda,
                        reinterpret_cast<hipblasComplex*>(B), ldb);
}

cublasStatus_t cublasZtrsm(cublasHandle_t handle, cublasSideMode_t size, cublasFillMode_t uplo, cublasOperation_t trans,
                           cublasDiagType_t diag, int m, int n, const cuDoubleComplex* alpha,
                           const cuDoubleComplex* A, int lda, cuDoubleComplex* B, int ldb) {
    return hipblasZtrsm(handle,
                        convert_hipblasSideMode_t(size),
                        convert_hipblasFillMode_t(uplo),
                        convert_hipblasOperation_t(trans),
                        convert_hipblasDiagType_t(diag),
                        m, n,
                        reinterpret_cast<const hipblasDoubleComplex*>(alpha),
                        reinterpret_cast<hipblasDoubleComplex*>(const_cast<cuDoubleComplex*>(A)), lda,
                        reinterpret_cast<hipblasDoubleComplex*>(B), ldb);
}


// BLAS extension
cublasStatus_t cublasSgeam(
        cublasHandle_t handle,
        cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, const float *alpha,
        const float *A, int lda, const float *beta, const float *B, int ldb,
        float *C, int ldc) {
    return hipblasSgeam(handle, convert_hipblasOperation_t(transa), convert_hipblasOperation_t(transb), m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}

cublasStatus_t cublasDgeam(
        cublasHandle_t handle,
        cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, const double *alpha,
        const double *A, int lda, const double *beta, const double *B, int ldb,
        double *C, int ldc) {
    return hipblasDgeam(handle, convert_hipblasOperation_t(transa), convert_hipblasOperation_t(transb), m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}

cublasStatus_t cublasCgeam(
        cublasHandle_t handle,
        cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, const cuComplex *alpha,
        const cuComplex *A, int lda, const cuComplex *beta, const cuComplex *B, int ldb,
        cuComplex *C, int ldc) {
    #if HIP_VERSION < 307
    return HIPBLAS_STATUS_NOT_SUPPORTED;
    #else
    return hipblasCgeam(handle, convert_hipblasOperation_t(transa), convert_hipblasOperation_t(transb), m, n,
                        reinterpret_cast<const hipblasComplex*>(alpha),
                        reinterpret_cast<const hipblasComplex*>(A),
                        lda,
                        reinterpret_cast<const hipblasComplex*>(beta),
                        reinterpret_cast<const hipblasComplex*>(B),
                        ldb,
                        reinterpret_cast<hipblasComplex*>(C),
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
    return HIPBLAS_STATUS_NOT_SUPPORTED;
    #else
    return hipblasZgeam(handle, convert_hipblasOperation_t(transa), convert_hipblasOperation_t(transb), m, n,
                        reinterpret_cast<const hipblasDoubleComplex*>(alpha),
                        reinterpret_cast<const hipblasDoubleComplex*>(A),
                        lda,
                        reinterpret_cast<const hipblasDoubleComplex*>(beta),
                        reinterpret_cast<const hipblasDoubleComplex*>(B),
                        ldb,
                        reinterpret_cast<hipblasDoubleComplex*>(C),
                        ldc);
    #endif
}

cublasStatus_t cublasSdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n,
                           const float *A, int lda,
                           const float *x, int incx,
                           float *C, int ldc) {
    #if HIP_VERSION < 306
    return HIPBLAS_STATUS_NOT_SUPPORTED;
    #else
    return hipblasSdgmm(handle, convert_hipblasSideMode_t(mode), m, n, A, lda, x, incx, C, ldc);
    #endif
}

cublasStatus_t cublasDdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n,
                           const double *A, int lda,
                           const double *x, int incx,
                           double *C, int ldc) {
    #if HIP_VERSION < 306
    return HIPBLAS_STATUS_NOT_SUPPORTED;
    #else
    return hipblasDdgmm(handle, convert_hipblasSideMode_t(mode), m, n, A, lda, x, incx, C, ldc);
    #endif
}

cublasStatus_t cublasCdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n,
                           const cuComplex *A, int lda,
                           const cuComplex *x, int incx,
                           cuComplex *C, int ldc) {
    #if HIP_VERSION < 306
    return HIPBLAS_STATUS_NOT_SUPPORTED;
    #else
    return hipblasCdgmm(handle, convert_hipblasSideMode_t(mode), m, n,
                        reinterpret_cast<const hipblasComplex*>(A), lda,
                        reinterpret_cast<const hipblasComplex*>(x), incx,
                        reinterpret_cast<hipblasComplex*>(C), ldc);
    #endif
}

cublasStatus_t cublasZdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n,
                           const cuDoubleComplex *A, int lda,
                           const cuDoubleComplex *x, int incx,
                           cuDoubleComplex *C, int ldc) {
    #if HIP_VERSION < 306
    return HIPBLAS_STATUS_NOT_SUPPORTED;
    #else
    return hipblasZdgmm(handle, convert_hipblasSideMode_t(mode), m, n,
                        reinterpret_cast<const hipblasDoubleComplex*>(A), lda,
                        reinterpret_cast<const hipblasDoubleComplex*>(x), incx,
                        reinterpret_cast<hipblasDoubleComplex*>(C), ldc);
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
    return HIPBLAS_STATUS_NOT_SUPPORTED;
    #else
    return hipblasSgetriBatched(handle, n, const_cast<float* const*>(A), lda, const_cast<int*>(P), C, ldc, info, batchSize);
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
    return HIPBLAS_STATUS_NOT_SUPPORTED;
    #else
    return hipblasDgetriBatched(handle, n, const_cast<double* const*>(A), lda, const_cast<int*>(P), C, ldc, info, batchSize);
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
    return HIPBLAS_STATUS_NOT_SUPPORTED;
    #else
    return hipblasCgetriBatched(handle, n,
                                reinterpret_cast<hipblasComplex* const*>(const_cast<cuComplex* const*>(A)),
                                lda, const_cast<int*>(P),
                                reinterpret_cast<hipblasComplex* const*>(C),
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
    return HIPBLAS_STATUS_NOT_SUPPORTED;
    #else
    return hipblasZgetriBatched(handle, n,
                                reinterpret_cast<hipblasDoubleComplex* const*>(const_cast<cuDoubleComplex* const*>(A)),
                                lda, const_cast<int*>(P),
                                reinterpret_cast<hipblasDoubleComplex* const*>(C),
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
    return hipblasSgemmStridedBatched(
        handle,
        convert_hipblasOperation_t(transa), convert_hipblasOperation_t(transb),
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
    return hipblasDgemmStridedBatched(
        handle,
        convert_hipblasOperation_t(transa), convert_hipblasOperation_t(transb),
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
    return hipblasCgemmStridedBatched(
        handle,
        convert_hipblasOperation_t(transa), convert_hipblasOperation_t(transb),
        m, n, k,
        reinterpret_cast<const hipblasComplex *>(alpha),
        reinterpret_cast<const hipblasComplex *>(A), lda, bsa,
        reinterpret_cast<const hipblasComplex *>(B), ldb, bsb,
        reinterpret_cast<const hipblasComplex *>(beta),
        reinterpret_cast<hipblasComplex *>(C), ldc, bsc,
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
    return hipblasZgemmStridedBatched(
        handle,
        convert_hipblasOperation_t(transa), convert_hipblasOperation_t(transb),
        m, n, k,
        reinterpret_cast<const hipblasDoubleComplex *>(alpha),
        reinterpret_cast<const hipblasDoubleComplex *>(A), lda, bsa,
        reinterpret_cast<const hipblasDoubleComplex *>(B), ldb, bsb,
        reinterpret_cast<const hipblasDoubleComplex *>(beta),
        reinterpret_cast<hipblasDoubleComplex *>(C), ldc, bsc,
        batchCount);
}

cublasStatus_t cublasStrttp(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasDtrttp(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasStpttr(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasDtpttr(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasSgetrfBatched(cublasHandle_t handle, int n, float **Aarray, int lda,
                                   int *PivotArray, int *infoArray, int batchSize) {
    return hipblasSgetrfBatched(handle, n, Aarray, lda, PivotArray, infoArray, batchSize);
}

cublasStatus_t cublasDgetrfBatched(cublasHandle_t handle, int n, double **Aarray, int lda,
                                   int *PivotArray, int *infoArray, int batchSize) {
    return hipblasDgetrfBatched(handle, n, Aarray, lda, PivotArray, infoArray, batchSize);
}

cublasStatus_t cublasCgetrfBatched(cublasHandle_t handle, int n, cuComplex **Aarray, int lda,
                                   int *PivotArray, int *infoArray, int batchSize) {
    return hipblasCgetrfBatched(handle, n,
                                reinterpret_cast<hipblasComplex** const>(Aarray), lda,
                                PivotArray, infoArray, batchSize);
}

cublasStatus_t cublasZgetrfBatched(cublasHandle_t handle, int n, cuDoubleComplex **Aarray, int lda,
                                   int *PivotArray, int *infoArray, int batchSize) {
    return hipblasZgetrfBatched(handle, n,
                                reinterpret_cast<hipblasDoubleComplex** const>(Aarray), lda,
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
    return hipblasSgetrsBatched(handle,
                                convert_hipblasOperation_t(trans),
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
    return hipblasDgetrsBatched(handle,
                                convert_hipblasOperation_t(trans),
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
    return hipblasCgetrsBatched(handle,
                                convert_hipblasOperation_t(trans),
                                n, nrhs,
                                reinterpret_cast<hipblasComplex* const*>(const_cast<cuComplex* const*>(Aarray)), lda,
                                devIpiv,
                                reinterpret_cast<hipblasComplex* const*>(Barray), ldb,
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
    return hipblasZgetrsBatched(handle,
                                convert_hipblasOperation_t(trans),
                                n, nrhs,
                                reinterpret_cast<hipblasDoubleComplex* const*>(const_cast<cuDoubleComplex* const*>(Aarray)), lda,
                                devIpiv,
                                reinterpret_cast<hipblasDoubleComplex* const*>(Barray), ldb,
                                info, batchSize);
}

} // extern "C"

#endif // #ifndef INCLUDE_GUARD_HIP_CUPY_HIPBLAS_H
