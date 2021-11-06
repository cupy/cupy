// This code was automatically generated. Do not modify it directly.

#ifndef INCLUDE_GUARD_HIP_CUPY_HIPBLAS_H
#define INCLUDE_GUARD_HIP_CUPY_HIPBLAS_H

#include "cupy_hip_common.h"
#include <hipblas.h>
#include <hip/hip_version.h>  // for HIP_VERSION
#include <stdexcept>  // for gcc 10

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

typedef hipblasHandle_t cublasHandle_t;

typedef hipblasStatus_t cublasStatus_t;
typedef enum {} cublasFillMode_t;
typedef enum {} cublasDiagType_t;
typedef enum {} cublasSideMode_t;
typedef enum {} cublasOperation_t;
typedef hipblasPointerMode_t cublasPointerMode_t;
typedef hipblasAtomicsMode_t cublasAtomicsMode_t;
typedef enum {} cublasGemmAlgo_t;
typedef enum {} cublasMath_t;
typedef enum {} cublasComputeType_t;

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

cublasStatus_t cublasCreate(cublasHandle_t* handle) {
    return hipblasCreate(handle);
}

cublasStatus_t cublasDestroy(cublasHandle_t handle) {
    return hipblasDestroy(handle);
}

cublasStatus_t cublasGetProperty(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

size_t cublasGetCudartVersion(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasSetStream(cublasHandle_t handle, cudaStream_t streamId) {
    return hipblasSetStream(handle, streamId);
}

cublasStatus_t cublasGetStream(cublasHandle_t handle, cudaStream_t* streamId) {
    return hipblasGetStream(handle, streamId);
}

cublasStatus_t cublasGetPointerMode(cublasHandle_t handle, cublasPointerMode_t* mode) {
    return hipblasGetPointerMode(handle, mode);
}

cublasStatus_t cublasSetPointerMode(cublasHandle_t handle, cublasPointerMode_t mode) {
    return hipblasSetPointerMode(handle, mode);
}

cublasStatus_t cublasGetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t* mode) {
    return hipblasGetAtomicsMode(handle, mode);
}

cublasStatus_t cublasSetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t mode) {
    return hipblasSetAtomicsMode(handle, mode);
}

cublasStatus_t cublasGetMathMode(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasSetMathMode(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasSetVector(int n, int elemSize, const void* x, int incx, void* devicePtr, int incy) {
    return hipblasSetVector(n, elemSize, x, incx, devicePtr, incy);
}

cublasStatus_t cublasGetVector(int n, int elemSize, const void* x, int incx, void* y, int incy) {
    return hipblasGetVector(n, elemSize, x, incx, y, incy);
}

cublasStatus_t cublasSetMatrix(int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb) {
    return hipblasSetMatrix(rows, cols, elemSize, A, lda, B, ldb);
}

cublasStatus_t cublasGetMatrix(int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb) {
    return hipblasGetMatrix(rows, cols, elemSize, A, lda, B, ldb);
}

cublasStatus_t cublasSetVectorAsync(int n, int elemSize, const void* hostPtr, int incx, void* devicePtr, int incy, cudaStream_t stream) {
    return hipblasSetVectorAsync(n, elemSize, hostPtr, incx, devicePtr, incy, stream);
}

cublasStatus_t cublasGetVectorAsync(int n, int elemSize, const void* devicePtr, int incx, void* hostPtr, int incy, cudaStream_t stream) {
    return hipblasGetVectorAsync(n, elemSize, devicePtr, incx, hostPtr, incy, stream);
}

cublasStatus_t cublasSetMatrixAsync(int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb, cudaStream_t stream) {
    return hipblasSetMatrixAsync(rows, cols, elemSize, A, lda, B, ldb, stream);
}

cublasStatus_t cublasGetMatrixAsync(int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb, cudaStream_t stream) {
    return hipblasGetMatrixAsync(rows, cols, elemSize, A, lda, B, ldb, stream);
}

cublasStatus_t cublasNrm2Ex(cublasHandle_t handle, int n, const void* x, cudaDataType xType, int incx, void* result, cudaDataType resultType, cudaDataType executionType) {
    #if HIP_VERSION < 401
        return HIPBLAS_STATUS_NOT_SUPPORTED;
    #else
        return hipblasNrm2Ex(handle, n, x, convert_hipblasDatatype_t(xType), incx, result, convert_hipblasDatatype_t(resultType), convert_hipblasDatatype_t(executionType));
    #endif
}

cublasStatus_t cublasSnrm2(cublasHandle_t handle, int n, const float* x, int incx, float* result) {
    return hipblasSnrm2(handle, n, x, incx, result);
}

cublasStatus_t cublasDnrm2(cublasHandle_t handle, int n, const double* x, int incx, double* result) {
    return hipblasDnrm2(handle, n, x, incx, result);
}

cublasStatus_t cublasScnrm2(cublasHandle_t handle, int n, const cuComplex* x, int incx, float* result) {
    return hipblasScnrm2(handle, n, reinterpret_cast<const hipblasComplex*>(x), incx, result);
}

cublasStatus_t cublasDznrm2(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, double* result) {
    return hipblasDznrm2(handle, n, reinterpret_cast<const hipblasDoubleComplex*>(x), incx, result);
}

cublasStatus_t cublasDotEx(cublasHandle_t handle, int n, const void* x, cudaDataType xType, int incx, const void* y, cudaDataType yType, int incy, void* result, cudaDataType resultType, cudaDataType executionType) {
    #if HIP_VERSION < 401
        return HIPBLAS_STATUS_NOT_SUPPORTED;
    #else
        return hipblasDotEx(handle, n, x, convert_hipblasDatatype_t(xType), incx, y, convert_hipblasDatatype_t(yType), incy, result, convert_hipblasDatatype_t(resultType), convert_hipblasDatatype_t(executionType));
    #endif
}

cublasStatus_t cublasDotcEx(cublasHandle_t handle, int n, const void* x, cudaDataType xType, int incx, const void* y, cudaDataType yType, int incy, void* result, cudaDataType resultType, cudaDataType executionType) {
    #if HIP_VERSION < 401
        return HIPBLAS_STATUS_NOT_SUPPORTED;
    #else
        return hipblasDotcEx(handle, n, x, convert_hipblasDatatype_t(xType), incx, y, convert_hipblasDatatype_t(yType), incy, result, convert_hipblasDatatype_t(resultType), convert_hipblasDatatype_t(executionType));
    #endif
}

cublasStatus_t cublasSdot(cublasHandle_t handle, int n, const float* x, int incx, const float* y, int incy, float* result) {
    return hipblasSdot(handle, n, x, incx, y, incy, result);
}

cublasStatus_t cublasDdot(cublasHandle_t handle, int n, const double* x, int incx, const double* y, int incy, double* result) {
    return hipblasDdot(handle, n, x, incx, y, incy, result);
}

cublasStatus_t cublasCdotu(cublasHandle_t handle, int n, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* result) {
    return hipblasCdotu(handle, n, reinterpret_cast<const hipblasComplex*>(x), incx, reinterpret_cast<const hipblasComplex*>(y), incy, reinterpret_cast<hipblasComplex*>(result));
}

cublasStatus_t cublasCdotc(cublasHandle_t handle, int n, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* result) {
    return hipblasCdotc(handle, n, reinterpret_cast<const hipblasComplex*>(x), incx, reinterpret_cast<const hipblasComplex*>(y), incy, reinterpret_cast<hipblasComplex*>(result));
}

cublasStatus_t cublasZdotu(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* result) {
    return hipblasZdotu(handle, n, reinterpret_cast<const hipblasDoubleComplex*>(x), incx, reinterpret_cast<const hipblasDoubleComplex*>(y), incy, reinterpret_cast<hipblasDoubleComplex*>(result));
}

cublasStatus_t cublasZdotc(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* result) {
    return hipblasZdotc(handle, n, reinterpret_cast<const hipblasDoubleComplex*>(x), incx, reinterpret_cast<const hipblasDoubleComplex*>(y), incy, reinterpret_cast<hipblasDoubleComplex*>(result));
}

cublasStatus_t cublasScalEx(cublasHandle_t handle, int n, const void* alpha, cudaDataType alphaType, void* x, cudaDataType xType, int incx, cudaDataType executionType) {
    #if HIP_VERSION < 401
        return HIPBLAS_STATUS_NOT_SUPPORTED;
    #else
        return hipblasScalEx(handle, n, alpha, convert_hipblasDatatype_t(alphaType), x, convert_hipblasDatatype_t(xType), incx, convert_hipblasDatatype_t(executionType));
    #endif
}

cublasStatus_t cublasSscal(cublasHandle_t handle, int n, const float* alpha, float* x, int incx) {
    return hipblasSscal(handle, n, alpha, x, incx);
}

cublasStatus_t cublasDscal(cublasHandle_t handle, int n, const double* alpha, double* x, int incx) {
    return hipblasDscal(handle, n, alpha, x, incx);
}

cublasStatus_t cublasCscal(cublasHandle_t handle, int n, const cuComplex* alpha, cuComplex* x, int incx) {
    return hipblasCscal(handle, n, reinterpret_cast<const hipblasComplex*>(alpha), reinterpret_cast<hipblasComplex*>(x), incx);
}

cublasStatus_t cublasCsscal(cublasHandle_t handle, int n, const float* alpha, cuComplex* x, int incx) {
    return hipblasCsscal(handle, n, alpha, reinterpret_cast<hipblasComplex*>(x), incx);
}

cublasStatus_t cublasZscal(cublasHandle_t handle, int n, const cuDoubleComplex* alpha, cuDoubleComplex* x, int incx) {
    return hipblasZscal(handle, n, reinterpret_cast<const hipblasDoubleComplex*>(alpha), reinterpret_cast<hipblasDoubleComplex*>(x), incx);
}

cublasStatus_t cublasZdscal(cublasHandle_t handle, int n, const double* alpha, cuDoubleComplex* x, int incx) {
    return hipblasZdscal(handle, n, alpha, reinterpret_cast<hipblasDoubleComplex*>(x), incx);
}

cublasStatus_t cublasAxpyEx(cublasHandle_t handle, int n, const void* alpha, cudaDataType alphaType, const void* x, cudaDataType xType, int incx, void* y, cudaDataType yType, int incy, cudaDataType executiontype) {
    #if HIP_VERSION < 401
        return HIPBLAS_STATUS_NOT_SUPPORTED;
    #else
        return hipblasAxpyEx(handle, n, alpha, convert_hipblasDatatype_t(alphaType), x, convert_hipblasDatatype_t(xType), incx, y, convert_hipblasDatatype_t(yType), incy, convert_hipblasDatatype_t(executiontype));
    #endif
}

cublasStatus_t cublasSaxpy(cublasHandle_t handle, int n, const float* alpha, const float* x, int incx, float* y, int incy) {
    return hipblasSaxpy(handle, n, alpha, x, incx, y, incy);
}

cublasStatus_t cublasDaxpy(cublasHandle_t handle, int n, const double* alpha, const double* x, int incx, double* y, int incy) {
    return hipblasDaxpy(handle, n, alpha, x, incx, y, incy);
}

cublasStatus_t cublasCaxpy(cublasHandle_t handle, int n, const cuComplex* alpha, const cuComplex* x, int incx, cuComplex* y, int incy) {
    return hipblasCaxpy(handle, n, reinterpret_cast<const hipblasComplex*>(alpha), reinterpret_cast<const hipblasComplex*>(x), incx, reinterpret_cast<hipblasComplex*>(y), incy);
}

cublasStatus_t cublasZaxpy(cublasHandle_t handle, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy) {
    return hipblasZaxpy(handle, n, reinterpret_cast<const hipblasDoubleComplex*>(alpha), reinterpret_cast<const hipblasDoubleComplex*>(x), incx, reinterpret_cast<hipblasDoubleComplex*>(y), incy);
}

cublasStatus_t cublasCopyEx(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasScopy(cublasHandle_t handle, int n, const float* x, int incx, float* y, int incy) {
    return hipblasScopy(handle, n, x, incx, y, incy);
}

cublasStatus_t cublasDcopy(cublasHandle_t handle, int n, const double* x, int incx, double* y, int incy) {
    return hipblasDcopy(handle, n, x, incx, y, incy);
}

cublasStatus_t cublasCcopy(cublasHandle_t handle, int n, const cuComplex* x, int incx, cuComplex* y, int incy) {
    return hipblasCcopy(handle, n, reinterpret_cast<const hipblasComplex*>(x), incx, reinterpret_cast<hipblasComplex*>(y), incy);
}

cublasStatus_t cublasZcopy(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy) {
    return hipblasZcopy(handle, n, reinterpret_cast<const hipblasDoubleComplex*>(x), incx, reinterpret_cast<hipblasDoubleComplex*>(y), incy);
}

cublasStatus_t cublasSswap(cublasHandle_t handle, int n, float* x, int incx, float* y, int incy) {
    return hipblasSswap(handle, n, x, incx, y, incy);
}

cublasStatus_t cublasDswap(cublasHandle_t handle, int n, double* x, int incx, double* y, int incy) {
    return hipblasDswap(handle, n, x, incx, y, incy);
}

cublasStatus_t cublasCswap(cublasHandle_t handle, int n, cuComplex* x, int incx, cuComplex* y, int incy) {
    return hipblasCswap(handle, n, reinterpret_cast<hipblasComplex*>(x), incx, reinterpret_cast<hipblasComplex*>(y), incy);
}

cublasStatus_t cublasZswap(cublasHandle_t handle, int n, cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy) {
    return hipblasZswap(handle, n, reinterpret_cast<hipblasDoubleComplex*>(x), incx, reinterpret_cast<hipblasDoubleComplex*>(y), incy);
}

cublasStatus_t cublasSwapEx(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasIsamax(cublasHandle_t handle, int n, const float* x, int incx, int* result) {
    return hipblasIsamax(handle, n, x, incx, result);
}

cublasStatus_t cublasIdamax(cublasHandle_t handle, int n, const double* x, int incx, int* result) {
    return hipblasIdamax(handle, n, x, incx, result);
}

cublasStatus_t cublasIcamax(cublasHandle_t handle, int n, const cuComplex* x, int incx, int* result) {
    return hipblasIcamax(handle, n, reinterpret_cast<const hipblasComplex*>(x), incx, result);
}

cublasStatus_t cublasIzamax(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, int* result) {
    return hipblasIzamax(handle, n, reinterpret_cast<const hipblasDoubleComplex*>(x), incx, result);
}

cublasStatus_t cublasIamaxEx(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasIsamin(cublasHandle_t handle, int n, const float* x, int incx, int* result) {
    return hipblasIsamin(handle, n, x, incx, result);
}

cublasStatus_t cublasIdamin(cublasHandle_t handle, int n, const double* x, int incx, int* result) {
    return hipblasIdamin(handle, n, x, incx, result);
}

cublasStatus_t cublasIcamin(cublasHandle_t handle, int n, const cuComplex* x, int incx, int* result) {
    return hipblasIcamin(handle, n, reinterpret_cast<const hipblasComplex*>(x), incx, result);
}

cublasStatus_t cublasIzamin(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, int* result) {
    return hipblasIzamin(handle, n, reinterpret_cast<const hipblasDoubleComplex*>(x), incx, result);
}

cublasStatus_t cublasIaminEx(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasAsumEx(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasSasum(cublasHandle_t handle, int n, const float* x, int incx, float* result) {
    return hipblasSasum(handle, n, x, incx, result);
}

cublasStatus_t cublasDasum(cublasHandle_t handle, int n, const double* x, int incx, double* result) {
    return hipblasDasum(handle, n, x, incx, result);
}

cublasStatus_t cublasScasum(cublasHandle_t handle, int n, const cuComplex* x, int incx, float* result) {
    return hipblasScasum(handle, n, reinterpret_cast<const hipblasComplex*>(x), incx, result);
}

cublasStatus_t cublasDzasum(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, double* result) {
    return hipblasDzasum(handle, n, reinterpret_cast<const hipblasDoubleComplex*>(x), incx, result);
}

cublasStatus_t cublasSrot(cublasHandle_t handle, int n, float* x, int incx, float* y, int incy, const float* c, const float* s) {
    return hipblasSrot(handle, n, x, incx, y, incy, c, s);
}

cublasStatus_t cublasDrot(cublasHandle_t handle, int n, double* x, int incx, double* y, int incy, const double* c, const double* s) {
    return hipblasDrot(handle, n, x, incx, y, incy, c, s);
}

cublasStatus_t cublasCrot(cublasHandle_t handle, int n, cuComplex* x, int incx, cuComplex* y, int incy, const float* c, const cuComplex* s) {
    return hipblasCrot(handle, n, reinterpret_cast<hipblasComplex*>(x), incx, reinterpret_cast<hipblasComplex*>(y), incy, c, reinterpret_cast<const hipblasComplex*>(s));
}

cublasStatus_t cublasCsrot(cublasHandle_t handle, int n, cuComplex* x, int incx, cuComplex* y, int incy, const float* c, const float* s) {
    return hipblasCsrot(handle, n, reinterpret_cast<hipblasComplex*>(x), incx, reinterpret_cast<hipblasComplex*>(y), incy, c, s);
}

cublasStatus_t cublasZrot(cublasHandle_t handle, int n, cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy, const double* c, const cuDoubleComplex* s) {
    return hipblasZrot(handle, n, reinterpret_cast<hipblasDoubleComplex*>(x), incx, reinterpret_cast<hipblasDoubleComplex*>(y), incy, c, reinterpret_cast<const hipblasDoubleComplex*>(s));
}

cublasStatus_t cublasZdrot(cublasHandle_t handle, int n, cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy, const double* c, const double* s) {
    return hipblasZdrot(handle, n, reinterpret_cast<hipblasDoubleComplex*>(x), incx, reinterpret_cast<hipblasDoubleComplex*>(y), incy, c, s);
}

cublasStatus_t cublasRotEx(cublasHandle_t handle, int n, void* x, cudaDataType xType, int incx, void* y, cudaDataType yType, int incy, const void* c, const void* s, cudaDataType csType, cudaDataType executiontype) {
    #if HIP_VERSION < 401
        return HIPBLAS_STATUS_NOT_SUPPORTED;
    #else
        return hipblasRotEx(handle, n, x, convert_hipblasDatatype_t(xType), incx, y, convert_hipblasDatatype_t(yType), incy, c, s, convert_hipblasDatatype_t(csType), convert_hipblasDatatype_t(executiontype));
    #endif
}

cublasStatus_t cublasSrotg(cublasHandle_t handle, float* a, float* b, float* c, float* s) {
    return hipblasSrotg(handle, a, b, c, s);
}

cublasStatus_t cublasDrotg(cublasHandle_t handle, double* a, double* b, double* c, double* s) {
    return hipblasDrotg(handle, a, b, c, s);
}

cublasStatus_t cublasCrotg(cublasHandle_t handle, cuComplex* a, cuComplex* b, float* c, cuComplex* s) {
    return hipblasCrotg(handle, reinterpret_cast<hipblasComplex*>(a), reinterpret_cast<hipblasComplex*>(b), c, reinterpret_cast<hipblasComplex*>(s));
}

cublasStatus_t cublasZrotg(cublasHandle_t handle, cuDoubleComplex* a, cuDoubleComplex* b, double* c, cuDoubleComplex* s) {
    return hipblasZrotg(handle, reinterpret_cast<hipblasDoubleComplex*>(a), reinterpret_cast<hipblasDoubleComplex*>(b), c, reinterpret_cast<hipblasDoubleComplex*>(s));
}

cublasStatus_t cublasRotgEx(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasSrotm(cublasHandle_t handle, int n, float* x, int incx, float* y, int incy, const float* param) {
    return hipblasSrotm(handle, n, x, incx, y, incy, param);
}

cublasStatus_t cublasDrotm(cublasHandle_t handle, int n, double* x, int incx, double* y, int incy, const double* param) {
    return hipblasDrotm(handle, n, x, incx, y, incy, param);
}

cublasStatus_t cublasRotmEx(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasSrotmg(cublasHandle_t handle, float* d1, float* d2, float* x1, const float* y1, float* param) {
    return hipblasSrotmg(handle, d1, d2, x1, y1, param);
}

cublasStatus_t cublasDrotmg(cublasHandle_t handle, double* d1, double* d2, double* x1, const double* y1, double* param) {
    return hipblasDrotmg(handle, d1, d2, x1, y1, param);
}

cublasStatus_t cublasRotmgEx(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasSgemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const float* alpha, const float* A, int lda, const float* x, int incx, const float* beta, float* y, int incy) {
    return hipblasSgemv(handle, convert_hipblasOperation_t(trans), m, n, alpha, A, lda, x, incx, beta, y, incy);
}

cublasStatus_t cublasDgemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const double* alpha, const double* A, int lda, const double* x, int incx, const double* beta, double* y, int incy) {
    return hipblasDgemv(handle, convert_hipblasOperation_t(trans), m, n, alpha, A, lda, x, incx, beta, y, incy);
}

cublasStatus_t cublasCgemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* x, int incx, const cuComplex* beta, cuComplex* y, int incy) {
    return hipblasCgemv(handle, convert_hipblasOperation_t(trans), m, n, reinterpret_cast<const hipblasComplex*>(alpha), reinterpret_cast<const hipblasComplex*>(A), lda, reinterpret_cast<const hipblasComplex*>(x), incx, reinterpret_cast<const hipblasComplex*>(beta), reinterpret_cast<hipblasComplex*>(y), incy);
}

cublasStatus_t cublasZgemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int incy) {
    return hipblasZgemv(handle, convert_hipblasOperation_t(trans), m, n, reinterpret_cast<const hipblasDoubleComplex*>(alpha), reinterpret_cast<const hipblasDoubleComplex*>(A), lda, reinterpret_cast<const hipblasDoubleComplex*>(x), incx, reinterpret_cast<const hipblasDoubleComplex*>(beta), reinterpret_cast<hipblasDoubleComplex*>(y), incy);
}

cublasStatus_t cublasSgbmv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku, const float* alpha, const float* A, int lda, const float* x, int incx, const float* beta, float* y, int incy) {
    return hipblasSgbmv(handle, convert_hipblasOperation_t(trans), m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
}

cublasStatus_t cublasDgbmv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku, const double* alpha, const double* A, int lda, const double* x, int incx, const double* beta, double* y, int incy) {
    return hipblasDgbmv(handle, convert_hipblasOperation_t(trans), m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
}

cublasStatus_t cublasCgbmv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* x, int incx, const cuComplex* beta, cuComplex* y, int incy) {
    return hipblasCgbmv(handle, convert_hipblasOperation_t(trans), m, n, kl, ku, reinterpret_cast<const hipblasComplex*>(alpha), reinterpret_cast<const hipblasComplex*>(A), lda, reinterpret_cast<const hipblasComplex*>(x), incx, reinterpret_cast<const hipblasComplex*>(beta), reinterpret_cast<hipblasComplex*>(y), incy);
}

cublasStatus_t cublasZgbmv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int incy) {
    return hipblasZgbmv(handle, convert_hipblasOperation_t(trans), m, n, kl, ku, reinterpret_cast<const hipblasDoubleComplex*>(alpha), reinterpret_cast<const hipblasDoubleComplex*>(A), lda, reinterpret_cast<const hipblasDoubleComplex*>(x), incx, reinterpret_cast<const hipblasDoubleComplex*>(beta), reinterpret_cast<hipblasDoubleComplex*>(y), incy);
}

cublasStatus_t cublasStrmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const float* A, int lda, float* x, int incx) {
    return hipblasStrmv(handle, convert_hipblasFillMode_t(uplo), convert_hipblasOperation_t(trans), convert_hipblasDiagType_t(diag), n, A, lda, x, incx);
}

cublasStatus_t cublasDtrmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const double* A, int lda, double* x, int incx) {
    return hipblasDtrmv(handle, convert_hipblasFillMode_t(uplo), convert_hipblasOperation_t(trans), convert_hipblasDiagType_t(diag), n, A, lda, x, incx);
}

cublasStatus_t cublasCtrmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuComplex* A, int lda, cuComplex* x, int incx) {
    return hipblasCtrmv(handle, convert_hipblasFillMode_t(uplo), convert_hipblasOperation_t(trans), convert_hipblasDiagType_t(diag), n, reinterpret_cast<const hipblasComplex*>(A), lda, reinterpret_cast<hipblasComplex*>(x), incx);
}

cublasStatus_t cublasZtrmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuDoubleComplex* A, int lda, cuDoubleComplex* x, int incx) {
    return hipblasZtrmv(handle, convert_hipblasFillMode_t(uplo), convert_hipblasOperation_t(trans), convert_hipblasDiagType_t(diag), n, reinterpret_cast<const hipblasDoubleComplex*>(A), lda, reinterpret_cast<hipblasDoubleComplex*>(x), incx);
}

cublasStatus_t cublasStbmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const float* A, int lda, float* x, int incx) {
    return hipblasStbmv(handle, convert_hipblasFillMode_t(uplo), convert_hipblasOperation_t(trans), convert_hipblasDiagType_t(diag), n, k, A, lda, x, incx);
}

cublasStatus_t cublasDtbmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const double* A, int lda, double* x, int incx) {
    return hipblasDtbmv(handle, convert_hipblasFillMode_t(uplo), convert_hipblasOperation_t(trans), convert_hipblasDiagType_t(diag), n, k, A, lda, x, incx);
}

cublasStatus_t cublasCtbmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const cuComplex* A, int lda, cuComplex* x, int incx) {
    return hipblasCtbmv(handle, convert_hipblasFillMode_t(uplo), convert_hipblasOperation_t(trans), convert_hipblasDiagType_t(diag), n, k, reinterpret_cast<const hipblasComplex*>(A), lda, reinterpret_cast<hipblasComplex*>(x), incx);
}

cublasStatus_t cublasZtbmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const cuDoubleComplex* A, int lda, cuDoubleComplex* x, int incx) {
    return hipblasZtbmv(handle, convert_hipblasFillMode_t(uplo), convert_hipblasOperation_t(trans), convert_hipblasDiagType_t(diag), n, k, reinterpret_cast<const hipblasDoubleComplex*>(A), lda, reinterpret_cast<hipblasDoubleComplex*>(x), incx);
}

cublasStatus_t cublasStpmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const float* AP, float* x, int incx) {
    return hipblasStpmv(handle, convert_hipblasFillMode_t(uplo), convert_hipblasOperation_t(trans), convert_hipblasDiagType_t(diag), n, AP, x, incx);
}

cublasStatus_t cublasDtpmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const double* AP, double* x, int incx) {
    return hipblasDtpmv(handle, convert_hipblasFillMode_t(uplo), convert_hipblasOperation_t(trans), convert_hipblasDiagType_t(diag), n, AP, x, incx);
}

cublasStatus_t cublasCtpmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuComplex* AP, cuComplex* x, int incx) {
    return hipblasCtpmv(handle, convert_hipblasFillMode_t(uplo), convert_hipblasOperation_t(trans), convert_hipblasDiagType_t(diag), n, reinterpret_cast<const hipblasComplex*>(AP), reinterpret_cast<hipblasComplex*>(x), incx);
}

cublasStatus_t cublasZtpmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuDoubleComplex* AP, cuDoubleComplex* x, int incx) {
    return hipblasZtpmv(handle, convert_hipblasFillMode_t(uplo), convert_hipblasOperation_t(trans), convert_hipblasDiagType_t(diag), n, reinterpret_cast<const hipblasDoubleComplex*>(AP), reinterpret_cast<hipblasDoubleComplex*>(x), incx);
}

cublasStatus_t cublasStrsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const float* A, int lda, float* x, int incx) {
    return hipblasStrsv(handle, convert_hipblasFillMode_t(uplo), convert_hipblasOperation_t(trans), convert_hipblasDiagType_t(diag), n, A, lda, x, incx);
}

cublasStatus_t cublasDtrsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const double* A, int lda, double* x, int incx) {
    return hipblasDtrsv(handle, convert_hipblasFillMode_t(uplo), convert_hipblasOperation_t(trans), convert_hipblasDiagType_t(diag), n, A, lda, x, incx);
}

cublasStatus_t cublasCtrsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuComplex* A, int lda, cuComplex* x, int incx) {
    return hipblasCtrsv(handle, convert_hipblasFillMode_t(uplo), convert_hipblasOperation_t(trans), convert_hipblasDiagType_t(diag), n, reinterpret_cast<const hipblasComplex*>(A), lda, reinterpret_cast<hipblasComplex*>(x), incx);
}

cublasStatus_t cublasZtrsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuDoubleComplex* A, int lda, cuDoubleComplex* x, int incx) {
    return hipblasZtrsv(handle, convert_hipblasFillMode_t(uplo), convert_hipblasOperation_t(trans), convert_hipblasDiagType_t(diag), n, reinterpret_cast<const hipblasDoubleComplex*>(A), lda, reinterpret_cast<hipblasDoubleComplex*>(x), incx);
}

cublasStatus_t cublasStpsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const float* AP, float* x, int incx) {
    return hipblasStpsv(handle, convert_hipblasFillMode_t(uplo), convert_hipblasOperation_t(trans), convert_hipblasDiagType_t(diag), n, AP, x, incx);
}

cublasStatus_t cublasDtpsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const double* AP, double* x, int incx) {
    return hipblasDtpsv(handle, convert_hipblasFillMode_t(uplo), convert_hipblasOperation_t(trans), convert_hipblasDiagType_t(diag), n, AP, x, incx);
}

cublasStatus_t cublasCtpsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuComplex* AP, cuComplex* x, int incx) {
    return hipblasCtpsv(handle, convert_hipblasFillMode_t(uplo), convert_hipblasOperation_t(trans), convert_hipblasDiagType_t(diag), n, reinterpret_cast<const hipblasComplex*>(AP), reinterpret_cast<hipblasComplex*>(x), incx);
}

cublasStatus_t cublasZtpsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuDoubleComplex* AP, cuDoubleComplex* x, int incx) {
    return hipblasZtpsv(handle, convert_hipblasFillMode_t(uplo), convert_hipblasOperation_t(trans), convert_hipblasDiagType_t(diag), n, reinterpret_cast<const hipblasDoubleComplex*>(AP), reinterpret_cast<hipblasDoubleComplex*>(x), incx);
}

cublasStatus_t cublasStbsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const float* A, int lda, float* x, int incx) {
    return hipblasStbsv(handle, convert_hipblasFillMode_t(uplo), convert_hipblasOperation_t(trans), convert_hipblasDiagType_t(diag), n, k, A, lda, x, incx);
}

cublasStatus_t cublasDtbsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const double* A, int lda, double* x, int incx) {
    return hipblasDtbsv(handle, convert_hipblasFillMode_t(uplo), convert_hipblasOperation_t(trans), convert_hipblasDiagType_t(diag), n, k, A, lda, x, incx);
}

cublasStatus_t cublasCtbsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const cuComplex* A, int lda, cuComplex* x, int incx) {
    return hipblasCtbsv(handle, convert_hipblasFillMode_t(uplo), convert_hipblasOperation_t(trans), convert_hipblasDiagType_t(diag), n, k, reinterpret_cast<const hipblasComplex*>(A), lda, reinterpret_cast<hipblasComplex*>(x), incx);
}

cublasStatus_t cublasZtbsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const cuDoubleComplex* A, int lda, cuDoubleComplex* x, int incx) {
    return hipblasZtbsv(handle, convert_hipblasFillMode_t(uplo), convert_hipblasOperation_t(trans), convert_hipblasDiagType_t(diag), n, k, reinterpret_cast<const hipblasDoubleComplex*>(A), lda, reinterpret_cast<hipblasDoubleComplex*>(x), incx);
}

cublasStatus_t cublasSsymv(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const float* A, int lda, const float* x, int incx, const float* beta, float* y, int incy) {
    return hipblasSsymv(handle, convert_hipblasFillMode_t(uplo), n, alpha, A, lda, x, incx, beta, y, incy);
}

cublasStatus_t cublasDsymv(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* alpha, const double* A, int lda, const double* x, int incx, const double* beta, double* y, int incy) {
    return hipblasDsymv(handle, convert_hipblasFillMode_t(uplo), n, alpha, A, lda, x, incx, beta, y, incy);
}

cublasStatus_t cublasCsymv(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* x, int incx, const cuComplex* beta, cuComplex* y, int incy) {
    return hipblasCsymv(handle, convert_hipblasFillMode_t(uplo), n, reinterpret_cast<const hipblasComplex*>(alpha), reinterpret_cast<const hipblasComplex*>(A), lda, reinterpret_cast<const hipblasComplex*>(x), incx, reinterpret_cast<const hipblasComplex*>(beta), reinterpret_cast<hipblasComplex*>(y), incy);
}

cublasStatus_t cublasZsymv(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int incy) {
    return hipblasZsymv(handle, convert_hipblasFillMode_t(uplo), n, reinterpret_cast<const hipblasDoubleComplex*>(alpha), reinterpret_cast<const hipblasDoubleComplex*>(A), lda, reinterpret_cast<const hipblasDoubleComplex*>(x), incx, reinterpret_cast<const hipblasDoubleComplex*>(beta), reinterpret_cast<hipblasDoubleComplex*>(y), incy);
}

cublasStatus_t cublasChemv(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* x, int incx, const cuComplex* beta, cuComplex* y, int incy) {
    return hipblasChemv(handle, convert_hipblasFillMode_t(uplo), n, reinterpret_cast<const hipblasComplex*>(alpha), reinterpret_cast<const hipblasComplex*>(A), lda, reinterpret_cast<const hipblasComplex*>(x), incx, reinterpret_cast<const hipblasComplex*>(beta), reinterpret_cast<hipblasComplex*>(y), incy);
}

cublasStatus_t cublasZhemv(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int incy) {
    return hipblasZhemv(handle, convert_hipblasFillMode_t(uplo), n, reinterpret_cast<const hipblasDoubleComplex*>(alpha), reinterpret_cast<const hipblasDoubleComplex*>(A), lda, reinterpret_cast<const hipblasDoubleComplex*>(x), incx, reinterpret_cast<const hipblasDoubleComplex*>(beta), reinterpret_cast<hipblasDoubleComplex*>(y), incy);
}

cublasStatus_t cublasSsbmv(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, const float* alpha, const float* A, int lda, const float* x, int incx, const float* beta, float* y, int incy) {
    return hipblasSsbmv(handle, convert_hipblasFillMode_t(uplo), n, k, alpha, A, lda, x, incx, beta, y, incy);
}

cublasStatus_t cublasDsbmv(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, const double* alpha, const double* A, int lda, const double* x, int incx, const double* beta, double* y, int incy) {
    return hipblasDsbmv(handle, convert_hipblasFillMode_t(uplo), n, k, alpha, A, lda, x, incx, beta, y, incy);
}

cublasStatus_t cublasChbmv(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* x, int incx, const cuComplex* beta, cuComplex* y, int incy) {
    return hipblasChbmv(handle, convert_hipblasFillMode_t(uplo), n, k, reinterpret_cast<const hipblasComplex*>(alpha), reinterpret_cast<const hipblasComplex*>(A), lda, reinterpret_cast<const hipblasComplex*>(x), incx, reinterpret_cast<const hipblasComplex*>(beta), reinterpret_cast<hipblasComplex*>(y), incy);
}

cublasStatus_t cublasZhbmv(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int incy) {
    return hipblasZhbmv(handle, convert_hipblasFillMode_t(uplo), n, k, reinterpret_cast<const hipblasDoubleComplex*>(alpha), reinterpret_cast<const hipblasDoubleComplex*>(A), lda, reinterpret_cast<const hipblasDoubleComplex*>(x), incx, reinterpret_cast<const hipblasDoubleComplex*>(beta), reinterpret_cast<hipblasDoubleComplex*>(y), incy);
}

cublasStatus_t cublasSspmv(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const float* AP, const float* x, int incx, const float* beta, float* y, int incy) {
    return hipblasSspmv(handle, convert_hipblasFillMode_t(uplo), n, alpha, AP, x, incx, beta, y, incy);
}

cublasStatus_t cublasDspmv(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* alpha, const double* AP, const double* x, int incx, const double* beta, double* y, int incy) {
    return hipblasDspmv(handle, convert_hipblasFillMode_t(uplo), n, alpha, AP, x, incx, beta, y, incy);
}

cublasStatus_t cublasChpmv(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* alpha, const cuComplex* AP, const cuComplex* x, int incx, const cuComplex* beta, cuComplex* y, int incy) {
    return hipblasChpmv(handle, convert_hipblasFillMode_t(uplo), n, reinterpret_cast<const hipblasComplex*>(alpha), reinterpret_cast<const hipblasComplex*>(AP), reinterpret_cast<const hipblasComplex*>(x), incx, reinterpret_cast<const hipblasComplex*>(beta), reinterpret_cast<hipblasComplex*>(y), incy);
}

cublasStatus_t cublasZhpmv(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* AP, const cuDoubleComplex* x, int incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int incy) {
    return hipblasZhpmv(handle, convert_hipblasFillMode_t(uplo), n, reinterpret_cast<const hipblasDoubleComplex*>(alpha), reinterpret_cast<const hipblasDoubleComplex*>(AP), reinterpret_cast<const hipblasDoubleComplex*>(x), incx, reinterpret_cast<const hipblasDoubleComplex*>(beta), reinterpret_cast<hipblasDoubleComplex*>(y), incy);
}

cublasStatus_t cublasSger(cublasHandle_t handle, int m, int n, const float* alpha, const float* x, int incx, const float* y, int incy, float* A, int lda) {
    return hipblasSger(handle, m, n, alpha, x, incx, y, incy, A, lda);
}

cublasStatus_t cublasDger(cublasHandle_t handle, int m, int n, const double* alpha, const double* x, int incx, const double* y, int incy, double* A, int lda) {
    return hipblasDger(handle, m, n, alpha, x, incx, y, incy, A, lda);
}

cublasStatus_t cublasCgeru(cublasHandle_t handle, int m, int n, const cuComplex* alpha, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* A, int lda) {
    return hipblasCgeru(handle, m, n, reinterpret_cast<const hipblasComplex*>(alpha), reinterpret_cast<const hipblasComplex*>(x), incx, reinterpret_cast<const hipblasComplex*>(y), incy, reinterpret_cast<hipblasComplex*>(A), lda);
}

cublasStatus_t cublasCgerc(cublasHandle_t handle, int m, int n, const cuComplex* alpha, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* A, int lda) {
    return hipblasCgerc(handle, m, n, reinterpret_cast<const hipblasComplex*>(alpha), reinterpret_cast<const hipblasComplex*>(x), incx, reinterpret_cast<const hipblasComplex*>(y), incy, reinterpret_cast<hipblasComplex*>(A), lda);
}

cublasStatus_t cublasZgeru(cublasHandle_t handle, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* A, int lda) {
    return hipblasZgeru(handle, m, n, reinterpret_cast<const hipblasDoubleComplex*>(alpha), reinterpret_cast<const hipblasDoubleComplex*>(x), incx, reinterpret_cast<const hipblasDoubleComplex*>(y), incy, reinterpret_cast<hipblasDoubleComplex*>(A), lda);
}

cublasStatus_t cublasZgerc(cublasHandle_t handle, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* A, int lda) {
    return hipblasZgerc(handle, m, n, reinterpret_cast<const hipblasDoubleComplex*>(alpha), reinterpret_cast<const hipblasDoubleComplex*>(x), incx, reinterpret_cast<const hipblasDoubleComplex*>(y), incy, reinterpret_cast<hipblasDoubleComplex*>(A), lda);
}

cublasStatus_t cublasSsyr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const float* x, int incx, float* A, int lda) {
    return hipblasSsyr(handle, convert_hipblasFillMode_t(uplo), n, alpha, x, incx, A, lda);
}

cublasStatus_t cublasDsyr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* alpha, const double* x, int incx, double* A, int lda) {
    return hipblasDsyr(handle, convert_hipblasFillMode_t(uplo), n, alpha, x, incx, A, lda);
}

cublasStatus_t cublasCsyr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* alpha, const cuComplex* x, int incx, cuComplex* A, int lda) {
    return hipblasCsyr(handle, convert_hipblasFillMode_t(uplo), n, reinterpret_cast<const hipblasComplex*>(alpha), reinterpret_cast<const hipblasComplex*>(x), incx, reinterpret_cast<hipblasComplex*>(A), lda);
}

cublasStatus_t cublasZsyr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, cuDoubleComplex* A, int lda) {
    return hipblasZsyr(handle, convert_hipblasFillMode_t(uplo), n, reinterpret_cast<const hipblasDoubleComplex*>(alpha), reinterpret_cast<const hipblasDoubleComplex*>(x), incx, reinterpret_cast<hipblasDoubleComplex*>(A), lda);
}

cublasStatus_t cublasCher(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const cuComplex* x, int incx, cuComplex* A, int lda) {
    return hipblasCher(handle, convert_hipblasFillMode_t(uplo), n, alpha, reinterpret_cast<const hipblasComplex*>(x), incx, reinterpret_cast<hipblasComplex*>(A), lda);
}

cublasStatus_t cublasZher(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* alpha, const cuDoubleComplex* x, int incx, cuDoubleComplex* A, int lda) {
    return hipblasZher(handle, convert_hipblasFillMode_t(uplo), n, alpha, reinterpret_cast<const hipblasDoubleComplex*>(x), incx, reinterpret_cast<hipblasDoubleComplex*>(A), lda);
}

cublasStatus_t cublasSspr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const float* x, int incx, float* AP) {
    return hipblasSspr(handle, convert_hipblasFillMode_t(uplo), n, alpha, x, incx, AP);
}

cublasStatus_t cublasDspr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* alpha, const double* x, int incx, double* AP) {
    return hipblasDspr(handle, convert_hipblasFillMode_t(uplo), n, alpha, x, incx, AP);
}

cublasStatus_t cublasChpr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const cuComplex* x, int incx, cuComplex* AP) {
    return hipblasChpr(handle, convert_hipblasFillMode_t(uplo), n, alpha, reinterpret_cast<const hipblasComplex*>(x), incx, reinterpret_cast<hipblasComplex*>(AP));
}

cublasStatus_t cublasZhpr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* alpha, const cuDoubleComplex* x, int incx, cuDoubleComplex* AP) {
    return hipblasZhpr(handle, convert_hipblasFillMode_t(uplo), n, alpha, reinterpret_cast<const hipblasDoubleComplex*>(x), incx, reinterpret_cast<hipblasDoubleComplex*>(AP));
}

cublasStatus_t cublasSsyr2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const float* x, int incx, const float* y, int incy, float* A, int lda) {
    return hipblasSsyr2(handle, convert_hipblasFillMode_t(uplo), n, alpha, x, incx, y, incy, A, lda);
}

cublasStatus_t cublasDsyr2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* alpha, const double* x, int incx, const double* y, int incy, double* A, int lda) {
    return hipblasDsyr2(handle, convert_hipblasFillMode_t(uplo), n, alpha, x, incx, y, incy, A, lda);
}

cublasStatus_t cublasCsyr2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* alpha, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* A, int lda) {
    return hipblasCsyr2(handle, convert_hipblasFillMode_t(uplo), n, reinterpret_cast<const hipblasComplex*>(alpha), reinterpret_cast<const hipblasComplex*>(x), incx, reinterpret_cast<const hipblasComplex*>(y), incy, reinterpret_cast<hipblasComplex*>(A), lda);
}

cublasStatus_t cublasZsyr2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* A, int lda) {
    return hipblasZsyr2(handle, convert_hipblasFillMode_t(uplo), n, reinterpret_cast<const hipblasDoubleComplex*>(alpha), reinterpret_cast<const hipblasDoubleComplex*>(x), incx, reinterpret_cast<const hipblasDoubleComplex*>(y), incy, reinterpret_cast<hipblasDoubleComplex*>(A), lda);
}

cublasStatus_t cublasCher2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* alpha, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* A, int lda) {
    return hipblasCher2(handle, convert_hipblasFillMode_t(uplo), n, reinterpret_cast<const hipblasComplex*>(alpha), reinterpret_cast<const hipblasComplex*>(x), incx, reinterpret_cast<const hipblasComplex*>(y), incy, reinterpret_cast<hipblasComplex*>(A), lda);
}

cublasStatus_t cublasZher2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* A, int lda) {
    return hipblasZher2(handle, convert_hipblasFillMode_t(uplo), n, reinterpret_cast<const hipblasDoubleComplex*>(alpha), reinterpret_cast<const hipblasDoubleComplex*>(x), incx, reinterpret_cast<const hipblasDoubleComplex*>(y), incy, reinterpret_cast<hipblasDoubleComplex*>(A), lda);
}

cublasStatus_t cublasSspr2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const float* x, int incx, const float* y, int incy, float* AP) {
    return hipblasSspr2(handle, convert_hipblasFillMode_t(uplo), n, alpha, x, incx, y, incy, AP);
}

cublasStatus_t cublasDspr2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* alpha, const double* x, int incx, const double* y, int incy, double* AP) {
    return hipblasDspr2(handle, convert_hipblasFillMode_t(uplo), n, alpha, x, incx, y, incy, AP);
}

cublasStatus_t cublasChpr2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* alpha, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* AP) {
    return hipblasChpr2(handle, convert_hipblasFillMode_t(uplo), n, reinterpret_cast<const hipblasComplex*>(alpha), reinterpret_cast<const hipblasComplex*>(x), incx, reinterpret_cast<const hipblasComplex*>(y), incy, reinterpret_cast<hipblasComplex*>(AP));
}

cublasStatus_t cublasZhpr2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* AP) {
    return hipblasZhpr2(handle, convert_hipblasFillMode_t(uplo), n, reinterpret_cast<const hipblasDoubleComplex*>(alpha), reinterpret_cast<const hipblasDoubleComplex*>(x), incx, reinterpret_cast<const hipblasDoubleComplex*>(y), incy, reinterpret_cast<hipblasDoubleComplex*>(AP));
}

cublasStatus_t cublasSgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc) {
    return hipblasSgemm(handle, convert_hipblasOperation_t(transa), convert_hipblasOperation_t(transb), m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

cublasStatus_t cublasDgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double* alpha, const double* A, int lda, const double* B, int ldb, const double* beta, double* C, int ldc) {
    return hipblasDgemm(handle, convert_hipblasOperation_t(transa), convert_hipblasOperation_t(transb), m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

cublasStatus_t cublasCgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const cuComplex* beta, cuComplex* C, int ldc) {
    return hipblasCgemm(handle, convert_hipblasOperation_t(transa), convert_hipblasOperation_t(transb), m, n, k, reinterpret_cast<const hipblasComplex*>(alpha), reinterpret_cast<const hipblasComplex*>(A), lda, reinterpret_cast<const hipblasComplex*>(B), ldb, reinterpret_cast<const hipblasComplex*>(beta), reinterpret_cast<hipblasComplex*>(C), ldc);
}

cublasStatus_t cublasCgemm3m(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasZgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc) {
    return hipblasZgemm(handle, convert_hipblasOperation_t(transa), convert_hipblasOperation_t(transb), m, n, k, reinterpret_cast<const hipblasDoubleComplex*>(alpha), reinterpret_cast<const hipblasDoubleComplex*>(A), lda, reinterpret_cast<const hipblasDoubleComplex*>(B), ldb, reinterpret_cast<const hipblasDoubleComplex*>(beta), reinterpret_cast<hipblasDoubleComplex*>(C), ldc);
}

cublasStatus_t cublasZgemm3m(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasCgemmEx(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasSsyrk(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float* alpha, const float* A, int lda, const float* beta, float* C, int ldc) {
    return hipblasSsyrk(handle, convert_hipblasFillMode_t(uplo), convert_hipblasOperation_t(trans), n, k, alpha, A, lda, beta, C, ldc);
}

cublasStatus_t cublasDsyrk(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double* alpha, const double* A, int lda, const double* beta, double* C, int ldc) {
    return hipblasDsyrk(handle, convert_hipblasFillMode_t(uplo), convert_hipblasOperation_t(trans), n, k, alpha, A, lda, beta, C, ldc);
}

cublasStatus_t cublasCsyrk(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* beta, cuComplex* C, int ldc) {
    return hipblasCsyrk(handle, convert_hipblasFillMode_t(uplo), convert_hipblasOperation_t(trans), n, k, reinterpret_cast<const hipblasComplex*>(alpha), reinterpret_cast<const hipblasComplex*>(A), lda, reinterpret_cast<const hipblasComplex*>(beta), reinterpret_cast<hipblasComplex*>(C), ldc);
}

cublasStatus_t cublasZsyrk(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc) {
    return hipblasZsyrk(handle, convert_hipblasFillMode_t(uplo), convert_hipblasOperation_t(trans), n, k, reinterpret_cast<const hipblasDoubleComplex*>(alpha), reinterpret_cast<const hipblasDoubleComplex*>(A), lda, reinterpret_cast<const hipblasDoubleComplex*>(beta), reinterpret_cast<hipblasDoubleComplex*>(C), ldc);
}

cublasStatus_t cublasCsyrkEx(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasCsyrk3mEx(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasCherk(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float* alpha, const cuComplex* A, int lda, const float* beta, cuComplex* C, int ldc) {
    return hipblasCherk(handle, convert_hipblasFillMode_t(uplo), convert_hipblasOperation_t(trans), n, k, alpha, reinterpret_cast<const hipblasComplex*>(A), lda, beta, reinterpret_cast<hipblasComplex*>(C), ldc);
}

cublasStatus_t cublasZherk(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double* alpha, const cuDoubleComplex* A, int lda, const double* beta, cuDoubleComplex* C, int ldc) {
    return hipblasZherk(handle, convert_hipblasFillMode_t(uplo), convert_hipblasOperation_t(trans), n, k, alpha, reinterpret_cast<const hipblasDoubleComplex*>(A), lda, beta, reinterpret_cast<hipblasDoubleComplex*>(C), ldc);
}

cublasStatus_t cublasCherkEx(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasCherk3mEx(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasSsyr2k(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc) {
    return hipblasSsyr2k(handle, convert_hipblasFillMode_t(uplo), convert_hipblasOperation_t(trans), n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

cublasStatus_t cublasDsyr2k(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double* alpha, const double* A, int lda, const double* B, int ldb, const double* beta, double* C, int ldc) {
    return hipblasDsyr2k(handle, convert_hipblasFillMode_t(uplo), convert_hipblasOperation_t(trans), n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

cublasStatus_t cublasCsyr2k(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const cuComplex* beta, cuComplex* C, int ldc) {
    return hipblasCsyr2k(handle, convert_hipblasFillMode_t(uplo), convert_hipblasOperation_t(trans), n, k, reinterpret_cast<const hipblasComplex*>(alpha), reinterpret_cast<const hipblasComplex*>(A), lda, reinterpret_cast<const hipblasComplex*>(B), ldb, reinterpret_cast<const hipblasComplex*>(beta), reinterpret_cast<hipblasComplex*>(C), ldc);
}

cublasStatus_t cublasZsyr2k(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc) {
    return hipblasZsyr2k(handle, convert_hipblasFillMode_t(uplo), convert_hipblasOperation_t(trans), n, k, reinterpret_cast<const hipblasDoubleComplex*>(alpha), reinterpret_cast<const hipblasDoubleComplex*>(A), lda, reinterpret_cast<const hipblasDoubleComplex*>(B), ldb, reinterpret_cast<const hipblasDoubleComplex*>(beta), reinterpret_cast<hipblasDoubleComplex*>(C), ldc);
}

cublasStatus_t cublasCher2k(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const float* beta, cuComplex* C, int ldc) {
    return hipblasCher2k(handle, convert_hipblasFillMode_t(uplo), convert_hipblasOperation_t(trans), n, k, reinterpret_cast<const hipblasComplex*>(alpha), reinterpret_cast<const hipblasComplex*>(A), lda, reinterpret_cast<const hipblasComplex*>(B), ldb, beta, reinterpret_cast<hipblasComplex*>(C), ldc);
}

cublasStatus_t cublasZher2k(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const double* beta, cuDoubleComplex* C, int ldc) {
    return hipblasZher2k(handle, convert_hipblasFillMode_t(uplo), convert_hipblasOperation_t(trans), n, k, reinterpret_cast<const hipblasDoubleComplex*>(alpha), reinterpret_cast<const hipblasDoubleComplex*>(A), lda, reinterpret_cast<const hipblasDoubleComplex*>(B), ldb, beta, reinterpret_cast<hipblasDoubleComplex*>(C), ldc);
}

cublasStatus_t cublasSsyrkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc) {
    return hipblasSsyrkx(handle, convert_hipblasFillMode_t(uplo), convert_hipblasOperation_t(trans), n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

cublasStatus_t cublasDsyrkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double* alpha, const double* A, int lda, const double* B, int ldb, const double* beta, double* C, int ldc) {
    return hipblasDsyrkx(handle, convert_hipblasFillMode_t(uplo), convert_hipblasOperation_t(trans), n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

cublasStatus_t cublasCsyrkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const cuComplex* beta, cuComplex* C, int ldc) {
    return hipblasCsyrkx(handle, convert_hipblasFillMode_t(uplo), convert_hipblasOperation_t(trans), n, k, reinterpret_cast<const hipblasComplex*>(alpha), reinterpret_cast<const hipblasComplex*>(A), lda, reinterpret_cast<const hipblasComplex*>(B), ldb, reinterpret_cast<const hipblasComplex*>(beta), reinterpret_cast<hipblasComplex*>(C), ldc);
}

cublasStatus_t cublasZsyrkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc) {
    return hipblasZsyrkx(handle, convert_hipblasFillMode_t(uplo), convert_hipblasOperation_t(trans), n, k, reinterpret_cast<const hipblasDoubleComplex*>(alpha), reinterpret_cast<const hipblasDoubleComplex*>(A), lda, reinterpret_cast<const hipblasDoubleComplex*>(B), ldb, reinterpret_cast<const hipblasDoubleComplex*>(beta), reinterpret_cast<hipblasDoubleComplex*>(C), ldc);
}

cublasStatus_t cublasCherkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const float* beta, cuComplex* C, int ldc) {
    return hipblasCherkx(handle, convert_hipblasFillMode_t(uplo), convert_hipblasOperation_t(trans), n, k, reinterpret_cast<const hipblasComplex*>(alpha), reinterpret_cast<const hipblasComplex*>(A), lda, reinterpret_cast<const hipblasComplex*>(B), ldb, beta, reinterpret_cast<hipblasComplex*>(C), ldc);
}

cublasStatus_t cublasZherkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const double* beta, cuDoubleComplex* C, int ldc) {
    return hipblasZherkx(handle, convert_hipblasFillMode_t(uplo), convert_hipblasOperation_t(trans), n, k, reinterpret_cast<const hipblasDoubleComplex*>(alpha), reinterpret_cast<const hipblasDoubleComplex*>(A), lda, reinterpret_cast<const hipblasDoubleComplex*>(B), ldb, beta, reinterpret_cast<hipblasDoubleComplex*>(C), ldc);
}

cublasStatus_t cublasSsymm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc) {
    return hipblasSsymm(handle, convert_hipblasSideMode_t(side), convert_hipblasFillMode_t(uplo), m, n, alpha, A, lda, B, ldb, beta, C, ldc);
}

cublasStatus_t cublasDsymm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const double* alpha, const double* A, int lda, const double* B, int ldb, const double* beta, double* C, int ldc) {
    return hipblasDsymm(handle, convert_hipblasSideMode_t(side), convert_hipblasFillMode_t(uplo), m, n, alpha, A, lda, B, ldb, beta, C, ldc);
}

cublasStatus_t cublasCsymm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const cuComplex* beta, cuComplex* C, int ldc) {
    return hipblasCsymm(handle, convert_hipblasSideMode_t(side), convert_hipblasFillMode_t(uplo), m, n, reinterpret_cast<const hipblasComplex*>(alpha), reinterpret_cast<const hipblasComplex*>(A), lda, reinterpret_cast<const hipblasComplex*>(B), ldb, reinterpret_cast<const hipblasComplex*>(beta), reinterpret_cast<hipblasComplex*>(C), ldc);
}

cublasStatus_t cublasZsymm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc) {
    return hipblasZsymm(handle, convert_hipblasSideMode_t(side), convert_hipblasFillMode_t(uplo), m, n, reinterpret_cast<const hipblasDoubleComplex*>(alpha), reinterpret_cast<const hipblasDoubleComplex*>(A), lda, reinterpret_cast<const hipblasDoubleComplex*>(B), ldb, reinterpret_cast<const hipblasDoubleComplex*>(beta), reinterpret_cast<hipblasDoubleComplex*>(C), ldc);
}

cublasStatus_t cublasChemm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const cuComplex* beta, cuComplex* C, int ldc) {
    return hipblasChemm(handle, convert_hipblasSideMode_t(side), convert_hipblasFillMode_t(uplo), m, n, reinterpret_cast<const hipblasComplex*>(alpha), reinterpret_cast<const hipblasComplex*>(A), lda, reinterpret_cast<const hipblasComplex*>(B), ldb, reinterpret_cast<const hipblasComplex*>(beta), reinterpret_cast<hipblasComplex*>(C), ldc);
}

cublasStatus_t cublasZhemm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc) {
    return hipblasZhemm(handle, convert_hipblasSideMode_t(side), convert_hipblasFillMode_t(uplo), m, n, reinterpret_cast<const hipblasDoubleComplex*>(alpha), reinterpret_cast<const hipblasDoubleComplex*>(A), lda, reinterpret_cast<const hipblasDoubleComplex*>(B), ldb, reinterpret_cast<const hipblasDoubleComplex*>(beta), reinterpret_cast<hipblasDoubleComplex*>(C), ldc);
}

cublasStatus_t cublasStrsm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const float* alpha, const float* A, int lda, float* B, int ldb) {
    return hipblasStrsm(handle, convert_hipblasSideMode_t(side), convert_hipblasFillMode_t(uplo), convert_hipblasOperation_t(trans), convert_hipblasDiagType_t(diag), m, n, alpha, const_cast<float*>(A), lda, B, ldb);
}

cublasStatus_t cublasDtrsm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const double* alpha, const double* A, int lda, double* B, int ldb) {
    return hipblasDtrsm(handle, convert_hipblasSideMode_t(side), convert_hipblasFillMode_t(uplo), convert_hipblasOperation_t(trans), convert_hipblasDiagType_t(diag), m, n, alpha, const_cast<double*>(A), lda, B, ldb);
}

cublasStatus_t cublasCtrsm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, cuComplex* B, int ldb) {
    return hipblasCtrsm(handle, convert_hipblasSideMode_t(side), convert_hipblasFillMode_t(uplo), convert_hipblasOperation_t(trans), convert_hipblasDiagType_t(diag), m, n, reinterpret_cast<const hipblasComplex*>(alpha), reinterpret_cast<hipblasComplex*>(const_cast<cuComplex*>(A)), lda, reinterpret_cast<hipblasComplex*>(B), ldb);
}

cublasStatus_t cublasZtrsm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, cuDoubleComplex* B, int ldb) {
    return hipblasZtrsm(handle, convert_hipblasSideMode_t(side), convert_hipblasFillMode_t(uplo), convert_hipblasOperation_t(trans), convert_hipblasDiagType_t(diag), m, n, reinterpret_cast<const hipblasDoubleComplex*>(alpha), reinterpret_cast<hipblasDoubleComplex*>(const_cast<cuDoubleComplex*>(A)), lda, reinterpret_cast<hipblasDoubleComplex*>(B), ldb);
}

cublasStatus_t cublasSgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float* alpha, const float* const Aarray[], int lda, const float* const Barray[], int ldb, const float* beta, float* const Carray[], int ldc, int batchCount) {
    return hipblasSgemmBatched(handle, convert_hipblasOperation_t(transa), convert_hipblasOperation_t(transb), m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
}

cublasStatus_t cublasDgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double* alpha, const double* const Aarray[], int lda, const double* const Barray[], int ldb, const double* beta, double* const Carray[], int ldc, int batchCount) {
    return hipblasDgemmBatched(handle, convert_hipblasOperation_t(transa), convert_hipblasOperation_t(transb), m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
}

cublasStatus_t cublasCgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex* alpha, const cuComplex* const Aarray[], int lda, const cuComplex* const Barray[], int ldb, const cuComplex* beta, cuComplex* const Carray[], int ldc, int batchCount) {
    return hipblasCgemmBatched(handle, convert_hipblasOperation_t(transa), convert_hipblasOperation_t(transb), m, n, k, reinterpret_cast<const hipblasComplex*>(alpha), reinterpret_cast<const hipblasComplex* const*>(Aarray), lda, reinterpret_cast<const hipblasComplex* const*>(Barray), ldb, reinterpret_cast<const hipblasComplex*>(beta), reinterpret_cast<hipblasComplex* const*>(Carray), ldc, batchCount);
}

cublasStatus_t cublasZgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* const Aarray[], int lda, const cuDoubleComplex* const Barray[], int ldb, const cuDoubleComplex* beta, cuDoubleComplex* const Carray[], int ldc, int batchCount) {
    return hipblasZgemmBatched(handle, convert_hipblasOperation_t(transa), convert_hipblasOperation_t(transb), m, n, k, reinterpret_cast<const hipblasDoubleComplex*>(alpha), reinterpret_cast<const hipblasDoubleComplex* const*>(Aarray), lda, reinterpret_cast<const hipblasDoubleComplex* const*>(Barray), ldb, reinterpret_cast<const hipblasDoubleComplex*>(beta), reinterpret_cast<hipblasDoubleComplex* const*>(Carray), ldc, batchCount);
}

cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float* alpha, const float* A, int lda, long long int strideA, const float* B, int ldb, long long int strideB, const float* beta, float* C, int ldc, long long int strideC, int batchCount) {
    return hipblasSgemmStridedBatched(handle, convert_hipblasOperation_t(transa), convert_hipblasOperation_t(transb), m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
}

cublasStatus_t cublasDgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double* alpha, const double* A, int lda, long long int strideA, const double* B, int ldb, long long int strideB, const double* beta, double* C, int ldc, long long int strideC, int batchCount) {
    return hipblasDgemmStridedBatched(handle, convert_hipblasOperation_t(transa), convert_hipblasOperation_t(transb), m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
}

cublasStatus_t cublasCgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, long long int strideA, const cuComplex* B, int ldb, long long int strideB, const cuComplex* beta, cuComplex* C, int ldc, long long int strideC, int batchCount) {
    return hipblasCgemmStridedBatched(handle, convert_hipblasOperation_t(transa), convert_hipblasOperation_t(transb), m, n, k, reinterpret_cast<const hipblasComplex*>(alpha), reinterpret_cast<const hipblasComplex*>(A), lda, strideA, reinterpret_cast<const hipblasComplex*>(B), ldb, strideB, reinterpret_cast<const hipblasComplex*>(beta), reinterpret_cast<hipblasComplex*>(C), ldc, strideC, batchCount);
}

cublasStatus_t cublasCgemm3mStridedBatched(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasZgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, long long int strideA, const cuDoubleComplex* B, int ldb, long long int strideB, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc, long long int strideC, int batchCount) {
    return hipblasZgemmStridedBatched(handle, convert_hipblasOperation_t(transa), convert_hipblasOperation_t(transb), m, n, k, reinterpret_cast<const hipblasDoubleComplex*>(alpha), reinterpret_cast<const hipblasDoubleComplex*>(A), lda, strideA, reinterpret_cast<const hipblasDoubleComplex*>(B), ldb, strideB, reinterpret_cast<const hipblasDoubleComplex*>(beta), reinterpret_cast<hipblasDoubleComplex*>(C), ldc, strideC, batchCount);
}

cublasStatus_t cublasSgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const float* alpha, const float* A, int lda, const float* beta, const float* B, int ldb, float* C, int ldc) {
    return hipblasSgeam(handle, convert_hipblasOperation_t(transa), convert_hipblasOperation_t(transb), m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}

cublasStatus_t cublasDgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const double* alpha, const double* A, int lda, const double* beta, const double* B, int ldb, double* C, int ldc) {
    return hipblasDgeam(handle, convert_hipblasOperation_t(transa), convert_hipblasOperation_t(transb), m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}

cublasStatus_t cublasCgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* beta, const cuComplex* B, int ldb, cuComplex* C, int ldc) {
    return hipblasCgeam(handle, convert_hipblasOperation_t(transa), convert_hipblasOperation_t(transb), m, n, reinterpret_cast<const hipblasComplex*>(alpha), reinterpret_cast<const hipblasComplex*>(A), lda, reinterpret_cast<const hipblasComplex*>(beta), reinterpret_cast<const hipblasComplex*>(B), ldb, reinterpret_cast<hipblasComplex*>(C), ldc);
}

cublasStatus_t cublasZgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* beta, const cuDoubleComplex* B, int ldb, cuDoubleComplex* C, int ldc) {
    return hipblasZgeam(handle, convert_hipblasOperation_t(transa), convert_hipblasOperation_t(transb), m, n, reinterpret_cast<const hipblasDoubleComplex*>(alpha), reinterpret_cast<const hipblasDoubleComplex*>(A), lda, reinterpret_cast<const hipblasDoubleComplex*>(beta), reinterpret_cast<const hipblasDoubleComplex*>(B), ldb, reinterpret_cast<hipblasDoubleComplex*>(C), ldc);
}

cublasStatus_t cublasSgetrfBatched(cublasHandle_t handle, int n, float* const A[], int lda, int* P, int* info, int batchSize) {
    return hipblasSgetrfBatched(handle, n, A, lda, P, info, batchSize);
}

cublasStatus_t cublasDgetrfBatched(cublasHandle_t handle, int n, double* const A[], int lda, int* P, int* info, int batchSize) {
    return hipblasDgetrfBatched(handle, n, A, lda, P, info, batchSize);
}

cublasStatus_t cublasCgetrfBatched(cublasHandle_t handle, int n, cuComplex* const A[], int lda, int* P, int* info, int batchSize) {
    return hipblasCgetrfBatched(handle, n, reinterpret_cast<hipblasComplex* const*>(A), lda, P, info, batchSize);
}

cublasStatus_t cublasZgetrfBatched(cublasHandle_t handle, int n, cuDoubleComplex* const A[], int lda, int* P, int* info, int batchSize) {
    return hipblasZgetrfBatched(handle, n, reinterpret_cast<hipblasDoubleComplex* const*>(A), lda, P, info, batchSize);
}

cublasStatus_t cublasSgetriBatched(cublasHandle_t handle, int n, const float* const A[], int lda, const int* P, float* const C[], int ldc, int* info, int batchSize) {
    return hipblasSgetriBatched(handle, n, const_cast<float* const*>(A), lda, const_cast<int*>(P), C, ldc, info, batchSize);
}

cublasStatus_t cublasDgetriBatched(cublasHandle_t handle, int n, const double* const A[], int lda, const int* P, double* const C[], int ldc, int* info, int batchSize) {
    return hipblasDgetriBatched(handle, n, const_cast<double* const*>(A), lda, const_cast<int*>(P), C, ldc, info, batchSize);
}

cublasStatus_t cublasCgetriBatched(cublasHandle_t handle, int n, const cuComplex* const A[], int lda, const int* P, cuComplex* const C[], int ldc, int* info, int batchSize) {
    return hipblasCgetriBatched(handle, n, reinterpret_cast<hipblasComplex* const*>(const_cast<cuComplex* const*>(A)), lda, const_cast<int*>(P), reinterpret_cast<hipblasComplex* const*>(C), ldc, info, batchSize);
}

cublasStatus_t cublasZgetriBatched(cublasHandle_t handle, int n, const cuDoubleComplex* const A[], int lda, const int* P, cuDoubleComplex* const C[], int ldc, int* info, int batchSize) {
    return hipblasZgetriBatched(handle, n, reinterpret_cast<hipblasDoubleComplex* const*>(const_cast<cuDoubleComplex* const*>(A)), lda, const_cast<int*>(P), reinterpret_cast<hipblasDoubleComplex* const*>(C), ldc, info, batchSize);
}

cublasStatus_t cublasSgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const float* const Aarray[], int lda, const int* devIpiv, float* const Barray[], int ldb, int* info, int batchSize) {
    return hipblasSgetrsBatched(handle, convert_hipblasOperation_t(trans), n, nrhs, const_cast<float* const*>(Aarray), lda, devIpiv, Barray, ldb, info, batchSize);
}

cublasStatus_t cublasDgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const double* const Aarray[], int lda, const int* devIpiv, double* const Barray[], int ldb, int* info, int batchSize) {
    return hipblasDgetrsBatched(handle, convert_hipblasOperation_t(trans), n, nrhs, const_cast<double* const*>(Aarray), lda, devIpiv, Barray, ldb, info, batchSize);
}

cublasStatus_t cublasCgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const cuComplex* const Aarray[], int lda, const int* devIpiv, cuComplex* const Barray[], int ldb, int* info, int batchSize) {
    return hipblasCgetrsBatched(handle, convert_hipblasOperation_t(trans), n, nrhs, reinterpret_cast<hipblasComplex* const*>(const_cast<cuComplex* const*>(Aarray)), lda, devIpiv, reinterpret_cast<hipblasComplex* const*>(Barray), ldb, info, batchSize);
}

cublasStatus_t cublasZgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const cuDoubleComplex* const Aarray[], int lda, const int* devIpiv, cuDoubleComplex* const Barray[], int ldb, int* info, int batchSize) {
    return hipblasZgetrsBatched(handle, convert_hipblasOperation_t(trans), n, nrhs, reinterpret_cast<hipblasDoubleComplex* const*>(const_cast<cuDoubleComplex* const*>(Aarray)), lda, devIpiv, reinterpret_cast<hipblasDoubleComplex* const*>(Barray), ldb, info, batchSize);
}

cublasStatus_t cublasStrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const float* alpha, const float* const A[], int lda, float* const B[], int ldb, int batchCount) {
    return hipblasStrsmBatched(handle, convert_hipblasSideMode_t(side), convert_hipblasFillMode_t(uplo), convert_hipblasOperation_t(trans), convert_hipblasDiagType_t(diag), m, n, alpha, const_cast<float* const*>(A), lda, const_cast<float**>(B), ldb, batchCount);
}

cublasStatus_t cublasDtrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const double* alpha, const double* const A[], int lda, double* const B[], int ldb, int batchCount) {
    return hipblasDtrsmBatched(handle, convert_hipblasSideMode_t(side), convert_hipblasFillMode_t(uplo), convert_hipblasOperation_t(trans), convert_hipblasDiagType_t(diag), m, n, alpha, const_cast<double* const*>(A), lda, const_cast<double**>(B), ldb, batchCount);
}

cublasStatus_t cublasCtrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuComplex* alpha, const cuComplex* const A[], int lda, cuComplex* const B[], int ldb, int batchCount) {
    return hipblasCtrsmBatched(handle, convert_hipblasSideMode_t(side), convert_hipblasFillMode_t(uplo), convert_hipblasOperation_t(trans), convert_hipblasDiagType_t(diag), m, n, reinterpret_cast<const hipblasComplex*>(alpha), reinterpret_cast<hipblasComplex* const*>(const_cast<cuComplex* const*>(A)), lda, reinterpret_cast<hipblasComplex**>(const_cast<cuComplex**>(B)), ldb, batchCount);
}

cublasStatus_t cublasZtrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* const A[], int lda, cuDoubleComplex* const B[], int ldb, int batchCount) {
    return hipblasZtrsmBatched(handle, convert_hipblasSideMode_t(side), convert_hipblasFillMode_t(uplo), convert_hipblasOperation_t(trans), convert_hipblasDiagType_t(diag), m, n, reinterpret_cast<const hipblasDoubleComplex*>(alpha), reinterpret_cast<hipblasDoubleComplex* const*>(const_cast<cuDoubleComplex* const*>(A)), lda, reinterpret_cast<hipblasDoubleComplex**>(const_cast<cuDoubleComplex**>(B)), ldb, batchCount);
}

cublasStatus_t cublasSmatinvBatched(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasDmatinvBatched(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasCmatinvBatched(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasZmatinvBatched(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasSgeqrfBatched(cublasHandle_t handle, int m, int n, float* const Aarray[], int lda, float* const TauArray[], int* info, int batchSize) {
    return hipblasSgeqrfBatched(handle, m, n, Aarray, lda, TauArray, info, batchSize);
}

cublasStatus_t cublasDgeqrfBatched(cublasHandle_t handle, int m, int n, double* const Aarray[], int lda, double* const TauArray[], int* info, int batchSize) {
    return hipblasDgeqrfBatched(handle, m, n, Aarray, lda, TauArray, info, batchSize);
}

cublasStatus_t cublasCgeqrfBatched(cublasHandle_t handle, int m, int n, cuComplex* const Aarray[], int lda, cuComplex* const TauArray[], int* info, int batchSize) {
    return hipblasCgeqrfBatched(handle, m, n, reinterpret_cast<hipblasComplex* const*>(Aarray), lda, reinterpret_cast<hipblasComplex* const*>(TauArray), info, batchSize);
}

cublasStatus_t cublasZgeqrfBatched(cublasHandle_t handle, int m, int n, cuDoubleComplex* const Aarray[], int lda, cuDoubleComplex* const TauArray[], int* info, int batchSize) {
    return hipblasZgeqrfBatched(handle, m, n, reinterpret_cast<hipblasDoubleComplex* const*>(Aarray), lda, reinterpret_cast<hipblasDoubleComplex* const*>(TauArray), info, batchSize);
}

cublasStatus_t cublasSgelsBatched(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasDgelsBatched(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasCgelsBatched(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasZgelsBatched(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasSdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, const float* A, int lda, const float* x, int incx, float* C, int ldc) {
    return hipblasSdgmm(handle, convert_hipblasSideMode_t(mode), m, n, A, lda, x, incx, C, ldc);
}

cublasStatus_t cublasDdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, const double* A, int lda, const double* x, int incx, double* C, int ldc) {
    return hipblasDdgmm(handle, convert_hipblasSideMode_t(mode), m, n, A, lda, x, incx, C, ldc);
}

cublasStatus_t cublasCdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, const cuComplex* A, int lda, const cuComplex* x, int incx, cuComplex* C, int ldc) {
    return hipblasCdgmm(handle, convert_hipblasSideMode_t(mode), m, n, reinterpret_cast<const hipblasComplex*>(A), lda, reinterpret_cast<const hipblasComplex*>(x), incx, reinterpret_cast<hipblasComplex*>(C), ldc);
}

cublasStatus_t cublasZdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx, cuDoubleComplex* C, int ldc) {
    return hipblasZdgmm(handle, convert_hipblasSideMode_t(mode), m, n, reinterpret_cast<const hipblasDoubleComplex*>(A), lda, reinterpret_cast<const hipblasDoubleComplex*>(x), incx, reinterpret_cast<hipblasDoubleComplex*>(C), ldc);
}

cublasStatus_t cublasStpttr(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasDtpttr(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasCtpttr(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasZtpttr(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasStrttp(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasDtrttp(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasCtrttp(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasZtrttp(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasSetWorkspace(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasGetVersion(cublasHandle_t handle, int *version) {
    // We use the rocBLAS version here because 1. it is the underlying workhorse,
    // and 2. we might get rid of the hipBLAS layer at some point (see TODO above).
    // ex: the rocBLAS version string is 2.22.0.2367-b2cceba in ROCm 3.5.0
    *version = 10000 * ROCBLAS_VERSION_MAJOR + 100 * ROCBLAS_VERSION_MINOR + ROCBLAS_VERSION_PATCH;
    return HIPBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasStrmm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const float* alpha, const float* A, int lda, const float* B, int ldb, float* C, int ldc) {
    // cuBLAS assumes an out-of-place implementation for this routine; however,
    // hipBLAS assumes an in-place implementation same as the BLAS API.
    if (B != C || ldb != ldc) {
	return HIPBLAS_STATUS_NOT_SUPPORTED;
    }
    return hipblasStrmm(handle, convert_hipblasSideMode_t(side), convert_hipblasFillMode_t(uplo), convert_hipblasOperation_t(trans), convert_hipblasDiagType_t(diag), m, n, alpha, A, lda, C, ldc);
}

cublasStatus_t cublasDtrmm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const double* alpha, const double* A, int lda, const double* B, int ldb, double* C, int ldc) {
    // cuBLAS assumes an out-of-place implementation for this routine; however,
    // hipBLAS assumes an in-place implementation same as the BLAS API.
    if (B != C || ldb != ldc) {
	return HIPBLAS_STATUS_NOT_SUPPORTED;
    }
    return hipblasDtrmm(handle, convert_hipblasSideMode_t(side), convert_hipblasFillMode_t(uplo), convert_hipblasOperation_t(trans), convert_hipblasDiagType_t(diag), m, n, alpha, A, lda, const_cast<double*>(B), ldb);
}

cublasStatus_t cublasCtrmm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, cuComplex* C, int ldc) {
    // cuBLAS assumes an out-of-place implementation for this routine; however,
    // hipBLAS assumes an in-place implementation same as the BLAS API.
    if (B != C || ldb != ldc) {
	return HIPBLAS_STATUS_NOT_SUPPORTED;
    }
    return hipblasCtrmm(handle, convert_hipblasSideMode_t(side), convert_hipblasFillMode_t(uplo), convert_hipblasOperation_t(trans), convert_hipblasDiagType_t(diag), m, n, reinterpret_cast<const hipblasComplex*>(alpha), reinterpret_cast<const hipblasComplex*>(A), lda, reinterpret_cast<hipblasComplex*>(const_cast<cuComplex*>(B)), ldb);
}

cublasStatus_t cublasZtrmm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, cuDoubleComplex* C, int ldc) {
    // cuBLAS assumes an out-of-place implementation for this routine; however,
    // hipBLAS assumes an in-place implementation same as the BLAS API.
    if (B != C || ldb != ldc) {
	return HIPBLAS_STATUS_NOT_SUPPORTED;
    }
    return hipblasZtrmm(handle, convert_hipblasSideMode_t(side), convert_hipblasFillMode_t(uplo), convert_hipblasOperation_t(trans), convert_hipblasDiagType_t(diag), m, n, reinterpret_cast<const hipblasDoubleComplex*>(alpha), reinterpret_cast<const hipblasDoubleComplex*>(A), lda, reinterpret_cast<hipblasDoubleComplex*>(const_cast<cuDoubleComplex*>(B)), ldb);
}

cublasStatus_t cublasSgemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
			     int m, int n, int k, const float *alpha,
			     const void *A, cudaDataType_t Atype, int lda,
			     const void *B, cudaDataType_t Btype, int ldb,
			     const float *beta,
			     void *C, cudaDataType_t Ctype, int ldc) {
    if (Atype != 0 || Btype != 0 || Ctype != 0) {  // CUDA_R_32F
        return HIPBLAS_STATUS_NOT_SUPPORTED;
    }
    return hipblasSgemm(handle, convert_hipblasOperation_t(transa), convert_hipblasOperation_t(transb),
                        m, n, k, alpha,
                        static_cast<const float*>(A), lda,
                        static_cast<const float*>(B), ldb, beta,
                        static_cast<float*>(C), ldc);
}

cublasStatus_t cublasGemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                            int m, int n, int k, const void *alpha,
                            const void *A, cudaDataType_t Atype, int lda,
                            const void *B, cudaDataType_t Btype, int ldb,
                            const void *beta,
                            void *C, cudaDataType_t Ctype, int ldc,
                            cudaDataType_t computeType, cublasGemmAlgo_t algo) {
    if (algo != -1) {  // must be CUBLAS_GEMM_DEFAULT
        return HIPBLAS_STATUS_NOT_SUPPORTED;
    }
    return hipblasGemmEx(handle, convert_hipblasOperation_t(transa), convert_hipblasOperation_t(transb),
                         m, n, k, alpha,
                         A, convert_hipblasDatatype_t(Atype), lda,
                         B, convert_hipblasDatatype_t(Btype), ldb,
                         beta,
                         C, convert_hipblasDatatype_t(Ctype), ldc,
                         convert_hipblasDatatype_t(computeType),
                         HIPBLAS_GEMM_DEFAULT);
}

cublasStatus_t cublasGemmEx_v11(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasGemmBatchedEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                                   int m, int n, int k, const void* alpha,
                                   const void* const A[], cudaDataType Atype, int lda,
                                   const void* const B[], cudaDataType Btype, int ldb,
                                   const void* beta,
                                   void* const C[],  cudaDataType Ctype, int ldc,
                                   int batchCount, cudaDataType_t computeType, cublasGemmAlgo_t algo) {
    if (algo != -1) {  // must be CUBLAS_GEMM_DEFAULT
        return HIPBLAS_STATUS_NOT_SUPPORTED;
    }
    return hipblasGemmBatchedEx(handle, convert_hipblasOperation_t(transa), convert_hipblasOperation_t(transb),
                                m, n, k, alpha,
                                const_cast<const void**>(A), convert_hipblasDatatype_t(Atype), lda,
                                const_cast<const void**>(B), convert_hipblasDatatype_t(Btype), ldb,
                                beta,
                                const_cast<void**>(C), convert_hipblasDatatype_t(Ctype), ldc,
                                batchCount, convert_hipblasDatatype_t(computeType),
				HIPBLAS_GEMM_DEFAULT);
}

cublasStatus_t cublasGemmBatchedEx_v11(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
					  int m, int n, int k, const void* alpha,
					  const void* A, cudaDataType Atype, int lda, long long int strideA,
					  const void* B, cudaDataType Btype, int ldb, long long int strideB,
					  const void* beta,
					  void* C, cudaDataType Ctype, int ldc, long long int strideC,
					  int batchCount, cudaDataType_t computeType, cublasGemmAlgo_t algo) {
    if (algo != -1) {  // must be CUBLAS_GEMM_DEFAULT
        return HIPBLAS_STATUS_NOT_SUPPORTED;
    }
    return hipblasGemmStridedBatchedEx(handle, convert_hipblasOperation_t(transa), convert_hipblasOperation_t(transb),
				       m, n, k, alpha,
				       A, convert_hipblasDatatype_t(Atype), lda, strideA,
				       B, convert_hipblasDatatype_t(Btype), ldb, strideB,
				       beta,
                                       C, convert_hipblasDatatype_t(Ctype), ldc, strideC,
                                       batchCount, convert_hipblasDatatype_t(computeType),
                                       HIPBLAS_GEMM_DEFAULT);
}

cublasStatus_t cublasGemmStridedBatchedEx_v11(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

} // extern "C"

#endif // #ifndef INCLUDE_GUARD_HIP_CUPY_HIPBLAS_H
