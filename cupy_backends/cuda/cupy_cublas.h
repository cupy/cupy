// This code was automatically generated. Do not modify it directly.

#ifndef INCLUDE_GUARD_CUDA_CUPY_CUBLAS_H
#define INCLUDE_GUARD_CUDA_CUPY_CUBLAS_H

#include <cuda.h>
#include <cublas_v2.h>

#if CUDA_VERSION < 11000
// Added in CUDA 11.0

typedef enum{} cublasComputeType_t;

cublasStatus_t cublasSetWorkspace(...) {
    return CUBLAS_STATUS_SUCCESS;
}

#endif  // #if CUDA_VERSION < 11000

#if CUDA_VERSION >= 11000

#define cublasGemmEx_v11 cublasGemmEx
#define cublasGemmBatchedEx_v11 cublasGemmBatchedEx
#define cublasGemmStridedBatchedEx_v11 cublasGemmStridedBatchedEx

#else

cublasStatus_t cublasGemmEx_v11(...) {
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasGemmBatchedEx_v11(...) {
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasGemmStridedBatchedEx_v11(...) {
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

#endif // if CUDA_VERSION >= 11000

#endif // #ifndef INCLUDE_GUARD_CUDA_CUPY_CUBLAS_H
