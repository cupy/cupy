#ifndef INCLUDE_GUARD_CUDA_CUPY_CUBLAS_H
#define INCLUDE_GUARD_CUDA_CUPY_CUBLAS_H

#include <cuda.h>
#include <cublas_v2.h>

#if CUDA_VERSION >= 11000

#define cublasGemmEx_v11 cublasGemmEx

#else

typedef enum{} cublasComputeType_t;
cublasStatus_t cublasGemmEx_v11(...) {
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

#endif // if CUDA_VERSION >= 11000

#endif // #ifndef INCLUDE_GUARD_CUDA_CUPY_CUBLAS_H
