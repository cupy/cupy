// This file is a stub header file of cuda for Read the Docs.

#ifndef INCLUDE_GUARD_CUPY_CUBLAS_H
#define INCLUDE_GUARD_CUPY_CUBLAS_H

#if CUPY_USE_HIP

#include "hip/cupy_hipblas.h"

#elif !defined(CUPY_NO_CUDA)

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

#else // #ifndef CUPY_NO_CUDA

#include "stub/cupy_cublas.h"

#endif // #ifndef CUPY_NO_CUDA
#endif // #ifndef INCLUDE_GUARD_CUPY_CUBLAS_H
