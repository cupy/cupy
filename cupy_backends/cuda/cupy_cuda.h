// This file is a stub header file of cuda for Read the Docs.

#ifndef INCLUDE_GUARD_CUPY_CUDA_H
#define INCLUDE_GUARD_CUPY_CUDA_H

#if CUPY_USE_HIP

#include "hip/cupy_cuda.h"

#elif !defined(CUPY_NO_CUDA)

#include <cuda.h>
#include <cuda_runtime.h>

extern "C" {

#if CUDA_VERSION < 9000

CUresult cuFuncSetAttribute(...) {
    return CUDA_ERROR_NOT_SUPPORTED;
}

typedef enum {} cublasMath_t;

cublasStatus_t cublasSetMathMode(...) {
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasGetMathMode(...) {
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

CUresult cuLaunchCooperativeKernel(...) {
    return CUDA_ERROR_NOT_SUPPORTED;
}

#endif // #if CUDA_VERSION < 9000

} // extern "C"

#else // #ifndef CUPY_NO_CUDA

#include "stub/cupy_cuda.h"

#endif // #ifndef CUPY_NO_CUDA
#endif // #ifndef INCLUDE_GUARD_CUPY_CUDA_H
