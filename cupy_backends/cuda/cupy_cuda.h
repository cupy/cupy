#ifndef INCLUDE_GUARD_CUDA_CUPY_CUDA_H
#define INCLUDE_GUARD_CUDA_CUPY_CUDA_H

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

#endif // #ifndef INCLUDE_GUARD_CUDA_CUPY_CUDA_H
