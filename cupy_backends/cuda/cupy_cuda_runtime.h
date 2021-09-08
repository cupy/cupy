#ifndef INCLUDE_GUARD_CUDA_CUPY_CUDA_RUNTIME_H
#define INCLUDE_GUARD_CUDA_CUPY_CUDA_RUNTIME_H

#include <cuda.h>  // for CUDA_VERSION
#include <cuda_runtime.h>

extern "C" {

bool hip_environment = false;

#if CUDA_VERSION < 10010
const int cudaErrorContextIsDestroyed = 709;
#endif

#if CUDA_VERSION < 11020

typedef void* cudaMemPool_t;
enum cudaMemPoolAttr {};

cudaError_t cudaDeviceGetDefaultMemPool(...) {
    return cudaSuccess;
}

cudaError_t cudaDeviceGetMemPool(...) {
    return cudaSuccess;
}

cudaError_t cudaDeviceSetMemPool(...) {
    return cudaSuccess;
}

cudaError_t cudaFreeAsync(...) {
    return cudaSuccess;
}

cudaError_t cudaMallocAsync(...) {
    return cudaSuccess;
}

cudaError_t cudaMemPoolTrimTo(...) {
    return cudaSuccess;
}

cudaError_t cudaMemPoolGetAttribute(...) {
    return cudaSuccess;
}

cudaError_t cudaMemPoolSetAttribute(...) {
    return cudaSuccess;
}

#endif


} // extern "C"

#endif // #ifndef INCLUDE_GUARD_CUDA_CUPY_CUDA_RUNTIME_H
