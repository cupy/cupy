#ifndef INCLUDE_GUARD_CUDA_CUPY_CUDA_RUNTIME_H
#define INCLUDE_GUARD_CUDA_CUPY_CUDA_RUNTIME_H

#include <cuda.h>  // for CUDA_VERSION
#include <cuda_runtime.h>

extern "C" {

bool hip_environment = false;

#if CUDA_VERSION < 10010
const int cudaErrorContextIsDestroyed = 709;
#endif

#if CUDA_VERSION < 11010
// APIs added in CUDA 11.1
cudaError_t cudaGraphUpload(...) {
    return cudaErrorUnknown;
}
#endif

#if CUDA_VERSION < 11020
// APIs added in CUDA 11.2
typedef void* cudaMemPool_t;
enum cudaMemPoolAttr {};
enum cudaMemAllocationType {};
enum cudaMemAllocationHandleType {};
enum cudaMemLocationType {};
struct cudaMemLocation {
    int id;
    cudaMemLocationType type;
};
struct cudaMemPoolProps {
    cudaMemAllocationType allocType;
    cudaMemAllocationHandleType handleTypes;
    struct cudaMemLocation location;
    unsigned char reserved[64];
    void* win32SecurityAttributes;
};

cudaError_t cudaMallocAsync(...) {
    return cudaErrorUnknown;
}

cudaError_t cudaMallocFromPoolAsync(...) {
    return cudaErrorUnknown;
}

cudaError_t cudaFreeAsync(...) {
    return cudaErrorUnknown;
}

cudaError_t cudaDeviceSetMemPool(...) {
    return cudaErrorUnknown;
}

cudaError_t cudaDeviceGetDefaultMemPool(...) {
    return cudaErrorUnknown;
}

cudaError_t cudaDeviceGetMemPool(...) {
    return cudaErrorUnknown;
}

cudaError_t cudaMemPoolCreate(...) {
    return cudaErrorUnknown;
}

cudaError_t cudaMemPoolDestroy(...) {
    return cudaErrorUnknown;
}

cudaError_t cudaMemPoolTrimTo(...) {
    return cudaErrorUnknown;
}

cudaError_t cudaMemPoolSetAttribute(...) {
    return cudaErrorUnknown;
}

cudaError_t cudaMemPoolGetAttribute(...) {
    return cudaErrorUnknown;
}

#endif

} // extern "C"

#endif // #ifndef INCLUDE_GUARD_CUDA_CUPY_CUDA_RUNTIME_H
