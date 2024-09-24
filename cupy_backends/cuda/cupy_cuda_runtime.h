#ifndef INCLUDE_GUARD_CUDA_CUPY_CUDA_RUNTIME_H
#define INCLUDE_GUARD_CUDA_CUPY_CUDA_RUNTIME_H

#include <cuda.h>  // for CUDA_VERSION
#include <cuda_runtime.h>

extern "C" {

bool hip_environment = false;

#if CUDA_VERSION < 10000
// APIs added in CUDA 10.0

cudaError_t cudaGraphCreate(...) {
    return cudaErrorUnknown;
}

#endif

#if CUDA_VERSION < 10010
// APIs added in CUDA 10.1

const int cudaErrorContextIsDestroyed = 709;

cudaError_t cudaStreamGetCaptureInfo(...) {
    return cudaErrorUnknown;
}

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

#if CUDA_VERSION < 11030
// APIs added in CUDA 11.3

enum cudaStreamUpdateCaptureDependenciesFlags {};

cudaError_t cudaGraphDebugDotPrint(...) {
    return cudaErrorUnknown;
}

cudaError_t cudaStreamUpdateCaptureDependencies(...) {
    return cudaErrorUnknown;
}

#endif

#if CUDA_VERSION < 12020
// APIs added in CUDA 12.2

cudaError_t cudaGraphAddNode(...) {
    return cudaErrorUnknown;
}

#endif

#if CUDA_VERSION < 12030
// APIs added in CUDA 12.3

enum cudaGraphConditionalNodeType {};
typedef unsigned long long cudaGraphConditionalHandle;
struct cudaConditionalNodeParams {
    cudaGraphConditionalHandle handle;
    cudaGraphConditionalNodeType type;
    unsigned int size;
    cudaGraph_t* phGraph_out;
};
struct cudaGraphNodeParams {
    /**
     * struct cudaGraphNodeParams is introduced at CUDA 12.2,
     * but at CUDA 12.2, this struct does not have attribute `conditional`,
     * which is required to build CuPy.
     * So we re-define cudaGraphNodeParams here for CUDA 12.2.
     */
    cudaGraphNodeType type;
    int reserved0[3];
    union {
        long long reserved1[29];
        struct cudaConditionalNodeParams conditional;
    };
    long long reserved2;
};
typedef struct cudaGraphEdgeData_st {
    unsigned char from_port;
    unsigned char to_port;
    unsigned char type;
    unsigned char reserved[5];
} cudaGraphEdgeData;

cudaError_t cudaStreamBeginCaptureToGraph(...) {
    return cudaErrorUnknown;
}

cudaError_t cudaGraphConditionalHandleCreate(...) {
    return cudaErrorUnknown;
}

#endif

} // extern "C"

#endif // #ifndef INCLUDE_GUARD_CUDA_CUPY_CUDA_RUNTIME_H
