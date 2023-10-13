// This file is a stub header file of cuda for Read the Docs.

#ifndef INCLUDE_GUARD_STUB_CUPY_CUDA_RUNTIME_H
#define INCLUDE_GUARD_STUB_CUPY_CUDA_RUNTIME_H

#include "cupy_cuda_common.h"

extern "C" {

bool hip_environment = false;

// Memory management
enum cudaMemAllocationType {};
enum cudaMemAllocationHandleType {};
enum cudaMemLocationType {};
struct cudaMemLocation {
    int id;
};
struct cudaMemPoolProps {
    cudaMemAllocationType allocType;
    cudaMemAllocationHandleType handleTypes;
    struct cudaMemLocation location;
    unsigned char reserved[64];
    void* win32SecurityAttributes;
};


// Stream and Event
enum cudaStreamCaptureMode {};
enum cudaStreamCaptureStatus {};


// Texture
cudaExtent make_cudaExtent(...) {
    struct cudaExtent ex = {0};
    return ex;
}

cudaPitchedPtr make_cudaPitchedPtr(...) {
    struct cudaPitchedPtr ptr = {0};
    return ptr;
}

cudaPos make_cudaPos(...) {
    struct cudaPos pos = {0};
    return pos;
}

} // extern "C"

#endif // #ifndef INCLUDE_GUARD_STUB_CUPY_CUDA_RUNTIME_H
