// This file is a stub header file of cuda for Read the Docs.

#ifndef INCLUDE_GUARD_STUB_CUPY_CUDA_RUNTIME_H
#define INCLUDE_GUARD_STUB_CUPY_CUDA_RUNTIME_H

#include "cupy_cuda_common.h"

extern "C" {

bool hip_environment = false;

// Error handling
const char* cudaGetErrorName(...) {
    return NULL;
}

const char* cudaGetErrorString(...) {
    return NULL;
}

cudaError_t cudaGetLastError() {
    return cudaSuccess;
}


// Initialization
cudaError_t cudaDriverGetVersion(...) {
    return cudaSuccess;
}

cudaError_t cudaRuntimeGetVersion(...) {
    return cudaSuccess;
}


// CUdevice operations
cudaError_t cudaGetDevice(...) {
    return cudaSuccess;
}

cudaError_t cudaDeviceGetAttribute(...) {
    return cudaSuccess;
}

cudaError_t cudaDeviceGetByPCIBusId(...) {
    return cudaSuccess;
}

cudaError_t cudaDeviceGetPCIBusId(...) {
    return cudaSuccess;
}

cudaError_t cudaGetDeviceCount(...) {
    return cudaSuccess;
}

cudaError_t cudaSetDevice(...) {
    return cudaSuccess;
}

cudaError_t cudaDeviceSynchronize() {
    return cudaSuccess;
}

cudaError_t cudaGetDeviceProperties(...) {
    return cudaSuccess;
}

cudaError_t cudaDeviceCanAccessPeer(...) {
    return cudaSuccess;
}

cudaError_t cudaDeviceEnablePeerAccess(...) {
    return cudaSuccess;
}

cudaError_t cudaDeviceGetLimit(...) {
    return cudaSuccess;
}

cudaError_t cudaDeviceSetLimit(...) {
    return cudaSuccess;
}

// IPC operations
cudaError_t cudaIpcCloseMemHandle(...){
    return cudaSuccess;
}

cudaError_t cudaIpcGetEventHandle(...){
    return cudaSuccess;
}

cudaError_t cudaIpcGetMemHandle(...){
    return cudaSuccess;
}

cudaError_t cudaIpcOpenEventHandle(...){
    return cudaSuccess;
}

cudaError_t cudaIpcOpenMemHandle(...){
    return cudaSuccess;
}

// Memory management
cudaError_t cudaMalloc(...) {
    return cudaSuccess;
}

cudaError_t cudaMalloc3DArray(...) {
    return cudaSuccess;
}

cudaError_t cudaMallocArray(...) {
    return cudaSuccess;
}

cudaError_t cudaMallocAsync(...) {
    return cudaSuccess;
}

cudaError_t cudaHostAlloc(...) {
    return cudaSuccess;
}

cudaError_t cudaHostRegister(...) {
    return cudaSuccess;
}

cudaError_t cudaHostUnregister(...) {
    return cudaSuccess;
}

cudaError_t cudaMallocManaged(...) {
    return cudaSuccess;
}

int cudaFree(...) {
    return cudaSuccess;
}

cudaError_t cudaFreeArray(...) {
    return cudaSuccess;
}

cudaError_t cudaFreeHost(...) {
    return cudaSuccess;
}

cudaError_t cudaFreeAsync(...) {
    return cudaSuccess;
}

int cudaMemGetInfo(...) {
    return cudaSuccess;
}

cudaError_t cudaMemcpy(...) {
    return cudaSuccess;
}

cudaError_t cudaMemcpyAsync(...) {
    return cudaSuccess;
}

cudaError_t cudaMemcpyPeer(...) {
    return cudaSuccess;
}

cudaError_t cudaMemcpyPeerAsync(...) {
    return cudaSuccess;
}

cudaError_t cudaMemcpy2D(...) {
    return cudaSuccess;
}

cudaError_t cudaMemcpy2DAsync(...) {
    return cudaSuccess;
}

cudaError_t cudaMemcpy2DFromArray(...) {
    return cudaSuccess;
}

cudaError_t cudaMemcpy2DFromArrayAsync(...) {
    return cudaSuccess;
}

cudaError_t cudaMemcpy2DToArray(...) {
    return cudaSuccess;
}

cudaError_t cudaMemcpy2DToArrayAsync(...) {
    return cudaSuccess;
}

cudaError_t cudaMemcpy3D(...) {
    return cudaSuccess;
}

cudaError_t cudaMemcpy3DAsync(...) {
    return cudaSuccess;
}

cudaError_t cudaMemset(...) {
    return cudaSuccess;
}

cudaError_t cudaMemsetAsync(...) {
    return cudaSuccess;
}

cudaError_t cudaMemAdvise(...) {
    return cudaSuccess;
}

cudaError_t cudaMemPrefetchAsync(...) {
    return cudaSuccess;
}

cudaError_t cudaPointerGetAttributes(...) {
    return cudaSuccess;
}

cudaError_t cudaDeviceGetDefaultMemPool(...) {
    return cudaSuccess;
}

cudaError_t cudaDeviceGetMemPool(...) {
    return cudaSuccess;
}

cudaError_t cudaDeviceSetMemPool(...) {
    return cudaSuccess;
}

cudaError_t cudaMemPoolTrimTo(...) {
    return cudaSuccess;
}


// Stream and Event
cudaError_t cudaStreamCreate(...) {
    return cudaSuccess;
}

cudaError_t cudaStreamCreateWithFlags(...) {
    return cudaSuccess;
}

cudaError_t cudaStreamDestroy(...) {
    return cudaSuccess;
}

cudaError_t cudaStreamSynchronize(...) {
    return cudaSuccess;
}

cudaError_t cudaStreamAddCallback(...) {
    return cudaSuccess;
}

cudaError_t cudaLaunchHostFunc(...) {
    return cudaSuccess;
}

cudaError_t cudaStreamQuery(...) {
    return cudaSuccess;
}

cudaError_t cudaStreamWaitEvent(...) {
    return cudaSuccess;
}

cudaError_t cudaEventCreate(...) {
    return cudaSuccess;
}

cudaError_t cudaEventCreateWithFlags(...) {
    return cudaSuccess;
}

cudaError_t cudaEventDestroy(...) {
    return cudaSuccess;
}

cudaError_t cudaEventElapsedTime(...) {
    return cudaSuccess;
}

cudaError_t cudaEventQuery(...) {
    return cudaSuccess;
}

cudaError_t cudaEventRecord(...) {
    return cudaSuccess;
}

cudaError_t cudaEventSynchronize(...) {
    return cudaSuccess;
}


// Texture
cudaError_t cudaCreateTextureObject(...) {
    return cudaSuccess;
}

cudaError_t cudaDestroyTextureObject(...) {
    return cudaSuccess;
}

cudaError_t cudaGetChannelDesc(...) {
    return cudaSuccess;
}

cudaError_t cudaGetTextureObjectResourceDesc(...) {
    return cudaSuccess;
}

cudaError_t cudaGetTextureObjectTextureDesc(...) {
    return cudaSuccess;
}

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

// Surface
cudaError_t cudaCreateSurfaceObject(...) {
    return cudaSuccess;
}

cudaError_t cudaDestroySurfaceObject(...) {
    return cudaSuccess;
}

} // extern "C"

#endif // #ifndef INCLUDE_GUARD_STUB_CUPY_CUDA_RUNTIME_H
