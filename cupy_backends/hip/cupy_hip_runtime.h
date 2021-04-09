#ifndef INCLUDE_GUARD_HIP_CUPY_RUNTIME_H
#define INCLUDE_GUARD_HIP_CUPY_RUNTIME_H

#include <hip/hip_runtime_api.h>
#include "cupy_hip_common.h"

extern "C" {

bool hip_environment = true;

// Error handling
const char* cudaGetErrorName(cudaError_t hipError) {
    return hipGetErrorName(hipError);
}

const char* cudaGetErrorString(cudaError_t hipError) {
    return hipGetErrorString(hipError);
}

cudaError_t cudaGetLastError() {
    return hipGetLastError();
}


// Initialization
cudaError_t cudaDriverGetVersion(int *driverVersion) {
    return hipDriverGetVersion(driverVersion);
}

cudaError_t cudaRuntimeGetVersion(int *runtimeVersion) {
    return hipRuntimeGetVersion(runtimeVersion);
}


// CUdevice operations
cudaError_t cudaGetDevice(int *deviceId) {
    return hipGetDevice(deviceId);
}

cudaError_t cudaDeviceGetAttribute(int* pi, cudaDeviceAttr attr,
                                   int deviceId) {
    return hipDeviceGetAttribute(pi, attr, deviceId);
}

cudaError_t cudaDeviceGetByPCIBusId(int *device, const char *pciBusId) {
    return hipDeviceGetByPCIBusId(device, pciBusId);
}

cudaError_t cudaDeviceGetPCIBusId(char *pciBusId, int len, int device) {
    return hipDeviceGetPCIBusId(pciBusId, len, device);
}

cudaError_t cudaGetDeviceCount(int *count) {
    return hipGetDeviceCount(count);
}

cudaError_t cudaSetDevice(int deviceId) {
    return hipSetDevice(deviceId);
}

cudaError_t cudaDeviceSynchronize() {
    return hipDeviceSynchronize();
}

cudaError_t cudaDeviceCanAccessPeer(int* canAccessPeer, int deviceId,
                                    int peerDeviceId) {
    return hipDeviceCanAccessPeer(canAccessPeer, deviceId, peerDeviceId);
}

cudaError_t cudaDeviceEnablePeerAccess(int peerDeviceId, unsigned int flags) {
    return hipDeviceEnablePeerAccess(peerDeviceId, flags);
}

cudaError_t cudaDeviceGetLimit(size_t* pValue, cudaLimit limit) {
    return hipDeviceGetLimit(pValue, limit);
}

cudaError_t cudaDeviceSetLimit(cudaLimit limit, size_t value) {
    // see https://github.com/ROCm-Developer-Tools/HIP/issues/1632
    return hipErrorUnknown;
}

// IPC operations
cudaError_t cudaIpcCloseMemHandle(void* devPtr) {
    return hipIpcCloseMemHandle(devPtr);
}

cudaError_t cudaIpcGetEventHandle(cudaIpcEventHandle_t* handle, cudaEvent_t event) {
    return hipErrorUnknown;

    // TODO(leofang): this is supported after ROCm-Developer-Tools/HIP#1996 is released;
    // as of ROCm 3.5.0 it is still not supported
    //return hipIpcGetEventHandle(handle, event);
}

cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t* handle, void* devPtr) {
    return hipIpcGetMemHandle(handle, devPtr);
}

cudaError_t cudaIpcOpenEventHandle(cudaEvent_t* event, cudaIpcEventHandle_t handle) {
    return hipErrorUnknown;

    // TODO(leofang): this is supported after ROCm-Developer-Tools/HIP#1996 is released;
    // as of ROCm 3.5.0 it is still not supported
    //return hipIpcOpenEventHandle(event, handle);
}

cudaError_t cudaIpcOpenMemHandle(void** devPtr, cudaIpcMemHandle_t handle, unsigned int flags) {
    return hipIpcOpenMemHandle(devPtr, handle, flags);
}

// Memory management
cudaError_t cudaMalloc(void** ptr, size_t size) {
    return hipMalloc(ptr, size);
}

cudaError_t cudaMalloc3DArray(...) {
    return hipErrorUnknown;
}

cudaError_t cudaMallocArray(...) {
    return hipErrorUnknown;
}

cudaError_t cudaMallocAsync(...) {
    return hipErrorUnknown;
}

cudaError_t cudaHostAlloc(void** ptr, size_t size, unsigned int flags) {
    return hipHostMalloc(ptr, size, flags);
}

cudaError_t cudaHostRegister(...) {
    return hipErrorUnknown;
}

cudaError_t cudaHostUnregister(...) {
    return hipErrorUnknown;
}

cudaError_t cudaMallocManaged(...) {
    return hipErrorUnknown;
}

int cudaFree(void* ptr) {
    return hipFree(ptr);
}

cudaError_t cudaFreeArray(...) {
    return hipErrorUnknown;
}

cudaError_t cudaFreeHost(void* ptr) {
    return hipHostFree(ptr);
}

cudaError_t cudaFreeAsync(...) {
    return hipErrorUnknown;
}

int cudaMemGetInfo(size_t* free, size_t* total) {
    return hipMemGetInfo(free, total);
}

cudaError_t cudaMemcpy(void* dst, const void* src, size_t sizeBytes,
                       hipMemcpyKind kind) {
    return hipMemcpy(dst, src, sizeBytes, kind);
}

cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t sizeBytes,
                            cudaMemcpyKind kind, cudaStream_t stream) {
    return hipMemcpyAsync(dst, src, sizeBytes, kind, stream);
}

cudaError_t cudaMemcpyPeer(void* dst, int dstDeviceId, const void* src,
                           int srcDeviceId, size_t sizeBytes) {
    return hipMemcpyPeer(dst, dstDeviceId, src, srcDeviceId, sizeBytes);
}

cudaError_t cudaMemcpyPeerAsync(void* dst, int dstDevice, const void* src,
                                int srcDevice, size_t sizeBytes,
                                cudaStream_t stream) {
    return hipMemcpyPeerAsync(dst, dstDevice, src, srcDevice, sizeBytes,
                              stream);
}

cudaError_t cudaMemcpy2D(...) {
    return hipErrorUnknown;
}

cudaError_t cudaMemcpy2DAsync(...) {
    return hipErrorUnknown;
}

cudaError_t cudaMemcpy2DFromArray(...) {
    return hipErrorUnknown;
}

cudaError_t cudaMemcpy2DFromArrayAsync(...) {
    return hipErrorUnknown;
}

cudaError_t cudaMemcpy2DToArray(...) {
    return hipErrorUnknown;
}

cudaError_t cudaMemcpy2DToArrayAsync(...) {
    return hipErrorUnknown;
}

cudaError_t cudaMemcpy3D(...) {
    return hipErrorUnknown;
}

cudaError_t cudaMemcpy3DAsync(...) {
    return hipErrorUnknown;
}

cudaError_t cudaMemset(void* dst, int value, size_t sizeBytes) {
    return hipMemset(dst, value, sizeBytes);
}

cudaError_t cudaMemsetAsync(void* dst, int value, size_t sizeBytes,
                            cudaStream_t stream) {
    return hipMemsetAsync(dst, value, sizeBytes, stream);
}

cudaError_t cudaMemAdvise(...) {
    return hipErrorUnknown;
}

cudaError_t cudaMemPrefetchAsync(...) {
    return hipErrorUnknown;
}

cudaError_t cudaPointerGetAttributes(cudaPointerAttributes *attributes,
                                     const void* ptr) {
    return hipPointerGetAttributes(attributes, ptr);
}

cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int device) {
    return hipGetDeviceProperties(prop, device);
}

cudaError_t cudaDeviceGetDefaultMemPool(...) {
    return hipErrorUnknown;
}

cudaError_t cudaDeviceGetMemPool(...) {
    return hipErrorUnknown;
}

cudaError_t cudaDeviceSetMemPool(...) {
    return hipErrorUnknown;
}

cudaError_t cudaMemPoolTrimTo(...) {
    return hipErrorUnknown;
}


// Stream and Event
cudaError_t cudaStreamCreate(cudaStream_t *stream) {
    return hipStreamCreate(stream);
}

cudaError_t cudaStreamCreateWithFlags(cudaStream_t *stream,
                                      unsigned int flags) {
    return hipStreamCreateWithFlags(stream, flags);
}

cudaError_t cudaStreamDestroy(cudaStream_t stream) {
    return hipStreamDestroy(stream);
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    return hipStreamSynchronize(stream);
}

cudaError_t cudaStreamAddCallback(cudaStream_t stream,
                                  cudaStreamCallback_t callback,
                                  void *userData, unsigned int flags) {
    return hipStreamAddCallback(stream, callback, userData, flags);
}

cudaError_t cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void* userData) {
    return hipErrorUnknown;
}

cudaError_t cudaStreamQuery(cudaStream_t stream) {
    return hipStreamQuery(stream);
}

cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event,
                                unsigned int flags) {
    return hipStreamWaitEvent(stream, event, flags);
}

cudaError_t cudaEventCreate(cudaEvent_t* event) {
    return hipEventCreate(event);
}

cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned flags) {
    return hipEventCreateWithFlags(event, flags);
}

cudaError_t cudaEventDestroy(cudaEvent_t event) {
    return hipEventDestroy(event);
}

cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start,
                                 cudaEvent_t stop){
    return hipEventElapsedTime(ms, start, stop);
}

cudaError_t cudaEventQuery(cudaEvent_t event) {
    return hipEventQuery(event);
}

cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    return hipEventRecord(event, stream);
}

cudaError_t cudaEventSynchronize(cudaEvent_t event) {
    return hipEventSynchronize(event);
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
    cudaExtent ex = {};
    return ex;
}

cudaPitchedPtr make_cudaPitchedPtr(...) {
    cudaPitchedPtr ptr = {};
    return ptr;
}

cudaPos make_cudaPos(...) {
    cudaPos pos = {};
    return pos;
}

// Surface
cudaError_t cudaCreateSurfaceObject(cudaSurfaceObject_t* pSurfObject,
                                    const cudaResourceDesc* pResDesc) {
    return hipCreateSurfaceObject(pSurfObject, pResDesc);
}

cudaError_t cudaDestroySurfaceObject(cudaSurfaceObject_t surfObject) {
    return hipDestroySurfaceObject(surfObject);
}

} // extern "C"

#endif // #ifndef INCLUDE_GUARD_HIP_CUPY_RUNTIME_H
