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

cudaError_t cudaDeviceDisablePeerAccess(int peerDeviceId) {
    return hipDeviceDisablePeerAccess(peerDeviceId);
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
enum cudaMemAllocationType {};  // stub
enum cudaMemAllocationHandleType {};  // stub
enum cudaMemLocationType {};  // stub
struct cudaMemLocation {  // stub
    int id;
    cudaMemLocationType type;
};
struct cudaMemPoolProps {  // stub
    cudaMemAllocationType allocType;
    cudaMemAllocationHandleType handleTypes;
    struct cudaMemLocation location;
    unsigned char reserved[64];
    void* win32SecurityAttributes;
};

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

cudaError_t cudaMallocManaged(void** ptr, size_t size, unsigned int flags) {
#if HIP_VERSION >= 40300000
    return hipMallocManaged(ptr, size, flags);
#else
    return hipErrorUnknown;
#endif
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

cudaError_t cudaMemAdvise(const void *devPtr, size_t count,
                          cudaMemoryAdvise advice, int device) {
#if HIP_VERSION >= 40300000
    return hipMemAdvise(devPtr, count, advice, device);
#else
    return hipErrorUnknown;
#endif
}

cudaError_t cudaMemPrefetchAsync(const void *devPtr, size_t count,
				 int dstDevice, cudaStream_t stream) {
#if HIP_VERSION >= 40300000
    return hipMemPrefetchAsync(devPtr, count, dstDevice, stream);
#else
    return hipErrorUnknown;
#endif
}

cudaError_t cudaPointerGetAttributes(cudaPointerAttributes *attributes,
                                     const void* ptr) {
    cudaError_t status = hipPointerGetAttributes(attributes, ptr);
    if (status == cudaSuccess) {
#if HIP_VERSION >= 600
        switch (attributes->type) {
            case 0 /* hipMemoryTypeHost */:
                attributes->type = (hipMemoryType)1; /* cudaMemoryTypeHost */
                return status;
            case 1 /* hipMemoryTypeDevice */:
                attributes->type = (hipMemoryType)2; /* cudaMemoryTypeDevice */
                return status;
#else
       switch (attributes->memoryType) {
            case 0 /* hipMemoryTypeHost */:
                attributes->memoryType = (hipMemoryType)1; /* cudaMemoryTypeHost */
                return status;
            case 1 /* hipMemoryTypeDevice */:
                attributes->memoryType = (hipMemoryType)2; /* cudaMemoryTypeDevice */
                return status;
#endif
            default:
                /* we don't care the rest of possibilities */
                return status;
        }
    } else {
        return status;
    }
}

cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int device) {
    return hipGetDeviceProperties(prop, device);
}

cudaError_t cudaMallocFromPoolAsync(...) {
    return hipErrorUnknown;
}

cudaError_t cudaMemPoolCreate(...) {
    return hipErrorUnknown;
}

cudaError_t cudaMemPoolDestroy(...) {
    return hipErrorUnknown;
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

cudaError_t cudaMemPoolGetAttribute(...) {
    return hipErrorUnknown;
}

cudaError_t cudaMemPoolSetAttribute(...) {
    return hipErrorUnknown;
}


// Stream and Event
#if HIP_VERSION >= 40300000
typedef hipStreamCaptureMode cudaStreamCaptureMode;
typedef hipStreamCaptureStatus cudaStreamCaptureStatus;
#else
enum cudaStreamCaptureMode {};
enum cudaStreamCaptureStatus {};
#endif

cudaError_t cudaStreamCreate(cudaStream_t *stream) {
    return hipStreamCreate(stream);
}

cudaError_t cudaStreamCreateWithFlags(cudaStream_t *stream,
                                      unsigned int flags) {
    return hipStreamCreateWithFlags(stream, flags);
}

cudaError_t cudaStreamCreateWithPriority(cudaStream_t *stream,
                                         unsigned int flags,
                                         int priority) {
    return hipStreamCreateWithPriority(stream, flags, priority);
}

cudaError_t cudaStreamGetFlags(cudaStream_t stream, unsigned int *flags) {
    return hipStreamGetFlags(stream, flags);
}

cudaError_t cudaStreamGetPriority(cudaStream_t stream, int *priority) {
    return hipStreamGetPriority(stream, priority);
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

cudaError_t cudaStreamBeginCapture(cudaStream_t stream,
                                   cudaStreamCaptureMode mode) {
#if HIP_VERSION >= 40300000
    return hipStreamBeginCapture(stream, mode);
#else
    return hipErrorUnknown;
#endif
}

cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t* pGraph) {
#if HIP_VERSION >= 40300000
    return hipStreamEndCapture(stream, pGraph);
#else
    return hipErrorUnknown;
#endif
}

cudaError_t cudaStreamIsCapturing(cudaStream_t stream,
                                  cudaStreamCaptureStatus* pCaptureStatus) {
#if HIP_VERSION >= 50000000
    return hipStreamIsCapturing(stream, pCaptureStatus);
#else
    return hipErrorUnknown;
#endif
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

// CUDA Graph
cudaError_t cudaGraphInstantiate(
	cudaGraphExec_t* pGraphExec,
	cudaGraph_t graph,
	cudaGraphNode_t* pErrorNode,
	char* pLogBuffer,
	size_t bufferSize) {
#if HIP_VERSION >= 40300000
    return hipGraphInstantiate(pGraphExec, graph, pErrorNode, pLogBuffer, bufferSize);
#else
    return hipErrorUnknown;
#endif
}

cudaError_t cudaGraphDestroy(cudaGraph_t graph) {
#if HIP_VERSION >= 40300000
    return hipGraphDestroy(graph);
#else
    return hipErrorUnknown;
#endif
}

cudaError_t cudaGraphExecDestroy(cudaGraphExec_t graphExec) {
#if HIP_VERSION >= 40300000
    return hipGraphExecDestroy(graphExec);
#else
    return hipErrorUnknown;
#endif
}

cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream) {
#if HIP_VERSION >= 40300000
    return hipGraphLaunch(graphExec, stream);
#else
    return hipErrorUnknown;
#endif
}

cudaError_t cudaGraphUpload(...) {
    return hipErrorUnknown;
}

} // extern "C"

#endif // #ifndef INCLUDE_GUARD_HIP_CUPY_RUNTIME_H
