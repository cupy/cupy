#ifndef INCLUDE_GUARD_ASCEND_CUPY_RUNTIME_H
#define INCLUDE_GUARD_ASCEND_CUPY_RUNTIME_H

#include "acl/acl.h"          // AscendCL主头文件
#include "acl/acl_mdl.h"      // model record
#include "cupy_ascend_common.h" // 假设的自定义头文件

constexpr int ASCEND_DEFAULT_STREAM_PRIORITY = 1; // TODO: value

extern "C" {

bool ascend_environment = true; //  backend name: Ascend

// IMPORTANT: AscendCL requires explicit initialization and finalization
// These calls (aclInit, aclFinalize) should be integrated into the main application lifecycle.

// Error handling
const char* cudaGetErrorName(cudaError_t aclError) {
    // WARNING: Missing direct equivalent in AscendCL for getting error name by code.
    // AscendCL typically handles errors via return codes and aclGetRecentErrMsg().
    return "ACL_ERROR_NAME_UNKNOWN"; // Placeholder
}

const char* cudaGetErrorString(cudaError_t aclError) {
    // Uses aclGetRecentErrMsg to get the last error string. [9](@ref)
    return aclGetRecentErrMsg();
}

cudaError_t cudaGetLastError() {
    // AscendCL does not have a direct "get last error" function like HIP/CUDA.
    // Error status is typically returned by each function.
    // WARNING: This implementation might not be thread-safe or fully equivalent.
    // Consider revising the error handling strategy for AscendCL.
    return ACL_ERROR_NONE; // Placeholder: AscendCL functions return error codes directly.
}

cudaError_t cudaDriverGetVersion(int *driverVersion) {
    // CANN driver package version might be obtained through other means (e.g., system info).
    *driverVersion = 820; // TODO: Assuming CANN 8.2
    return ACL_SUCCESS;
}

cudaError_t cudaRuntimeGetVersion(int *runtimeVersion) {
    // TODO: aclsysGetCANNVersion(aclCANNPackageName name, aclCANNPackageVersion *version);
    *runtimeVersion = 820; // Placeholder: Assuming CANN 8.2
    return ACL_SUCCESS;
}

// Primary context management, cuDevicePrimaryCtxRetain()
CUresult cuDevicePrimaryCtxRelease(CUdevice dev) {
    // WARNING: Missing direct equivalent in AscendCL for primary context release.
    // Context management in AscendCL is explicit via aclrtDestroyContext.
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

// Context management: 
CUresult cuCtxGetCurrent(CUcontext *ctx) {
    aclError ret = aclrtGetCurrentContext(ctx);
    return ret;
}

CUresult cuCtxSetCurrent(CUcontext ctx) {
    aclError ret = aclrtSetCurrentContext(ctx);
    return ret;
}

CUresult cuCtxCreate(CUcontext* pctx, unsigned int flags, CUdevice dev) {
    // ASCEND create context and stream when SetDevice
    // while CUDA create Context and binding to Device
    aclError ret = aclrtSetDevice(dev);
    ret = aclrtGetCurrentContext(pctx);
    // Note: AscendCL's create context might not use flags the same way
    // aclrtCtxSetSysParamOpt() could be the API to set flags
    return ret;
}

CUresult cuCtxDestroy(CUcontext ctx) {
    aclError ret = aclrtDestroyContext(ctx);
    return ret;
}


#ifndef CUPY_INSTALL_USE_ASCEND
cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int device) {
    // // WARNING: AscendCL device properties structure (aclrtDeviceProp) is different from cudaDeviceProp.
    // // Requires manual mapping of fields.
    // aclrtDeviceProp aclProp;
    // aclError ret = aclrtGetDeviceProperties(&aclProp, device);
    // if (ret != ACL_SUCCESS) {
    //     return ret;
    // }
    // // Map relevant fields from aclProp to prop (e.g., name, compute capability)
    // // prop->name = aclProp.name; // Example
    // // ... other fields
    // return ACL_SUCCESS;
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

cudaError_t cudaDeviceGetAttribute(int* pi, cudaDeviceAttr attr,
                                   int deviceId) {
    // // WARNING: AscendCL device attributes are different from CUDA/HIP.
    // // Use aclrtGetDeviceProperties or specific aclGet* functions. 
    // // This is a placeholder and requires mapping CUDA attributes to AscendCL properties.
    // aclrtDeviceProp prop;
    // aclError ret = aclrtGetDeviceProperties(&prop, deviceId);
    // if (ret != ACL_SUCCESS) {
    //     return ret;
    // }
    // // Map specific attr to prop fields (example for compute capability)
    // switch(attr) {
    //     // case cudaDevAttrComputeCapabilityMajor: // WARNING: No direct equivalent
    //     //     *pi = 0; // Placeholder
    //     //     break;
    //     default:
    //         return ACL_ERROR_INVALID_PARAM;
    // }
    // return ACL_SUCCESS;
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

cudaError_t cudaDeviceGetByPCIBusId(int *device, const char *pciBusId) {
    // WARNING: Missing direct equivalent in AscendCL for PCI Bus ID lookup.
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

cudaError_t cudaDeviceGetPCIBusId(char *pciBusId, int len, int device) {
    // WARNING: Missing direct equivalent in AscendCL for getting PCI Bus ID.
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}
#endif

cudaError_t cudaGetDeviceCount(int *count) {
    uint32_t cc = 0;
    aclError ret = aclrtGetDeviceCount(&cc);
    *count = cc;
    return ret;
}

cudaError_t cudaSetDevice(int deviceId) {
    aclError ret = aclrtSetDevice(deviceId); 
    return ret;
}

// Returns the device handle (int) for the current context
cudaError_t cudaGetDevice(CUdevice *deviceId) {
    aclError ret = aclrtGetDevice(deviceId); 
    return ret;
}

cudaError_t cudaDeviceSynchronize() {
    aclError ret = aclrtSynchronizeDevice(); // Synchronizes all streams on the current device. 
    return ret;
}

cudaError_t cudaDeviceCanAccessPeer(int* canAccessPeer, int deviceId,
                                    int peerDeviceId) {
    // Ascend hardware and software stack may have different interconnect capabilities.
    aclError ret = aclrtDeviceCanAccessPeer(canAccessPeer, deviceId, peerDeviceId);
    return ret;
}

cudaError_t cudaDeviceEnablePeerAccess(int peerDeviceId, unsigned int flags) {
    aclError ret = aclrtDeviceEnablePeerAccess(peerDeviceId, flags);
    return ret;
}

cudaError_t cudaDeviceDisablePeerAccess(int peerDeviceId) {
    aclError ret = aclrtDeviceDisablePeerAccess(peerDeviceId);
    return ret;
}

#ifndef CUPY_INSTALL_USE_ASCEND
cudaError_t cudaDeviceGetLimit(size_t* pValue, cudaLimit limit) {
    // WARNING: No direct equivalent concept of "limits" in AscendCL.
    return ACL_ERROR_INVALID_PARAM;
}

cudaError_t cudaDeviceSetLimit(cudaLimit limit, size_t value) {
    // WARNING: No direct equivalent concept of "limits" in AscendCL.
    return ACL_ERROR_INVALID_PARAM;
}
#endif

#ifndef CUPY_INSTALL_USE_ASCEND
    // IPC operations
    cudaError_t cudaIpcCloseMemHandle(void* devPtr) {
        // WARNING: Inter-Process Communication (IPC) handles likely not directly portable to AscendCL. 
        // AscendCL may have different mechanisms for resource sharing between processes.
        return ACL_ERROR_FEATURE_UNSUPPORTED;
    }

    cudaError_t cudaIpcGetEventHandle(cudaIpcEventHandle_t* handle, cudaEvent_t event) {
        // WARNING: Missing direct equivalent in AscendCL for event IPC.
        return ACL_ERROR_FEATURE_UNSUPPORTED;
    }

    cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t* handle, void* devPtr) {
        // WARNING: Missing direct equivalent in AscendCL for memory IPC.
        return ACL_ERROR_FEATURE_UNSUPPORTED;
    }

    cudaError_t cudaIpcOpenEventHandle(cudaEvent_t* event, cudaIpcEventHandle_t handle) {
        // WARNING: Missing direct equivalent in AscendCL for opening event IPC handle.
        return ACL_ERROR_FEATURE_UNSUPPORTED;
    }

    cudaError_t cudaIpcOpenMemHandle(void** devPtr, cudaIpcMemHandle_t handle, unsigned int flags) {
        // WARNING: Missing direct equivalent in AscendCL for opening memory IPC handle.
        return ACL_ERROR_FEATURE_UNSUPPORTED;
    }
#endif

// Memory management
// Stub enums and structs remain, but their usage might differ in AscendCL.
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
    aclError ret = aclrtMalloc(ptr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    return ret;
}

cudaError_t cudaMallocAsync(...) {
    // WARNING: Missing direct equivalent in AscendCL for asynchronous allocation.
    throw std::runtime_error("cudaMallocAsync() is not supported on ASCEND");
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

cudaError_t cudaHostAlloc(void** ptr, size_t size, unsigned int flags) {
    // AscendCL: Use aclrtMallocHost for pinned host memory.
    aclError ret = aclrtMallocHost(ptr, size);
    return ret;
}

cudaError_t cudaHostRegister(...) {
    // WARNING: Missing direct equivalent in AscendCL for host memory registration.
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

cudaError_t cudaHostUnregister(...) {
    // WARNING: Missing direct equivalent in AscendCL for host memory unregistration.
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

cudaError_t cudaMallocManaged(void** ptr, size_t size, unsigned int flags) {
    // WARNING: Unified Memory Management (UVM) might be handled differently in AscendCL.
    // Check AscendCL documentation for similar concepts (e.g., aclrtMalloc with specific flags).
    throw std::runtime_error("cudaMallocManaged() is not supported on ASCEND");
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

int cudaFree(void* ptr) {
    aclError ret = aclrtFree(ptr); 
    return ret;
}

cudaError_t cudaFreeHost(void* ptr) {
    aclError ret = aclrtFreeHost(ptr); 
    return ret;
}

cudaError_t cudaFreeAsync(...) {
    // WARNING: Missing direct equivalent in AscendCL for asynchronous free.
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

int cudaMemGetInfo(size_t* free, size_t* total) {
    // TODO: Edge devices such as 310P has no HBM, but DDR
    size_t free_ddr = 0;
    size_t total_ddr = 0;
    aclError ret = aclrtGetMemInfo(ACL_DDR_MEM, &free_ddr, &total_ddr);

    size_t free_hbm = 0;
    size_t total_hbm = 0;
    ret = aclrtGetMemInfo(ACL_HBM_MEM, &free_hbm, &total_hbm);
    *free = std::max(free_ddr, free_hbm);
    *total = std::max(total_ddr, total_hbm);
    return ret;
}

inline aclrtMemcpyKind _convertMemcpyKind(cudaMemcpyKind kind){
    aclrtMemcpyKind aclKind;
    switch (kind) {
        case cudaMemcpyHostToHost:   aclKind = ACL_MEMCPY_HOST_TO_HOST; break;
        case cudaMemcpyHostToDevice: aclKind = ACL_MEMCPY_HOST_TO_DEVICE; break;
        case cudaMemcpyDeviceToHost: aclKind = ACL_MEMCPY_DEVICE_TO_HOST; break;
        case cudaMemcpyDeviceToDevice: aclKind = ACL_MEMCPY_DEVICE_TO_DEVICE; break;
        case cudaMemcpyDefault: aclKind = ACL_MEMCPY_DEFAULT; break;
        default: return ACL_MEMCPY_DEFAULT;  // ACL_ERROR_INVALID_PARAM;
    }
    return aclKind;
}

cudaError_t cudaMemcpy(void* dst, const void* src, size_t sizeBytes,
                       cudaMemcpyKind kind) {
    aclrtMemcpyKind aclKind = _convertMemcpyKind(kind);
    aclError ret = aclrtMemcpy(dst, sizeBytes, src, sizeBytes, aclKind); 
    return ret;
}

cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t sizeBytes,
                            cudaMemcpyKind kind, cudaStream_t stream) {
    aclrtMemcpyKind aclKind = _convertMemcpyKind(kind);
    aclError ret = aclrtMemcpyAsync(dst, sizeBytes, src, sizeBytes, aclKind, stream); 
    return ret;
}

cudaError_t cudaMemcpy2D(void *dst, size_t dpitch,
    const void *src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind) {
    aclrtMemcpyKind aclKind = _convertMemcpyKind(kind);
    aclError ret = aclrtMemcpy2d(dst, dpitch, src, spitch, width, height, aclKind); 
    return ret;
}

cudaError_t cudaMemcpy2DAsync(void *dst, size_t dpitch,
    const void *src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind,
    cudaStream_t stream) {
    aclrtMemcpyKind aclKind = _convertMemcpyKind(kind);
    aclError ret = aclrtMemcpy2dAsync(dst, dpitch, src, spitch, width, height, aclKind, stream); 
    return ret;
}

cudaError_t cudaMemcpyPeer(void* dst, int dstDeviceId, const void* src,
                           int srcDeviceId, size_t sizeBytes) {
    // TODO: Peer-to-peer memory copy might be handled differently in AscendCL.
    // It might require explicit enablement or use different functions.
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

cudaError_t cudaMemcpyPeerAsync(void* dst, int dstDevice, const void* src,
                                int srcDevice, size_t sizeBytes,
                                cudaStream_t stream) {
    // TODO: Peer-to-peer memory copy might be handled differently in AscendCL.
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

#ifndef CUPY_INSTALL_USE_ASCEND
cudaError_t cudaMalloc3DArray(...) {
    // WARNING: Missing direct equivalent in AscendCL for 3D array allocation.
    // AscendCL uses aclDataBuffer and aclTensorDesc for data management.
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

cudaError_t cudaMallocArray(...) {
    // WARNING: Missing direct equivalent in AscendCL for array allocation.
    // AscendCL uses aclDataBuffer and aclTensorDesc for data management.
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

cudaError_t cudaFreeArray(...) {
    // WARNING: Missing direct equivalent in AscendCL for freeing arrays.
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

cudaError_t cudaMemcpy2DFromArray(...) {
    // WARNING: Missing direct equivalent in AscendCL.
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

cudaError_t cudaMemcpy2DFromArrayAsync(...) {
    // WARNING: Missing direct equivalent in AscendCL.
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

cudaError_t cudaMemcpy2DToArray(...) {
    // WARNING: Missing direct equivalent in AscendCL.
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

cudaError_t cudaMemcpy2DToArrayAsync(...) {
    // WARNING: Missing direct equivalent in AscendCL.
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

cudaError_t cudaMemcpy3D(...) {
    // WARNING: Missing direct equivalent in AscendCL for 3D memory copy.
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

cudaError_t cudaMemcpy3DAsync(...) {
    // WARNING: Missing direct equivalent in AscendCL for async 3D memory copy.
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}
#endif

cudaError_t cudaMemset(void* dst, int value, size_t sizeBytes) {
    aclError ret = aclrtMemset(dst, sizeBytes, value, sizeBytes); 
    return ret;
}

cudaError_t cudaMemsetAsync(void* dst, int value, size_t sizeBytes,
                            cudaStream_t stream) {
    aclError ret = aclrtMemsetAsync(dst, sizeBytes, value, sizeBytes, stream); 
    return ret;
}

cudaError_t cudaMemAdvise(const void *devPtr, size_t count,
                          cudaMemoryAdvise advice, int device) {
    // WARNING: Missing direct equivalent in AscendCL for memory advice.
    throw std::runtime_error("cudaMemAdvise() is not supported on ASCEND");
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

cudaError_t cudaMemPrefetchAsync(const void *devPtr, size_t count,
				 int dstDevice, cudaStream_t stream) {
    // WARNING: Missing direct equivalent in AscendCL for async memory prefetch.
    
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

cudaError_t cudaPointerGetAttributes(cudaPointerAttributes *attributes,
                                     const void* ptr) {
    // WARNING: AscendCL's pointer attribute query might differ.
    // Use aclrtGetPointerInfo or similar functions if available. 
    // This is a complex function to map, requires detailed understanding of AscendCL memory model.
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

#ifndef CUPY_INSTALL_USE_ASCEND
// ====================== MemPool is supported by ascend ===================
// ====================== MemPool is also not supported on ROCm ===================
cudaError_t cudaMallocFromPoolAsync(...) {
    // WARNING: Missing direct equivalent in AscendCL for memory pools.
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

cudaError_t cudaMemPoolCreate(...) {
    // WARNING: Missing direct equivalent in AscendCL for memory pool creation.
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

cudaError_t cudaMemPoolDestroy(...) {
    // WARNING: Missing direct equivalent in AscendCL for memory pool destruction.
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

cudaError_t cudaDeviceGetDefaultMemPool(...) {
    // WARNING: Missing direct equivalent in AscendCL.
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

cudaError_t cudaDeviceGetMemPool(...) {
    // WARNING: Missing direct equivalent in AscendCL.
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

cudaError_t cudaDeviceSetMemPool(...) {
    // WARNING: Missing direct equivalent in AscendCL.
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

cudaError_t cudaMemPoolTrimTo(...) {
    // WARNING: Missing direct equivalent in AscendCL.
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

cudaError_t cudaMemPoolGetAttribute(...) {
    // WARNING: Missing direct equivalent in AscendCL.
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

cudaError_t cudaMemPoolSetAttribute(...) {
    // WARNING: Missing direct equivalent in AscendCL.
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}
#endif


// Stream and Event
// AscendCL has its own stream and event types (aclrtStream, aclrtEvent).

cudaError_t cudaStreamCreate(cudaStream_t stream) {
    aclError ret = aclrtCreateStream(&stream); 
    return ret;
}

cudaError_t cudaStreamCreateWithFlags(cudaStream_t stream,
                                      unsigned int flags) {
    // TODO: AscendCL's aclrtCreateStream might not use the same flags as HIP.
    aclError ret = aclrtCreateStreamWithConfig(&stream, ASCEND_DEFAULT_STREAM_PRIORITY, flags); // Check exact API
    return ret;
}

cudaError_t cudaStreamCreateWithPriority(cudaStream_t stream,
                                         unsigned int flags,
                                         int priority) {
    // WARNING: AscendCL might have different priority mechanisms.
    aclError ret = aclrtCreateStreamWithConfig(&stream, priority, flags); // Check exact API
    return ret;
}

cudaError_t cudaStreamGetFlags(cudaStream_t stream, unsigned int *flags) {
    // WARNING: Missing direct equivalent in AscendCL for getting stream flags.
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

cudaError_t cudaStreamGetPriority(cudaStream_t stream, int *priority) {
    // WARNING: Missing direct equivalent in AscendCL for getting stream priority.
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

cudaError_t cudaStreamDestroy(cudaStream_t stream) {
    aclError ret = aclrtDestroyStream(stream); 
    return ret;
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    aclError ret = aclrtSynchronizeStream(stream); 
    return ret;
}

cudaError_t cudaStreamAddCallback(cudaStream_t stream,
                                  cudaStreamCallback_t callback,
                                  void *userData, unsigned int flags) {
    // WARNING: Missing direct equivalent in AscendCL for stream callbacks.
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

cudaError_t cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void* userData) {
    // TODO:  aclError aclrtSubscribeHostFunc(uint64_t hostFuncThreadId, aclrtStream exeStream);
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

// Returns cudaSuccess if all operations in stream have completed, or cudaErrorNotReady if not.
cudaError_t cudaStreamQuery(cudaStream_t stream) {
    aclrtStreamStatus status;
    aclError ret = aclrtStreamQuery(stream, &status); // TODO
    return ret;
}

cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event,
                                unsigned int flags) {
    aclError ret = aclrtStreamWaitEvent(stream, event);
    return ret;
}

cudaError_t cudaEventCreate(cudaEvent_t event) {
    aclError ret = aclrtCreateEvent(&event);
    return ret;
}

cudaError_t cudaEventCreateWithFlags(cudaEvent_t event, unsigned flags) {
    // WARNING: AscendCL's aclrtCreateEvent might not use the same flags. 
    aclError ret = aclrtCreateEventWithFlag(&event, flags); // Check exact API
    return ret;
}

cudaError_t cudaEventDestroy(cudaEvent_t event) {
    aclError ret = aclrtDestroyEvent(event); 
    return ret;
}

cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start,
                                 cudaEvent_t stop){
    aclError ret = aclrtEventElapsedTime(ms, start, stop); 
    return ret;
}

// TODO: convert the cudaEventStatus?
cudaError_t cudaEventQuery(cudaEvent_t event) {
    aclrtEventRecordedStatus status;
    aclError ret = aclrtQueryEventStatus(event, &status); // extra arg: status
    return ret;
}

cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    aclError ret = aclrtRecordEvent(event, stream); 
    return ret;
}

cudaError_t cudaEventSynchronize(cudaEvent_t event) {
    aclError ret = aclrtSynchronizeEvent(event); 
    return ret;
}

typedef aclmdlRICaptureMode cudaStreamCaptureMode;
typedef aclmdlRICaptureStatus cudaStreamCaptureStatus;
typedef aclmdlRI cudaGraph_t;

cudaError_t cudaStreamBeginCapture(cudaStream_t stream,
                                   cudaStreamCaptureMode mode) {
    // WARNING: Missing direct equivalent in AscendCL for stream capture.
    return aclmdlRICaptureBegin(stream, mode);
}

cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t* pGraph) {
    // WARNING: Missing direct equivalent in AscendCL for stream capture.
    return aclmdlRICaptureEnd(stream, pGraph);
}

cudaError_t cudaStreamIsCapturing(cudaStream_t stream,
                                  cudaStreamCaptureStatus* pCaptureStatus) {
    // TODO:  aclmdlRICaptureGetInfo
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

// =================================================
// Ascend has graph, but not sure if they are related with cuGraph
cudaError_t cudaGraphInstantiate(
    cudaGraphExec_t* pGraphExec,
	cudaGraph_t graph,
	cudaGraphNode_t* pErrorNode,
	char* pLogBuffer,
	size_t bufferSize) {
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

cudaError_t cudaGraphDestroy(cudaGraph_t graph) {
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

cudaError_t cudaGraphExecDestroy(cudaGraphExec_t graphExec) {
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream) {
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

cudaError_t cudaGraphUpload(...) {
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

cudaError_t cudaGraphDebugDotPrint(cudaGraph_t graph, const char* path, unsigned int flags) {
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

// ================ GPU render API is not supported on ASCEND NPU============================
#ifndef CUPY_INSTALL_USE_ASCEND
// Texture is not supported on NPU
cudaError_t cudaCreateTextureObject(...) {
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

cudaError_t cudaDestroyTextureObject(...) {
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

cudaError_t cudaGetChannelDesc(...) {
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

cudaError_t cudaGetTextureObjectResourceDesc(...) {
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

cudaError_t cudaGetTextureObjectTextureDesc(...) {
    return ACL_ERROR_FEATURE_UNSUPPORTED;
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
    // return hipCreateSurfaceObject(pSurfObject, pResDesc);
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

cudaError_t cudaDestroySurfaceObject(cudaSurfaceObject_t surfObject) {
    // return hipDestroySurfaceObject(surfObject);
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}
#endif

} // extern "C"

#endif // #ifndef INCLUDE_GUARD_ASCEND_CUPY_RUNTIME_H
