#ifndef INCLUDE_GUARD_HIP_CUPY_CUDA_H
#define INCLUDE_GUARD_HIP_CUPY_CUDA_H

#include "cupy_hip_common.h"
#include "cupy_cuComplex.h"

extern "C" {

// Error handling
CUPY_HIP_EXPORT CUresult cuGetErrorName(CUresult hipError, const char** pStr) {
    *pStr = hipGetErrorName(hipError);
    return CUDA_SUCCESS;
}

CUPY_HIP_EXPORT CUresult cuGetErrorString(CUresult hipError, const char** pStr) {
    *pStr = hipGetErrorString(hipError);
    return CUDA_SUCCESS;
}

// Primary context management
CUPY_HIP_EXPORT CUresult cuDevicePrimaryCtxRelease(CUdevice dev) {
    // HIP documents this as a no-op that always returns hipSuccess;
    // marked [[deprecated]] in recent ROCm. Replicate inline.
    (void)dev;
    return CUDA_SUCCESS;
}

// Context management
CUPY_HIP_EXPORT CUresult cuCtxGetCurrent(CUcontext *ctx) {
    // deprecated api
    //return hipCtxGetCurrent(ctx);
    return hipErrorUnknown;
}

CUPY_HIP_EXPORT CUresult cuCtxSetCurrent(CUcontext ctx) {
    // deprecated api
    //return hipCtxSetCurrent(ctx);
    return hipErrorUnknown;
}

CUPY_HIP_EXPORT CUresult cuCtxCreate(CUcontext* pctx, unsigned int flags, CUdevice dev) {
    // deprecated api
    //return hipCtxCreate(pctx, flags, dev);
    return hipErrorUnknown;
}

CUPY_HIP_EXPORT CUresult cuCtxDestroy(CUcontext ctx) {
    // deprecated api
    // return hipCtxDestroy(ctx);
    return hipErrorUnknown;
}

CUPY_HIP_EXPORT CUresult cuCtxGetDevice(...) {
    return hipErrorUnknown;
}

// Module load and kernel execution
CUPY_HIP_EXPORT CUresult cuLinkCreate(...) {
    return hipErrorUnknown;
}

CUPY_HIP_EXPORT CUresult cuLinkAddData(...) {
    return hipErrorUnknown;
}

CUPY_HIP_EXPORT CUresult cuLinkAddFile(...) {
    return hipErrorUnknown;
}

CUPY_HIP_EXPORT CUresult cuLinkComplete(...) {
    return hipErrorUnknown;
}

CUPY_HIP_EXPORT CUresult cuLinkDestroy(...) {
    return hipErrorUnknown;
}

CUPY_HIP_EXPORT CUresult cuModuleLoad(CUmodule *module, const char *fname) {
    return hipModuleLoad(module, fname);
}

CUPY_HIP_EXPORT CUresult cuModuleLoadData(CUmodule *module, const void *image) {
    return hipModuleLoadData(module, image);
}

CUPY_HIP_EXPORT CUresult cuModuleUnload(CUmodule module) {
    return hipModuleUnload(module);
}

CUPY_HIP_EXPORT CUresult cuModuleGetFunction(CUfunction *function, CUmodule module,
                             const char *kname) {
    return hipModuleGetFunction(function, module, kname);
}

CUPY_HIP_EXPORT CUresult cuModuleGetGlobal(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod,
                           const char *name) {
    return hipModuleGetGlobal(dptr, bytes, hmod, name);
}

CUPY_HIP_EXPORT CUresult cuLaunchKernel(CUfunction f, uint32_t gridDimX, uint32_t gridDimY,
                        uint32_t gridDimZ, uint32_t blockDimX,
                        uint32_t blockDimY, uint32_t blockDimZ,
                        uint32_t sharedMemBytes, cudaStream_t hStream,
                        void **kernelParams, void **extra) {
    return hipModuleLaunchKernel(f, gridDimX, gridDimY, gridDimZ,
                                 blockDimX, blockDimY, blockDimZ,
                                 sharedMemBytes, hStream, kernelParams, extra);
}

CUPY_HIP_EXPORT CUresult cuLaunchCooperativeKernel(...) {
    return hipErrorUnknown;
}

// Function attribute
CUPY_HIP_EXPORT CUresult cuFuncGetAttribute(int* pi, CUfunction_attribute attrib, CUfunction hfunc) {
    return hipFuncGetAttribute(pi, attrib, hfunc);
}

CUPY_HIP_EXPORT CUresult cuFuncSetAttribute(...) {
    return hipErrorUnknown;
}

// Occupancy
typedef size_t (*CUoccupancyB2DSize)(int);

CUPY_HIP_EXPORT CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(...) {
    return hipErrorUnknown;
}

CUPY_HIP_EXPORT CUresult cuOccupancyMaxPotentialBlockSize(...) {
    return hipErrorUnknown;
}

// Stream
CUPY_HIP_EXPORT CUresult cuStreamGetCtx(...) {
    return hipErrorUnknown;
}

} // extern "C"

#endif // #ifndef INCLUDE_GUARD_HIP_CUPY_CUDA_H
