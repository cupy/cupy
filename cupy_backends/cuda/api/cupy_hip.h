// This file is a stub header file of hip for Read the Docs.

#ifndef INCLUDE_GUARD_CUPY_HIP_H
#define INCLUDE_GUARD_CUPY_HIP_H

#include <hip/hip_runtime_api.h>
#include "../cupy_hip_common.h"

extern "C" {

// Error handling
CUresult cuGetErrorName(CUresult hipError, const char** pStr) {
    *pStr = hipGetErrorName(hipError);
    return CUDA_SUCCESS;
}

CUresult cuGetErrorString(CUresult hipError, const char** pStr) {
    *pStr = hipGetErrorString(hipError);
    return CUDA_SUCCESS;
}

// Primary context management
CUresult cuDevicePrimaryCtxRelease(CUdevice dev) {
    return hipDevicePrimaryCtxRelease(dev);
}

// Context management
CUresult cuCtxGetCurrent(CUcontext *ctx) {
    // deprecated api
    //return hipCtxGetCurrent(ctx);
    return hipErrorUnknown;
}

CUresult cuCtxSetCurrent(CUcontext ctx) {
    // deprecated api
    //return hipCtxSetCurrent(ctx);
    return hipErrorUnknown;
}

CUresult cuCtxCreate(CUcontext* pctx, unsigned int flags, CUdevice dev) {
    // deprecated api
    //return hipCtxCreate(pctx, flags, dev);
    return hipErrorUnknown;
}

CUresult cuCtxDestroy(CUcontext ctx) {
    // deprecated api
    // return hipCtxDestroy(ctx);
    return hipErrorUnknown;
}


// Module load and kernel execution
CUresult cuLinkCreate(...) {
    return hipErrorUnknown;
}

CUresult cuLinkAddData(...) {
    return hipErrorUnknown;
}

CUresult cuLinkAddFile(...) {
    return hipErrorUnknown;
}

CUresult cuLinkComplete(...) {
    return hipErrorUnknown;
}

CUresult cuLinkDestroy(...) {
    return hipErrorUnknown;
}

CUresult cuModuleLoad(CUmodule *module, const char *fname) {
    return hipModuleLoad(module, fname);
}

CUresult cuModuleLoadData(CUmodule *module, const void *image) {
    return hipModuleLoadData(module, image);
}

CUresult cuModuleUnload(CUmodule module) {
    return hipModuleUnload(module);
}

CUresult cuModuleGetFunction(CUfunction *function, CUmodule module,
                             const char *kname) {
    return hipModuleGetFunction(function, module, kname);
}

CUresult cuModuleGetGlobal(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod,
                           const char *name) {
    return hipModuleGetGlobal(dptr, bytes, hmod, name);
}

CUresult cuModuleGetTexRef(...) {
    return hipErrorUnknown;
}

CUresult cuLaunchKernel(CUfunction f, uint32_t gridDimX, uint32_t gridDimY,
                        uint32_t gridDimZ, uint32_t blockDimX,
                        uint32_t blockDimY, uint32_t blockDimZ,
                        uint32_t sharedMemBytes, cudaStream_t hStream,
                        void **kernelParams, void **extra) {
    return hipModuleLaunchKernel(f, gridDimX, gridDimY, gridDimZ,
                                 blockDimX, blockDimY, blockDimZ,
                                 sharedMemBytes, hStream, kernelParams, extra);
}

CUresult cuLaunchCooperativeKernel(...) {
    return hipErrorUnknown;
}


// Function attribute
CUresult cuFuncGetAttribute(...) {
    return hipErrorUnknown;
}

CUresult cuFuncSetAttribute(...) {
    return hipErrorUnknown;
}


// Texture reference
CUresult cuTexRefSetAddress (...) {
    return hipErrorUnknown;
}

CUresult cuTexRefSetAddress2D (...) {
    return hipErrorUnknown;
}

CUresult cuTexRefSetAddressMode (...) {
    return hipErrorUnknown;
}

CUresult cuTexRefSetArray (...) {
    return hipErrorUnknown;
}

CUresult cuTexRefSetBorderColor (...) {
    return hipErrorUnknown;
}

CUresult cuTexRefSetFilterMode (...) {
    return hipErrorUnknown;
}

CUresult cuTexRefSetFlags (...) {
    return hipErrorUnknown;
}

CUresult cuTexRefSetFormat (...) {
    return hipErrorUnknown;
}

CUresult cuTexRefSetMaxAnisotropy (...) {
    return hipErrorUnknown;
}

// Occupancy
typedef size_t (*CUoccupancyB2DSize)(int);

CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(...) {
    return hipErrorUnknown;
}

CUresult cuOccupancyMaxPotentialBlockSize(...) {
    return hipErrorUnknown;
}

} // extern "C"

#endif // #ifndef INCLUDE_GUARD_CUPY_HIP_H
