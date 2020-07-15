// This file is a stub header file of cuda for Read the Docs.

#ifndef INCLUDE_GUARD_CUPY_CUDA_H
#define INCLUDE_GUARD_CUPY_CUDA_H

#if CUPY_USE_HIP

#include "cupy_hip.h"

#elif !defined(CUPY_NO_CUDA)

#include <cuda.h>
#include <cuda_runtime.h>

extern "C" {

#if CUDA_VERSION < 9000

CUresult cuFuncSetAttribute(...) {
    return CUDA_ERROR_NOT_SUPPORTED;
}

typedef enum {} cublasMath_t;

cublasStatus_t cublasSetMathMode(...) {
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasGetMathMode(...) {
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

CUresult cuLaunchCooperativeKernel(...) {
    return CUDA_ERROR_NOT_SUPPORTED;
}

#endif // #if CUDA_VERSION < 9000

} // extern "C"

#else // #ifndef CUPY_NO_CUDA

#include "../cupy_cuda_common.h"

extern "C" {

// Error handling
CUresult cuGetErrorName(...) {
    return CUDA_SUCCESS;
}

CUresult cuGetErrorString(...) {
    return CUDA_SUCCESS;
}

// Primary context management
CUresult cuDevicePrimaryCtxRelease(...) {
    return CUDA_SUCCESS;
}

// Context management
CUresult cuCtxGetCurrent(...) {
    return CUDA_SUCCESS;
}

CUresult cuCtxSetCurrent(...) {
    return CUDA_SUCCESS;
}

CUresult cuCtxCreate(...) {
    return CUDA_SUCCESS;
}

CUresult cuCtxDestroy(...) {
    return CUDA_SUCCESS;
}


// Module load and kernel execution
CUresult cuLinkCreate (...) {
    return CUDA_SUCCESS;
}

CUresult cuLinkAddData(...) {
    return CUDA_SUCCESS;
}

CUresult cuLinkAddFile(...) {
    return CUDA_SUCCESS;
}

CUresult cuLinkComplete(...) {
    return CUDA_SUCCESS;
}

CUresult cuLinkDestroy(...) {
    return CUDA_SUCCESS;
}

CUresult cuModuleLoad(...) {
    return CUDA_SUCCESS;
}

CUresult cuModuleLoadData(...) {
    return CUDA_SUCCESS;
}

CUresult cuModuleUnload(...) {
    return CUDA_SUCCESS;
}

CUresult cuModuleGetFunction(...) {
    return CUDA_SUCCESS;
}

CUresult cuModuleGetGlobal(...) {
    return CUDA_SUCCESS;
}

CUresult cuModuleGetTexRef(...) {
    return CUDA_SUCCESS;
}

CUresult cuLaunchKernel(...) {
    return CUDA_SUCCESS;
}

CUresult cuLaunchCooperativeKernel(...) {
    return CUDA_SUCCESS;
}

// Function attribute
CUresult cuFuncGetAttribute(...) {
    return CUDA_SUCCESS;
}

CUresult cuFuncSetAttribute(...) {
    return CUDA_SUCCESS;
}

// Texture reference
CUresult cuTexRefSetAddress (...) {
    return CUDA_SUCCESS;
}

CUresult cuTexRefSetAddress2D (...) {
    return CUDA_SUCCESS;
}

CUresult cuTexRefSetAddressMode (...) {
    return CUDA_SUCCESS;
}

CUresult cuTexRefSetArray (...) {
    return CUDA_SUCCESS;
}

CUresult cuTexRefSetBorderColor (...) {
    return CUDA_SUCCESS;
}

CUresult cuTexRefSetFilterMode (...) {
    return CUDA_SUCCESS;
}

CUresult cuTexRefSetFlags (...) {
    return CUDA_SUCCESS;
}

CUresult cuTexRefSetFormat (...) {
    return CUDA_SUCCESS;
}

CUresult cuTexRefSetMaxAnisotropy (...) {
    return CUDA_SUCCESS;
}

// Occupancy
typedef size_t (*CUoccupancyB2DSize)(int);

CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(...) {
    return CUDA_SUCCESS;
}

CUresult cuOccupancyMaxPotentialBlockSize(...) {
    return CUDA_SUCCESS;
}

} // extern "C"

#endif // #ifndef CUPY_NO_CUDA
#endif // #ifndef INCLUDE_GUARD_CUPY_CUDA_H
