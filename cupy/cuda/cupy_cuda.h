// This file is a stub header file of cuda for Read the Docs.

#ifndef INCLUDE_GUARD_CUPY_CUDA_H
#define INCLUDE_GUARD_CUPY_CUDA_H

#include <stdint.h>

#if CUPY_USE_HIP

#include "cupy_hip.h"
#include "cupy_cuComplex.h"

#elif !defined(CUPY_NO_CUDA)

#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <curand.h>
#ifndef CUPY_NO_NVTX
#include <nvToolsExt.h>
#endif // #ifndef CUPY_NO_NVTX

extern "C" {

bool hip_environment = false;

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

#include "cupy_cuda_common.h"
#include "cupy_cuComplex.h"

extern "C" {

bool hip_environment = false;

///////////////////////////////////////////////////////////////////////////////
// cuda.h
///////////////////////////////////////////////////////////////////////////////

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

///////////////////////////////////////////////////////////////////////////////
// cuda_runtime.h
///////////////////////////////////////////////////////////////////////////////

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

cudaError_t cudaDeviceCanAccessPeer(...) {
    return cudaSuccess;
}

cudaError_t cudaDeviceEnablePeerAccess(...) {
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

///////////////////////////////////////////////////////////////////////////////
// cuComplex.h
///////////////////////////////////////////////////////////////////////////////

#include "cupy_cuComplex.h"

///////////////////////////////////////////////////////////////////////////////
// cublas_v2.h
///////////////////////////////////////////////////////////////////////////////

// Context
cublasStatus_t cublasCreate(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDestroy(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetVersion(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetPointerMode(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetPointerMode(...) {
    return CUBLAS_STATUS_SUCCESS;
}

// Stream
cublasStatus_t cublasSetStream(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetStream(...) {
    return CUBLAS_STATUS_SUCCESS;
}

// Math Mode
cublasStatus_t cublasSetMathMode(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetMathMode(...) {
    return CUBLAS_STATUS_SUCCESS;
}

// BLAS Level 1
cublasStatus_t cublasIsamax(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasIsamin(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSasum(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSaxpy(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDaxpy(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSdot(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDdot(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCdotu(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCdotc(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZdotc(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZdotu(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSnrm2(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSscal(...) {
    return CUBLAS_STATUS_SUCCESS;
}


// BLAS Level 2
cublasStatus_t cublasSgemv(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDgemv(...) {
    return CUBLAS_STATUS_SUCCESS;
}


cublasStatus_t cublasCgemv(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZgemv(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSger(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDger(...) {
    return CUBLAS_STATUS_SUCCESS;
}

int cublasCgeru(...) {
    return 0;
}

int cublasCgerc(...) {
    return 0;
}

int cublasZgeru(...) {
    return 0;
}

int cublasZgerc(...) {
    return 0;
}

// BLAS Level 3
cublasStatus_t cublasSgemm(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDgemm(...) {
    return CUBLAS_STATUS_SUCCESS;
}


cublasStatus_t cublasCgemm(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZgemm(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSgemmBatched(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDgemmBatched(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCgemmBatched(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZgemmBatched(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSgemmStridedBatched(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDgemmStridedBatched(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCgemmStridedBatched(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZgemmStridedBatched(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSgemmEx(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGemmEx(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasStrsm(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDtrsm(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCtrsm(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZtrsm(...) {
    return CUBLAS_STATUS_SUCCESS;
}


// BLAS extension
cublasStatus_t cublasSgeam(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDgeam(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSdgmm(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSgetrfBatched(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDgetrfBatched(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCgetrfBatched(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZgetrfBatched(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSgetriBatched(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDgetriBatched(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCgetriBatched(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZgetriBatched(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasStrttp(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDtrttp(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasStpttr(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDtpttr(...) {
    return CUBLAS_STATUS_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
// curand.h
///////////////////////////////////////////////////////////////////////////////

typedef enum {} curandOrdering_t;
typedef enum {} curandRngType_t;
typedef enum {
    CURAND_STATUS_SUCCESS = 0,
} curandStatus_t;

typedef void* curandGenerator_t;


// curandGenerator_t
curandStatus_t curandCreateGenerator(...) {
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandDestroyGenerator(...) {
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGetVersion(...) {
    return CURAND_STATUS_SUCCESS;
}


// Stream
curandStatus_t curandSetStream(...) {
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandSetPseudoRandomGeneratorSeed(...) {
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandSetGeneratorOffset(...) {
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandSetGeneratorOrdering(...) {
    return CURAND_STATUS_SUCCESS;
}


// Generation functions
curandStatus_t curandGenerate(...) {
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGenerateLongLong(...) {
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGenerateUniform(...) {
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGenerateUniformDouble(...) {
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGenerateNormal(...) {
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGenerateNormalDouble(...) {
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGenerateLogNormal(...) {
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGenerateLogNormalDouble(...) {
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGeneratePoisson(...) {
    return CURAND_STATUS_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
// cuda_profiler_api.h
///////////////////////////////////////////////////////////////////////////////

typedef enum {} cudaOutputMode_t;

cudaError_t cudaProfilerInitialize(...) {
  return cudaSuccess;
}

cudaError_t cudaProfilerStart() {
  return cudaSuccess;
}

cudaError_t cudaProfilerStop() {
  return cudaSuccess;
}

///////////////////////////////////////////////////////////////////////////////
// nvToolsExt.h
///////////////////////////////////////////////////////////////////////////////

#define NVTX_VERSION 1

typedef enum nvtxColorType_t
{
    NVTX_COLOR_UNKNOWN  = 0,
    NVTX_COLOR_ARGB     = 1
} nvtxColorType_t;

typedef enum nvtxMessageType_t
{
    NVTX_MESSAGE_UNKNOWN          = 0,
    NVTX_MESSAGE_TYPE_ASCII       = 1,
    NVTX_MESSAGE_TYPE_UNICODE     = 2,
} nvtxMessageType_t;

typedef union nvtxMessageValue_t
{
    const char* ascii;
    const wchar_t* unicode;
} nvtxMessageValue_t;

typedef struct nvtxEventAttributes_v1
{
    uint16_t version;
    uint16_t size;
    uint32_t category;
    int32_t colorType;
    uint32_t color;
    int32_t payloadType;
    int32_t reserved0;
    union payload_t
    {
        uint64_t ullValue;
        int64_t llValue;
        double dValue;
    } payload;
    int32_t messageType;
    nvtxMessageValue_t message;
} nvtxEventAttributes_v1;

typedef nvtxEventAttributes_v1 nvtxEventAttributes_t;

void nvtxMarkA(...) {
}

void nvtxMarkEx(...) {
}

int nvtxRangePushA(...) {
    return 0;
}

int nvtxRangePushEx(...) {
    return 0;
}

int nvtxRangePop() {
    return 0;
}

uint64_t nvtxRangeStartEx(...) {
    return 0;
}

void nvtxRangeEnd(...) {
}

} // extern "C"

#endif // #ifndef CUPY_NO_CUDA
#endif // #ifndef INCLUDE_GUARD_CUPY_CUDA_H
