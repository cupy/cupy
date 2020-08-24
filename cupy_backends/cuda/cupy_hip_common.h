// This file is a stub header file of hip for Read the Docs.

#ifndef INCLUDE_GUARD_CUPY_HIP_COMMON_H
#define INCLUDE_GUARD_CUPY_HIP_COMMON_H

#include <hip/hip_runtime_api.h>
#include <hipblas.h>

#define CUDA_VERSION 0

extern "C" {

///////////////////////////////////////////////////////////////////////////////
// cuda.h
///////////////////////////////////////////////////////////////////////////////

typedef int CUdevice;
typedef hipError_t CUresult;
const CUresult CUDA_SUCCESS = static_cast<CUresult>(0);
enum CUjit_option {};
enum CUjitInputType {};
enum CUfunction_attribute {};
enum CUarray_format {};
enum CUaddress_mode {};
enum CUfilter_mode {};


typedef hipDeviceptr_t CUdeviceptr;
struct CUlinkState_st;


typedef hipCtx_t CUcontext;
typedef hipEvent_t cudaEvent_t;
typedef hipFunction_t CUfunction;
typedef hipModule_t CUmodule;
typedef hipStream_t cudaStream_t;
typedef struct CUlinkState_st* CUlinkState;
typedef struct CUtexref_st* CUtexref;
typedef struct CUarray_st* CUarray;
struct CUDA_ARRAY_DESCRIPTOR {
    CUarray_format Format;
    size_t Height;
    unsigned int NumChannels;
    size_t Width;
};


///////////////////////////////////////////////////////////////////////////////
// cuda_runtime.h
///////////////////////////////////////////////////////////////////////////////

enum {
    cudaDevAttrComputeCapabilityMajor
        = hipDeviceAttributeComputeCapabilityMajor,
    cudaDevAttrComputeCapabilityMinor
        = hipDeviceAttributeComputeCapabilityMinor,
};

typedef hipError_t cudaError_t;
const CUresult cudaSuccess = static_cast<CUresult>(0);
const CUresult cudaErrorInvalidValue = hipErrorInvalidValue;
const CUresult cudaErrorMemoryAllocation = hipErrorMemoryAllocation;
const CUresult cudaErrorPeerAccessAlreadyEnabled = hipErrorPeerAccessAlreadyEnabled;
typedef enum {} cudaDataType;
typedef hipDeviceAttribute_t cudaDeviceAttr;
typedef hipLimit_t cudaLimit;
enum cudaMemoryAdvise {};
typedef hipMemcpyKind cudaMemcpyKind;


typedef hipStreamCallback_t cudaStreamCallback_t;
typedef hipPointerAttribute_t cudaPointerAttributes;

typedef hipChannelFormatKind cudaChannelFormatKind;
typedef hipTextureObject_t cudaTextureObject_t;
typedef hipSurfaceObject_t cudaSurfaceObject_t;
typedef hipResourceType cudaResourceType;
typedef hipTextureAddressMode cudaTextureAddressMode;
typedef hipTextureFilterMode cudaTextureFilterMode;
typedef hipTextureReadMode cudaTextureReadMode;
typedef hipResourceViewDesc cudaResourceViewDesc;
typedef hipArray_t cudaArray_t;
typedef hipExtent cudaExtent;
typedef hipPos cudaPos;
typedef hipPitchedPtr cudaPitchedPtr;
typedef hipMipmappedArray_t cudaMipmappedArray_t;
typedef hipMemcpy3DParms cudaMemcpy3DParms;
typedef hipChannelFormatDesc cudaChannelFormatDesc;
typedef hipResourceDesc cudaResourceDesc;
typedef hipTextureDesc cudaTextureDesc;

// IPC operations
typedef hipIpcMemHandle_st cudaIpcMemHandle_t;
typedef hipIpcEventHandle_st cudaIpcEventHandle_t;


///////////////////////////////////////////////////////////////////////////////
// blas
///////////////////////////////////////////////////////////////////////////////

typedef hipblasHandle_t cublasHandle_t;

typedef hipblasDiagType_t cublasDiagType_t;
typedef hipblasFillMode_t cublasFillMode_t;
typedef hipblasOperation_t cublasOperation_t;
typedef hipblasPointerMode_t cublasPointerMode_t;
typedef hipblasSideMode_t cublasSideMode_t;
typedef enum {} cublasGemmAlgo_t;
typedef enum {} cublasMath_t;
typedef int cudaDataType_t;
typedef hipblasStatus_t cublasStatus_t;

// TODO(leofang): as of ROCm 3.5.0 this does not exist yet
typedef enum {} cublasComputeType_t;


///////////////////////////////////////////////////////////////////////////////
// library_types.h
// (needed for supporting cusolver)
///////////////////////////////////////////////////////////////////////////////

typedef enum libraryPropertyType_t {
    MAJOR_VERSION,
    MINOR_VERSION,
    PATCH_LEVEL
} libraryPropertyType;


///////////////////////////////////////////////////////////////////////////////
// roctx
///////////////////////////////////////////////////////////////////////////////

// this is to make roctxMarkA etc work; ROCm does not yet support the "Ex" APIs
#define NVTX_VERSION (100 * ROCTX_VERSION_MAJOR + 10 * ROCTX_VERSION_MINOR)

// ----- stubs that are no-ops (copied from cupy_backends/cuda/cupy_cuda.h) -----
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

} // extern "C"

#endif // #ifndef INCLUDE_GUARD_CUPY_HIP_COMMON_H
