#ifndef INCLUDE_GUARD_HIP_CUPY_COMMON_H
#define INCLUDE_GUARD_HIP_CUPY_COMMON_H

#include <hip/hip_runtime_api.h>
#include <hipblas.h>
#include <rocsolver.h>

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
enum CUarray_format {};
enum CUaddress_mode {};
enum CUfilter_mode {};


typedef hipDeviceptr_t CUdeviceptr;
struct CUlinkState_st;


typedef hipCtx_t CUcontext;
typedef hipEvent_t cudaEvent_t;
typedef hipFunction_t CUfunction;
typedef hipFunction_attribute CUfunction_attribute;
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
const CUresult cudaErrorInvalidResourceHandle = hipErrorInvalidResourceHandle;
const CUresult cudaErrorContextIsDestroyed = hipErrorUnknown;  // no counterpart in HIP
const CUresult cudaErrorPeerAccessAlreadyEnabled = hipErrorPeerAccessAlreadyEnabled;
typedef enum {} cudaDataType;
typedef hipDeviceAttribute_t cudaDeviceAttr;
typedef hipLimit_t cudaLimit;
enum cudaMemoryAdvise {};
typedef hipMemcpyKind cudaMemcpyKind;
typedef hipDeviceProp_t cudaDeviceProp;
typedef void* cudaMemPool_t;

typedef hipStreamCallback_t cudaStreamCallback_t;
typedef void (*cudaHostFn_t)(void* userData);
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
// blas & lapack (hipBLAS/rocBLAS & rocSOLVER)
///////////////////////////////////////////////////////////////////////////////

/* As of ROCm 3.5.0 (this may have started earlier) many rocSOLVER helper functions
 * are deprecated and using their counterparts from rocBLAS is recommended. In
 * particular, rocSOLVER simply uses rocBLAS's handle for its API calls. This means
 * they are much more integrated than cuBLAS and cuSOLVER do, so it is better to
 * put all of the relevant function in one place.
 */

// TODO(leofang): investigate if we should just remove the hipBLAS layer and use
// rocBLAS directly, since we need to expose its handle anyway


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

typedef rocblas_status cusolverStatus_t;
typedef rocblas_handle cusolverDnHandle_t;


///////////////////////////////////////////////////////////////////////////////
// library_types.h
// (needed for supporting cusolver)
///////////////////////////////////////////////////////////////////////////////

typedef enum libraryPropertyType_t {
    MAJOR_VERSION,
    MINOR_VERSION,
    PATCH_LEVEL
} libraryPropertyType;

} // extern "C"

#endif // #ifndef INCLUDE_GUARD_HIP_CUPY_COMMON_H
