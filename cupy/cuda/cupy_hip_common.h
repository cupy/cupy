// This file is a stub header file of hip for Read the Docs.

#ifndef INCLUDE_GUARD_CUPY_HIP_COMMON_H
#define INCLUDE_GUARD_CUPY_HIP_COMMON_H

#include <hip/hip_runtime_api.h>
#include <hipblas.h>
#include <hiprand/hiprand.h>

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


typedef hipDeviceptr_t CUdeviceptr;
//struct CUevent_st;
//struct CUfunc_st;
//struct CUmod_st;
struct CUlinkState_st;


typedef hipCtx_t CUcontext;
typedef hipEvent_t cudaEvent_t;
typedef hipFunction_t CUfunction;
typedef hipModule_t CUmodule;
typedef hipStream_t cudaStream_t;
typedef struct CUlinkState_st* CUlinkState;


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
typedef enum {} cudaDataType;
typedef hipDeviceAttribute_t cudaDeviceAttr;
enum cudaMemoryAdvise {};
typedef hipMemcpyKind cudaMemcpyKind;


typedef hipStreamCallback_t cudaStreamCallback_t;
typedef hipPointerAttribute_t cudaPointerAttributes;


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
typedef hipblasStatus_t cublasStatus_t;

} // extern "C"

#endif // #ifndef INCLUDE_GUARD_CUPY_HIP_COMMON_H
