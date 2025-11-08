#ifndef INCLUDE_GUARD_ASCEND_CUPY_COMMON_H
#define INCLUDE_GUARD_ASCEND_CUPY_COMMON_H

#include <acl/acl.h>  // has version
#include <acl/acl_base.h>
#if __has_include(<blas_api.h>) // CANN 8.2 provide BlAS by NNAL's AsdSip namepace
#include <blas_api.h>
#endif

#define CUDA_VERSION 0

extern "C" {

bool hip_environment = false;
bool cann_environment = true;

typedef aclError CUresult; // AscendCL错误类型
// Conditionally define CUDA_SUCCESS only if it's not defined
#ifndef CUDA_SUCCESS
const CUresult CUDA_SUCCESS = ACL_SUCCESS; // 使用AscendCL成功码
#endif
// 以下枚举在AscendCL中无直接等效，通常通过其他方式配置
typedef aclError cudaError_t; // Use AscendCL error type
const CUresult cudaSuccess = ACL_SUCCESS;
const CUresult cudaErrorInvalidValue = ACL_ERROR_INVALID_PARAM;
const CUresult cudaErrorMemoryAllocation = ACL_ERROR_BAD_ALLOC;
const CUresult cudaErrorInvalidResourceHandle = ACL_ERROR_INVALID_RESOURCE_HANDLE;
const CUresult cudaErrorContextIsDestroyed = ACL_ERROR_INVALID_RESOURCE_HANDLE; // WARNING: Potential equivalent
const CUresult cudaErrorPeerAccessAlreadyEnabled = ACL_ERROR_FEATURE_UNSUPPORTED; // WARNING: Peer access might be handled differently or not supported


typedef int CUdevice;  // AscendCL device is also int id
typedef aclDataBuffer* CUdeviceptr; // AscendCL数据缓冲区指针
typedef aclrtContext CUcontext; // AscendCL上下文
typedef aclrtStream CUstream; // AscendCL Stream
typedef aclrtStream cudaStream_t;
typedef aclrtEvent CUevent; // AscendCL事件
typedef aclrtEvent cudaEvent_t;
typedef void (*cudaStreamCallback_t)(aclrtStream* stream, aclError status, void* userData); // Adapted to AscendCL stream callback signature
typedef void (*cudaHostFn_t)(void* userData); // Can be implemented via threads/tasks

// TODO: ASCEND GraphExecutation, ascend ge has similar functions
#ifndef CUPY_INSTALL_USE_ASCEND
typedef void* cudaGraph_t;
typedef void* cudaGraphNode_t;
typedef void* cudaGraphExec_t;
#else
typedef void* cudaGraph_t;
typedef void* cudaGraphNode_t;
typedef void* cudaGraphExec_t;
#endif

///////////////////////////////////////////////////////////////////////////////
// CUDA 2DArray Interface, cuda.h (can be done via dlpack to exchange buffer with torch)
///////////////////////////////////////////////////////////////////////////////
#ifndef CUPY_INSTALL_USE_ASCEND
enum CUarray_format {}; // this enum is defined in <cuda.h>
typedef struct CUarray_st* CUarray; // AscendCL uses aclDataBuffer/aclTensorDesc for data management
struct CUDA_ARRAY_DESCRIPTOR {
    size_t Width;             /**< Width of array */
    size_t Height;            /**< Height of array */
    CUarray_format Format;    /**< Array format */
    unsigned int NumChannels; /**< Channels per array element */
};

///////////////////////////////////////////////////////////////////////////////
// JIT cuda_runtime.h (not needed in numpy-ascend)
///////////////////////////////////////////////////////////////////////////////
// WARNING: Missing direct equivalent - Kernels are typically pre-compiled into *.om models
typedef void* CUfunction; // Placeholder, kernel execution handled via model/operator interface
typedef int CUfunction_attribute; // WARNING: Missing direct equivalent
// typedef hipModule_t CUmodule; // WARNING: Missing direct equivalent - Models are loaded via aclmdlLoadFromFile/etc.
typedef void* CUmodule; // Placeholder
struct CUlinkState_st {}; // WARNING: Missing direct equivalent in AscendCL for link state - model building is typically offline
typedef struct CUlinkState_st *CUlinkState;

enum CUjit_option {}; // WARNING: Missing direct equivalent in AscendCL for JIT options - configuration often done via offline model generation (om)
enum CUjitInputType {}; // WARNING: Missing direct equivalent, use empty def
enum {
    cudaDevAttrComputeCapabilityMajor
        = 0, // WARNING: No direct equivalent attribute in AscendCL. Compute capability is Ascend chip specific.
    cudaDevAttrComputeCapabilityMinor
        = 1, // WARNING: No direct equivalent
};

typedef struct {} cudaDeviceProp; // Use aclDeviceDesc for device properties (partial info)
typedef int cudaDeviceAttr; // WARNING: No direct equivalent set of attributes, use aclDeviceGetDesc

#endif

///////////////////////////////////////////////////////////////////////////////
// Memory cuda_runtime.h
///////////////////////////////////////////////////////////////////////////////
typedef enum {} cudaLimit; //  resource limit enum: cudaLimitPrintfFifoSize
typedef enum {} cudaMemoryAdvise; // WARNING: No direct equivalent for unified memory advice
typedef enum {
    cudaMemcpyHostToHost,
    cudaMemcpyHostToDevice,
    cudaMemcpyDeviceToHost,
    cudaMemcpyDeviceToDevice,
    cudaMemcpyDefault
} cudaMemcpyKind; // AscendCL memory copy direction, compatible with cuda

typedef void* cudaMemPool_t; // WARNING: Missing direct equivalent for memory pool, use void* placeholder
enum cudaMemPoolAttr {}; // WARNING: Missing direct equivalent, use empty enum def

// ​cudaMemoryTypeHost（主机内存）、cudaMemoryTypeDevice（设备内存）和 cudaMemoryTypeManaged（统一内存）
typedef struct {
    int device;
    void *devicePointer;
    void *hostPointer;
} cudaPointerAttributes; // AscendCL has no such, use empty struct to get compiled

#ifndef CUPY_INSTALL_USE_ASCEND
///////////////////////////////////////////////////////////////////////////////
// ascend NPU AICORE does not support, dvpp may has similar feature
// The following texture/surface related types and enums have no direct equivalent in AscendCL
// use void*, empty struct/enum to hold place
///////////////////////////////////////////////////////////////////////////////

enum CUaddress_mode {}; // WARNING: Missing direct equivalent for texture address mode
enum CUfilter_mode {}; // WARNING: Missing direct equivalent for texture filter mode

// AscendCL uses different mechanisms for data processing (e.g., aclDataBuffer, aclTensorDesc).
typedef int cudaChannelFormatKind; // WARNING: Missing direct equivalent
typedef void* cudaTextureObject_t; // WARNING: Missing direct equivalent
typedef void* cudaSurfaceObject_t; // WARNING: Missing direct equivalent
typedef int cudaResourceType; // WARNING: Missing direct equivalent
typedef int cudaTextureAddressMode; // WARNING: Missing direct equivalent
typedef int cudaTextureFilterMode; // WARNING: Missing direct equivalent
typedef int cudaTextureReadMode; // WARNING: Missing direct equivalent
typedef struct {} cudaResourceViewDesc; // WARNING: Missing direct equivalent
typedef void* cudaArray_t; // WARNING: Missing direct equivalent

// WARNING: Missing direct equivalent
struct cudaExtent {
  size_t width;
  size_t height;
  size_t depth;
};

// WARNING: Missing direct equivalent, but use cuda's def here
struct cudaPos {
  size_t x;
  size_t y;
  size_t z;
};
typedef void* cudaPitchedPtr; // WARNING: Missing direct equivalent
typedef void* cudaMipmappedArray_t; // WARNING: Missing direct equivalent
typedef struct {}  cudaMemcpy3DParms; // WARNING: Missing direct equivalent - use aclrtMemcpy in specific cases
typedef struct {}  cudaChannelFormatDesc; // WARNING: Missing direct equivalent
typedef struct {}  cudaResourceDesc; // WARNING: Missing direct equivalent
typedef struct {}  cudaTextureDesc; // WARNING: Missing direct equivalent

// IPC operations
// AscendCL may have different mechanisms for resource sharing between processes.
typedef void* cudaIpcMemHandle_t; // WARNING: Missing direct equivalent
typedef void* cudaIpcEventHandle_t; // WARNING: Missing direct equivalent
#endif

///////////////////////////////////////////////////////////////////////////////
// blas & lapack
///////////////////////////////////////////////////////////////////////////////
/* AscendCL provides its own set of libraries for linear algebra and other operations.
 * These are not direct API-for-API replacements but serve similar functions.
 */
#if __has_include(<blas_api.h>)
typedef AsdSip::asdBlasHandle cublasHandle_t; // AscendCL BLAS handle
typedef AsdSip::asdBlasStatus cublasStatus_t; // Use AscendCL BLAS error type enum

// translation of these enum (int) will be done in higher layer, from numpy.dtype directly into aclDataType
typedef aclDataType cudaDataType_t; // Map to aclDataType
// BLAS enumeration types - AscendCL may use different enums or parameters
typedef AsdSip::asdBlasDiagType_t cublasDiagType_t;
typedef AsdSip::asdBlasFillMode_t cublasFillMode_t;
typedef AsdSip::asdBlasSideMode_t cublasSideMode_t;
typedef AsdSip::asdBlasOperation_t cublasOperation_t;

// TODO: ASCEND
typedef int cublasPointerMode_t; // WARNING: Check aclblasPointerMode enum if available
typedef enum {} cublasGemmAlgo_t; // WARNING: Missing direct equivalent - algorithm selection might differ
typedef enum {} cublasMath_t; // WARNING: Missing direct equivalent
typedef int cublasComputeType_t; // WARNING: Missing direct equivalent in AscendCL BLAS
#endif

// SOLVER (rocSOLVER/cuSOLVER replacement)
// AscendCL's support for direct solver routines (like rocSOLVER) is under develop
// might be limited or offered through different interfaces (e.g., specific operators in om models).
// Consider using Ascend's provided operators or leveraging MindSpore etc. for higher-level solver functionality.
#ifndef CUPY_INSTALL_USE_ASCEND
typedef aclError cusolverStatus_t; // Use AscendCL error type
typedef void* cusolverDnHandle_t; // WARNING: Missing direct equivalent handle for dense solvers
typedef void* cusolverSpHandle_t; // WARNING: Missing direct equivalent handle for sparse solvers
#endif

} // extern "C"

#endif /* INCLUDE_GUARD_ASCEND_CUPY_COMMON_H */