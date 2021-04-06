// This file is a stub header file of cuda for Read the Docs.

#ifndef INCLUDE_GUARD_STUB_CUPY_CUDA_COMMON_H
#define INCLUDE_GUARD_STUB_CUPY_CUDA_COMMON_H

#define CUDA_VERSION 0

extern "C" {

///////////////////////////////////////////////////////////////////////////////
// cuda.h
///////////////////////////////////////////////////////////////////////////////

typedef int CUdevice;
typedef enum {
    CUDA_SUCCESS = 0,
} CUresult;
enum CUjit_option {};
enum CUjitInputType {};
enum CUfunction_attribute {};;
enum CUarray_format {};
enum CUaddress_mode {};
enum CUfilter_mode {};


typedef void* CUdeviceptr;
struct CUctx_st;
struct CUevent_st;
struct CUfunc_st;
struct CUmod_st;
struct CUstream_st;
struct CUlinkState_st;


typedef struct CUctx_st* CUcontext;
typedef struct CUevent_st* cudaEvent_t;
typedef struct CUfunc_st* CUfunction;
typedef struct CUmod_st* CUmodule;
typedef struct CUstream_st* cudaStream_t;
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
    cudaDevAttrComputeCapabilityMajor = 75,
    cudaDevAttrComputeCapabilityMinor = 76,
};

typedef enum {
    cudaSuccess = 0,
    cudaErrorInvalidValue = 1,
    cudaErrorMemoryAllocation = 2,
    cudaErrorInvalidResourceHandle = 400,
    cudaErrorContextIsDestroyed = 709,
    cudaErrorPeerAccessAlreadyEnabled = 704,
} cudaError_t;
typedef enum {} cudaDataType;
enum cudaDeviceAttr {};
enum cudaLimit {};
enum cudaMemoryAdvise {};
enum cudaMemcpyKind {};


typedef void (*cudaStreamCallback_t)(
    cudaStream_t stream, cudaError_t status, void* userData);


struct cudaPointerAttributes{
    int device;
    void* devicePointer;
    void* hostPointer;
};


enum cudaChannelFormatKind {};
typedef unsigned long long cudaTextureObject_t;
typedef unsigned long long cudaSurfaceObject_t;
enum cudaResourceType {};
enum cudaTextureAddressMode {};
enum cudaTextureFilterMode {};
enum cudaTextureReadMode {};
struct cudaResourceViewDesc;
typedef void* cudaArray_t;
struct cudaExtent {
    size_t width, height, depth;
};
struct cudaPos {
    size_t x, y, z;
};
struct cudaPitchedPtr {
    size_t pitch;
    void* ptr;
    size_t xsize, ysize;
};
typedef void* cudaMipmappedArray_t;
struct cudaMemcpy3DParms {
    cudaArray_t srcArray;
    struct cudaPos srcPos;
    struct cudaPitchedPtr srcPtr;
    cudaArray_t dstArray;
    struct cudaPos dstPos;
    struct cudaPitchedPtr dstPtr;
    struct cudaExtent extent;
    enum cudaMemcpyKind kind;
};
struct cudaChannelFormatDesc {
    int x, y, z, w;
    enum cudaChannelFormatKind f;
};
struct cudaResourceDesc {
    enum cudaResourceType resType;

    union {
        struct {
            cudaArray_t array;
        } array;
        struct {
            cudaMipmappedArray_t mipmap;
        } mipmap;
        struct {
            void *devPtr;
            struct cudaChannelFormatDesc desc;
            size_t sizeInBytes;
        } linear;
        struct {
            void *devPtr;
            struct cudaChannelFormatDesc desc;
            size_t width;
            size_t height;
            size_t pitchInBytes;
        } pitch2D;
    } res;
};
struct cudaTextureDesc {
    enum cudaTextureAddressMode addressMode[3];
    enum cudaTextureFilterMode filterMode;
    enum cudaTextureReadMode readMode;
    int sRGB;
    float borderColor[4];
    int normalizedCoords;
    unsigned int maxAnisotropy;
    enum cudaTextureFilterMode mipmapFilterMode;
    float mipmapLevelBias;
    float minMipmapLevelClamp;
    float maxMipmapLevelClamp;
};

// IPC operations
typedef struct {
    unsigned char reserved[64];
} cudaIpcMemHandle_t;

// IPC operations
typedef struct {
    unsigned char reserved[64];
} cudaIpcEventHandle_t;

typedef struct {
     char name[256];
     size_t totalGlobalMem;
     size_t sharedMemPerBlock;
     int regsPerBlock;
     int warpSize;
     int maxThreadsPerBlock;
     int maxThreadsDim[3];
     int maxGridSize[3];
     int clockRate;
     int memoryClockRate;
     int memoryBusWidth;
     size_t totalConstMem;
     int major;
     int minor;
     int multiProcessorCount;
     int l2CacheSize;
     int maxThreadsPerMultiProcessor;
     int computeMode;
     int clockInstructionRate;
     int concurrentKernels;
     int pciBusID;
     int pciDeviceID;
     size_t maxSharedMemoryPerMultiProcessor;
     int isMultiGpuBoard;
     int canMapHostMemory;
} cudaDeviceProp;

typedef void* cudaMemPool_t;


///////////////////////////////////////////////////////////////////////////////
// library_types.h
///////////////////////////////////////////////////////////////////////////////

typedef enum libraryPropertyType_t {
	MAJOR_VERSION,
	MINOR_VERSION,
	PATCH_LEVEL
} libraryPropertyType;


///////////////////////////////////////////////////////////////////////////////
// cublas_v2.h
///////////////////////////////////////////////////////////////////////////////

typedef void* cublasHandle_t;

typedef enum {} cublasDiagType_t;
typedef enum {} cublasFillMode_t;
typedef enum {} cublasOperation_t;
typedef enum {} cublasPointerMode_t;
typedef enum {} cublasSideMode_t;
typedef enum {} cublasGemmAlgo_t;
typedef enum {} cublasMath_t;
typedef enum {
    CUBLAS_STATUS_SUCCESS=0,
} cublasStatus_t;

} // extern "C"

#endif // #ifndef INCLUDE_GUARD_STUB_CUPY_CUDA_COMMON_H
