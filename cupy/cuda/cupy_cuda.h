// This file is a stub header file of cuda for Read the Docs.

#ifndef INCLUDE_GUARD_CUPY_CUDA_H
#define INCLUDE_GUARD_CUPY_CUDA_H

#include "cupy_stdint.h"

#ifndef CUPY_NO_CUDA
#include <cuda.h>
#endif

#ifdef __APPLE__
#if CUDA_VERSION == 7050
// To avoid redefinition error of cudaDataType_t
// caused by including library_types.h.
// https://github.com/pfnet/chainer/issues/1700
// https://github.com/pfnet/chainer/pull/1819
#define __LIBRARY_TYPES_H__
#endif // #if CUDA_VERSION == 7050
#endif // #ifdef __APPLE__

#ifndef CUPY_NO_CUDA
#include <cublas_v2.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <curand.h>
#ifndef _WIN32
#include <nvToolsExt.h>
#endif

extern "C" {

#if CUDA_VERSION < 8000
#if CUDA_VERSION >= 7050
typedef cublasDataType_t cudaDataType;
#else
enum cudaDataType_t {};
typedef enum cudaDataType_t cudaDataType;
#endif // #if CUDA_VERSION >= 7050
#endif // #if CUDA_VERSION < 8000

#if CUDA_VERSION < 7050
cublasStatus_t cublasSgemmEx(
        cublasHandle_t handle, cublasOperation_t transa,
        cublasOperation_t transb, int m, int n, int k,
        const float *alpha, const void *A, cudaDataType Atype,
        int lda, const void *B, cudaDataType Btype, int ldb,
        const float *beta, void *C, cudaDataType Ctype, int ldc) {
    return CUBLAS_STATUS_NOT_SUPPORTED;
}
#endif // #if CUDA_VERSION < 7050

} // extern "C"

#else // #ifndef CUPY_NO_CUDA

extern "C" {

///////////////////////////////////////////////////////////////////////////////
// cuda.h
///////////////////////////////////////////////////////////////////////////////

typedef int CUdevice;
typedef enum {
    CUDA_SUCCESS = 0,
} CUresult;


typedef void* CUdeviceptr;
struct CUevent_st;
struct CUfunc_st;
struct CUmod_st;
struct CUstream_st;

typedef struct CUevent_st* cudaEvent_t;
typedef struct CUfunc_st* CUfunction;
typedef struct CUmod_st* CUmodule;
typedef struct CUstream_st* cudaStream_t;


// Error handling
CUresult cuGetErrorName(CUresult error, const char** pStr) {
    return CUDA_SUCCESS;
}

CUresult cuGetErrorString(CUresult error, const char** pStr) {
    return CUDA_SUCCESS;
}


// Module load and kernel execution
CUresult cuModuleLoad(CUmodule* module, char* fname) {
    return CUDA_SUCCESS;
}

CUresult cuModuleLoadData(CUmodule* module, void* image) {
    return CUDA_SUCCESS;
}

CUresult cuModuleUnload(CUmodule hmod) {
    return CUDA_SUCCESS;
}

CUresult cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, char* name) {
    return CUDA_SUCCESS;
}

CUresult cuModuleGetGlobal(CUdeviceptr* dptr, size_t* bytes, CUmodule hmod,
                      char* name) {
    return CUDA_SUCCESS;
}

CUresult cuLaunchKernel(
        CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
        unsigned int gridDimZ, unsigned int blockDimX,
        unsigned int blockDimY, unsigned int blockDimZ,
        unsigned int sharedMemBytes, cudaStream_t hStream,
        void** kernelParams, void** extra) {
    return CUDA_SUCCESS;
}


///////////////////////////////////////////////////////////////////////////////
// cuda_runtime.h
///////////////////////////////////////////////////////////////////////////////

typedef enum {
    cudaSuccess = 0,
} cudaError_t;
typedef enum {} cudaDataType;
enum cudaDeviceAttr {};
enum cudaMemcpyKind {};


typedef void (*cudaStreamCallback_t)(
    cudaStream_t stream, cudaError_t status, void* userData);

typedef cudaStreamCallback_t StreamCallback;


struct cudaPointerAttributes{
    int device;
    void* devicePointer;
    void* hostPointer;
    int isManaged;
    int memoryType;
};

typedef cudaPointerAttributes _PointerAttributes;


// Error handling
const char* cudaGetErrorName(cudaError_t error) {
    return NULL;
}

const char* cudaGetErrorString(cudaError_t error) {
    return NULL;
}


// Initialization
cudaError_t cudaDriverGetVersion(int* driverVersion) {
    return cudaSuccess;
}

cudaError_t cudaRuntimeGetVersion(int* runtimeVersion) {
    return cudaSuccess;
}


// CUdevice operations
cudaError_t cudaGetDevice(int* device) {
    return cudaSuccess;
}

cudaError_t cudaDeviceGetAttribute(
        int* value, cudaDeviceAttr attr, int device) {
    return cudaSuccess;
}

cudaError_t cudaGetDeviceCount(int* count) {
    return cudaSuccess;
}

cudaError_t cudaSetDevice(int device) {
    return cudaSuccess;
}

cudaError_t cudaDeviceSynchronize() {
    return cudaSuccess;
}

cudaError_t cudaDeviceCanAccessPeer(
        int* canAccessPeer, int device, int peerDevice) {
    return cudaSuccess;
}

cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags) {
    return cudaSuccess;
}


// Memory management
cudaError_t cudaMalloc(void** devPtr, size_t size) {
    return cudaSuccess;
}

cudaError_t cudaHostAlloc(void** ptr, size_t size, unsigned int flags) {
    return cudaSuccess;
}

int cudaFree(void* devPtr) {
    return cudaSuccess;
}

cudaError_t cudaFreeHost(void* ptr) {
    return cudaSuccess;
}

int cudaMemGetInfo(size_t* free, size_t* total) {
    return cudaSuccess;
}

cudaError_t cudaMemcpy(
          void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
    return cudaSuccess;
}

cudaError_t cudaMemcpyAsync(
        void* dst, const void* src, size_t count, cudaMemcpyKind kind,
        cudaStream_t stream) {
    return cudaSuccess;
}

cudaError_t cudaMemcpyPeer(
        void* dst, int dstDevice, const void* src, int srcDevice,
        size_t count) {
    return cudaSuccess;
}

cudaError_t cudaMemcpyPeerAsync(
          void* dst, int dstDevice, const void* src, int srcDevice,
          size_t count, cudaStream_t stream) {
    return cudaSuccess;
}

cudaError_t cudaMemset(void* devPtr, int value, size_t count) {
    return cudaSuccess;
}

cudaError_t cudaMemsetAsync(
        void* devPtr, int value, size_t count, cudaStream_t stream) {
    return cudaSuccess;
}

cudaError_t cudaPointerGetAttributes(
        _PointerAttributes* attributes, const void* ptr) {
    return cudaSuccess;
}


// Stream and Event
cudaError_t cudaStreamCreate(cudaStream_t* pStream) {
    return cudaSuccess;
}

cudaError_t cudaStreamCreateWithFlags(
        cudaStream_t* pStream, unsigned int flags) {
    return cudaSuccess;
}

cudaError_t cudaStreamDestroy(cudaStream_t stream) {
    return cudaSuccess;
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    return cudaSuccess;
}

cudaError_t cudaStreamAddCallback(
        cudaStream_t stream, StreamCallback callback,
        void* userData, unsigned int flags) {
    return cudaSuccess;
}

cudaError_t cudaStreamQuery(cudaStream_t stream) {
    return cudaSuccess;
}

cudaError_t cudaStreamWaitEvent(
        cudaStream_t stream, cudaEvent_t event, unsigned int flags) {
    return cudaSuccess;
}

cudaError_t cudaEventCreate(cudaEvent_t* event) {
    return cudaSuccess;
}

cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags) {
    return cudaSuccess;
}

cudaError_t cudaEventDestroy(cudaEvent_t event) {
    return cudaSuccess;
}

cudaError_t cudaEventElapsedTime(
        float* ms, cudaEvent_t start, cudaEvent_t end) {
    return cudaSuccess;
}

cudaError_t cudaEventQuery(cudaEvent_t event) {
    return cudaSuccess;
}

cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    return cudaSuccess;
}

cudaError_t cudaEventSynchronize(cudaEvent_t event) {
    return cudaSuccess;
}


///////////////////////////////////////////////////////////////////////////////
// cublas_v2.h
///////////////////////////////////////////////////////////////////////////////

typedef void* cublasHandle_t;

typedef enum {} cublasOperation_t;
typedef enum {} cublasPointerMode_t;
typedef enum {} cublasSideMode_t;
typedef enum {
    CUBLAS_STATUS_SUCCESS=0,
} cublasStatus_t;


// Context
cublasStatus_t cublasCreate(cublasHandle_t* handle) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDestroy(cublasHandle_t handle) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetVersion(cublasHandle_t handle, int* version) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetPointerMode(
        cublasHandle_t handle, cublasPointerMode_t* mode) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetPointerMode(
        cublasHandle_t handle, cublasPointerMode_t mode) {
    return CUBLAS_STATUS_SUCCESS;
}

// Stream
cublasStatus_t cublasSetStream(cublasHandle_t handle, cudaStream_t streamId) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetStream(cublasHandle_t handle, cudaStream_t* streamId) {
    return CUBLAS_STATUS_SUCCESS;
}

// BLAS Level 1
cublasStatus_t cublasIsamax(
        cublasHandle_t handle, int n, float* x, int incx, int* result) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasIsamin(
        cublasHandle_t handle, int n, float* x, int incx, int* result) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSasum(cublasHandle_t handle, int n, float* x, int incx,
                float* result) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSaxpy(
      cublasHandle_t handle, int n, float* alpha, float* x,
      int incx, float* y, int incy) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDaxpy(
        cublasHandle_t handle, int n, double* alpha, double* x,
        int incx, double* y, int incy) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSdot(
        cublasHandle_t handle, int n, float* x, int incx,
        float* y, int incy, float* result) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDdot(
        cublasHandle_t handle, int n, double* x, int incx,
        double* y, int incy, double* result) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSnrm2(
        cublasHandle_t handle, int n, float* x, int incx, float* result) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSscal(
        cublasHandle_t handle, int n, float* alpha, float* x, int incx) {
    return CUBLAS_STATUS_SUCCESS;
}


// BLAS Level 2
cublasStatus_t cublasSgemv(
        cublasHandle_t handle, cublasOperation_t trans, int m, int n,
        float* alpha, float* A, int lda, float* x, int incx, float* beta,
        float* y, int incy) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDgemv(
        cublasHandle_t handle, cublasOperation_t trans, int m, int n,
        double* alpha, double* A, int lda, double* x, int incx, double* beta,
        double* y, int incy) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSger(
        cublasHandle_t handle, int m, int n, float* alpha, float* x, int incx,
        float* y, int incy, float* A, int lda) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDger(
        cublasHandle_t handle, int m, int n, double* alpha, double* x,
        int incx, double* y, int incy, double* A, int lda) {
    return CUBLAS_STATUS_SUCCESS;
}

// BLAS Level 3
cublasStatus_t cublasSgemm(
        cublasHandle_t handle, cublasOperation_t transa,
        cublasOperation_t transb, int m, int n, int k, float* alpha,
        float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDgemm(
        cublasHandle_t handle, cublasOperation_t transa,
        cublasOperation_t transb, int m, int n, int k, double* alpha,
        double* A, int lda, double* B, int ldb, double* beta, double* C,
        int ldc) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSgemmBatched(
        cublasHandle_t handle, cublasOperation_t transa,
        cublasOperation_t transb, int m, int n, int k, const float* alpha,
        const float** Aarray, int lda, const float** Barray, int ldb,
        const float* beta, float** Carray, int ldc, int batchCount) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSgemmEx(
        cublasHandle_t handle, cublasOperation_t transa,
        cublasOperation_t transb, int m, int n, int k,
        const float *alpha, const void *A, cudaDataType Atype,
        int lda, const void *B, cudaDataType Btype, int ldb,
        const float *beta, void *C, cudaDataType Ctype, int ldc) {
    return CUBLAS_STATUS_SUCCESS;
}


// BLAS extension
cublasStatus_t cublasSdgmm(
        cublasHandle_t handle, cublasSideMode_t mode, int m, int n, float* A,
        int lda, float* x, int incx, float* C, int ldc) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSgetrfBatched(
        cublasHandle_t handle, int n, float *Aarray[], int lda,
        int *PivotArray, int *infoArray, int batchSize) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSgetriBatched(
        cublasHandle_t handle, int n, const float *Aarray[], int lda,
        int *PivotArray, float *Carray[], int ldc, int *infoArray,
        int batchSize) {
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
curandStatus_t curandCreateGenerator(
        curandGenerator_t* generator, curandRngType_t rng_type) {
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandDestroyGenerator(curandGenerator_t generator) {
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGetVersion(int* version) {
    return CURAND_STATUS_SUCCESS;
}


// Stream
curandStatus_t curandSetStream(
        curandGenerator_t generator, cudaStream_t stream) {
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandSetPseudoRandomGeneratorSeed(
        curandGenerator_t generator, unsigned long long seed) {
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandSetGeneratorOffset(
        curandGenerator_t generator, unsigned long long offset) {
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandSetGeneratorOrdering(
        curandGenerator_t generator, curandOrdering_t order) {
    return CURAND_STATUS_SUCCESS;
}


// Generation functions
curandStatus_t curandGenerate(
        curandGenerator_t generator, unsigned int* outputPtr, size_t num) {
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGenerateLongLong(
        curandGenerator_t generator, unsigned long long* outputPtr,
        size_t num) {
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGenerateUniform(
        curandGenerator_t generator, float* outputPtr, size_t num) {
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGenerateUniformDouble(
        curandGenerator_t generator, double* outputPtr, size_t num) {
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGenerateNormal(
        curandGenerator_t generator, float* outputPtr, size_t num,
        float mean, float stddev) {
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGenerateNormalDouble(
        curandGenerator_t generator, double* outputPtr, size_t n,
        double mean, double stddev) {
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGenerateLogNormal(
        curandGenerator_t generator, float* outputPtr, size_t n,
        float mean, float stddev) {
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGenerateLogNormalDouble(
        curandGenerator_t generator, double* outputPtr, size_t n,
        double mean, double stddev) {
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGeneratePoisson(
        curandGenerator_t generator, unsigned int* outputPtr, size_t n,
        double lam) {
    return CURAND_STATUS_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
// cuda_profiler_api.h
///////////////////////////////////////////////////////////////////////////////

typedef enum {} cudaOutputMode_t;

cudaError_t cudaProfilerInitialize(const char *configFile,
                           const char *outputFile,
                           cudaOutputMode_t outputMode) {
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

void nvtxMarkA(const char *message) {
}

void nvtxMarkEx(const nvtxEventAttributes_t *eventAttrib) {
}

int nvtxRangePushA(const char *message) {
    return 0;
}

int nvtxRangePushEx(const nvtxEventAttributes_t *eventAttrib) {
    return 0;
}

int nvtxRangePop() {
    return 0;
}

} // extern "C"

#endif // #ifndef CUPY_NO_CUDA
#endif // #ifndef INCLUDE_GUARD_CUPY_CUDA_H
