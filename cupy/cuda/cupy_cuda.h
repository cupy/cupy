// This file is a stub header file of cuda for Read the Docs.

#ifndef INCLUDE_GUARD_CUPY_CUDA_H
#define INCLUDE_GUARD_CUPY_CUDA_H

#ifndef CUPY_NO_CUDA
#include <cublas_v2.h>
#include <cuda.h>
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
int cublasSgemmEx(
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
typedef int CUresult;

typedef int Device;
typedef int Result;


typedef void* CUdeviceptr;
struct CUevent_st {};
struct CUfunc_st {};
struct CUmod_st {};
struct CUstream_st {};

typedef CUdeviceptr Deviceptr;
typedef struct CUevent_st* Event;
typedef struct CUfunc_st* Function;
typedef struct CUmod_st* Module;
typedef struct CUstream_st* Stream;


// Error handling
int cuGetErrorName(Result error, const char** pStr) {
    return 0;
}

int cuGetErrorString(Result error, const char** pStr) {
    return 0;
}


// Module load and kernel execution
int cuModuleLoad(Module* module, char* fname) {
    return 0;
}

int cuModuleLoadData(Module* module, void* image) {
    return 0;
}

int cuModuleUnload(Module hmod) {
    return 0;
}

int cuModuleGetFunction(Function* hfunc, Module hmod, char* name) {
    return 0;
}

int cuModuleGetGlobal(Deviceptr* dptr, size_t* bytes, Module hmod,
                      char* name) {
    return 0;
}

int cuLaunchKernel(
        Function f, unsigned int gridDimX, unsigned int gridDimY,
        unsigned int gridDimZ, unsigned int blockDimX,
        unsigned int blockDimY, unsigned int blockDimZ,
        unsigned int sharedMemBytes, Stream hStream,
        void** kernelParams, void** extra) {
    return 0;
}


///////////////////////////////////////////////////////////////////////////////
// cuda.h
///////////////////////////////////////////////////////////////////////////////

typedef int cudaError_t;
enum cudaDataType_t {};
enum cudaDeviceAttr {};
enum cudaMemcpyKind {};

typedef int Error;
typedef enum cudaDataType_t cudaDataType;
typedef enum cudaDeviceAttr DeviceAttr;
typedef enum cudaMemcpyKind MemoryKind;

typedef void (*cudaStreamCallback_t)(
    Stream stream, Error status, void* userData);

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
const char* cudaGetErrorName(Error error) {
    return NULL;
}

const char* cudaGetErrorString(Error error) {
    return NULL;
}


// Initialization
int cudaDriverGetVersion(int* driverVersion) {
    return 0;
}

int cudaRuntimeGetVersion(int* runtimeVersion) {
    return 0;
}


// Device operations
int cudaGetDevice(int* device) {
    return 0;
}

int cudaDeviceGetAttribute(int* value, DeviceAttr attr, int device ) {
    return 0;
}

int cudaGetDeviceCount(int* count) {
    return 0;
}

int cudaSetDevice(int device) {
    return 0;
}

int cudaDeviceSynchronize() {
    return 0;
}

int cudaDeviceCanAccessPeer(int* canAccessPeer, int device, int peerDevice) {
    return 0;
}

int cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags) {
    return 0;
}


// Memory management
int cudaMalloc(void** devPtr, size_t size) {
    return 0;
}

int cudaFree(void* devPtr) {
    return 0;
}

int cudaMemGetInfo(size_t* free, size_t* total) {
    return 0;
}

int cudaMemcpy(void* dst, const void* src, size_t count,
               MemoryKind kind) {
    return 0;
}

int cudaMemcpyAsync(void* dst, const void* src, size_t count,
                    MemoryKind kind, Stream stream) {
    return 0;
}

int cudaMemcpyPeer(void* dst, int dstDevice, const void* src,
                   int srcDevice, size_t count) {
    return 0;
}

int cudaMemcpyPeerAsync(void* dst, int dstDevice, const void* src,
                   int srcDevice, size_t count, Stream stream) {
    return 0;
}

int cudaMemset(void* devPtr, int value, size_t count) {
    return 0;
}

int cudaMemsetAsync(void* devPtr, int value, size_t count,
                    Stream stream) {
    return 0;
}

int cudaPointerGetAttributes(_PointerAttributes* attributes,
                             const void* ptr) {
    return 0;
}


// Stream and Event
int cudaStreamCreate(Stream* pStream) {
    return 0;
}

int cudaStreamCreateWithFlags(Stream* pStream, unsigned int flags) {
    return 0;
}

int cudaStreamDestroy(Stream stream) {
    return 0;
}

int cudaStreamSynchronize(Stream stream) {
    return 0;
}

int cudaStreamAddCallback(Stream stream, StreamCallback callback,
                          void* userData, unsigned int flags) {
    return 0;
}

int cudaStreamQuery(Stream stream) {
    return 0;
}

int cudaStreamWaitEvent(Stream stream, Event event,
                        unsigned int flags) {
    return 0;
}

int cudaEventCreate(Event* event) {
    return 0;
}

int cudaEventCreateWithFlags(Event* event, unsigned int flags) {
    return 0;
}

int cudaEventDestroy(Event event) {
    return 0;
}

int cudaEventElapsedTime(float* ms, Event start, Event end) {
    return 0;
}

int cudaEventQuery(Event event) {
    return 0;
}

int cudaEventRecord(Event event, Stream stream) {
    return 0;
}

int cudaEventSynchronize(Event event) {
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
// cuComplex.h
///////////////////////////////////////////////////////////////////////////////

#include "cupy_cuComplex.h"

///////////////////////////////////////////////////////////////////////////////
// cublas_v2.h
///////////////////////////////////////////////////////////////////////////////

typedef void* cublasHandle_t;

typedef void* Handle;


typedef int cublasOperation_t;
typedef int cublasPointerMode_t;
typedef int cublasSideMode_t;

typedef int Operation;
typedef int PointerMode;
typedef int SideMode;


// Context
int cublasCreate(Handle* handle) {
    return 0;
}

int cublasDestroy(Handle handle) {
    return 0;
}

int cublasGetVersion(Handle handle, int* version) {
    return 0;
}

int cublasGetPointerMode(Handle handle, PointerMode* mode) {
    return 0;
}

int cublasSetPointerMode(Handle handle, PointerMode mode) {
    return 0;
}

// Stream
int cublasSetStream(Handle handle, Stream streamId) {
    return 0;
}

int cublasGetStream(Handle handle, Stream* streamId) {
    return 0;
}

// BLAS Level 1
int cublasIsamax(Handle handle, int n, float* x, int incx,
                 int* result) {
    return 0;
}

int cublasIsamin(Handle handle, int n, float* x, int incx,
                 int* result) {
    return 0;
}

int cublasSasum(Handle handle, int n, float* x, int incx,
                float* result) {
    return 0;
}

int cublasSaxpy(Handle handle, int n, float* alpha, float* x,
                int incx, float* y, int incy) {
    return 0;
}

int cublasDaxpy(Handle handle, int n, double* alpha, double* x,
                int incx, double* y, int incy) {
    return 0;
}

int cublasSdot(Handle handle, int n, float* x, int incx,
               float* y, int incy, float* result) {
    return 0;
}

int cublasDdot(Handle handle, int n, double* x, int incx,
               double* y, int incy, double* result) {
    return 0;
}

int cublasCdotu(Handle handle, int n, cuComplex* x, int incx,
               cuComplex* y, int incy, cuComplex* result) {
    return 0;
}

int cublasCdotc(Handle handle, int n, cuComplex* x, int incx,
               cuComplex* y, int incy, cuComplex* result) {
    return 0;
}

int cublasZdotc(Handle handle, int n, cuDoubleComplex* x, int incx,
               cuDoubleComplex* y, int incy, cuDoubleComplex* result) {
    return 0;
}

int cublasZdotu(Handle handle, int n, cuDoubleComplex* x, int incx,
               cuComplex* y, int incy, cuDoubleComplex* result) {
    return 0;
}

int cublasSnrm2(Handle handle, int n, float* x, int incx,
                float* result) {
    return 0;
}

int cublasSscal(Handle handle, int n, float* alpha, float* x,
                int incx) {
    return 0;
}


// BLAS Level 2
int cublasSgemv(
        Handle handle, Operation trans, int m, int n, float* alpha,
        float* A, int lda, float* x, int incx, float* beta,
        float* y, int incy) {
    return 0;
}

int cublasDgemv(
        Handle handle, Operation trans, int m, int n, double* alpha,
        double* A, int lda, double* x, int incx, double* beta,
        double* y, int incy) {
    return 0;
}

int cublasCgemv(
        Handle handle, Operation trans, int m, int n, cuComplex* alpha,
        cuComplex* A, int lda, cuComplex* x, int incx, cuComplex* beta,
        cuComplex* y, int incy) {
    return 0;
}

int cublasZgemv(
        Handle handle, Operation trans, int m, int n, cuDoubleComplex* alpha,
        cuDoubleComplex* A, int lda, double* x, int incx, cuDoubleComplex* beta,
        cuDoubleComplex* y, int incy) {
    return 0;
}

int cublasSger(
        Handle handle, int m, int n, float* alpha, float* x, int incx,
        float* y, int incy, float* A, int lda) {
    return 0;
}

int cublasDger(
        Handle handle, int m, int n, double* alpha, double* x,
        int incx, double* y, int incy, double* A, int lda) {
    return 0;
}

int cublasCgeru(
        Handle handle, int m, int n, cuComplex* alpha, cuComplex* x,
        int incx, cuComplex* y, int incy, cuComplex* A, int lda) {
    return 0;
}

int cublasCgerc(
        Handle handle, int m, int n, cuComplex* alpha, cuComplex* x,
        int incx, cuComplex* y, int incy, cuComplex* A, int lda) {
    return 0;
}

int cublasZgeru(
        Handle handle, int m, int n, cuDoubleComplex* alpha, cuDoubleComplex* x,
        int incx, cuDoubleComplex* y, int incy, cuDoubleComplex* A, int lda) {
    return 0;
}

int cublasZgerc(
        Handle handle, int m, int n, cuDoubleComplex* alpha, cuDoubleComplex* x,
        int incx, cuDoubleComplex* y, int incy, cuDoubleComplex* A, int lda) {
    return 0;
}

// BLAS Level 3
int cublasSgemm(
        Handle handle, Operation transa, Operation transb, int m,
        int n, int k, float* alpha, float* A, int lda, float* B,
        int ldb, float* beta, float* C, int ldc) {
    return 0;
}

int cublasDgemm(
        Handle handle, Operation transa, Operation transb, int m,
        int n, int k, double* alpha, double* A, int lda, double* B,
        int ldb, double* beta, double* C, int ldc) {
    return 0;
}

int cublasCgemm(
        Handle handle, Operation transa, Operation transb, int m,
        int n, int k, cuComplex* alpha, cuComplex* A, int lda, cuComplex* B,
        int ldb, cuComplex* beta, cuComplex* C, int ldc) {
    return 0;
}

int cublasZgemm(
        Handle handle, Operation transa, Operation transb, int m,
        int n, int k, cuDoubleComplex* alpha, cuDoubleComplex* A, int lda,
        cuDoubleComplex* B, int ldb, cuDoubleComplex* beta, double* C,
        int ldc) {
    return 0;
}

int cublasSgemmBatched(
        Handle handle, Operation transa, Operation transb, int m,
        int n, int k, const float* alpha, const float** Aarray,
        int lda, const float** Barray, int ldb, const float* beta,
        float** Carray, int ldc, int batchCount) {
    return 0;
}

int cublasDgemmBatched(
        Handle handle, Operation transa, Operation transb, int m,
        int n, int k, const double* alpha, const double** Aarray,
        int lda, const double** Barray, int ldb, const double* beta,
        double** Carray, int ldc, int batchCount) {
    return 0;
}

int cublasCgemmBatched(
    Handle handle, Operation transa, Operation transb, int m,
    int n, int k, const cuComplex* alpha, const cuComplex** Aarray,
    int lda, const cuComplex** Barray, int ldb, const cuComplex* beta,
    cuComplexPtrPtr Carray, int ldc, int batchCount) {
    return 0;
}

int cublasZgemmBatched(
    Handle handle, Operation transa, Operation transb, int m,
    int n, int k, const cuDoubleComplex* alpha,
    const cuDoubleComplex** Aarray, int lda,
    const cuDoubleComplex** Barray, int ldb,
    const cuDoubleComplex* beta, cuDoubleComplex** Carray, int ldc,
    int batchCount) {
    return 0;
}

int cublasSgemmEx(
        cublasHandle_t handle, cublasOperation_t transa,
        cublasOperation_t transb, int m, int n, int k,
        const float *alpha, const void *A, cudaDataType Atype,
        int lda, const void *B, cudaDataType Btype, int ldb,
        const float *beta, void *C, cudaDataType Ctype, int ldc) {
    return 0;
}


// BLAS extension
int cublasSdgmm(
        Handle handle, SideMode mode, int m, int n, float* A, int lda,
        float* x, int incx, float* C, int ldc) {
    return 0;
}

int cublasSgetrfBatched(
        Handle handle, int n, float *Aarray[], int lda, int *PivotArray,
        int *infoArray, int batchSize) {
    return 0;
}

int cublasSgetriBatched(
        Handle handle, int n, const float *Aarray[], int lda, int *PivotArray,
        float *Carray[], int ldc, int *infoArray, int batchSize) {
    return 0;
}


///////////////////////////////////////////////////////////////////////////////
// curand.h
///////////////////////////////////////////////////////////////////////////////

typedef int curandOrdering_t;
typedef int curandRngType_t;

typedef int Ordering;
typedef int RngType;


typedef void* curandGenerator_t;

typedef void* Generator;


// Generator
int curandCreateGenerator(Generator* generator, int rng_type) {
    return 0;
}

int curandDestroyGenerator(Generator generator) {
    return 0;
}

int curandGetVersion(int* version) {
    return 0;
}


// Stream
int curandSetStream(Generator generator, Stream stream) {
    return 0;
}

int curandSetPseudoRandomGeneratorSeed(
    Generator generator, unsigned long long seed) {
    return 0;
}

int curandSetGeneratorOffset(
    Generator generator, unsigned long long offset) {
    return 0;
}

int curandSetGeneratorOrdering(Generator generator, Ordering order) {
    return 0;
}


// Generation functions
int curandGenerate(
        Generator generator, unsigned int* outputPtr, size_t num) {
    return 0;
}

int curandGenerateLongLong(
        Generator generator, unsigned long long* outputPtr,
        size_t num) {
    return 0;
}

int curandGenerateUniform(
        Generator generator, float* outputPtr, size_t num) {
    return 0;
}

int curandGenerateUniformDouble(
        Generator generator, double* outputPtr, size_t num) {
    return 0;
}

int curandGenerateNormal(
        Generator generator, float* outputPtr, size_t num,
        float mean, float stddev) {
    return 0;
}

int curandGenerateNormalDouble(
        Generator generator, double* outputPtr, size_t n,
        double mean, double stddev) {
    return 0;
}

int curandGenerateLogNormal(
        Generator generator, float* outputPtr, size_t n,
        float mean, float stddev) {
    return 0;
}

int curandGenerateLogNormalDouble(
        Generator generator, double* outputPtr, size_t n,
        double mean, double stddev) {
    return 0;
}

int curandGeneratePoisson(
        Generator generator, unsigned int* outputPtr, size_t n,
        double lam) {
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
// cuda_profiler_api.h
///////////////////////////////////////////////////////////////////////////////

typedef int cudaOutputMode_t;

int cudaProfilerInitialize(const char *configFile, 
                           const char *outputFile, 
                           cudaOutputMode_t outputMode) {
  return 0;
}

int cudaProfilerStart() {
  return 0;
}

int cudaProfilerStop() {
  return 0;
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
