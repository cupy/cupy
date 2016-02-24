// This file is a stub header file of cuda for Read the Docs.

#ifndef INCLUDE_GUARD_CUPY_CUDA_H
#define INCLUDE_GUARD_CUPY_CUDA_H

#ifndef CUPY_NO_CUDA
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>

#else // #ifndef CUPY_NO_CUDA

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
enum cudaDeviceAttr {};
enum cudaMemcpyKind {};

typedef int Error;
typedef enum cudaDeviceAttr DeviceAttr;
typedef enum cudaMemcpyKind MemoryKind;


typedef void* _Pointer;


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
int cudaDriverGetVersion(int* driverVersion ) {
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

int cublasSgemmBatched(
        Handle handle, Operation transa, Operation transb, int m,
        int n, int k, const float* alpha, const float** Aarray,
        int lda, const float** Barray, int ldb, const float* beta,
        float** Carray, int ldc, int batchCount) {
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
// cublas_v2.h
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


#endif // #ifndef CUPY_NO_CUDA
#endif // #ifndef INCLUDE_GUARD_CUPY_CUDA_H
