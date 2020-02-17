// This file is a stub header file of hip for Read the Docs.

#ifndef INCLUDE_GUARD_CUPY_HIP_H
#define INCLUDE_GUARD_CUPY_HIP_H

#include <hiprand/hiprand.h>
#include "cupy_hip_common.h"

extern "C" {

bool hip_environment = true;

///////////////////////////////////////////////////////////////////////////////
// cuda.h
///////////////////////////////////////////////////////////////////////////////

// Error handling
CUresult cuGetErrorName(CUresult hipError, const char** pStr) {
    *pStr = hipGetErrorName(hipError);
    return CUDA_SUCCESS;
}

CUresult cuGetErrorString(CUresult hipError, const char** pStr) {
    *pStr = hipGetErrorString(hipError);
    return CUDA_SUCCESS;
}

// Primary context management
CUresult cuDevicePrimaryCtxRelease(CUdevice dev) {
    return hipDevicePrimaryCtxRelease(dev);
}

// Context management
CUresult cuCtxGetCurrent(CUcontext *ctx) {
    // deprecated api
    //return hipCtxGetCurrent(ctx);
    return hipErrorUnknown;
}

CUresult cuCtxSetCurrent(CUcontext ctx) {
    // deprecated api
    //return hipCtxSetCurrent(ctx);
    return hipErrorUnknown;
}

CUresult cuCtxCreate(CUcontext* pctx, unsigned int flags, CUdevice dev) {
    // deprecated api
    //return hipCtxCreate(pctx, flags, dev);
    return hipErrorUnknown;
}

CUresult cuCtxDestroy(CUcontext ctx) {
    // deprecated api
    // return hipCtxDestroy(ctx);
    return hipErrorUnknown;
}


// Module load and kernel execution
CUresult cuLinkCreate(...) {
    return hipErrorUnknown;
}

CUresult cuLinkAddData(...) {
    return hipErrorUnknown;
}

CUresult cuLinkAddFile(...) {
    return hipErrorUnknown;
}

CUresult cuLinkComplete(...) {
    return hipErrorUnknown;
}

CUresult cuLinkDestroy(...) {
    return hipErrorUnknown;
}

CUresult cuModuleLoad(CUmodule *module, const char *fname) {
    return hipModuleLoad(module, fname);
}

CUresult cuModuleLoadData(CUmodule *module, const void *image) {
    return hipModuleLoadData(module, image);
}

CUresult cuModuleUnload(CUmodule module) {
    return hipModuleUnload(module);
}

CUresult cuModuleGetFunction(CUfunction *function, CUmodule module,
                             const char *kname) {
    return hipModuleGetFunction(function, module, kname);
}

CUresult cuModuleGetGlobal(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod,
                           const char *name) {
    return hipModuleGetGlobal(dptr, bytes, hmod, name);
}

CUresult cuModuleGetTexRef(...) {
    return hipErrorUnknown;
}

CUresult cuLaunchKernel(CUfunction f, uint32_t gridDimX, uint32_t gridDimY,
                        uint32_t gridDimZ, uint32_t blockDimX,
                        uint32_t blockDimY, uint32_t blockDimZ,
                        uint32_t sharedMemBytes, cudaStream_t hStream,
                        void **kernelParams, void **extra) {
    return hipModuleLaunchKernel(f, gridDimX, gridDimY, gridDimZ,
                                 blockDimX, blockDimY, blockDimZ,
                                 sharedMemBytes, hStream, kernelParams, extra);
}

CUresult cuLaunchCooperativeKernel(...) {
    return hipErrorUnknown;
}


// Function attribute
CUresult cuFuncGetAttribute(...) {
    return hipErrorUnknown;
}

CUresult cuFuncSetAttribute(...) {
    return hipErrorUnknown;
}


// Texture reference
CUresult cuTexRefSetAddress (...) {
    return hipErrorUnknown;
}

CUresult cuTexRefSetAddress2D (...) {
    return hipErrorUnknown;
}

CUresult cuTexRefSetAddressMode (...) {
    return hipErrorUnknown;
}

CUresult cuTexRefSetArray (...) {
    return hipErrorUnknown;
}

CUresult cuTexRefSetBorderColor (...) {
    return hipErrorUnknown;
}

CUresult cuTexRefSetFilterMode (...) {
    return hipErrorUnknown;
}

CUresult cuTexRefSetFlags (...) {
    return hipErrorUnknown;
}

CUresult cuTexRefSetFormat (...) {
    return hipErrorUnknown;
}

CUresult cuTexRefSetMaxAnisotropy (...) {
    return hipErrorUnknown;
}

// Occupancy
typedef size_t (*CUoccupancyB2DSize)(int);

CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(...) {
    return hipErrorUnknown;
}

CUresult cuOccupancyMaxPotentialBlockSize(...) {
    return hipErrorUnknown;
}


///////////////////////////////////////////////////////////////////////////////
// cuda_runtime.h
///////////////////////////////////////////////////////////////////////////////

// Error handling
const char* cudaGetErrorName(cudaError_t hipError) {
    return hipGetErrorName(hipError);
}

const char* cudaGetErrorString(cudaError_t hipError) {
    return hipGetErrorString(hipError);
}

cudaError_t cudaGetLastError() {
    return hipGetLastError();
}


// Initialization
cudaError_t cudaDriverGetVersion(int *driverVersion) {
    return hipDriverGetVersion(driverVersion);
}

cudaError_t cudaRuntimeGetVersion(int *runtimeVersion) {
    return hipRuntimeGetVersion(runtimeVersion);
}


// CUdevice operations
cudaError_t cudaGetDevice(int *deviceId) {
    return hipGetDevice(deviceId);
}

cudaError_t cudaDeviceGetAttribute(int* pi, cudaDeviceAttr attr,
                                   int deviceId) {
    return hipDeviceGetAttribute(pi, attr, deviceId);
}

cudaError_t cudaDeviceGetByPCIBusId(int *device, const char *pciBusId) {
    return hipDeviceGetByPCIBusId(device, pciBusId);
}

cudaError_t cudaDeviceGetPCIBusId(char *pciBusId, int len, int device) {
    return hipDeviceGetPCIBusId(pciBusId, len, device);
}

cudaError_t cudaGetDeviceCount(int *count) {
    return hipGetDeviceCount(count);
}

cudaError_t cudaSetDevice(int deviceId) {
    return hipSetDevice(deviceId);
}

cudaError_t cudaDeviceSynchronize() {
    return hipDeviceSynchronize();
}

cudaError_t cudaDeviceCanAccessPeer(int* canAccessPeer, int deviceId,
                                    int peerDeviceId) {
    return hipDeviceCanAccessPeer(canAccessPeer, deviceId, peerDeviceId);
}

cudaError_t cudaDeviceEnablePeerAccess(int peerDeviceId, unsigned int flags) {
    return hipDeviceEnablePeerAccess(peerDeviceId, flags);
}


// Memory management
cudaError_t cudaMalloc(void** ptr, size_t size) {
    return hipMalloc(ptr, size);
}

cudaError_t cudaMalloc3DArray(...) {
    return hipErrorUnknown;
}

cudaError_t cudaMallocArray(...) {
    return hipErrorUnknown;
}

cudaError_t cudaHostAlloc(void** ptr, size_t size, unsigned int flags) {
    return hipHostMalloc(ptr, size, flags);
}

cudaError_t cudaHostRegister(...) {
    return hipErrorUnknown;
}

cudaError_t cudaHostUnregister(...) {
    return hipErrorUnknown;
}

cudaError_t cudaMallocManaged(...) {
    return hipErrorUnknown;
}

int cudaFree(void* ptr) {
    return hipFree(ptr);
}

cudaError_t cudaFreeArray(...) {
    return hipErrorUnknown;
}

cudaError_t cudaFreeHost(void* ptr) {
    return hipHostFree(ptr);
}

int cudaMemGetInfo(size_t* free, size_t* total) {
    return hipMemGetInfo(free, total);
}

cudaError_t cudaMemcpy(void* dst, const void* src, size_t sizeBytes,
                       hipMemcpyKind kind) {
    return hipMemcpy(dst, src, sizeBytes, kind);
}

cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t sizeBytes,
                            cudaMemcpyKind kind, cudaStream_t stream) {
    return hipMemcpyAsync(dst, src, sizeBytes, kind, stream);
}

cudaError_t cudaMemcpyPeer(void* dst, int dstDeviceId, const void* src,
                           int srcDeviceId, size_t sizeBytes) {
    return hipMemcpyPeer(dst, dstDeviceId, src, srcDeviceId, sizeBytes);
}

cudaError_t cudaMemcpyPeerAsync(void* dst, int dstDevice, const void* src,
                                int srcDevice, size_t sizeBytes,
                                cudaStream_t stream) {
    return hipMemcpyPeerAsync(dst, dstDevice, src, srcDevice, sizeBytes,
                              stream);
}

cudaError_t cudaMemcpy2D(...) {
    return hipErrorUnknown;
}

cudaError_t cudaMemcpy2DAsync(...) {
    return hipErrorUnknown;
}

cudaError_t cudaMemcpy2DFromArray(...) {
    return hipErrorUnknown;
}

cudaError_t cudaMemcpy2DFromArrayAsync(...) {
    return hipErrorUnknown;
}

cudaError_t cudaMemcpy2DToArray(...) {
    return hipErrorUnknown;
}

cudaError_t cudaMemcpy2DToArrayAsync(...) {
    return hipErrorUnknown;
}

cudaError_t cudaMemcpy3D(...) {
    return hipErrorUnknown;
}

cudaError_t cudaMemcpy3DAsync(...) {
    return hipErrorUnknown;
}

cudaError_t cudaMemset(void* dst, int value, size_t sizeBytes) {
    return hipMemset(dst, value, sizeBytes);
}

cudaError_t cudaMemsetAsync(void* dst, int value, size_t sizeBytes,
                            cudaStream_t stream) {
    return hipMemsetAsync(dst, value, sizeBytes, stream);
}

cudaError_t cudaMemAdvise(...) {
    return hipErrorUnknown;
}

cudaError_t cudaMemPrefetchAsync(...) {
    return hipErrorUnknown;
}

cudaError_t cudaPointerGetAttributes(cudaPointerAttributes *attributes,
                                     const void* ptr) {
    return hipPointerGetAttributes(attributes, ptr);
}


// Stream and Event
cudaError_t cudaStreamCreate(cudaStream_t *stream) {
    return hipStreamCreate(stream);
}

cudaError_t cudaStreamCreateWithFlags(cudaStream_t *stream,
                                      unsigned int flags) {
    return hipStreamCreateWithFlags(stream, flags);
}

cudaError_t cudaStreamDestroy(cudaStream_t stream) {
    return hipStreamDestroy(stream);
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    return hipStreamSynchronize(stream);
}

cudaError_t cudaStreamAddCallback(cudaStream_t stream,
                                  cudaStreamCallback_t callback,
                                  void *userData, unsigned int flags) {
    return hipStreamAddCallback(stream, callback, userData, flags);
}

cudaError_t cudaStreamQuery(cudaStream_t stream) {
    return hipStreamQuery(stream);
}

cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event,
                                unsigned int flags) {
    return hipStreamWaitEvent(stream, event, flags);
}

cudaError_t cudaEventCreate(cudaEvent_t* event) {
    return hipEventCreate(event);
}

cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned flags) {
    return hipEventCreateWithFlags(event, flags);
}

cudaError_t cudaEventDestroy(cudaEvent_t event) {
    return hipEventDestroy(event);
}

cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start,
                                 cudaEvent_t stop){
    return hipEventElapsedTime(ms, start, stop);
}

cudaError_t cudaEventQuery(cudaEvent_t event) {
    return hipEventQuery(event);
}

cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    return hipEventRecord(event, stream);
}

cudaError_t cudaEventSynchronize(cudaEvent_t event) {
    return hipEventSynchronize(event);
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
    cudaExtent ex = {};
    return ex;
}

cudaPitchedPtr make_cudaPitchedPtr(...) {
    cudaPitchedPtr ptr = {};
    return ptr;
}

cudaPos make_cudaPos(...) {
    cudaPos pos = {};
    return pos;
}


///////////////////////////////////////////////////////////////////////////////
// blas
///////////////////////////////////////////////////////////////////////////////

static hipblasOperation_t convert_hipblasOperation_t(hipblasOperation_t op) {
    return static_cast<hipblasOperation_t>(static_cast<int>(op) + 111);
}

// Context
cublasStatus_t cublasCreate(cublasHandle_t* handle) {
    return hipblasCreate(handle);
}

cublasStatus_t cublasDestroy(cublasHandle_t handle) {
    return hipblasDestroy(handle);
}

cublasStatus_t cublasGetVersion(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasSetPointerMode(cublasHandle_t handle, cublasPointerMode_t mode) {
    return hipblasSetPointerMode(handle, mode);
}

cublasStatus_t cublasGetPointerMode(cublasHandle_t handle, cublasPointerMode_t *mode) {
    return hipblasGetPointerMode(handle, mode);
}

// Stream
cublasStatus_t cublasSetStream(cublasHandle_t handle, cudaStream_t streamId) {
    return hipblasSetStream(handle, streamId);
}

cublasStatus_t cublasGetStream(cublasHandle_t handle, cudaStream_t *streamId) {
    return hipblasGetStream(handle, streamId);
}

// Math Mode
cublasStatus_t cublasSetMathMode(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasGetMathMode(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// BLAS Level 1
cublasStatus_t cublasIsamax(cublasHandle_t handle, int n, const float *x, int incx, int *result) {
    return hipblasIsamax(handle, n, x, incx, result);
}

cublasStatus_t cublasIsamin(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasSasum(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasSaxpy(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasDaxpy(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasSdot(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasDdot(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasCdotu(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasCdotc(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasZdotc(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasZdotu(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasSnrm2(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasSscal(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}


// BLAS Level 2
cublasStatus_t cublasSgemv(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasDgemv(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}


cublasStatus_t cublasCgemv(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasZgemv(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasSger(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasDger(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
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
cublasStatus_t cublasSgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                            int m, int n, int k, const float *alpha,
                            const float *A, int lda,
                            const float *B, int ldb,
                            const float *beta,
                            float *C, int ldc) {
    return hipblasSgemm(handle, convert_hipblasOperation_t(transa), convert_hipblasOperation_t(transb), m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

cublasStatus_t cublasDgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k, const double *alpha,
                           const double *A, int lda,
                           const double *B, int ldb,
                           const double *beta, double *C, int ldc) {
    return hipblasDgemm(handle, convert_hipblasOperation_t(transa), convert_hipblasOperation_t(transb), m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}


cublasStatus_t cublasCgemm(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasZgemm(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasSgemmBatched(
        cublasHandle_t handle,
        cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, int k,  const float *alpha,
        const float *A[], int lda,
        const float *B[], int ldb,
        const float *beta,
        float *C[], int ldc, int batchCount) {
    return hipblasSgemmBatched(handle, convert_hipblasOperation_t(transa), convert_hipblasOperation_t(transb), m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
}

cublasStatus_t cublasDgemmBatched(
        cublasHandle_t handle,
        cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, int k,  const double *alpha,
        const double *A[], int lda,
        const double *B[], int ldb,
        const double *beta,
        double *C[], int ldc, int batchCount) {
    return hipblasDgemmBatched(handle, convert_hipblasOperation_t(transa), convert_hipblasOperation_t(transb), m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
}

cublasStatus_t cublasCgemmBatched(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasZgemmBatched(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasSgemmEx(
        cublasHandle_t handle,
        cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, int k,
        const float *alpha, const void *A, cudaDataType_t Atype, int lda,
        const void *B, cudaDataType_t Btype, int ldb, const float *beta,
        void *C, cudaDataType_t Ctype, int ldc) {
    if (Atype != 0 || Btype != 0 || Ctype != 0) {
        return HIPBLAS_STATUS_NOT_SUPPORTED;
    }
    return hipblasSgemm(
        handle,
        convert_hipblasOperation_t(transa), convert_hipblasOperation_t(transb),
        m, n, k,
        alpha, static_cast<const float*>(A), lda,
        static_cast<const float*>(B), ldb, beta,
        static_cast<float*>(C), ldc);
}

cublasStatus_t cublasGemmEx(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasStrsm(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasDtrsm(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasCtrsm(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasZtrsm(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}


// BLAS extension
cublasStatus_t cublasSgeam(
        cublasHandle_t handle,
        cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, const float *alpha,
        const float *A, int lda, const float *beta, const float *B, int ldb,
        float *C, int ldc) {
    return hipblasSgeam(handle, convert_hipblasOperation_t(transa), convert_hipblasOperation_t(transb), m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}

cublasStatus_t cublasDgeam(
        cublasHandle_t handle,
        cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, const double *alpha,
        const double *A, int lda, const double *beta, const double *B, int ldb,
        double *C, int ldc) {
    return hipblasDgeam(handle, convert_hipblasOperation_t(transa), convert_hipblasOperation_t(transb), m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}

cublasStatus_t cublasSdgmm(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasSgetrfBatched(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasDgetrfBatched(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasCgetrfBatched(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasZgetrfBatched(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasSgetriBatched(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasDgetriBatched(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasCgetriBatched(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasZgetriBatched(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasSgemmStridedBatched(
        cublasHandle_t handle,
        cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, int k, const float *alpha,
        const float *A, int lda, long long bsa,
        const float *B, int ldb, long long bsb, const float *beta,
        float *C, int ldc, long long bsc, int batchCount) {
    return hipblasSgemmStridedBatched(
        handle,
        convert_hipblasOperation_t(transa), convert_hipblasOperation_t(transb),
        m, n, k, alpha,  A, lda, bsa, B, ldb, bsb, beta, C, ldc, bsc,
        batchCount);
}

cublasStatus_t cublasDgemmStridedBatched(
        cublasHandle_t handle,
        cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, int k, const double *alpha,
        const double *A, int lda, long long bsa,
        const double *B, int ldb, long long bsb, const double *beta,
        double *C, int ldc, long long bsc, int batchCount) {
    return hipblasDgemmStridedBatched(
        handle,
        convert_hipblasOperation_t(transa), convert_hipblasOperation_t(transb),
        m, n, k, alpha,  A, lda, bsa, B, ldb, bsb, beta, C, ldc, bsc,
        batchCount);
}

cublasStatus_t cublasCgemmStridedBatched(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasZgemmStridedBatched(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasStrttp(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasDtrttp(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasStpttr(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasDtpttr(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

///////////////////////////////////////////////////////////////////////////////
// rand
///////////////////////////////////////////////////////////////////////////////

typedef enum {} curandOrdering_t;
typedef hiprandRngType curandRngType_t;
typedef hiprandStatus_t curandStatus_t;

typedef hiprandGenerator_t curandGenerator_t;

curandRngType_t convert_hiprandRngType(curandRngType_t t) {
    switch(static_cast<int>(t)) {
    case 100: return HIPRAND_RNG_PSEUDO_DEFAULT;
    case 101: return HIPRAND_RNG_PSEUDO_XORWOW;
    case 121: return HIPRAND_RNG_PSEUDO_MRG32K3A;
    case 141: return HIPRAND_RNG_PSEUDO_MTGP32;
    case 142: return HIPRAND_RNG_PSEUDO_MT19937;
    case 161: return HIPRAND_RNG_PSEUDO_PHILOX4_32_10;
    case 200: return HIPRAND_RNG_QUASI_DEFAULT;
    case 201: return HIPRAND_RNG_QUASI_SOBOL32;
    case 202: return HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL32;
    case 203: return HIPRAND_RNG_QUASI_SOBOL64;
    case 204: return HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL64;
    }
    return HIPRAND_RNG_TEST;
}

// curandGenerator_t
curandStatus_t curandCreateGenerator(curandGenerator_t *generator, curandRngType_t rng_type) {
    rng_type = convert_hiprandRngType(rng_type);
    return hiprandCreateGenerator(generator, rng_type);
}

curandStatus_t curandDestroyGenerator(curandGenerator_t generator) {
    return hiprandDestroyGenerator(generator);
}

curandStatus_t curandGetVersion(int *version) {
    return hiprandGetVersion(version);
}


// Stream
curandStatus_t curandSetStream(curandGenerator_t generator, cudaStream_t stream) {
    return hiprandSetStream(generator, stream);
}

curandStatus_t curandSetPseudoRandomGeneratorSeed(curandGenerator_t generator, unsigned long long seed) {
    return hiprandSetPseudoRandomGeneratorSeed(generator, seed);
}

curandStatus_t curandSetGeneratorOffset(curandGenerator_t generator, unsigned long long offset) {
    return hiprandSetGeneratorOffset(generator, offset);
}

curandStatus_t curandSetGeneratorOrdering(...) {
    return HIPRAND_STATUS_NOT_IMPLEMENTED;
}


// Generation functions
curandStatus_t curandGenerate(curandGenerator_t generator, unsigned int *output_data, size_t n) {
    return hiprandGenerate(generator, output_data, n);
}

curandStatus_t curandGenerateLongLong(...) {
    return HIPRAND_STATUS_NOT_IMPLEMENTED;
}

curandStatus_t curandGenerateUniform(curandGenerator_t generator, float *output_data, size_t n) {
    return hiprandGenerateUniform(generator, output_data, n);
}

curandStatus_t curandGenerateUniformDouble(curandGenerator_t generator, double *output_data, size_t n) {
    return hiprandGenerateUniformDouble(generator, output_data, n);
}

curandStatus_t curandGenerateNormal(curandGenerator_t generator, float *output_data, size_t n, float mean, float stddev) {
    return hiprandGenerateNormal(generator, output_data, n, mean, stddev);
}

curandStatus_t curandGenerateNormalDouble(curandGenerator_t generator, double *output_data, size_t n, double mean, double stddev) {
    return hiprandGenerateNormalDouble(generator, output_data, n, mean, stddev);
}

curandStatus_t curandGenerateLogNormal(curandGenerator_t generator, float *output_data, size_t n, float mean, float stddev) {
    return hiprandGenerateLogNormal(generator, output_data, n, mean, stddev);
}

curandStatus_t curandGenerateLogNormalDouble(curandGenerator_t generator, double *output_data, size_t n, double mean, double stddev) {
    return hiprandGenerateLogNormalDouble(generator, output_data, n, mean, stddev);
}

curandStatus_t curandGeneratePoisson(curandGenerator_t generator, unsigned int *output_data, size_t n, double lambda) {
    return hiprandGeneratePoisson(generator, output_data, n, lambda);
}

///////////////////////////////////////////////////////////////////////////////
// cuda_profiler_api.h
///////////////////////////////////////////////////////////////////////////////

typedef enum {} cudaOutputMode_t;

cudaError_t cudaProfilerInitialize(...) {
  return cudaSuccess;
}

cudaError_t cudaProfilerStart() {
  return hipProfilerStart();
}

cudaError_t cudaProfilerStop() {
  return hipProfilerStop();
}

} // extern "C"

#endif // #ifndef INCLUDE_GUARD_CUPY_HIP_H
