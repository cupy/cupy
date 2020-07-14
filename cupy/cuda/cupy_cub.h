#ifndef INCLUDE_GUARD_CUPY_CUDA_CUB_H
#define INCLUDE_GUARD_CUPY_CUDA_CUB_H

#define CUPY_CUB_SUM     0
#define CUPY_CUB_MIN     1
#define CUPY_CUB_MAX     2
#define CUPY_CUB_ARGMIN  3
#define CUPY_CUB_ARGMAX  4
#define CUPY_CUB_CUMSUM  5
#define CUPY_CUB_CUMPROD 6
#define CUPY_CUB_PROD    7

// this is defined during the build process
#ifndef CUPY_CUB_VERSION_CODE
#define CUPY_CUB_VERSION_CODE 0
#endif

#ifndef CUPY_NO_CUDA
#include <cuda_runtime.h>  // for cudaStream_t

void cub_device_reduce(void*, size_t&, void*, void*, int, cudaStream_t, int, int);
void cub_device_segmented_reduce(void*, size_t&, void*, void*, int, void*, void*, cudaStream_t, int, int);
void cub_device_spmv(void*, size_t&, void*, void*, void*, void*, void*, int, int, int, cudaStream_t, int);
void cub_device_scan(void*, size_t&, void*, void*, int, cudaStream_t, int, int);
void cub_device_histogram_range(void*, size_t&, void*, void*, int, void*, size_t, cudaStream_t, int);
size_t cub_device_reduce_get_workspace_size(void*, void*, int, cudaStream_t, int, int);
size_t cub_device_segmented_reduce_get_workspace_size(void*, void*, int, void*, void*, cudaStream_t, int, int);
size_t cub_device_spmv_get_workspace_size(void*, void*, void*, void*, void*, int, int, int, cudaStream_t, int);
size_t cub_device_scan_get_workspace_size(void*, void*, int, cudaStream_t, int, int);
size_t cub_device_histogram_range_get_workspace_size(void*, void*, int, void*, size_t, cudaStream_t, int);

// This is for CUB's HistogramRange
#ifdef __CUDA_ARCH__
__device__ long long atomicAdd(long long *address, long long val) {
    return atomicAdd(reinterpret_cast<unsigned long long*>(address),
                     static_cast<unsigned long long>(val));
}
#endif // __CUDA_ARCH__

#else // CUPY_NO_CUDA

typedef struct CUstream_st *cudaStream_t;

void cub_device_reduce(...) {
}

void cub_device_segmented_reduce(...) {
}

void cub_device_spmv(...) {
}

void cub_device_scan(...) {
}

void cub_device_histogram_range(...) {
}

size_t cub_device_reduce_get_workspace_size(...) {
    return 0;
}

size_t cub_device_segmented_reduce_get_workspace_size(...) {
    return 0;
}

size_t cub_device_spmv_get_workspace_size(...) {
    return 0;
}

size_t cub_device_scan_get_workspace_size(...) {
    return 0;
}

size_t cub_device_histogram_range_get_workspace_size(...) {
    return 0;
}

#endif // #ifndef CUPY_NO_CUDA

#endif // #ifndef INCLUDE_GUARD_CUPY_CUDA_CUB_H
