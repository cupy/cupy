#ifndef INCLUDE_GUARD_CUPY_CUDA_CUB_H
#define INCLUDE_GUARD_CUPY_CUDA_CUB_H

#define CUPY_CUB_INT8        0
#define CUPY_CUB_UINT8       1
#define CUPY_CUB_INT16       2
#define CUPY_CUB_UINT16      3
#define CUPY_CUB_INT32       4
#define CUPY_CUB_UINT32      5
#define CUPY_CUB_INT64       6
#define CUPY_CUB_UINT64      7
#define CUPY_CUB_FLOAT16     8
#define CUPY_CUB_FLOAT32     9
#define CUPY_CUB_FLOAT64    10
#define CUPY_CUB_COMPLEX64  11
#define CUPY_CUB_COMPLEX128 12

#define CUPY_CUB_SUM     0
#define CUPY_CUB_MIN     1
#define CUPY_CUB_MAX     2
#define CUPY_CUB_ARGMIN  3
#define CUPY_CUB_ARGMAX  4
#define CUPY_CUB_CUMSUM  5
#define CUPY_CUB_CUMPROD 6
#define CUPY_CUB_PROD    7

#ifndef CUPY_NO_CUDA
#include <cuda_runtime.h>  // for cudaStream_t

void cub_device_reduce(void*, size_t&, void*, void*, int, cudaStream_t, int, int);
void cub_device_segmented_reduce(void*, size_t&, void*, void*, int, void*, void*, cudaStream_t, int, int);
void cub_device_spmv(void*, size_t&, void*, void*, void*, void*, void*, int, int, int, cudaStream_t, int);
void cub_device_scan(void*, size_t&, void*, void*, int, cudaStream_t, int, int);
size_t cub_device_reduce_get_workspace_size(void*, void*, int, cudaStream_t, int, int);
size_t cub_device_segmented_reduce_get_workspace_size(void*, void*, int, void*, void*, cudaStream_t, int, int);
size_t cub_device_spmv_get_workspace_size(void*, void*, void*, void*, void*, int, int, int, cudaStream_t, int);
size_t cub_device_scan_get_workspace_size(void*, void*, int, cudaStream_t, int, int);

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

#endif // #ifndef CUPY_NO_CUDA

#endif // #ifndef INCLUDE_GUARD_CUPY_CUDA_CUB_H
