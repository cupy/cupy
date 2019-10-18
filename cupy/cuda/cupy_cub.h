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

#ifndef CUPY_NO_CUDA

void cub_reduce_sum(void *, void *, int, void *, size_t &, int);
void cub_reduce_min(void *, void *, int, void *, size_t &, int);
void cub_reduce_max(void *, void *, int, void *, size_t &, int);

size_t cub_reduce_sum_get_workspace_size(void *, void *, int, int);
size_t cub_reduce_min_get_workspace_size(void *, void *, int, int);
size_t cub_reduce_max_get_workspace_size(void *, void *, int, int);

#else // CUPY_NO_CUDA

void cub_reduce_sum(void *, void *, int, void *, size_t &, int) {
    return;
}

void cub_reduce_min(void *, void *, int, void *, size_t &, int) {
    return;
}

void cub_reduce_max(void *, void *, int, void *, size_t &, int) {
    return;
}

size_t cub_reduce_sum_get_workspace_size(void *, void *, int, int) {
    return 0;
}

size_t cub_reduce_min_get_workspace_size(void *, void *, int, int) {
    return 0;
}

size_t cub_reduce_max_get_workspace_size(void *, void *, int, int) {
    return 0;
}

#endif // #ifndef CUPY_NO_CUDA

#endif // #ifndef INCLUDE_GUARD_CUPY_CUDA_CUB_H
