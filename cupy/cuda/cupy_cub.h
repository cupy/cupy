#ifndef INCLUDE_GUARD_CUPY_CUDA_CUB_H
#define INCLUDE_GUARD_CUPY_CUDA_CUB_H

#ifndef CUPY_NO_CUDA

namespace cupy {

namespace cub {

template <typename T>
void _reduce_sum(void *, void *, int, void *, size_t);

template <typename T>
void _reduce_min(void *, void *, int, void *, size_t);

template <typename T>
void _reduce_max(void *, void *, int, void *, size_t);

template <typename T>
size_t _reduce_sum_get_workspace_size(void *, void *, int);

template <typename T>
size_t _reduce_min_get_workspace_size(void *, void *, int);

template <typename T>
size_t _reduce_max_get_workspace_size(void *, void *, int);

} // namespace cub

} // namespace cupy

#else // CUPY_NO_CUDA

#include "cupy_common.h"

namespace cupy {

namespace cub {

template <typename T>
void _reduce_sum(void *, void *, int, void *, size_t) {
    return;
}

template <typename T>
void _reduce_min(void *, void *, int, void *, size_t) {
    return;
}

template <typename T>
void _reduce_max(void *, void *, int, void *, size_t) {
    return;
}

template <typename T>
size_t _reduce_sum_get_workspace_size(void *, void *, int) {
    return 0;
}

template <typename T>
size_t _reduce_min_get_workspace_size(void *, void *, int) {
    return 0;
}

template <typename T>
size_t _reduce_max_get_workspace_size(void *, void *, int) {
    return 0;
}

} // namespace cub

} // namespace cupy

#endif // #ifndef CUPY_NO_CUDA

#endif // #ifndef INCLUDE_GUARD_CUPY_CUDA_CUB_H
