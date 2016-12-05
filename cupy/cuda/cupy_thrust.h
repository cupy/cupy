#ifndef INCLUDE_GUARD_CUPY_CUDA_THRUST_H
#define INCLUDE_GUARD_CUPY_CUDA_THRUST_H

#ifndef CUPY_NO_CUDA

namespace cupy {

namespace thrust {

template <typename T> void stable_sort(void *, ssize_t);

} // namespace thrust

} // namespace cupy

#else // CUPY_NO_CUDA

#include "cupy_common.h"

namespace cupy {

namespace thrust {

template <typename T>
void stable_sort(void *, ssize_t) {
    return;
}

} // namespace thrust

} // namespace cupy

#endif // #ifndef CUPY_NO_CUDA

#endif // INCLUDE_GUARD_CUPY_CUDA_THRUST_H
