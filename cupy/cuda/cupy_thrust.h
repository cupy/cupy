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

template void cupy::thrust::stable_sort<cpy_byte>(void *, ssize_t);
template void cupy::thrust::stable_sort<cpy_ubyte>(void *, ssize_t);
template void cupy::thrust::stable_sort<cpy_short>(void *, ssize_t);
template void cupy::thrust::stable_sort<cpy_ushort>(void *, ssize_t);
template void cupy::thrust::stable_sort<cpy_int>(void *, ssize_t);
template void cupy::thrust::stable_sort<cpy_uint>(void *, ssize_t);
template void cupy::thrust::stable_sort<cpy_long>(void *, ssize_t);
template void cupy::thrust::stable_sort<cpy_ulong>(void *, ssize_t);
template void cupy::thrust::stable_sort<cpy_float>(void *, ssize_t);
template void cupy::thrust::stable_sort<cpy_double>(void *, ssize_t);

} // namespace thrust

} // namespace cupy

#endif // #ifndef CUPY_NO_CUDA

#endif // INCLUDE_GUARD_CUPY_CUDA_THRUST_H
