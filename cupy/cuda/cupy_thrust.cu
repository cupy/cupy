#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include "cupy_common.h"
#include "cupy_thrust.h"

using namespace thrust;

template <typename T>
void cupy::thrust::stable_sort(void *start, ssize_t num) {
    device_ptr<T> dp_first = device_pointer_cast((T *)start);
    device_ptr<T> dp_last  = device_pointer_cast((T *)start + num);
    stable_sort< device_ptr<T> >(dp_first, dp_last);
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
