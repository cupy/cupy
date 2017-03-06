#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include "cupy_common.h"
#include "cupy_thrust.h"

using namespace thrust;

template <typename T>
void cupy::thrust::sort(void *start, ptrdiff_t num) {
    device_ptr<T> dp_first = device_pointer_cast((T *)start);
    device_ptr<T> dp_last  = device_pointer_cast((T *)start + num);
    stable_sort< device_ptr<T> >(dp_first, dp_last);
}

template void cupy::thrust::sort<cpy_byte>(void *, ptrdiff_t);
template void cupy::thrust::sort<cpy_ubyte>(void *, ptrdiff_t);
template void cupy::thrust::sort<cpy_short>(void *, ptrdiff_t);
template void cupy::thrust::sort<cpy_ushort>(void *, ptrdiff_t);
template void cupy::thrust::sort<cpy_int>(void *, ptrdiff_t);
template void cupy::thrust::sort<cpy_uint>(void *, ptrdiff_t);
template void cupy::thrust::sort<cpy_long>(void *, ptrdiff_t);
template void cupy::thrust::sort<cpy_ulong>(void *, ptrdiff_t);
template void cupy::thrust::sort<cpy_float>(void *, ptrdiff_t);
template void cupy::thrust::sort<cpy_double>(void *, ptrdiff_t);
