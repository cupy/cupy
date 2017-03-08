#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include "cupy_common.h"
#include "cupy_thrust.h"

using namespace thrust;

template <typename T>
void cupy::thrust::stable_sort(T *start, T *last) {
    device_ptr<T> dp_first = device_pointer_cast(start);
    device_ptr<T> dp_last  = device_pointer_cast(last);
    stable_sort< device_ptr<T> >(dp_first, dp_last);
}

template void cupy::thrust::stable_sort<cpy_byte>(cpy_byte *, cpy_byte *);
template void cupy::thrust::stable_sort<cpy_ubyte>(cpy_ubyte *, cpy_ubyte *);
template void cupy::thrust::stable_sort<cpy_short>(cpy_short *, cpy_short *);
template void cupy::thrust::stable_sort<cpy_ushort>(cpy_ushort *, cpy_ushort *);
template void cupy::thrust::stable_sort<cpy_int>(cpy_int *, cpy_int *);
template void cupy::thrust::stable_sort<cpy_uint>(cpy_uint *, cpy_uint *);
template void cupy::thrust::stable_sort<cpy_long>(cpy_long *, cpy_long *);
template void cupy::thrust::stable_sort<cpy_ulong>(cpy_ulong *, cpy_ulong *);
template void cupy::thrust::stable_sort<cpy_float>(cpy_float *, cpy_float *);
template void cupy::thrust::stable_sort<cpy_double>(cpy_double *, cpy_double *);
