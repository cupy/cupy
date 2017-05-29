#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include "cupy_common.h"
#include "cupy_thrust.h"

using namespace thrust;


/*
 * sort
 */

template <typename T>
void cupy::thrust::_sort(void *start, ptrdiff_t num) {
    device_ptr<T> dp_first = device_pointer_cast((T *)start);
    device_ptr<T> dp_last  = device_pointer_cast((T *)start + num);
    stable_sort< device_ptr<T> >(dp_first, dp_last);
}

template void cupy::thrust::_sort<cpy_byte>(void *, ptrdiff_t);
template void cupy::thrust::_sort<cpy_ubyte>(void *, ptrdiff_t);
template void cupy::thrust::_sort<cpy_short>(void *, ptrdiff_t);
template void cupy::thrust::_sort<cpy_ushort>(void *, ptrdiff_t);
template void cupy::thrust::_sort<cpy_int>(void *, ptrdiff_t);
template void cupy::thrust::_sort<cpy_uint>(void *, ptrdiff_t);
template void cupy::thrust::_sort<cpy_long>(void *, ptrdiff_t);
template void cupy::thrust::_sort<cpy_ulong>(void *, ptrdiff_t);
template void cupy::thrust::_sort<cpy_float>(void *, ptrdiff_t);
template void cupy::thrust::_sort<cpy_double>(void *, ptrdiff_t);


/*
 * argsort
 */

template <typename T>
class elem_less {
public:
    elem_less(void *data):_data((const T*)data) {}
    __device__ bool operator()(size_t i, size_t j) { return _data[i] < _data[j]; }
private:
    const T *_data;
};

template <typename T>
void cupy::thrust::_argsort(size_t *idx_start, void *data_start, size_t num) {
    /* idx_start is the beggining of the output array where the indexes that
       would sort the data will be placed. The original contents of idx_start
       will be destroyed. */
    device_ptr<size_t> dp_first = device_pointer_cast(idx_start);
    device_ptr<size_t> dp_last  = device_pointer_cast(idx_start + num);
    sequence(dp_first, dp_last);
    stable_sort< device_ptr<size_t> >(dp_first, dp_last, elem_less<T>(data_start));
}

template void cupy::thrust::_argsort<cpy_byte>(size_t *, void *, size_t);
template void cupy::thrust::_argsort<cpy_ubyte>(size_t *, void *, size_t);
template void cupy::thrust::_argsort<cpy_short>(size_t *, void *, size_t);
template void cupy::thrust::_argsort<cpy_ushort>(size_t *, void *, size_t);
template void cupy::thrust::_argsort<cpy_int>(size_t *, void *, size_t);
template void cupy::thrust::_argsort<cpy_uint>(size_t *, void *, size_t);
template void cupy::thrust::_argsort<cpy_long>(size_t *, void *, size_t);
template void cupy::thrust::_argsort<cpy_ulong>(size_t *, void *, size_t);
template void cupy::thrust::_argsort<cpy_float>(size_t *, void *, size_t);
template void cupy::thrust::_argsort<cpy_double>(size_t *, void *, size_t);
