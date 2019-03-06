#include <cub/device/device_reduce.cuh>
#include "cupy_common.h"
#include "cupy_cub.h"

using namespace cub;

//
// **** reduce_sum ****
//
template <typename T>
void cupy::cub::_reduce_sum(void *x, void *y, int num_items, void *workspace,
		 size_t workspace_size) {
    DeviceReduce::Sum(workspace, workspace_size, reinterpret_cast<T*>(x),
		      reinterpret_cast<T*>(y), num_items);
}

template <typename T>
size_t cupy::cub::_reduce_sum_get_workspace_size(void *x, void *y, int num_items) {
    size_t workspace_size;
    DeviceReduce::Sum(NULL, workspace_size, reinterpret_cast<T*>(x),
		      reinterpret_cast<T*>(y), num_items);
    return workspace_size;
}

template void cupy::cub::_reduce_sum<cpy_byte>(void *, void *, int, void *, size_t);
template void cupy::cub::_reduce_sum<cpy_ubyte>(void *, void *, int, void *, size_t);
template void cupy::cub::_reduce_sum<cpy_short>(void *, void *, int, void *, size_t);
template void cupy::cub::_reduce_sum<cpy_ushort>(void *, void *, int, void *, size_t);
template void cupy::cub::_reduce_sum<cpy_int>(void *, void *, int, void *, size_t);
template void cupy::cub::_reduce_sum<cpy_uint>(void *, void *, int, void *, size_t);
template void cupy::cub::_reduce_sum<cpy_long>(void *, void *, int, void *, size_t);
template void cupy::cub::_reduce_sum<cpy_ulong>(void *, void *, int, void *, size_t);
template void cupy::cub::_reduce_sum<cpy_float>(void *, void *, int, void *, size_t);
template void cupy::cub::_reduce_sum<cpy_double>(void *, void *, int, void *, size_t);

template size_t cupy::cub::_reduce_sum_get_workspace_size<cpy_byte>(void *, void *, int);
template size_t cupy::cub::_reduce_sum_get_workspace_size<cpy_ubyte>(void *, void *, int);
template size_t cupy::cub::_reduce_sum_get_workspace_size<cpy_short>(void *, void *, int);
template size_t cupy::cub::_reduce_sum_get_workspace_size<cpy_ushort>(void *, void *, int);
template size_t cupy::cub::_reduce_sum_get_workspace_size<cpy_int>(void *, void *, int);
template size_t cupy::cub::_reduce_sum_get_workspace_size<cpy_uint>(void *, void *, int);
template size_t cupy::cub::_reduce_sum_get_workspace_size<cpy_long>(void *, void *, int);
template size_t cupy::cub::_reduce_sum_get_workspace_size<cpy_ulong>(void *, void *, int);
template size_t cupy::cub::_reduce_sum_get_workspace_size<cpy_float>(void *, void *, int);
template size_t cupy::cub::_reduce_sum_get_workspace_size<cpy_double>(void *, void *, int);

//
// **** reduce_min ****
//
template <typename T>
void cupy::cub::_reduce_min(void *x, void *y, int num_items, void *workspace,
		 size_t workspace_size) {
    DeviceReduce::Min(workspace, workspace_size, reinterpret_cast<T*>(x),
		      reinterpret_cast<T*>(y), num_items);
}

template <typename T>
size_t cupy::cub::_reduce_min_get_workspace_size(void *x, void *y, int num_items) {
    size_t workspace_size;
    DeviceReduce::Min(NULL, workspace_size, reinterpret_cast<T*>(x),
		      reinterpret_cast<T*>(y), num_items);
    return workspace_size;
}

template void cupy::cub::_reduce_min<cpy_byte>(void *, void *, int, void *, size_t);
template void cupy::cub::_reduce_min<cpy_ubyte>(void *, void *, int, void *, size_t);
template void cupy::cub::_reduce_min<cpy_short>(void *, void *, int, void *, size_t);
template void cupy::cub::_reduce_min<cpy_ushort>(void *, void *, int, void *, size_t);
template void cupy::cub::_reduce_min<cpy_int>(void *, void *, int, void *, size_t);
template void cupy::cub::_reduce_min<cpy_uint>(void *, void *, int, void *, size_t);
template void cupy::cub::_reduce_min<cpy_long>(void *, void *, int, void *, size_t);
template void cupy::cub::_reduce_min<cpy_ulong>(void *, void *, int, void *, size_t);
template void cupy::cub::_reduce_min<cpy_float>(void *, void *, int, void *, size_t);
template void cupy::cub::_reduce_min<cpy_double>(void *, void *, int, void *, size_t);

template size_t cupy::cub::_reduce_min_get_workspace_size<cpy_byte>(void *, void *, int);
template size_t cupy::cub::_reduce_min_get_workspace_size<cpy_ubyte>(void *, void *, int);
template size_t cupy::cub::_reduce_min_get_workspace_size<cpy_short>(void *, void *, int);
template size_t cupy::cub::_reduce_min_get_workspace_size<cpy_ushort>(void *, void *, int);
template size_t cupy::cub::_reduce_min_get_workspace_size<cpy_int>(void *, void *, int);
template size_t cupy::cub::_reduce_min_get_workspace_size<cpy_uint>(void *, void *, int);
template size_t cupy::cub::_reduce_min_get_workspace_size<cpy_long>(void *, void *, int);
template size_t cupy::cub::_reduce_min_get_workspace_size<cpy_ulong>(void *, void *, int);
template size_t cupy::cub::_reduce_min_get_workspace_size<cpy_float>(void *, void *, int);
template size_t cupy::cub::_reduce_min_get_workspace_size<cpy_double>(void *, void *, int);

//
// **** reduce_max ****
//
template <typename T>
void cupy::cub::_reduce_max(void *x, void *y, int num_items, void *workspace,
		 size_t workspace_size) {
    DeviceReduce::Max(workspace, workspace_size, reinterpret_cast<T*>(x),
		      reinterpret_cast<T*>(y), num_items);
}

template <typename T>
size_t cupy::cub::_reduce_max_get_workspace_size(void *x, void *y, int num_items) {
    size_t workspace_size;
    DeviceReduce::Max(NULL, workspace_size, reinterpret_cast<T*>(x),
		      reinterpret_cast<T*>(y), num_items);
    return workspace_size;
}

template void cupy::cub::_reduce_max<cpy_byte>(void *, void *, int, void *, size_t);
template void cupy::cub::_reduce_max<cpy_ubyte>(void *, void *, int, void *, size_t);
template void cupy::cub::_reduce_max<cpy_short>(void *, void *, int, void *, size_t);
template void cupy::cub::_reduce_max<cpy_ushort>(void *, void *, int, void *, size_t);
template void cupy::cub::_reduce_max<cpy_int>(void *, void *, int, void *, size_t);
template void cupy::cub::_reduce_max<cpy_uint>(void *, void *, int, void *, size_t);
template void cupy::cub::_reduce_max<cpy_long>(void *, void *, int, void *, size_t);
template void cupy::cub::_reduce_max<cpy_ulong>(void *, void *, int, void *, size_t);
template void cupy::cub::_reduce_max<cpy_float>(void *, void *, int, void *, size_t);
template void cupy::cub::_reduce_max<cpy_double>(void *, void *, int, void *, size_t);

template size_t cupy::cub::_reduce_max_get_workspace_size<cpy_byte>(void *, void *, int);
template size_t cupy::cub::_reduce_max_get_workspace_size<cpy_ubyte>(void *, void *, int);
template size_t cupy::cub::_reduce_max_get_workspace_size<cpy_short>(void *, void *, int);
template size_t cupy::cub::_reduce_max_get_workspace_size<cpy_ushort>(void *, void *, int);
template size_t cupy::cub::_reduce_max_get_workspace_size<cpy_int>(void *, void *, int);
template size_t cupy::cub::_reduce_max_get_workspace_size<cpy_uint>(void *, void *, int);
template size_t cupy::cub::_reduce_max_get_workspace_size<cpy_long>(void *, void *, int);
template size_t cupy::cub::_reduce_max_get_workspace_size<cpy_ulong>(void *, void *, int);
template size_t cupy::cub::_reduce_max_get_workspace_size<cpy_float>(void *, void *, int);
template size_t cupy::cub::_reduce_max_get_workspace_size<cpy_double>(void *, void *, int);
