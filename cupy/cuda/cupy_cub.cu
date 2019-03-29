#include <cub/device/device_reduce.cuh>
#include "cupy_cub.h"

using namespace cub;

//
// **** cub_reduce_sum ****
//
template <typename T>
void _cub_reduce_sum(void *x, void *y, int num_items,
		     void *workspace, size_t &workspace_size)
{
    DeviceReduce::Sum(workspace, workspace_size, reinterpret_cast<T*>(x),
		      reinterpret_cast<T*>(y), num_items);
}

void cub_reduce_sum(void *x, void *y, int num_items,
		    void *workspace, size_t &workspace_size, int dtype_id)
{
    void (*f)(void*, void*, int, void*, size_t&);
    switch (dtype_id) {
    case CUPY_CUB_INT8:	   f = _cub_reduce_sum<char>; break;
    case CUPY_CUB_INT16:   f = _cub_reduce_sum<short>; break;
    case CUPY_CUB_INT32:   f = _cub_reduce_sum<int>; break;
    case CUPY_CUB_INT64:   f = _cub_reduce_sum<long>; break;
    case CUPY_CUB_UINT8:   f = _cub_reduce_sum<unsigned char>; break;
    case CUPY_CUB_UINT16:  f = _cub_reduce_sum<unsigned short>; break;
    case CUPY_CUB_UINT32:  f = _cub_reduce_sum<unsigned int>; break;
    case CUPY_CUB_UINT64:  f = _cub_reduce_sum<unsigned long>; break;
    case CUPY_CUB_FLOAT32: f = _cub_reduce_sum<float>; break;
    case CUPY_CUB_FLOAT64: f = _cub_reduce_sum<double>; break;
    default:
	std::cerr << "Unsupported dtype ID: " << dtype_id << std::endl;
	break;
    }
    (*f)(x, y, num_items, workspace, workspace_size);
}

size_t cub_reduce_sum_get_workspace_size(void *x, void *y, int num_items,
					 int dtype_id)
{
    size_t workspace_size;
    cub_reduce_sum(x, y, num_items, NULL, workspace_size, dtype_id);
    return workspace_size;
}

//
// **** cub_reduce_min ****
//
template <typename T>
void _cub_reduce_min(void *x, void *y, int num_items,
		     void *workspace, size_t &workspace_size)
{
    DeviceReduce::Min(workspace, workspace_size, reinterpret_cast<T*>(x),
		      reinterpret_cast<T*>(y), num_items);
}

void cub_reduce_min(void *x, void *y, int num_items,
		    void *workspace, size_t &workspace_size, int dtype_id)
{
    void (*f)(void*, void*, int, void*, size_t&);
    switch (dtype_id) {
    case CUPY_CUB_INT8:	   f = _cub_reduce_min<char>; break;
    case CUPY_CUB_INT16:   f = _cub_reduce_min<short>; break;
    case CUPY_CUB_INT32:   f = _cub_reduce_min<int>; break;
    case CUPY_CUB_INT64:   f = _cub_reduce_min<long>; break;
    case CUPY_CUB_UINT8:   f = _cub_reduce_min<unsigned char>; break;
    case CUPY_CUB_UINT16:  f = _cub_reduce_min<unsigned short>; break;
    case CUPY_CUB_UINT32:  f = _cub_reduce_min<unsigned int>; break;
    case CUPY_CUB_UINT64:  f = _cub_reduce_min<unsigned long>; break;
    case CUPY_CUB_FLOAT32: f = _cub_reduce_min<float>; break;
    case CUPY_CUB_FLOAT64: f = _cub_reduce_min<double>; break;
    default:
	std::cerr << "Unsupported dtype ID: " << dtype_id << std::endl;
	break;
    }
    (*f)(x, y, num_items, workspace, workspace_size);
}

size_t cub_reduce_min_get_workspace_size(void *x, void *y, int num_items,
					 int dtype_id)
{
    size_t workspace_size;
    cub_reduce_min(x, y, num_items, NULL, workspace_size, dtype_id);
    return workspace_size;
}

//
// **** cub_reduce_max ****
//
template <typename T>
void _cub_reduce_max(void *x, void *y, int num_items,
		     void *workspace, size_t &workspace_size)
{
    DeviceReduce::Max(workspace, workspace_size, reinterpret_cast<T*>(x),
		      reinterpret_cast<T*>(y), num_items);
}

void cub_reduce_max(void *x, void *y, int num_items,
		    void *workspace, size_t &workspace_size, int dtype_id)
{
    void (*f)(void*, void*, int, void*, size_t&);
    switch (dtype_id) {
    case CUPY_CUB_INT8:	   f = _cub_reduce_max<char>; break;
    case CUPY_CUB_INT16:   f = _cub_reduce_max<short>; break;
    case CUPY_CUB_INT32:   f = _cub_reduce_max<int>; break;
    case CUPY_CUB_INT64:   f = _cub_reduce_max<long>; break;
    case CUPY_CUB_UINT8:   f = _cub_reduce_max<unsigned char>; break;
    case CUPY_CUB_UINT16:  f = _cub_reduce_max<unsigned short>; break;
    case CUPY_CUB_UINT32:  f = _cub_reduce_max<unsigned int>; break;
    case CUPY_CUB_UINT64:  f = _cub_reduce_max<unsigned long>; break;
    case CUPY_CUB_FLOAT32: f = _cub_reduce_max<float>; break;
    case CUPY_CUB_FLOAT64: f = _cub_reduce_max<double>; break;
    default:
	std::cerr << "Unsupported dtype ID: " << dtype_id << std::endl;
	break;
    }
    (*f)(x, y, num_items, workspace, workspace_size);
}

size_t cub_reduce_max_get_workspace_size(void *x, void *y, int num_items,
					 int dtype_id)
{
    size_t workspace_size;
    cub_reduce_max(x, y, num_items, NULL, workspace_size, dtype_id);
    return workspace_size;
}
