#include <cupy/complex.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_segmented_reduce.cuh>
#include "cupy_cub.h"
#include <stdexcept>

using namespace cub;

// Minimum boilerplate to support complex numbers in sum(), min(), and max():
// - This works only because all data fields in the *Traits struct are not
//   used in <cub/device/device_reduce.cuh>.
// - DO NOT USE THIS STUB for supporting CUB sorting!!!!!!
// - The Max() and Lowest() below are chosen to comply with NumPy's lexical
//   ordering; note that std::numeric_limits<T> does not support complex
//   numbers as in general the comparison is ill defined.
template <>
struct FpLimits<complex<float>>
{
    static __host__ __device__ __forceinline__ complex<float> Max() {
        return (complex<float>(FLT_MAX, FLT_MAX));
    }

    static __host__ __device__ __forceinline__ complex<float> Lowest() {
        return (complex<float>(FLT_MAX * float(-1), FLT_MAX * float(-1)));
    }
};

template <>
struct FpLimits<complex<double>>
{
    static __host__ __device__ __forceinline__ complex<double> Max() {
        return (complex<double>(DBL_MAX, DBL_MAX));
    }

    static __host__ __device__ __forceinline__ complex<double> Lowest() {
        return (complex<double>(DBL_MAX * double(-1), DBL_MAX * double(-1)));
    }
};

template <> struct NumericTraits<complex<float>>  : BaseTraits<FLOATING_POINT, true, false, unsigned int, complex<float>> {};
template <> struct NumericTraits<complex<double>> : BaseTraits<FLOATING_POINT, true, false, unsigned long long, complex<double>> {};
// end of boilerplate


//
// **** dtype_dispatcher ****
//
// This is implemented with reference to the following implementation.
// https://github.com/rapidsai/cudf/blob/branch-0.6/cpp/src/utilities/type_dispatcher.hpp
//
template <class functor_t, typename... Ts>
void dtype_dispatcher(int dtype_id, functor_t f, Ts&&... args)
{
    switch (dtype_id) {
    case CUPY_CUB_INT8:	      return f.template operator()<char>(std::forward<Ts>(args)...);
    case CUPY_CUB_INT16:      return f.template operator()<short>(std::forward<Ts>(args)...);
    case CUPY_CUB_INT32:      return f.template operator()<int>(std::forward<Ts>(args)...);
    case CUPY_CUB_INT64:      return f.template operator()<long>(std::forward<Ts>(args)...);
    case CUPY_CUB_UINT8:      return f.template operator()<unsigned char>(std::forward<Ts>(args)...);
    case CUPY_CUB_UINT16:     return f.template operator()<unsigned short>(std::forward<Ts>(args)...);
    case CUPY_CUB_UINT32:     return f.template operator()<unsigned int>(std::forward<Ts>(args)...);
    case CUPY_CUB_UINT64:     return f.template operator()<unsigned long>(std::forward<Ts>(args)...);
    case CUPY_CUB_FLOAT32:    return f.template operator()<float>(std::forward<Ts>(args)...);
    case CUPY_CUB_FLOAT64:    return f.template operator()<double>(std::forward<Ts>(args)...);
    case CUPY_CUB_COMPLEX64:  return f.template operator()<complex<float>>(std::forward<Ts>(args)...);
    case CUPY_CUB_COMPLEX128: return f.template operator()<complex<double>>(std::forward<Ts>(args)...);
    default:
	throw std::runtime_error("Unsupported dtype ID");
    }
}

//
// **** CUB Sum ****
//
struct _cub_reduce_sum {
    template <typename T>
    void operator()(void* workspace, size_t& workspace_size, void* x, void* y,
        int num_items, cudaStream_t s)
    {
        DeviceReduce::Sum(workspace, workspace_size, static_cast<T*>(x),
            static_cast<T*>(y), num_items, s);
    }
};

struct _cub_segmented_reduce_sum {
    template <typename T>
    void operator()(void* workspace, size_t& workspace_size, void* x, void* y,
        int num_segments, void* offset_start, void* offset_end, cudaStream_t s)
    {
        DeviceSegmentedReduce::Sum(workspace, workspace_size,
            static_cast<T*>(x), static_cast<T*>(y), num_segments,
            static_cast<int*>(offset_start),
            static_cast<int*>(offset_end), s);
    }
};

//
// **** CUB Min ****
//
struct _cub_reduce_min {
    template <typename T>
    void operator()(void* workspace, size_t& workspace_size, void* x, void* y,
        int num_items, cudaStream_t s)
    {
        DeviceReduce::Min(workspace, workspace_size, static_cast<T*>(x),
            static_cast<T*>(y), num_items, s);
    }
};

struct _cub_segmented_reduce_min {
    template <typename T>
    void operator()(void* workspace, size_t& workspace_size, void* x, void* y,
        int num_segments, void* offset_start, void* offset_end, cudaStream_t s)
    {
        DeviceSegmentedReduce::Min(workspace, workspace_size,
            static_cast<T*>(x), static_cast<T*>(y), num_segments,
            static_cast<int*>(offset_start),
            static_cast<int*>(offset_end), s);
    }
};

//
// **** CUB Max ****
//
struct _cub_reduce_max {
    template <typename T>
    void operator()(void* workspace, size_t& workspace_size, void* x, void* y,
        int num_items, cudaStream_t s)
    {
        DeviceReduce::Max(workspace, workspace_size, static_cast<T*>(x),
            static_cast<T*>(y), num_items, s);
    }
};

struct _cub_segmented_reduce_max {
    template <typename T>
    void operator()(void* workspace, size_t& workspace_size, void* x, void* y,
        int num_segments, void* offset_start, void* offset_end, cudaStream_t s)
    {
        DeviceSegmentedReduce::Max(workspace, workspace_size,
            static_cast<T*>(x), static_cast<T*>(y), num_segments,
            static_cast<int*>(offset_start),
            static_cast<int*>(offset_end), s);
    }
};

//
// APIs exposed to CuPy
//

/* -------- device reduce -------- */

void cub_device_reduce(void* workspace, size_t& workspace_size, void* x, void* y,
    int num_items, cudaStream_t stream, int op, int dtype_id)
{
    switch(op) {
    case CUPY_CUB_SUM:  return dtype_dispatcher(dtype_id, _cub_reduce_sum(),
                            workspace, workspace_size, x, y, num_items, stream);
    case CUPY_CUB_MIN:  return dtype_dispatcher(dtype_id, _cub_reduce_min(),
                            workspace, workspace_size, x, y, num_items, stream);
    case CUPY_CUB_MAX:  return dtype_dispatcher(dtype_id, _cub_reduce_max(),
                            workspace, workspace_size, x, y, num_items, stream);
    default:            throw std::runtime_error("Unsupported operation");
    }
}

size_t cub_device_reduce_get_workspace_size(void* x, void* y, int num_items,
    cudaStream_t stream, int op, int dtype_id)
{
    size_t workspace_size = 0;
    cub_device_reduce(NULL, workspace_size, x, y, num_items, stream,
                      op, dtype_id);
    return workspace_size;
}

/* -------- device segmented reduce -------- */

void cub_device_segmented_reduce(void* workspace, size_t& workspace_size,
    void* x, void* y, int num_segments, void* offset_start, void* offset_end,
    cudaStream_t stream, int op, int dtype_id)
{
    switch(op) {
    case CUPY_CUB_SUM:
        return dtype_dispatcher(dtype_id, _cub_segmented_reduce_sum(),
                   workspace, workspace_size, x, y, num_segments, offset_start,
                   offset_end, stream);
    case CUPY_CUB_MIN:
        return dtype_dispatcher(dtype_id, _cub_segmented_reduce_min(),
                   workspace, workspace_size, x, y, num_segments, offset_start,
                   offset_end, stream);
    case CUPY_CUB_MAX:
        return dtype_dispatcher(dtype_id, _cub_segmented_reduce_max(),
                   workspace, workspace_size, x, y, num_segments, offset_start,
                   offset_end, stream);
    default:
        throw std::runtime_error("Unsupported operation");
    }
}

size_t cub_device_segmented_reduce_get_workspace_size(void* x, void* y,
    int num_segments, void* offset_start, void* offset_end,
    cudaStream_t stream, int op, int dtype_id)
{
    size_t workspace_size = 0;
    cub_device_segmented_reduce(NULL, workspace_size, x, y, num_segments,
                                offset_start, offset_end, stream,
                                op, dtype_id);
    return workspace_size;
}
