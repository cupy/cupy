#include <cupy/complex.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_segmented_reduce.cuh>
#include <cub/device/device_spmv.cuh>
#include "cupy_cub.h"
#include <stdexcept>

using namespace cub;

/* ------------------------------------ Minimum boilerplate to support complex numbers ------------------------------------ */
// - This works only because all data fields in the *Traits struct are not
//   used in <cub/device/device_reduce.cuh>.
// - The Max() and Lowest() below are chosen to comply with NumPy's lexical
//   ordering; note that std::numeric_limits<T> does not support complex
//   numbers as in general the comparison is ill defined.
// - DO NOT USE THIS STUB for supporting CUB sorting!!!!!!
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
/* ------------------------------------ end of boilerplate ------------------------------------ */


/* ------------------------------------ "Patches" to CUB ------------------------------------
   These stubs are needed because CUB does not handle NaNs properly, while NumPy has certain
   behaviors with which we must comply.
   TODO(leofang): support half precision?
*/

//
// Max()
//

// specialization for float for handling NaNs
template <>
__host__ __device__ __forceinline__ float Max::operator()(const float &a, const float &b) const
{
    // NumPy behavior: NaN is always chosen!
    if (isnan(a)) {return a;}
    else if (isnan(b)) {return b;}
    else {return CUB_MAX(a, b);}
}

// specialization for double for handling NaNs
template <>
__host__ __device__ __forceinline__ double Max::operator()(const double &a, const double &b) const
{
    // NumPy behavior: NaN is always chosen!
    if (isnan(a)) {return a;}
    else if (isnan(b)) {return b;}
    else {return CUB_MAX(a, b);}
}

// specialization for complex<float> for handling NaNs
template <>
__host__ __device__ __forceinline__ complex<float> Max::operator()(const complex<float> &a, const complex<float> &b) const
{
    // - TODO(leofang): just call max() here when the bug in cupy/complex.cuh is fixed
    // - NumPy behavior: If both a and b contain NaN, the first argument is chosen
    // - isnan() and max() are defined in cupy/complex.cuh
    if (isnan(a)) {return a;}
    else if (isnan(b)) {return b;}
    else {return max(a, b);}
}

// specialization for complex<double> for handling NaNs
template <>
__host__ __device__ __forceinline__ complex<double> Max::operator()(const complex<double> &a, const complex<double> &b) const
{
    // - TODO(leofang): just call max() here when the bug in cupy/complex.cuh is fixed
    // - NumPy behavior: If both a and b contain NaN, the first argument is chosen
    // - isnan() and max() are defined in cupy/complex.cuh
    if (isnan(a)) {return a;}
    else if (isnan(b)) {return b;}
    else {return max(a, b);}
}

//
// Min()
//

// specialization for float for handling NaNs
template <>
__host__ __device__ __forceinline__ float Min::operator()(const float &a, const float &b) const
{
    // NumPy behavior: NaN is always chosen!
    if (isnan(a)) {return a;}
    else if (isnan(b)) {return b;}
    else {return CUB_MIN(a, b);}
}

// specialization for double for handling NaNs
template <>
__host__ __device__ __forceinline__ double Min::operator()(const double &a, const double &b) const
{
    // NumPy behavior: NaN is always chosen!
    if (isnan(a)) {return a;}
    else if (isnan(b)) {return b;}
    else {return CUB_MIN(a, b);}
}

// specialization for complex<float> for handling NaNs
template <>
__host__ __device__ __forceinline__ complex<float> Min::operator()(const complex<float> &a, const complex<float> &b) const
{
    // - TODO(leofang): just call min() here when the bug in cupy/complex.cuh is fixed
    // - NumPy behavior: If both a and b contain NaN, the first argument is chosen
    // - isnan() and min() are defined in cupy/complex.cuh
    if (isnan(a)) {return a;}
    else if (isnan(b)) {return b;}
    else {return min(a, b);}
}

// specialization for complex<double> for handling NaNs
template <>
__host__ __device__ __forceinline__ complex<double> Min::operator()(const complex<double> &a, const complex<double> &b) const
{
    // - TODO(leofang): just call min() here when the bug in cupy/complex.cuh is fixed
    // - NumPy behavior: If both a and b contain NaN, the first argument is chosen
    // - isnan() and min() are defined in cupy/complex.cuh
    if (isnan(a)) {return a;}
    else if (isnan(b)) {return b;}
    else {return min(a, b);}
}

//
// ArgMax()
//

// specialization for float for handling NaNs
template <>
__host__ __device__ __forceinline__ KeyValuePair<int, float> ArgMax::operator()(
    const KeyValuePair<int, float> &a,
    const KeyValuePair<int, float> &b) const
{
    if (isnan(a.value))
        return a;
    else if (isnan(b.value))
        return b;
    else if ((b.value > a.value) || ((a.value == b.value) && (b.key < a.key)))
        return b;
    else
        return a;
}

// specialization for double for handling NaNs
template <>
__host__ __device__ __forceinline__ KeyValuePair<int, double> ArgMax::operator()(
    const KeyValuePair<int, double> &a,
    const KeyValuePair<int, double> &b) const
{
    if (isnan(a.value))
        return a;
    else if (isnan(b.value))
        return b;
    else if ((b.value > a.value) || ((a.value == b.value) && (b.key < a.key)))
        return b;
    else
        return a;
}

// specialization for complex<float> for handling NaNs
template <>
__host__ __device__ __forceinline__ KeyValuePair<int, complex<float>> ArgMax::operator()(
    const KeyValuePair<int, complex<float>> &a,
    const KeyValuePair<int, complex<float>> &b) const
{
    if (isnan(a.value))
        return a;
    else if (isnan(b.value))
        return b;
    else if ((b.value > a.value) || ((a.value == b.value) && (b.key < a.key)))
        return b;
    else
        return a;
}

// specialization for complex<double> for handling NaNs
template <>
__host__ __device__ __forceinline__ KeyValuePair<int, complex<double>> ArgMax::operator()(
    const KeyValuePair<int, complex<double>> &a,
    const KeyValuePair<int, complex<double>> &b) const
{
    if (isnan(a.value))
        return a;
    else if (isnan(b.value))
        return b;
    else if ((b.value > a.value) || ((a.value == b.value) && (b.key < a.key)))
        return b;
    else
        return a;
}

//
// ArgMin()
//

// specialization for float for handling NaNs
template <>
__host__ __device__ __forceinline__ KeyValuePair<int, float> ArgMin::operator()(
    const KeyValuePair<int, float> &a,
    const KeyValuePair<int, float> &b) const
{
    if (isnan(a.value))
        return a;
    else if (isnan(b.value))
        return b;
    else if ((b.value < a.value) || ((a.value == b.value) && (b.key < a.key)))
        return b;
    else
        return a;
}

// specialization for double for handling NaNs
template <>
__host__ __device__ __forceinline__ KeyValuePair<int, double> ArgMin::operator()(
    const KeyValuePair<int, double> &a,
    const KeyValuePair<int, double> &b) const
{
    if (isnan(a.value))
        return a;
    else if (isnan(b.value))
        return b;
    else if ((b.value < a.value) || ((a.value == b.value) && (b.key < a.key)))
        return b;
    else
        return a;
}

// specialization for complex<float> for handling NaNs
template <>
__host__ __device__ __forceinline__ KeyValuePair<int, complex<float>> ArgMin::operator()(
    const KeyValuePair<int, complex<float>> &a,
    const KeyValuePair<int, complex<float>> &b) const
{
    if (isnan(a.value))
        return a;
    else if (isnan(b.value))
        return b;
    else if ((b.value < a.value) || ((a.value == b.value) && (b.key < a.key)))
        return b;
    else
        return a;
}

// specialization for complex<double> for handling NaNs
template <>
__host__ __device__ __forceinline__ KeyValuePair<int, complex<double>> ArgMin::operator()(
    const KeyValuePair<int, complex<double>> &a,
    const KeyValuePair<int, complex<double>> &b) const
{
    if (isnan(a.value))
        return a;
    else if (isnan(b.value))
        return b;
    else if ((b.value < a.value) || ((a.value == b.value) && (b.key < a.key)))
        return b;
    else
        return a;
}
/* ------------------------------------ End of "patches" ------------------------------------ */


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
// **** CUB ArgMin ****
//
struct _cub_reduce_argmin {
    template <typename T>
    void operator()(void* workspace, size_t& workspace_size, void* x, void* y,
        int num_items, cudaStream_t s)
    {
        DeviceReduce::ArgMin(workspace, workspace_size, static_cast<T*>(x),
            static_cast<KeyValuePair<int, T>*>(y), num_items, s);
    }
};

// TODO(leofang): add _cub_segmented_reduce_argmin

//
// **** CUB ArgMax ****
//
struct _cub_reduce_argmax {
    template <typename T>
    void operator()(void* workspace, size_t& workspace_size, void* x, void* y,
        int num_items, cudaStream_t s)
    {
        DeviceReduce::ArgMax(workspace, workspace_size, static_cast<T*>(x),
            static_cast<KeyValuePair<int, T>*>(y), num_items, s);
    }
};

// TODO(leofang): add _cub_segmented_reduce_argmax

//
// **** CUB SpMV ****
//
struct _cub_device_spmv {
    template <typename T>
    void operator()(void* workspace, size_t& workspace_size, void* values,
        void* row_offsets, void* column_indices, void* x, void* y,
        int num_rows, int num_cols, int num_nonzeros, cudaStream_t stream)
    {
        DeviceSpmv::CsrMV(workspace, workspace_size, static_cast<T*>(values),
            static_cast<int*>(row_offsets), static_cast<int*>(column_indices),
            static_cast<T*>(x), static_cast<T*>(y), num_rows, num_cols,
            num_nonzeros, stream);
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
    case CUPY_CUB_SUM:      return dtype_dispatcher(dtype_id, _cub_reduce_sum(),
                                workspace, workspace_size, x, y, num_items, stream);
    case CUPY_CUB_MIN:      return dtype_dispatcher(dtype_id, _cub_reduce_min(),
                                workspace, workspace_size, x, y, num_items, stream);
    case CUPY_CUB_MAX:      return dtype_dispatcher(dtype_id, _cub_reduce_max(),
                                workspace, workspace_size, x, y, num_items, stream);
    case CUPY_CUB_ARGMIN:   return dtype_dispatcher(dtype_id, _cub_reduce_argmin(),
                                workspace, workspace_size, x, y, num_items, stream);
    case CUPY_CUB_ARGMAX:   return dtype_dispatcher(dtype_id, _cub_reduce_argmax(),
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

/*--------- device spmv (sparse-matrix dense-vector multiply) ---------*/

void cub_device_spmv(void* workspace, size_t& workspace_size, void* values,
    void* row_offsets, void* column_indices, void* x, void* y, int num_rows,
    int num_cols, int num_nonzeros, cudaStream_t stream,
    int dtype_id)
{
    return dtype_dispatcher(dtype_id, _cub_device_spmv(),
                            workspace, workspace_size, values, row_offsets,
                            column_indices, x, y, num_rows, num_cols,
                            num_nonzeros, stream);
}

size_t cub_device_spmv_get_workspace_size(void* values, void* row_offsets,
    void* column_indices, void* x, void* y, int num_rows, int num_cols,
    int num_nonzeros, cudaStream_t stream, int dtype_id)
{
    size_t workspace_size = 0;
    cub_device_spmv(NULL, workspace_size, values, row_offsets, column_indices,
                    x, y, num_rows, num_cols, num_nonzeros, stream, dtype_id);
    return workspace_size;
}
