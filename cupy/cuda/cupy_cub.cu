#include "cupy_cub.h"  // need to make atomicAdd visible to CUB templates early
#include <cupy/type_dispatcher.cuh>

#ifndef CUPY_USE_HIP
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_segmented_reduce.cuh>
#include <cub/device/device_spmv.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_histogram.cuh>
#include <cub/iterator/counting_input_iterator.cuh>
#include <cub/iterator/transform_input_iterator.cuh>
#else
#include <hipcub/device/device_reduce.hpp>
#include <hipcub/device/device_segmented_reduce.hpp>
#include <hipcub/device/device_scan.hpp>
#include <hipcub/device/device_histogram.hpp>
#include <rocprim/iterator/counting_iterator.hpp>
#include <hipcub/iterator/transform_input_iterator.hpp>
#endif


/* ------------------------------------ Minimum boilerplate to support complex numbers ------------------------------------ */
#ifndef CUPY_USE_HIP
// - This works only because all data fields in the *Traits struct are not
//   used in <cub/device/device_reduce.cuh>.
// - The Max() and Lowest() below are chosen to comply with NumPy's lexical
//   ordering; note that std::numeric_limits<T> does not support complex
//   numbers as in general the comparison is ill defined.
// - DO NOT USE THIS STUB for supporting CUB sorting!!!!!!
using namespace cub;

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

#else

// hipCUB internally uses std::numeric_limits, so we should provide specializations for the complex numbers.
// Note that there's std::complex, so to avoid name collision we must use the full decoration (thrust::complex)!
// TODO(leofang): wrap CuPy's thrust namespace with another one (say, cupy::thrust) for safer scope resolution?

namespace std {
template <>
class numeric_limits<thrust::complex<float>> {
  public:
    static __host__ __device__ thrust::complex<float> max() noexcept {
        return thrust::complex<float>(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    }

    static __host__ __device__ thrust::complex<float> lowest() noexcept {
        return thrust::complex<float>(-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max());
    }
};

template <>
class numeric_limits<thrust::complex<double>> {
  public:
    static __host__ __device__ thrust::complex<double> max() noexcept {
        return thrust::complex<double>(std::numeric_limits<double>::max(), std::numeric_limits<double>::max());
    }

    static __host__ __device__ thrust::complex<double> lowest() noexcept {
        return thrust::complex<double>(-std::numeric_limits<double>::max(), -std::numeric_limits<double>::max());
    }
};

// Copied from https://github.com/ROCmSoftwarePlatform/hipCUB/blob/master-rocm-3.5/hipcub/include/hipcub/backend/rocprim/device/device_reduce.hpp
// (For some reason the specialization for __half defined in the above file does not work, so we have to go
// through the same route as we did above for complex numbers.)
template <>
class numeric_limits<__half> {
  public:
    static __host__ __device__ __half max() noexcept {
        unsigned short max_half = 0x7bff;
        __half max_value = *reinterpret_cast<__half*>(&max_half);
        return max_value;
    }

    static __host__ __device__ __half lowest() noexcept {
        unsigned short lowest_half = 0xfbff;
        __half lowest_value = *reinterpret_cast<__half*>(&lowest_half);
        return lowest_value;
    }
};
}  // namespace std

using namespace hipcub;

#endif  // ifndef CUPY_USE_HIP
/* ------------------------------------ end of boilerplate ------------------------------------ */


/* ------------------------------------ "Patches" to CUB ------------------------------------
   This stub is needed because CUB does not have a built-in "prod" operator
*/

//
// product functor
//
struct _multiply
{
    template <typename T>
    __host__ __device__ __forceinline__ T operator()(const T &a, const T &b) const
    {
        return a * b;
    }
};

//
// arange functor: arange(0, n+1) -> arange(0, n+1, step_size)
//
struct _arange
{
    private:
        int step_size;

    public:
    __host__ __device__ __forceinline__ _arange(int i): step_size(i) {}
    __host__ __device__ __forceinline__ int operator()(const int &in) const {
        return step_size * in;
    }
};

#ifndef CUPY_USE_HIP
typedef TransformInputIterator<int, _arange, CountingInputIterator<int>> seg_offset_itr;
#else
typedef TransformInputIterator<int, _arange, rocprim::counting_iterator<int>> seg_offset_itr;
#endif

/*
   These stubs are needed because CUB does not handle NaNs properly, while NumPy has certain
   behaviors with which we must comply.
*/

#if ((__CUDACC_VER_MAJOR__ > 9 || (__CUDACC_VER_MAJOR__ == 9 && __CUDACC_VER_MINOR__ == 2)) \
    && (__CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__))) || (defined(__HIPCC__) || defined(CUPY_USE_HIP))
__host__ __device__ __forceinline__ bool half_isnan(const __half& x) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return __hisnan(x);
#else
    // TODO: avoid cast to float
    return isnan(__half2float(x));
#endif
}

__host__ __device__ __forceinline__ bool half_less(const __half& l, const __half& r) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return l < r;
#else
    // TODO: avoid cast to float
    return __half2float(l) < __half2float(r);
#endif
}

__host__ __device__ __forceinline__ bool half_equal(const __half& l, const __half& r) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return l == r;
#else
    // TODO: avoid cast to float
    return __half2float(l) == __half2float(r);
#endif
}
#endif

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
    else {return a < b ? b : a;}
}

// specialization for double for handling NaNs
template <>
__host__ __device__ __forceinline__ double Max::operator()(const double &a, const double &b) const
{
    // NumPy behavior: NaN is always chosen!
    if (isnan(a)) {return a;}
    else if (isnan(b)) {return b;}
    else {return a < b ? b : a;}
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
    else {return a < b ? b : a;}
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
    else {return a < b ? b : a;}
}

#if ((__CUDACC_VER_MAJOR__ > 9 || (__CUDACC_VER_MAJOR__ == 9 && __CUDACC_VER_MINOR__ == 2)) \
    && (__CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__))) || (defined(__HIPCC__) || defined(CUPY_USE_HIP))
// specialization for half for handling NaNs
template <>
__host__ __device__ __forceinline__ __half Max::operator()(const __half &a, const __half &b) const
{
    // NumPy behavior: NaN is always chosen!
    if (half_isnan(a)) {return a;}
    else if (half_isnan(b)) {return b;}
    else { return half_less(a, b) ? b : a; }
}
#endif

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
    else {return a < b ? a : b;}
}

// specialization for double for handling NaNs
template <>
__host__ __device__ __forceinline__ double Min::operator()(const double &a, const double &b) const
{
    // NumPy behavior: NaN is always chosen!
    if (isnan(a)) {return a;}
    else if (isnan(b)) {return b;}
    else {return a < b ? a : b;}
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
    else {return a < b ? a : b;}
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
    else {return a < b ? a : b;}
}

#if ((__CUDACC_VER_MAJOR__ > 9 || (__CUDACC_VER_MAJOR__ == 9 && __CUDACC_VER_MINOR__ == 2)) \
    && (__CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__))) || (defined(__HIPCC__) || defined(CUPY_USE_HIP))
// specialization for half for handling NaNs
template <>
__host__ __device__ __forceinline__ __half Min::operator()(const __half &a, const __half &b) const
{
    // NumPy behavior: NaN is always chosen!
    if (half_isnan(a)) {return a;}
    else if (half_isnan(b)) {return b;}
    else { return half_less(a, b) ? a : b; }
}
#endif

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

#if ((__CUDACC_VER_MAJOR__ > 9 || (__CUDACC_VER_MAJOR__ == 9 && __CUDACC_VER_MINOR__ == 2)) \
    && (__CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__))) || (defined(__HIPCC__) || defined(CUPY_USE_HIP))
// specialization for half for handling NaNs
template <>
__host__ __device__ __forceinline__ KeyValuePair<int, __half> ArgMax::operator()(
    const KeyValuePair<int, __half> &a,
    const KeyValuePair<int, __half> &b) const
{
    if (half_isnan(a.value))
        return a;
    else if (half_isnan(b.value))
        return b;
    else if ((half_less(a.value, b.value)) ||
             (half_equal(a.value, b.value) && (b.key < a.key)))
        return b;
    else
        return a;
}
#endif

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

#if ((__CUDACC_VER_MAJOR__ > 9 || (__CUDACC_VER_MAJOR__ == 9 && __CUDACC_VER_MINOR__ == 2)) \
    && (__CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__))) || (defined(__HIPCC__) || defined(CUPY_USE_HIP))
// specialization for half for handling NaNs
template <>
__host__ __device__ __forceinline__ KeyValuePair<int, __half> ArgMin::operator()(
    const KeyValuePair<int, __half> &a,
    const KeyValuePair<int, __half> &b) const
{
    if (half_isnan(a.value))
        return a;
    else if (half_isnan(b.value))
        return b;
    else if ((half_less(b.value, a.value)) ||
             (half_equal(a.value, b.value) && (b.key < a.key)))
        return b;
    else
        return a;

}
#endif

/* ------------------------------------ End of "patches" ------------------------------------ */

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
        int num_segments, seg_offset_itr offset_start, cudaStream_t s)
    {
        DeviceSegmentedReduce::Sum(workspace, workspace_size,
            static_cast<T*>(x), static_cast<T*>(y), num_segments,
            offset_start, offset_start+1, s);
    }
};

//
// **** CUB Prod ****
//
struct _cub_reduce_prod {
    template <typename T>
    void operator()(void* workspace, size_t& workspace_size, void* x, void* y,
        int num_items, cudaStream_t s)
    {
        _multiply product_op;
        // the init value is cast from 1.0f because on host __half can only be
        // initialized by float or double; static_cast<__half>(1) = 0 on host.
        DeviceReduce::Reduce(workspace, workspace_size, static_cast<T*>(x),
            static_cast<T*>(y), num_items, product_op, static_cast<T>(1.0f), s);
    }
};

struct _cub_segmented_reduce_prod {
    template <typename T>
    void operator()(void* workspace, size_t& workspace_size, void* x, void* y,
        int num_segments, seg_offset_itr offset_start, cudaStream_t s)
    {
        _multiply product_op;
        // the init value is cast from 1.0f because on host __half can only be
        // initialized by float or double; static_cast<__half>(1) = 0 on host.
        DeviceSegmentedReduce::Reduce(workspace, workspace_size,
            static_cast<T*>(x), static_cast<T*>(y), num_segments,
            offset_start, offset_start+1,
            product_op, static_cast<T>(1.0f), s);
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
        int num_segments, seg_offset_itr offset_start, cudaStream_t s)
    {
        DeviceSegmentedReduce::Min(workspace, workspace_size,
            static_cast<T*>(x), static_cast<T*>(y), num_segments,
            offset_start, offset_start+1, s);
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
        int num_segments, seg_offset_itr offset_start, cudaStream_t s)
    {
        DeviceSegmentedReduce::Max(workspace, workspace_size,
            static_cast<T*>(x), static_cast<T*>(y), num_segments,
            offset_start, offset_start+1, s);
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
        #ifndef CUPY_USE_HIP
        DeviceSpmv::CsrMV(workspace, workspace_size, static_cast<T*>(values),
            static_cast<int*>(row_offsets), static_cast<int*>(column_indices),
            static_cast<T*>(x), static_cast<T*>(y), num_rows, num_cols,
            num_nonzeros, stream);
        #endif
    }
};

//
// **** CUB InclusiveSum  ****
//
struct _cub_inclusive_sum {
    template <typename T>
    void operator()(void* workspace, size_t& workspace_size, void* input, void* output,
        int num_items, cudaStream_t s)
    {
        DeviceScan::InclusiveSum(workspace, workspace_size, static_cast<T*>(input),
            static_cast<T*>(output), num_items, s);
    }
};

//
// **** CUB inclusive product  ****
//
struct _cub_inclusive_product {
    template <typename T>
    void operator()(void* workspace, size_t& workspace_size, void* input, void* output,
        int num_items, cudaStream_t s)
    {
        _multiply product_op;
        DeviceScan::InclusiveScan(workspace, workspace_size, static_cast<T*>(input),
            static_cast<T*>(output), product_op, num_items, s);
    }
};

//
// **** CUB histogram range ****
//
struct _cub_histogram_range {
    template <typename sampleT,
              typename binT = typename If<std::is_integral<sampleT>::value, double, sampleT>::Type>
    void operator()(void* workspace, size_t& workspace_size, void* input, void* output,
        int n_bins, void* bins, size_t n_samples, cudaStream_t s) const
    {
        // Ugly hack to avoid specializing complex types, which cub::DeviceHistogram does not support.
        // The If and Equals templates are from cub/util_type.cuh.
        // TODO(leofang): revisit this part when complex support is added to cupy.histogram()
        #ifndef CUPY_USE_HIP
        typedef typename If<(Equals<sampleT, complex<float>>::VALUE || Equals<sampleT, complex<double>>::VALUE),
                            double,
                            sampleT>::Type h_sampleT;
        typedef typename If<(Equals<binT, complex<float>>::VALUE || Equals<binT, complex<double>>::VALUE),
                            double,
                            binT>::Type h_binT;
        #else
        typedef typename std::conditional<(std::is_same<sampleT, complex<float>>::value || std::is_same<sampleT, complex<double>>::value),
                                          double,
                                          sampleT>::type h_sampleT;
        typedef typename std::conditional<(std::is_same<binT, complex<float>>::value || std::is_same<binT, complex<double>>::value),
                                          double,
                                          binT>::type h_binT;
        #endif

        // TODO(leofang): CUB has a bug that when specializing n_samples with type size_t,
        // it would error out. Before the fix (thrust/cub#38) is merged we disable the code
        // path splitting for now. A type/range check must be done in the caller.
        // TODO(leofang): check if hipCUB has the same bug or not

        // if (n_samples < (1ULL << 31)) {
            int num_samples = n_samples;
            DeviceHistogram::HistogramRange(workspace, workspace_size, static_cast<h_sampleT*>(input),
                #ifndef CUPY_USE_HIP
                static_cast<long long*>(output), n_bins, static_cast<h_binT*>(bins), num_samples, s);
                #else
                // rocPRIM looks up atomic_add() from the namespace rocprim::detail; there's no way we can
                // inject a "long long" version as we did for CUDA, so we must do it in "unsigned long long"
                // and convert later...
                static_cast<unsigned long long*>(output), n_bins, static_cast<h_binT*>(bins), num_samples, s);
                #endif
        // } else {
        //     DeviceHistogram::HistogramRange(workspace, workspace_size, static_cast<h_sampleT*>(input),
        //         static_cast<long long*>(output), n_bins, static_cast<h_binT*>(bins), n_samples, s);
        // }
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
    case CUPY_CUB_PROD:     return dtype_dispatcher(dtype_id, _cub_reduce_prod(),
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
    void* x, void* y, int num_segments, int segment_size,
    cudaStream_t stream, int op, int dtype_id)
{
    // CUB internally use int for offset...
    // This iterates over [0, segment_size, 2*segment_size, 3*segment_size, ...]
    #ifndef CUPY_USE_HIP
    CountingInputIterator<int> count_itr(0);
    #else
    rocprim::counting_iterator<int> count_itr(0);
    #endif
    _arange scaling(segment_size);
    seg_offset_itr itr(count_itr, scaling);

    switch(op) {
    case CUPY_CUB_SUM:
        return dtype_dispatcher(dtype_id, _cub_segmented_reduce_sum(),
                   workspace, workspace_size, x, y, num_segments, itr, stream);
    case CUPY_CUB_MIN:
        return dtype_dispatcher(dtype_id, _cub_segmented_reduce_min(),
                   workspace, workspace_size, x, y, num_segments, itr, stream);
    case CUPY_CUB_MAX:
        return dtype_dispatcher(dtype_id, _cub_segmented_reduce_max(),
                   workspace, workspace_size, x, y, num_segments, itr, stream);
    case CUPY_CUB_PROD:
        return dtype_dispatcher(dtype_id, _cub_segmented_reduce_prod(),
                   workspace, workspace_size, x, y, num_segments, itr, stream);
    default:
        throw std::runtime_error("Unsupported operation");
    }
}

size_t cub_device_segmented_reduce_get_workspace_size(void* x, void* y,
    int num_segments, int segment_size,
    cudaStream_t stream, int op, int dtype_id)
{
    size_t workspace_size = 0;
    cub_device_segmented_reduce(NULL, workspace_size, x, y,
                                num_segments, segment_size, stream,
                                op, dtype_id);
    return workspace_size;
}

/*--------- device spmv (sparse-matrix dense-vector multiply) ---------*/

void cub_device_spmv(void* workspace, size_t& workspace_size, void* values,
    void* row_offsets, void* column_indices, void* x, void* y, int num_rows,
    int num_cols, int num_nonzeros, cudaStream_t stream,
    int dtype_id)
{
    #ifndef CUPY_USE_HIP
    return dtype_dispatcher(dtype_id, _cub_device_spmv(),
                            workspace, workspace_size, values, row_offsets,
                            column_indices, x, y, num_rows, num_cols,
                            num_nonzeros, stream);
    #endif
}

size_t cub_device_spmv_get_workspace_size(void* values, void* row_offsets,
    void* column_indices, void* x, void* y, int num_rows, int num_cols,
    int num_nonzeros, cudaStream_t stream, int dtype_id)
{
    size_t workspace_size = 0;
    #ifndef CUPY_USE_HIP
    cub_device_spmv(NULL, workspace_size, values, row_offsets, column_indices,
                    x, y, num_rows, num_cols, num_nonzeros, stream, dtype_id);
    #endif
    return workspace_size;
}

/* -------- device scan -------- */

void cub_device_scan(void* workspace, size_t& workspace_size, void* x, void* y,
    int num_items, cudaStream_t stream, int op, int dtype_id)
{
    switch(op) {
    case CUPY_CUB_CUMSUM:
        return dtype_dispatcher(dtype_id, _cub_inclusive_sum(),
                                workspace, workspace_size, x, y, num_items, stream);
    case CUPY_CUB_CUMPROD:
        return dtype_dispatcher(dtype_id, _cub_inclusive_product(),
                                workspace, workspace_size, x, y, num_items, stream);
    default:
        throw std::runtime_error("Unsupported operation");
    }
}

size_t cub_device_scan_get_workspace_size(void* x, void* y, int num_items,
    cudaStream_t stream, int op, int dtype_id)
{
    size_t workspace_size = 0;
    cub_device_scan(NULL, workspace_size, x, y, num_items, stream,
                    op, dtype_id);
    return workspace_size;
}

/* -------- device histogram -------- */

void cub_device_histogram_range(void* workspace, size_t& workspace_size, void* x, void* y,
    int n_bins, void* bins, size_t n_samples, cudaStream_t stream, int dtype_id)
{
    // TODO(leofang): support complex
    if (dtype_id == CUPY_TYPE_COMPLEX64 || dtype_id == CUPY_TYPE_COMPLEX128) {
	    throw std::runtime_error("complex dtype is not yet supported");
    }

    // TODO(leofang): n_samples is of type size_t, but if it's < 2^31 we cast it to int later
    return dtype_dispatcher(dtype_id, _cub_histogram_range(),
                            workspace, workspace_size, x, y, n_bins, bins, n_samples, stream);
}

size_t cub_device_histogram_range_get_workspace_size(void* x, void* y, int n_bins,
    void* bins, size_t n_samples, cudaStream_t stream, int dtype_id)
{
    size_t workspace_size = 0;
    cub_device_histogram_range(NULL, workspace_size, x, y, n_bins, bins, n_samples,
                               stream, dtype_id);
    return workspace_size;
}
