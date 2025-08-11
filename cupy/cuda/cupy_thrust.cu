#include "cupy_thrust.h"
#include <cupy/type_dispatcher.cuh>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/execution_policy.h>
#include <type_traits>
#if (__CUDACC_VER_MAJOR__ >11 || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 2) || HIP_VERSION >= 402)
// This is used to avoid a problem with constexpr in functions declarations introduced in
// cuda 11.2, MSVC 15 does not fully support it so we need a dummy constexpr declaration
// that is provided by this header. However optional.h is only available
// starting CUDA 10.1
#include <thrust/optional.h>

#ifdef _MSC_VER
#define THRUST_OPTIONAL_CPP11_CONSTEXPR_LESS constexpr
#else
#define THRUST_OPTIONAL_CPP11_CONSTEXPR_LESS THRUST_OPTIONAL_CPP11_CONSTEXPR
#endif

#endif


#if CUPY_USE_HIP
typedef hipStream_t cudaStream_t;
namespace cuda {
    using thrust::hip::par;
}
#else // #if CUPY_USE_HIP
namespace cuda {
    using thrust::cuda::par;
}
#endif // #if CUPY_USE_HIP


extern "C" char *cupy_malloc(void *, size_t);
extern "C" int cupy_free(void *, char *);


class cupy_allocator {
private:
    void* memory;

public:
    typedef char value_type;

    cupy_allocator(void* memory) : memory(memory) {}

    char *allocate(size_t num_bytes) {
        return cupy_malloc(memory, num_bytes);
    }

    void deallocate(char *ptr, size_t n) {
        cupy_free(memory, ptr);
    }
};


/*
 * ------------------------------------- Minimum boilerplate for NumPy compatibility --------------------------------------
 * We need a specialized operator< here in order to match the NumPy behavior:
 * "The sort order for complex numbers is lexicographic. If both the real and imaginary parts are non-nan then the order is
 * determined by the real parts except when they are equal, in which case the order is determined by the imaginary parts.
 *
 * In numpy versions >= 1.4.0 nan values are sorted to the end. The extended sort order is:
 *     Real: [R, nan]
 *     Complex: [R + Rj, R + nanj, nan + Rj, nan + nanj]
 * where R is a non-nan real value. Complex values with the same nan placements are sorted according to the non-nan part if
 * it exists. Non-nan values are sorted as before."
 * Ref: https://numpy.org/doc/stable/reference/generated/numpy.sort.html
 */

#if ((__CUDACC_VER_MAJOR__ > 9 || (__CUDACC_VER_MAJOR__ == 9 && __CUDACC_VER_MINOR__ == 2)) \
&& (__CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__))) || (defined(__HIPCC__) || defined(CUPY_USE_HIP))
    #define ENABLE_HALF
#endif

#if (__CUDACC_VER_MAJOR__ >11 || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 2))
    #define CONSTEXPR_FUNC THRUST_OPTIONAL_CPP11_CONSTEXPR
#else
    #define CONSTEXPR_FUNC
#endif

#if (__CUDACC_VER_MAJOR__ >11 || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 2) || HIP_VERSION >= 402)
    #define CONSTEXPR_COMPARATOR THRUST_OPTIONAL_CPP11_CONSTEXPR_LESS
#else
    #define CONSTEXPR_COMPARATOR
#endif

#ifdef ENABLE_HALF
__host__ __device__ __forceinline__ bool isnan(const __half& x) {
    #if (defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__))
    return __hisnan(x);
    #else
    return false;  // This will never be called on the host
    #endif
}
#endif // ENABLE_HALF

template <typename T>
__host__ __device__ __forceinline__ CONSTEXPR_FUNC
static bool real_less(const T& lhs, const T& rhs) {
#if  (defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__))
    if (isnan(lhs)) {
        return false;
    } else if (isnan(rhs)) {
        return true;
    } else {
        return lhs < rhs;
    }
#else
    return false;  // This will be never executed in the host
#endif
}

template <typename T>
__host__ __device__ __forceinline__ CONSTEXPR_FUNC
static bool tuple_less(const thrust::tuple<size_t, T>& lhs,
		 const thrust::tuple<size_t, T>& rhs) {
    const size_t& lhs_k = thrust::get<0>(lhs);
    const size_t& rhs_k = thrust::get<0>(rhs);
    const T& lhs_v = thrust::get<1>(lhs);
    const T& rhs_v = thrust::get<1>(rhs);

    // tuple's comparison rule: compare the 1st member, then 2nd, then 3rd, ...,
    // which should be respected
    if (lhs_k < rhs_k) {
        return true;
    } else if (lhs_k == rhs_k) {
        // same key, compare values
        // note that we can't rely on native operator< due to NaN, so we rely on our custom comparison object
        return real_less(lhs_v, rhs_v);
    } else {
        return false;
    }
}

/*
 * ********** complex numbers **********
 * We need a custom comparator because we can't overload operator< for complex numbers...
 */

template <typename T>
__host__ __device__ __forceinline__ CONSTEXPR_FUNC
static bool complex_less(const T& lhs, const T& rhs) {
    const bool lhsRe = isnan(lhs.real());
    const bool lhsIm = isnan(lhs.imag());
    const bool rhsRe = isnan(rhs.real());
    const bool rhsIm = isnan(rhs.imag());

    // neither side has nan
    if (!lhsRe && !lhsIm && !rhsRe && !rhsIm) {
        return lhs < rhs;
    }

    // one side has nan, and the other does not
    if (!lhsRe && !lhsIm && (rhsRe || rhsIm)) {
        return true;
    }
    if ((lhsRe || lhsIm) && !rhsRe && !rhsIm) {
        return false;
    }

    // pick 2 from 3 possibilities (R + nanj, nan + Rj, nan + nanj)
    if (lhsRe && !rhsRe) {
        return false;
    }
    if (!lhsRe && rhsRe) {
        return true;
    }
    if (lhsIm && !rhsIm) {
        return false;
    }
    if (!lhsIm && rhsIm) {
        return true;
    }

    // pick 1 from 3 and compare the numerical values (nan+nan*I compares to itself as false)
    return (((lhsIm && rhsIm) && (lhs.real() < rhs.real())) || ((lhsRe && rhsRe) && (lhs.imag() < rhs.imag())));
}

// Type function giving us the right comparison operator. We use a custom one for all the specializations below,
// but otherwise just default to thrust::less. We notable do not define a specialization for float and double, since
// thrust uses radix sort for them and sorts NaNs to the back.
template <typename T>
struct select_less {
    using type = thrust::less<T>;
};

// complex numbers

template <>
struct select_less<complex<float>> {
    struct type {
        __host__ __device__ __forceinline__ CONSTEXPR_COMPARATOR
        bool operator() (
            const complex<float>& lhs, const complex<float>& rhs) const {
            return complex_less(lhs, rhs);
        }
    };
};

template <>
struct select_less<complex<double>> {
    struct type {
        __host__ __device__ __forceinline__ CONSTEXPR_COMPARATOR
        bool operator() (
            const complex<double>& lhs, const complex<double>& rhs) const {
            return complex_less(lhs, rhs);
        }
    };
};

template <>
struct select_less<thrust::tuple<size_t, complex<float>>> {
    struct type {
        __host__ __device__ __forceinline__ CONSTEXPR_COMPARATOR
        bool operator() (
            const thrust::tuple<size_t, complex<float>>& lhs, const thrust::tuple<size_t, complex<float>>& rhs) const {
            return tuple_less(lhs, rhs);
        }
    };
};

template <>
struct select_less<thrust::tuple<size_t, complex<double>>> {
    struct type {
        __host__ __device__ __forceinline__ CONSTEXPR_COMPARATOR
        bool operator() (
            const thrust::tuple<size_t, complex<double>>& lhs, const thrust::tuple<size_t, complex<double>>& rhs) const {
            return tuple_less(lhs, rhs);
        }
    };
};

template <>
struct select_less<thrust::tuple<size_t, float>> {
    struct type {
        __host__ __device__ __forceinline__ CONSTEXPR_COMPARATOR
        bool operator() (
            const thrust::tuple<size_t, float>& lhs, const thrust::tuple<size_t, float>& rhs) const {
            return tuple_less(lhs, rhs);
        }
    };
};

template <>
struct select_less<thrust::tuple<size_t, double>> {
    struct type {
        __host__ __device__ __forceinline__ CONSTEXPR_COMPARATOR
        bool operator() (
            const thrust::tuple<size_t, double>& lhs, const thrust::tuple<size_t, double>& rhs) const {
            return tuple_less(lhs, rhs);
        }
    };
};

// floating points

template <>
struct select_less<float> {
    struct type {
        __host__ __device__ __forceinline__ CONSTEXPR_COMPARATOR
        bool operator() (const float& lhs, const float& rhs) const {
            return real_less(lhs, rhs);
        }
    };
};

template <>
struct select_less<double> {
    struct type {
        __host__ __device__ __forceinline__ CONSTEXPR_COMPARATOR
        bool operator() (const double& lhs, const double& rhs) const {
            return real_less(lhs, rhs);
        }
    };
};

#ifdef ENABLE_HALF
template <>
struct select_less<__half> {
    struct type {
        __host__ __device__ __forceinline__ CONSTEXPR_COMPARATOR
        bool operator() (const __half& lhs, const __half& rhs) const {
            return real_less(lhs, rhs);
        }
    };
};

template <>
struct select_less<thrust::tuple<size_t, __half>> {
    struct type {
        __host__ __device__ __forceinline__ CONSTEXPR_COMPARATOR
        bool operator() (
            const thrust::tuple<size_t, __half>& lhs, const thrust::tuple<size_t, __half>& rhs) const {

            return tuple_less(lhs, rhs);
        }
    };
};
#endif  // ENABLE_HALF

/*
 * -------------------------------------------------- end of boilerplate --------------------------------------------------
 */


/*
 * sort
 */

struct _sort {
    template <typename T>
    __forceinline__ void operator()(void *data_start, size_t *keys_start,
                                    const std::vector<ptrdiff_t>& shape, intptr_t stream,
                                    void* memory) {
        size_t ndim = shape.size();
        ptrdiff_t size;
        thrust::device_ptr<T> dp_data_first, dp_data_last;
        thrust::device_ptr<size_t> dp_keys_first, dp_keys_last;
        cudaStream_t stream_ = (cudaStream_t)stream;
        cupy_allocator alloc(memory);

        // Compute the total size of the array.
        size = shape[0];
        for (size_t i = 1; i < ndim; ++i) {
            size *= shape[i];
        }

        dp_data_first = thrust::device_pointer_cast(static_cast<T*>(data_start));
        dp_data_last  = thrust::device_pointer_cast(static_cast<T*>(data_start) + size);

        if (ndim == 1) {
            // we use thrust::less directly to sort floating points, because then it can use radix sort, which happens to sort NaNs to the back
            using compare_op = std::conditional_t<std::is_floating_point<T>::value, thrust::less<T>, typename select_less<T>::type>;
            stable_sort(cuda::par(alloc).on(stream_), dp_data_first, dp_data_last, compare_op{});
        } else {
            // Generate key indices.
            dp_keys_first = thrust::device_pointer_cast(keys_start);
            dp_keys_last  = thrust::device_pointer_cast(keys_start + size);
            transform(cuda::par(alloc).on(stream_),
                      #ifdef __HIP_PLATFORM_HCC__
                      rocprim::make_counting_iterator<size_t>(0),
                      rocprim::make_counting_iterator<size_t>(size),
                      rocprim::make_constant_iterator<ptrdiff_t>(shape[ndim-1]),
                      #else
                      thrust::make_counting_iterator<size_t>(0),
                      thrust::make_counting_iterator<size_t>(size),
                      thrust::make_constant_iterator<ptrdiff_t>(shape[ndim-1]),
                      #endif
                      dp_keys_first,
                      thrust::divides<size_t>());

            stable_sort(
                cuda::par(alloc).on(stream_),
                make_zip_iterator(dp_keys_first, dp_data_first),
                make_zip_iterator(dp_keys_last, dp_data_last),
                typename select_less<thrust::tuple<size_t, T>>::type{});
        }
    }
};


/*
 * lexsort
 */

template <typename T>
class elem_less {
public:
    elem_less(const T *data):_data(data) {}
    __device__ __forceinline__ bool operator()(size_t i, size_t j) const {
        return typename select_less<T>::type{}(_data[i], _data[j]);
    }
private:
    const T *_data;
};

struct _lexsort {
    template <typename T>
    __forceinline__ void operator()(size_t *idx_start, void *keys_start, size_t k,
                                    size_t n, intptr_t stream, void *memory) {
        /* idx_start is the beginning of the output array where the indexes that
           would sort the data will be placed. The original contents of idx_start
           will be destroyed. */
        thrust::device_ptr<size_t> dp_first = thrust::device_pointer_cast(idx_start);
        thrust::device_ptr<size_t> dp_last  = thrust::device_pointer_cast(idx_start + n);
        cudaStream_t stream_ = (cudaStream_t)stream;
        cupy_allocator alloc(memory);
        sequence(cuda::par(alloc).on(stream_), dp_first, dp_last);
        for (size_t i = 0; i < k; ++i) {
            T *key_start = static_cast<T*>(keys_start) + i * n;
            stable_sort(
                cuda::par(alloc).on(stream_),
                dp_first,
                dp_last,
                elem_less<T>(key_start)
            );
        }
    }
};


/*
 * argsort
 */

struct _argsort {
    template <typename T>
    __forceinline__ void operator()(size_t *idx_start, void *data_start,
                                    void *keys_start,
                                    const std::vector<ptrdiff_t>& shape,
                                    intptr_t stream, void *memory) {
        /* idx_start is the beginning of the output array where the indexes that
           would sort the data will be placed. The original contents of idx_start
           will be destroyed. */

        size_t ndim = shape.size();
        ptrdiff_t size;
        cudaStream_t stream_ = (cudaStream_t)stream;
        cupy_allocator alloc(memory);

        thrust::device_ptr<size_t> dp_idx_first, dp_idx_last;
        thrust::device_ptr<T> dp_data_first, dp_data_last;
        thrust::device_ptr<size_t> dp_keys_first, dp_keys_last;

        // Compute the total size of the data array.
        size = shape[0];
        for (size_t i = 1; i < ndim; ++i) {
            size *= shape[i];
        }

        // Cast device pointers of data.
        dp_data_first = thrust::device_pointer_cast(static_cast<T*>(data_start));
        dp_data_last  = thrust::device_pointer_cast(static_cast<T*>(data_start) + size);

        // Generate an index sequence.
        dp_idx_first = thrust::device_pointer_cast(static_cast<size_t*>(idx_start));
        dp_idx_last  = thrust::device_pointer_cast(static_cast<size_t*>(idx_start) + size);
        transform(cuda::par(alloc).on(stream_),
                  #ifdef __HIP_PLATFORM_HCC__
                  rocprim::make_counting_iterator<size_t>(0),
                  rocprim::make_counting_iterator<size_t>(size),
                  rocprim::make_constant_iterator<ptrdiff_t>(shape[ndim-1]),
                  #else
                  thrust::make_counting_iterator<size_t>(0),
                  thrust::make_counting_iterator<size_t>(size),
                  thrust::make_constant_iterator<ptrdiff_t>(shape[ndim-1]),
                  #endif
                  dp_idx_first,
                  thrust::modulus<size_t>());

        if (ndim == 1) {
            // we use thrust::less directly to sort floating points, because then it can use radix sort, which happens to sort NaNs to the back
            using compare_op = std::conditional_t<std::is_floating_point<T>::value, thrust::less<T>, typename select_less<T>::type>;
            // Sort the index sequence by data.
            stable_sort_by_key(cuda::par(alloc).on(stream_),
                               dp_data_first,
                               dp_data_last,
                               dp_idx_first,
                               compare_op{});
        } else {
            // Generate key indices.
            dp_keys_first = thrust::device_pointer_cast(static_cast<size_t*>(keys_start));
            dp_keys_last  = thrust::device_pointer_cast(static_cast<size_t*>(keys_start) + size);
            transform(cuda::par(alloc).on(stream_),
                      #ifdef __HIP_PLATFORM_HCC__
                      rocprim::make_counting_iterator<size_t>(0),
                      rocprim::make_counting_iterator<size_t>(size),
                      rocprim::make_constant_iterator<ptrdiff_t>(shape[ndim-1]),
                      #else
                      thrust::make_counting_iterator<size_t>(0),
                      thrust::make_counting_iterator<size_t>(size),
                      thrust::make_constant_iterator<ptrdiff_t>(shape[ndim-1]),
                      #endif
                      dp_keys_first,
                      thrust::divides<size_t>());

            stable_sort_by_key(
                cuda::par(alloc).on(stream_),
                make_zip_iterator(dp_keys_first, dp_data_first),
                make_zip_iterator(dp_keys_last, dp_data_last),
                dp_idx_first,
                typename select_less<thrust::tuple<size_t, T>>::type{});
        }
    }
};


//
// APIs exposed to CuPy
//

/* -------- sort -------- */

void thrust_sort(int dtype_id, void *data_start, size_t *keys_start,
    const std::vector<ptrdiff_t>& shape, intptr_t stream, void* memory) {

    _sort op;
    return dtype_dispatcher(dtype_id, op, data_start, keys_start, shape, stream, memory);
}


/* -------- lexsort -------- */
void thrust_lexsort(int dtype_id, size_t *idx_start, void *keys_start, size_t k,
    size_t n, intptr_t stream, void *memory) {

    _lexsort op;
    return dtype_dispatcher(dtype_id, op, idx_start, keys_start, k, n, stream, memory);
}


/* -------- argsort -------- */
void thrust_argsort(int dtype_id, size_t *idx_start, void *data_start,
    void *keys_start, const std::vector<ptrdiff_t>& shape, intptr_t stream, void *memory) {

    _argsort op;
    return dtype_dispatcher(dtype_id, op, idx_start, data_start, keys_start, shape,
                            stream, memory);
}

