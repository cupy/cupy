#include <cupy/type_dispatcher.cuh>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/execution_policy.h>
#if (__CUDACC_VER_MAJOR__ >11 || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 2))
// This is used to avoid a problem with constexpr in functions declarations introduced in
// cuda 11.2, MSVC 15 does not fully support it so we need a dummy constexpr declaration
// that is provided by this header. However optional.h is only available
// starting CUDA 10.1
#include <thrust/optional.h>
#endif
#include "cupy_thrust.h"


using namespace thrust;


#if CUPY_USE_HIP
typedef hipStream_t cudaStream_t;
namespace cuda {
    using thrust::hip::par;
}

#endif // #if CUPY_USE_HIP


extern "C" char *cupy_malloc(void *, size_t);
extern "C" void cupy_free(void *, char *);


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
 * Ref: https://docs.scipy.org/doc/numpy/reference/generated/numpy.sort.html
 */

template <typename T>
__host__ __device__ __forceinline__ 
#if (__CUDACC_VER_MAJOR__ >11 || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 2))
THRUST_OPTIONAL_CPP11_CONSTEXPR
#endif
bool _tuple_less(const tuple<size_t, T>& lhs,
                                                     const tuple<size_t, T>& rhs) {
    const size_t& lhs_k = lhs.template get<0>();
    const size_t& rhs_k = rhs.template get<0>();
    const T& lhs_v = lhs.template get<1>();
    const T& rhs_v = rhs.template get<1>();
    const less<T> _less;

    // tuple's comparison rule: compare the 1st member, then 2nd, then 3rd, ...,
    // which should be respected
    if (lhs_k < rhs_k) {
        return true;
    } else if (lhs_k == rhs_k) {
        // same key, compare values
        // note that we can't rely on native operator< due to NaN, so we rely on
        // thrust::less() to be specialized shortly
        return _less(lhs_v, rhs_v);
    } else {
        return false;
    }
}

/*
 * ********** complex numbers **********
 * We need to specialize thrust::less because obviously we can't overload operator< for complex numbers...
 */

template <typename T>
__host__ __device__ __forceinline__
#if (__CUDACC_VER_MAJOR__ >11 || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 2))
THRUST_OPTIONAL_CPP11_CONSTEXPR
#endif
bool _cmp_less(const T& lhs, const T& rhs) {
    bool lhsRe = isnan(lhs.real());
    bool lhsIm = isnan(lhs.imag());
    bool rhsRe = isnan(rhs.real());
    bool rhsIm = isnan(rhs.imag());

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

// specialize thrust::less for single complex
template <>
__host__ __device__ __forceinline__
#if (__CUDACC_VER_MAJOR__ >11 || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 2))
constexpr
#endif
bool less<complex<float>>::operator() (
    const complex<float>& lhs, const complex<float>& rhs) const {

    return _cmp_less<complex<float>>(lhs, rhs);
}

// specialize thrust::less for double complex
template <>
__host__ __device__ __forceinline__
#if (__CUDACC_VER_MAJOR__ >11 || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 2))
constexpr
#endif
bool less<complex<double>>::operator() (
    const complex<double>& lhs, const complex<double>& rhs) const {

    return _cmp_less<complex<double>>(lhs, rhs);
}

// specialize thrust::less for tuple<size_t, complex<float>>
template <>
__host__ __device__ __forceinline__
#if (__CUDACC_VER_MAJOR__ >11 || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 2))
constexpr
#endif
bool less< tuple<size_t, complex<float>> >::operator() (
    const tuple<size_t, complex<float>>& lhs, const tuple<size_t, complex<float>>& rhs) const {

    return _tuple_less<complex<float>>(lhs, rhs);
}

// specialize thrust::less for tuple<size_t, complex<double>>
template <>
__host__ __device__ __forceinline__
#if (__CUDACC_VER_MAJOR__ >11 || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 2))
constexpr
#endif
bool less< tuple<size_t, complex<double>> >::operator() (
    const tuple<size_t, complex<double>>& lhs, const tuple<size_t, complex<double>>& rhs) const {

    return _tuple_less<complex<double>>(lhs, rhs);
}

/*
 * ********** real numbers (templates) **********
 * We need to specialize thrust::less because obviously we can't overload operator< for floating point numbers...
 */

template <typename T>
__host__ __device__ __forceinline__
#if (__CUDACC_VER_MAJOR__ >11 || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 2))
THRUST_OPTIONAL_CPP11_CONSTEXPR
#endif
bool _real_less(const T& lhs, const T& rhs) {
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

/*
 * ********** real numbers (specializations for single & double precisions) **********
 */

// specialize thrust::less for float
template <>
__host__ __device__ __forceinline__
#if (__CUDACC_VER_MAJOR__ >11 || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 2))
constexpr
#endif
bool less<float>::operator() (
    const float& lhs, const float& rhs) const {

    return _real_less<float>(lhs, rhs);
}

// specialize thrust::less for double
template <>
__host__ __device__ __forceinline__
#if (__CUDACC_VER_MAJOR__ >11 || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 2))
constexpr
#endif
bool less<double>::operator() (
    const double& lhs, const double& rhs) const {

    return _real_less<double>(lhs, rhs);
}

// specialize thrust::less for tuple<size_t, float>
template <>
__host__ __device__ __forceinline__
#if (__CUDACC_VER_MAJOR__ >11 || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 2))
constexpr
#endif
bool less< tuple<size_t, float> >::operator() (
    const tuple<size_t, float>& lhs, const tuple<size_t, float>& rhs) const {

    return _tuple_less<float>(lhs, rhs);
}

// specialize thrust::less for tuple<size_t, double>
template <>
__host__ __device__ __forceinline__
#if (__CUDACC_VER_MAJOR__ >11 || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 2))
constexpr
#endif
bool less< tuple<size_t, double> >::operator() (
    const tuple<size_t, double>& lhs, const tuple<size_t, double>& rhs) const {

    return _tuple_less<double>(lhs, rhs);
}

/*
 * ********** real numbers (specializations for half precision) **********
 */

#if ((__CUDACC_VER_MAJOR__ > 9 || (__CUDACC_VER_MAJOR__ == 9 && __CUDACC_VER_MINOR__ == 2)) \
     && (__CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__))) || (defined(__HIPCC__) || defined(CUPY_USE_HIP))

// it seems Thrust doesn't care the code path on host, so we just need a wrapper for device
__host__ __device__ __forceinline__ bool isnan(const __half& x) {
    #if (defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__))
    return __hisnan(x);
    #else
    return false;  // This will never be called on the host
    #endif
}

// specialize thrust::less for __half
template <>
__host__ __device__ __forceinline__
#if (__CUDACC_VER_MAJOR__ >11 || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 2))
constexpr
#endif
bool less<__half>::operator() (const __half& lhs, const __half& rhs) const {
    return _real_less<__half>(lhs, rhs);
}

// specialize thrust::less for tuple<size_t, __half>
template <>
__host__ __device__ __forceinline__
#if (__CUDACC_VER_MAJOR__ >11 || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 2))
constexpr
#endif
bool less< tuple<size_t, __half> >::operator() (
    const tuple<size_t, __half>& lhs, const tuple<size_t, __half>& rhs) const {

    return _tuple_less<__half>(lhs, rhs);
}

#endif  // include cupy_fp16.h

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
        device_ptr<T> dp_data_first, dp_data_last;
        device_ptr<size_t> dp_keys_first, dp_keys_last;
        cudaStream_t stream_ = (cudaStream_t)stream;
        cupy_allocator alloc(memory);

        // Compute the total size of the array.
        size = shape[0];
        for (size_t i = 1; i < ndim; ++i) {
            size *= shape[i];
        }

        dp_data_first = device_pointer_cast(static_cast<T*>(data_start));
        dp_data_last  = device_pointer_cast(static_cast<T*>(data_start) + size);

        if (ndim == 1) {
            stable_sort(cuda::par(alloc).on(stream_), dp_data_first, dp_data_last, less<T>());
        } else {
            // Generate key indices.
            dp_keys_first = device_pointer_cast(keys_start);
            dp_keys_last  = device_pointer_cast(keys_start + size);
            transform(cuda::par(alloc).on(stream_),
                      make_counting_iterator<size_t>(0),
                      make_counting_iterator<size_t>(size),
                      make_constant_iterator<ptrdiff_t>(shape[ndim-1]),
                      dp_keys_first,
                      divides<size_t>());

            stable_sort(
                cuda::par(alloc).on(stream_),
                make_zip_iterator(make_tuple(dp_keys_first, dp_data_first)),
                make_zip_iterator(make_tuple(dp_keys_last, dp_data_last)),
                less< tuple<size_t, T> >());
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
        return less<T>()(_data[i], _data[j]);
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
        device_ptr<size_t> dp_first = device_pointer_cast(idx_start);
        device_ptr<size_t> dp_last  = device_pointer_cast(idx_start + n);
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

        device_ptr<size_t> dp_idx_first, dp_idx_last;
        device_ptr<T> dp_data_first, dp_data_last;
        device_ptr<size_t> dp_keys_first, dp_keys_last;

        // Compute the total size of the data array.
        size = shape[0];
        for (size_t i = 1; i < ndim; ++i) {
            size *= shape[i];
        }

        // Cast device pointers of data.
        dp_data_first = device_pointer_cast(static_cast<T*>(data_start));
        dp_data_last  = device_pointer_cast(static_cast<T*>(data_start) + size);

        // Generate an index sequence.
        dp_idx_first = device_pointer_cast(static_cast<size_t*>(idx_start));
        dp_idx_last  = device_pointer_cast(static_cast<size_t*>(idx_start) + size);
        transform(cuda::par(alloc).on(stream_),
                  make_counting_iterator<size_t>(0),
                  make_counting_iterator<size_t>(size),
                  make_constant_iterator<ptrdiff_t>(shape[ndim-1]),
                  dp_idx_first,
                  modulus<size_t>());

        if (ndim == 1) {
            // Sort the index sequence by data.
            stable_sort_by_key(cuda::par(alloc).on(stream_),
                               dp_data_first,
                               dp_data_last,
                               dp_idx_first);
        } else {
            // Generate key indices.
            dp_keys_first = device_pointer_cast(static_cast<size_t*>(keys_start));
            dp_keys_last  = device_pointer_cast(static_cast<size_t*>(keys_start) + size);
            transform(cuda::par(alloc).on(stream_),
                      make_counting_iterator<size_t>(0),
                      make_counting_iterator<size_t>(size),
                      make_constant_iterator<ptrdiff_t>(shape[ndim-1]),
                      dp_keys_first,
                      divides<size_t>());

            stable_sort_by_key(
                cuda::par(alloc).on(stream_),
                make_zip_iterator(make_tuple(dp_keys_first, dp_data_first)),
                make_zip_iterator(make_tuple(dp_keys_last, dp_data_last)),
                dp_idx_first);
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
