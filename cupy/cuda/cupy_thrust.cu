#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/execution_policy.h>
#include "cupy_common.h"
#include "cupy_thrust.h"
#if (__CUDACC_VER_MAJOR__ > 9 || (__CUDACC_VER_MAJOR__ == 9 && __CUDACC_VER_MINOR__ == 2)) \
    && (__CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__))
#include <cuda_fp16.h>
#endif

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

/*
 * ********** complex numbers **********
 */

template <typename T>
__host__ __device__ __forceinline__ bool _cmp_less(const T& lhs, const T& rhs) {
    bool lhsRe = isnan(lhs.x);
    bool lhsIm = isnan(lhs.y);
    bool rhsRe = isnan(rhs.x);
    bool rhsIm = isnan(rhs.y);

    // neither side has nan
    if (!lhsRe && !lhsIm && !rhsRe && !rhsIm) {
        return (lhs.x < rhs.x || ((lhs.x == rhs.x) && (lhs.y < rhs.y)));
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
    return (((lhsIm && rhsIm) && (lhs.x < rhs.x)) || ((lhsRe && rhsRe) && (lhs.y < rhs.y)));
}

/*
 * Unfortunately we need explicit (instead of templated) definitions here, because the template specializations would
 * go through some wild routes in Thrust that passing by reference to device functions is not working...
 */

__host__ __device__ __forceinline__ bool operator<(const cuComplex& lhs, const cuComplex& rhs) {
    return _cmp_less(lhs, rhs);
}

__host__ __device__ __forceinline__ bool operator<(const cuDoubleComplex& lhs, const cuDoubleComplex& rhs) {
    return _cmp_less(lhs, rhs);
}

/*
 * ********** real numbers (templates) **********
 * We need to specialize thrust::less because obviously we can't overload operator< for floating point numbers...
 */

template <typename T>
__host__ __device__ __forceinline__ bool _real_less(const T& lhs, const T& rhs) {
    if (isnan(lhs)) {
        return false;
    } else if (isnan(rhs)) {
        return true;
    } else {
        return lhs < rhs;
    }
}

template <typename T>
__host__ __device__ __forceinline__ bool _tuple_real_less(const tuple<size_t, T>& lhs,
                                                          const tuple<size_t, T>& rhs) {
    const size_t& lhs_k = lhs.template get<0>();
    const size_t& rhs_k = rhs.template get<0>();
    const T& lhs_v = lhs.template get<1>();
    const T& rhs_v = rhs.template get<1>();

    // tuple's comparison rule: compare the 1st member, then 2nd, then 3rd, ...,
    // which should be respected
    if (lhs_k < rhs_k) {
        return true;
    } else if (lhs_k == rhs_k) {
        // same key, compare values
        // note that we can't rely on native operator< due to NaN
        return _real_less<T>(lhs_v, rhs_v);
    } else {
        return false;
    }
}

/*
 * ********** real numbers (specializations for single & double precisions) **********
 */

// specialize thrust::less for float
template <>
__host__ __device__ __forceinline__ bool less<float>::operator() (const float& lhs,
                                                                  const float& rhs) const {
    return _real_less<float>(lhs, rhs);
}

// specialize thrust::less for double
template <>
__host__ __device__ __forceinline__ bool less<double>::operator() (const double& lhs, const double& rhs) const {
    return _real_less<double>(lhs, rhs);
}

// specialize thrust::less for tuple<size_t, float>
template <>
__host__ __device__ __forceinline__ bool less< tuple<size_t, float> >::operator() (const tuple<size_t, float>& lhs,
                                                                                   const tuple<size_t, float>& rhs) const {
    return _tuple_real_less<float>(lhs, rhs);
}

// specialize thrust::less for tuple<size_t, double>
template <>
__host__ __device__ __forceinline__ bool less< tuple<size_t, double> >::operator() (const tuple<size_t, double>& lhs,
                                                                                    const tuple<size_t, double>& rhs) const {
    return _tuple_real_less<double>(lhs, rhs);
}

/*
 * ********** real numbers (specializations for half precision) **********
 */

#if (__CUDACC_VER_MAJOR__ > 9 || (__CUDACC_VER_MAJOR__ == 9 && __CUDACC_VER_MINOR__ == 2)) \
    && (__CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__))

// it seems Thrust doesn't care the code path on host, so we just need a wrapper for device
__device__ __forceinline__ bool isnan(const __half& x) {
    return __hisnan(x);
}

// specialize thrust::less for __half
template <>
__host__ __device__ __forceinline__ bool less<__half>::operator() (const __half& lhs, const __half& rhs) const {
    return _real_less<__half>(lhs, rhs);
}

// specialize thrust::less for tuple<size_t, __half>
template <>
__host__ __device__ __forceinline__ bool less< tuple<size_t, __half> >::operator() (const tuple<size_t, __half>& lhs,
                                                                                    const tuple<size_t, __half>& rhs) const {
    return _tuple_real_less<__half>(lhs, rhs);
}

#endif  // include cupy_fp16.h

/*
 * -------------------------------------------------- end of boilerplate --------------------------------------------------
 */


/*
 * sort
 */

template <typename T>
void cupy::thrust::_sort(void *data_start, size_t *keys_start,
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

template void cupy::thrust::_sort<cpy_byte>(
    void *, size_t *, const std::vector<ptrdiff_t>& shape, intptr_t, void *);
template void cupy::thrust::_sort<cpy_ubyte>(
    void *, size_t *, const std::vector<ptrdiff_t>& shape, intptr_t, void *);
template void cupy::thrust::_sort<cpy_short>(
    void *, size_t *, const std::vector<ptrdiff_t>& shape, intptr_t, void *);
template void cupy::thrust::_sort<cpy_ushort>(
    void *, size_t *, const std::vector<ptrdiff_t>& shape, intptr_t, void *);
template void cupy::thrust::_sort<cpy_int>(
    void *, size_t *, const std::vector<ptrdiff_t>& shape, intptr_t, void *);
template void cupy::thrust::_sort<cpy_uint>(
    void *, size_t *, const std::vector<ptrdiff_t>& shape, intptr_t, void *);
template void cupy::thrust::_sort<cpy_long>(
    void *, size_t *, const std::vector<ptrdiff_t>& shape, intptr_t, void *);
template void cupy::thrust::_sort<cpy_ulong>(
    void *, size_t *, const std::vector<ptrdiff_t>& shape, intptr_t, void *);
template void cupy::thrust::_sort<cpy_float>(
    void *, size_t *, const std::vector<ptrdiff_t>& shape, intptr_t, void *);
template void cupy::thrust::_sort<cpy_double>(
    void *, size_t *, const std::vector<ptrdiff_t>& shape, intptr_t, void *);
template void cupy::thrust::_sort<cpy_complex64>(
    void *, size_t *, const std::vector<ptrdiff_t>& shape, intptr_t, void *);
template void cupy::thrust::_sort<cpy_complex128>(
    void *, size_t *, const std::vector<ptrdiff_t>& shape, intptr_t, void *);
template void cupy::thrust::_sort<cpy_bool>(
    void *, size_t *, const std::vector<ptrdiff_t>& shape, intptr_t, void *);
void cupy::thrust::_sort_fp16(void *data_start, size_t *keys_start,
                              const std::vector<ptrdiff_t>& shape, intptr_t stream,
                              void* memory) {
#if (__CUDACC_VER_MAJOR__ > 9 || (__CUDACC_VER_MAJOR__ == 9 && __CUDACC_VER_MINOR__ == 2)) \
    && (__CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__))
    cupy::thrust::_sort<__half>(data_start, keys_start, shape, stream, memory);
#endif
}


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

template <typename T>
void cupy::thrust::_lexsort(size_t *idx_start, void *keys_start, size_t k,
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

template void cupy::thrust::_lexsort<cpy_byte>(
    size_t *, void *, size_t, size_t, intptr_t, void *);
template void cupy::thrust::_lexsort<cpy_ubyte>(
    size_t *, void *, size_t, size_t, intptr_t, void *);
template void cupy::thrust::_lexsort<cpy_short>(
    size_t *, void *, size_t, size_t, intptr_t, void *);
template void cupy::thrust::_lexsort<cpy_ushort>(
    size_t *, void *, size_t, size_t, intptr_t, void *);
template void cupy::thrust::_lexsort<cpy_int>(
    size_t *, void *, size_t, size_t, intptr_t, void *);
template void cupy::thrust::_lexsort<cpy_uint>(
    size_t *, void *, size_t, size_t, intptr_t, void *);
template void cupy::thrust::_lexsort<cpy_long>(
    size_t *, void *, size_t, size_t, intptr_t, void *);
template void cupy::thrust::_lexsort<cpy_ulong>(
    size_t *, void *, size_t, size_t, intptr_t, void *);
template void cupy::thrust::_lexsort<cpy_float>(
    size_t *, void *, size_t, size_t, intptr_t, void *);
template void cupy::thrust::_lexsort<cpy_double>(
    size_t *, void *, size_t, size_t, intptr_t, void *);
template void cupy::thrust::_lexsort<cpy_complex64>(
    size_t *, void *, size_t, size_t, intptr_t, void *);
template void cupy::thrust::_lexsort<cpy_complex128>(
    size_t *, void *, size_t, size_t, intptr_t, void *);
template void cupy::thrust::_lexsort<cpy_bool>(
    size_t *, void *, size_t, size_t, intptr_t, void *);
void cupy::thrust::_lexsort_fp16(size_t *idx_start, void *keys_start, size_t k,
                                 size_t n, intptr_t stream, void *memory) {
#if (__CUDACC_VER_MAJOR__ > 9 || (__CUDACC_VER_MAJOR__ == 9 && __CUDACC_VER_MINOR__ == 2)) \
    && (__CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__))
    cupy::thrust::_lexsort<__half>(idx_start, keys_start, k, n, stream, memory);
#endif
}


/*
 * argsort
 */

template <typename T>
void cupy::thrust::_argsort(size_t *idx_start, void *data_start,
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

template void cupy::thrust::_argsort<cpy_byte>(
    size_t *, void *, void *, const std::vector<ptrdiff_t>& shape, intptr_t,
    void *);
template void cupy::thrust::_argsort<cpy_ubyte>(
    size_t *, void *, void *, const std::vector<ptrdiff_t>& shape, intptr_t,
    void *);
template void cupy::thrust::_argsort<cpy_short>(
    size_t *, void *, void *, const std::vector<ptrdiff_t>& shape, intptr_t,
    void *);
template void cupy::thrust::_argsort<cpy_ushort>(
    size_t *, void *, void *, const std::vector<ptrdiff_t>& shape, intptr_t,
    void *);
template void cupy::thrust::_argsort<cpy_int>(
    size_t *, void *, void *, const std::vector<ptrdiff_t>& shape, intptr_t,
    void *);
template void cupy::thrust::_argsort<cpy_uint>(
    size_t *, void *, void *, const std::vector<ptrdiff_t>& shape, intptr_t,
    void *);
template void cupy::thrust::_argsort<cpy_long>(
    size_t *, void *, void *, const std::vector<ptrdiff_t>& shape, intptr_t,
    void *);
template void cupy::thrust::_argsort<cpy_ulong>(
    size_t *, void *, void *, const std::vector<ptrdiff_t>& shape, intptr_t,
    void *);
template void cupy::thrust::_argsort<cpy_float>(
    size_t *, void *, void *, const std::vector<ptrdiff_t>& shape, intptr_t,
    void *);
template void cupy::thrust::_argsort<cpy_double>(
    size_t *, void *, void *, const std::vector<ptrdiff_t>& shape, intptr_t,
    void *);
template void cupy::thrust::_argsort<cpy_complex64>(
    size_t *, void *, void *, const std::vector<ptrdiff_t>& shape, intptr_t,
    void *);
template void cupy::thrust::_argsort<cpy_complex128>(
    size_t *, void *, void *, const std::vector<ptrdiff_t>& shape, intptr_t,
    void *);
template void cupy::thrust::_argsort<cpy_bool>(
    size_t *, void *, void *, const std::vector<ptrdiff_t>& shape, intptr_t,
    void *);
void cupy::thrust::_argsort_fp16(size_t *idx_start, void *data_start,
                                 void *keys_start,
                                 const std::vector<ptrdiff_t>& shape,
                                 intptr_t stream, void *memory) {
#if (__CUDACC_VER_MAJOR__ > 9 || (__CUDACC_VER_MAJOR__ == 9 && __CUDACC_VER_MINOR__ == 2)) \
    && (__CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__))
    cupy::thrust::_argsort<__half>(idx_start, data_start, keys_start, shape, stream, memory);
#endif
}
