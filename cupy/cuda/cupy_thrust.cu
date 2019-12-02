#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/execution_policy.h>
//#include <thrust/complex.h>
#include "cupy_common.h"
#include "cupy_thrust.h"
#include "cupy_cuComplex.h"

using namespace thrust;


#if CUPY_USE_HIP
typedef hipStream_t cudaStream_t;
namespace cuda {
    using thrust::hip::par;
}

#endif // #if CUPY_USE_HIP


extern "C" char *cupy_malloc(void *, ptrdiff_t);
extern "C" void cupy_free(void *, char *);


class cupy_allocator {
private:
    void* memory;

public:
    typedef char value_type;

    cupy_allocator(void* memory) : memory(memory) {}

    char *allocate(std::ptrdiff_t num_bytes) {
        return cupy_malloc(memory, num_bytes);
    }

    void deallocate(char *ptr, size_t n) {
        cupy_free(memory, ptr);
    }
};


/* ------------------------------------ Minimum boilerplate to support complex numbers ------------------------------------ */
// copied from cupy/core/include/cupy/complex/arithmetic.h
//template <typename T>
//__host__ __device__ inline T thrust::complex<T>::real(const device_reference<complex<T>>& z) {
//  return z.real();
//}
//
//template <typename T>
//__host__ __device__ inline bool thrust::operator<(const complex<T>& lhs,
//                                                           const complex<T>& rhs) {
//  if (lhs == rhs) {
//      return false;
//  } else if (lhs.real() < rhs.real()) {
//      return true;
//  } else if (lhs.real() == rhs.real()) {
//      return lhs.imag() < rhs.imag();
//  } else {
//      return false;
//  }
//}
__host__ __device__ inline bool isnan(const cuComplex& z) {
    return isnan(z.x) || isnan(z.y);
}

__host__ __device__ inline bool isnan(const cuDoubleComplex& z) {
    return isnan(z.x) || isnan(z.y);
}

__host__ __device__ inline bool operator<(const cuComplex& lhs,
                                          const cuComplex& rhs) {
  if (isnan(lhs)) {
      if (!isnan(rhs)) {
          return false;
      } else if (isnan(lhs.x) && !isnan(rhs.x)) {
          return false;
      } else if (!isnan(lhs.x) && isnan(rhs.x)) {
          return true;
      } else if (isnan(lhs.y) && !isnan(rhs.y)) {
          return false;
      } else if (!isnan(lhs.y) && isnan(rhs.y)) {
          return true;
      } else if (isnan(lhs.y) && isnan(rhs.y)) {
          return lhs.x < rhs.x;
      } else if (isnan(lhs.x) && isnan(rhs.x)) {
          return lhs.y < rhs.y;
      } else { // both lhs & rhs = nan + nan I
          return true;
      }
  } else { // !isnan(lhs)
      if (isnan(rhs)) {
          return true;
      }
  }

  if (lhs.x == rhs.x && lhs.y == rhs.y) {
      return false;
  } else if (lhs.x < rhs.x) {
      return true;
  } else if (lhs.x == rhs.x) {
      return lhs.y < rhs.y;
  } else {
      return false;
  }
}
__host__ __device__ inline bool operator<(const cuDoubleComplex& lhs,
                                          const cuDoubleComplex& rhs) {
  if (isnan(lhs)) {
      if (!isnan(rhs)) {
          return false;
      } else if (isnan(lhs.x) && !isnan(rhs.x)) {
          return false;
      } else if (!isnan(lhs.x) && isnan(rhs.x)) {
          return true;
      } else if (isnan(lhs.y) && !isnan(rhs.y)) {
          return false;
      } else if (!isnan(lhs.y) && isnan(rhs.y)) {
          return true;
      } else if (isnan(lhs.y) && isnan(rhs.y)) {
          return lhs.x < rhs.x;
      } else if (isnan(lhs.x) && isnan(rhs.x)) {
          return lhs.y < rhs.y;
      } else { // both lhs & rhs = nan + nan I
          return true;
      }
  } else { // !isnan(lhs)
      if (isnan(rhs)) {
          return true;
      }
  }

  if (lhs.x == rhs.x && lhs.y == rhs.y) {
      return false;
  } else if (lhs.x < rhs.x) {
      return true;
  } else if (lhs.x == rhs.x) {
      return lhs.y < rhs.y;
  } else {
      return false;
  }
}

//template <typename ValueType>
//__host__ __device__ inline bool operator<(const ValueType& lhs,
//                                 const complex<ValueType>& rhs) {
//    return complex<ValueType>(lhs) < rhs;
//}
//
//template <typename ValueType>
//__host__ __device__ inline bool operator<(const complex<ValueType>& lhs,
//                                 const ValueType& rhs) {
//    return lhs < complex<ValueType>(rhs);
//}
/* ------------------------------------ end of boilerplate ------------------------------------ */


/*
 * sort
 */

template <typename T>
void cupy::thrust::_sort(void *data_start, size_t *keys_start,
                         const std::vector<ptrdiff_t>& shape, size_t stream,
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
        stable_sort(cuda::par(alloc).on(stream_), dp_data_first, dp_data_last);
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
            make_zip_iterator(make_tuple(dp_keys_last, dp_data_last)));
    }
}

//template void cupy::thrust::_sort<cpy_byte>(
//    void *, size_t *, const std::vector<ptrdiff_t>& shape, size_t, void *);
//template void cupy::thrust::_sort<cpy_ubyte>(
//    void *, size_t *, const std::vector<ptrdiff_t>& shape, size_t, void *);
//template void cupy::thrust::_sort<cpy_short>(
//    void *, size_t *, const std::vector<ptrdiff_t>& shape, size_t, void *);
//template void cupy::thrust::_sort<cpy_ushort>(
//    void *, size_t *, const std::vector<ptrdiff_t>& shape, size_t, void *);
//template void cupy::thrust::_sort<cpy_int>(
//    void *, size_t *, const std::vector<ptrdiff_t>& shape, size_t, void *);
//template void cupy::thrust::_sort<cpy_uint>(
//    void *, size_t *, const std::vector<ptrdiff_t>& shape, size_t, void *);
//template void cupy::thrust::_sort<cpy_long>(
//    void *, size_t *, const std::vector<ptrdiff_t>& shape, size_t, void *);
//template void cupy::thrust::_sort<cpy_ulong>(
//    void *, size_t *, const std::vector<ptrdiff_t>& shape, size_t, void *);
//template void cupy::thrust::_sort<cpy_float>(
//    void *, size_t *, const std::vector<ptrdiff_t>& shape, size_t, void *);
template void cupy::thrust::_sort<cpy_double>(
    void *, size_t *, const std::vector<ptrdiff_t>& shape, size_t, void *);
//template void cupy::thrust::_sort<complex<float>>(
//    void *, size_t *, const std::vector<ptrdiff_t>& shape, size_t, void *);
//template void cupy::thrust::_sort<complex<double>>(
//    void *, size_t *, const std::vector<ptrdiff_t>& shape, size_t, void *);
template void cupy::thrust::_sort<cuComplex>(
    void *, size_t *, const std::vector<ptrdiff_t>& shape, size_t, void *);
template void cupy::thrust::_sort<cuDoubleComplex>(
    void *, size_t *, const std::vector<ptrdiff_t>& shape, size_t, void *);


/*
 * lexsort
 */

template <typename T>
class elem_less {
public:
    elem_less(const T *data):_data(data) {}
    __device__ bool operator()(size_t i, size_t j) {
        return _data[i] < _data[j];
    }
private:
    const T *_data;
};

template <typename T>
void cupy::thrust::_lexsort(size_t *idx_start, void *keys_start, size_t k,
                            size_t n, size_t stream, void *memory) {
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

//template void cupy::thrust::_lexsort<cpy_byte>(
//    size_t *, void *, size_t, size_t, size_t, void *);
//template void cupy::thrust::_lexsort<cpy_ubyte>(
//    size_t *, void *, size_t, size_t, size_t, void *);
//template void cupy::thrust::_lexsort<cpy_short>(
//    size_t *, void *, size_t, size_t, size_t, void *);
//template void cupy::thrust::_lexsort<cpy_ushort>(
//    size_t *, void *, size_t, size_t, size_t, void *);
//template void cupy::thrust::_lexsort<cpy_int>(
//    size_t *, void *, size_t, size_t, size_t, void *);
//template void cupy::thrust::_lexsort<cpy_uint>(
//    size_t *, void *, size_t, size_t, size_t, void *);
//template void cupy::thrust::_lexsort<cpy_long>(
//    size_t *, void *, size_t, size_t, size_t, void *);
//template void cupy::thrust::_lexsort<cpy_ulong>(
//    size_t *, void *, size_t, size_t, size_t, void *);
//template void cupy::thrust::_lexsort<cpy_float>(
//    size_t *, void *, size_t, size_t, size_t, void *);
template void cupy::thrust::_lexsort<cpy_double>(
    size_t *, void *, size_t, size_t, size_t, void *);


/*
 * argsort
 */

template <typename T>
void cupy::thrust::_argsort(size_t *idx_start, void *data_start,
                            void *keys_start,
                            const std::vector<ptrdiff_t>& shape,
                            size_t stream, void *memory) {
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

//template void cupy::thrust::_argsort<cpy_byte>(
//    size_t *, void *, void *, const std::vector<ptrdiff_t>& shape, size_t,
//    void *);
//template void cupy::thrust::_argsort<cpy_ubyte>(
//    size_t *, void *, void *, const std::vector<ptrdiff_t>& shape, size_t,
//    void *);
//template void cupy::thrust::_argsort<cpy_short>(
//    size_t *, void *, void *, const std::vector<ptrdiff_t>& shape, size_t,
//    void *);
//template void cupy::thrust::_argsort<cpy_ushort>(
//    size_t *, void *, void *, const std::vector<ptrdiff_t>& shape, size_t,
//    void *);
//template void cupy::thrust::_argsort<cpy_int>(
//    size_t *, void *, void *, const std::vector<ptrdiff_t>& shape, size_t,
//    void *);
//template void cupy::thrust::_argsort<cpy_uint>(
//    size_t *, void *, void *, const std::vector<ptrdiff_t>& shape, size_t,
//    void *);
//template void cupy::thrust::_argsort<cpy_long>(
//    size_t *, void *, void *, const std::vector<ptrdiff_t>& shape, size_t,
//    void *);
//template void cupy::thrust::_argsort<cpy_ulong>(
//    size_t *, void *, void *, const std::vector<ptrdiff_t>& shape, size_t,
//    void *);
//template void cupy::thrust::_argsort<cpy_float>(
//    size_t *, void *, void *, const std::vector<ptrdiff_t>& shape, size_t,
//    void *);
template void cupy::thrust::_argsort<cpy_double>(
    size_t *, void *, void *, const std::vector<ptrdiff_t>& shape, size_t,
    void *);
