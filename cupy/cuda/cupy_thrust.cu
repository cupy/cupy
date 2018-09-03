#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/execution_policy.h>
#include "cupy_common.h"
#include "cupy_thrust.h"

using namespace thrust;


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

template void cupy::thrust::_sort<cpy_byte>(
    void *, size_t *, const std::vector<ptrdiff_t>& shape, size_t, void *);
template void cupy::thrust::_sort<cpy_ubyte>(
    void *, size_t *, const std::vector<ptrdiff_t>& shape, size_t, void *);
template void cupy::thrust::_sort<cpy_short>(
    void *, size_t *, const std::vector<ptrdiff_t>& shape, size_t, void *);
template void cupy::thrust::_sort<cpy_ushort>(
    void *, size_t *, const std::vector<ptrdiff_t>& shape, size_t, void *);
template void cupy::thrust::_sort<cpy_int>(
    void *, size_t *, const std::vector<ptrdiff_t>& shape, size_t, void *);
template void cupy::thrust::_sort<cpy_uint>(
    void *, size_t *, const std::vector<ptrdiff_t>& shape, size_t, void *);
template void cupy::thrust::_sort<cpy_long>(
    void *, size_t *, const std::vector<ptrdiff_t>& shape, size_t, void *);
template void cupy::thrust::_sort<cpy_ulong>(
    void *, size_t *, const std::vector<ptrdiff_t>& shape, size_t, void *);
template void cupy::thrust::_sort<cpy_float>(
    void *, size_t *, const std::vector<ptrdiff_t>& shape, size_t, void *);
template void cupy::thrust::_sort<cpy_double>(
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

template void cupy::thrust::_lexsort<cpy_byte>(
    size_t *, void *, size_t, size_t, size_t, void *);
template void cupy::thrust::_lexsort<cpy_ubyte>(
    size_t *, void *, size_t, size_t, size_t, void *);
template void cupy::thrust::_lexsort<cpy_short>(
    size_t *, void *, size_t, size_t, size_t, void *);
template void cupy::thrust::_lexsort<cpy_ushort>(
    size_t *, void *, size_t, size_t, size_t, void *);
template void cupy::thrust::_lexsort<cpy_int>(
    size_t *, void *, size_t, size_t, size_t, void *);
template void cupy::thrust::_lexsort<cpy_uint>(
    size_t *, void *, size_t, size_t, size_t, void *);
template void cupy::thrust::_lexsort<cpy_long>(
    size_t *, void *, size_t, size_t, size_t, void *);
template void cupy::thrust::_lexsort<cpy_ulong>(
    size_t *, void *, size_t, size_t, size_t, void *);
template void cupy::thrust::_lexsort<cpy_float>(
    size_t *, void *, size_t, size_t, size_t, void *);
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
    /* idx_start is the beggining of the output array where the indexes that
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
    size_t *, void *, void *, const std::vector<ptrdiff_t>& shape, size_t,
    void *);
template void cupy::thrust::_argsort<cpy_ubyte>(
    size_t *, void *, void *, const std::vector<ptrdiff_t>& shape, size_t,
    void *);
template void cupy::thrust::_argsort<cpy_short>(
    size_t *, void *, void *, const std::vector<ptrdiff_t>& shape, size_t,
    void *);
template void cupy::thrust::_argsort<cpy_ushort>(
    size_t *, void *, void *, const std::vector<ptrdiff_t>& shape, size_t,
    void *);
template void cupy::thrust::_argsort<cpy_int>(
    size_t *, void *, void *, const std::vector<ptrdiff_t>& shape, size_t,
    void *);
template void cupy::thrust::_argsort<cpy_uint>(
    size_t *, void *, void *, const std::vector<ptrdiff_t>& shape, size_t,
    void *);
template void cupy::thrust::_argsort<cpy_long>(
    size_t *, void *, void *, const std::vector<ptrdiff_t>& shape, size_t,
    void *);
template void cupy::thrust::_argsort<cpy_ulong>(
    size_t *, void *, void *, const std::vector<ptrdiff_t>& shape, size_t,
    void *);
template void cupy::thrust::_argsort<cpy_float>(
    size_t *, void *, void *, const std::vector<ptrdiff_t>& shape, size_t,
    void *);
template void cupy::thrust::_argsort<cpy_double>(
    size_t *, void *, void *, const std::vector<ptrdiff_t>& shape, size_t,
    void *);
