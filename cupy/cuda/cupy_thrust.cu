#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include "cupy_common.h"
#include "cupy_thrust.h"

using namespace thrust;


/*
 * sort
 */

template <typename T>
void cupy::thrust::_sort(void *start, const std::vector<ptrdiff_t>& shape) {

    size_t ndim = shape.size();
    ptrdiff_t size;
    device_ptr<T> dp_first, dp_last;

    // Compute the total size of the array.
    size = shape[0];
    for (size_t i = 1; i < ndim; ++i) {
        size *= shape[i];
    }

    dp_first = device_pointer_cast(static_cast<T*>(start));
    dp_last  = device_pointer_cast(static_cast<T*>(start) + size);

    if (ndim == 1) {
        stable_sort(dp_first, dp_last);
    } else {
        device_vector<size_t> d_keys(size);

        // Generate key indices.
        transform(make_counting_iterator<size_t>(0),
                  make_counting_iterator<size_t>(size),
                  make_constant_iterator<ptrdiff_t>(shape[ndim-1]),
                  d_keys.begin(),
                  divides<size_t>());

        // Sorting with back-to-back approach.
        stable_sort_by_key(dp_first,
                           dp_last,
                           d_keys.begin(),
                           less<T>());

        stable_sort_by_key(d_keys.begin(),
                           d_keys.end(),
                           dp_first,
                           less<size_t>());
    }
}

template void cupy::thrust::_sort<cpy_byte>(void *, const std::vector<ptrdiff_t>& shape);
template void cupy::thrust::_sort<cpy_ubyte>(void *, const std::vector<ptrdiff_t>& shape);
template void cupy::thrust::_sort<cpy_short>(void *, const std::vector<ptrdiff_t>& shape);
template void cupy::thrust::_sort<cpy_ushort>(void *, const std::vector<ptrdiff_t>& shape);
template void cupy::thrust::_sort<cpy_int>(void *, const std::vector<ptrdiff_t>& shape);
template void cupy::thrust::_sort<cpy_uint>(void *, const std::vector<ptrdiff_t>& shape);
template void cupy::thrust::_sort<cpy_long>(void *, const std::vector<ptrdiff_t>& shape);
template void cupy::thrust::_sort<cpy_ulong>(void *, const std::vector<ptrdiff_t>& shape);
template void cupy::thrust::_sort<cpy_float>(void *, const std::vector<ptrdiff_t>& shape);
template void cupy::thrust::_sort<cpy_double>(void *, const std::vector<ptrdiff_t>& shape);


/*
 * lexsort
 */

template <typename T>
class elem_less {
public:
    elem_less(const T *data):_data(data) {}
    __device__ bool operator()(size_t i, size_t j) { return _data[i] < _data[j]; }
private:
    const T *_data;
};

template <typename T>
void cupy::thrust::_lexsort(size_t *idx_start, void *keys_start, size_t k, size_t n) {
    /* idx_start is the beginning of the output array where the indexes that
       would sort the data will be placed. The original contents of idx_start
       will be destroyed. */
    device_ptr<size_t> dp_first = device_pointer_cast(idx_start);
    device_ptr<size_t> dp_last  = device_pointer_cast(idx_start + n);
    sequence(dp_first, dp_last);
    for (size_t i = 0; i < k; ++i) {
        T *key_start = static_cast<T*>(keys_start) + i * n;
        stable_sort< device_ptr<size_t> >(dp_first, dp_last, elem_less<T>(key_start));
    }
}

template void cupy::thrust::_lexsort<cpy_byte>(size_t *, void *, size_t, size_t);
template void cupy::thrust::_lexsort<cpy_ubyte>(size_t *, void *, size_t, size_t);
template void cupy::thrust::_lexsort<cpy_short>(size_t *, void *, size_t, size_t);
template void cupy::thrust::_lexsort<cpy_ushort>(size_t *, void *, size_t, size_t);
template void cupy::thrust::_lexsort<cpy_int>(size_t *, void *, size_t, size_t);
template void cupy::thrust::_lexsort<cpy_uint>(size_t *, void *, size_t, size_t);
template void cupy::thrust::_lexsort<cpy_long>(size_t *, void *, size_t, size_t);
template void cupy::thrust::_lexsort<cpy_ulong>(size_t *, void *, size_t, size_t);
template void cupy::thrust::_lexsort<cpy_float>(size_t *, void *, size_t, size_t);
template void cupy::thrust::_lexsort<cpy_double>(size_t *, void *, size_t, size_t);


/*
 * argsort
 */

template <typename T>
void cupy::thrust::_argsort(size_t *idx_start, void *data_start, void *keys_start, void *buff_start, const std::vector<ptrdiff_t>& shape) {
    /* idx_start is the beggining of the output array where the indexes that
       would sort the data will be placed. The original contents of idx_start
       will be destroyed. */

    size_t ndim = shape.size();
    ptrdiff_t size;

    device_ptr<size_t> dp_idx_first, dp_idx_last;
    device_ptr<T> dp_data_first, dp_data_last;
    device_ptr<size_t> dp_keys_first, dp_keys_last;
    device_ptr<T> dp_buff_first, dp_buff_last;

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
    transform(make_counting_iterator<size_t>(0),
              make_counting_iterator<size_t>(size),
              make_constant_iterator<ptrdiff_t>(shape[ndim-1]),
              dp_idx_first,
              modulus<size_t>());

    if (ndim == 1) {
        // Sort the index sequence by data.
        stable_sort_by_key(dp_data_first,
                           dp_data_last,
                           dp_idx_first,
                           less<T>());
    } else {
        // Generate key indices.
        dp_keys_first = device_pointer_cast(static_cast<size_t*>(keys_start));
        dp_keys_last  = device_pointer_cast(static_cast<size_t*>(keys_start) + size);
        transform(make_counting_iterator<size_t>(0),
                  make_counting_iterator<size_t>(size),
                  make_constant_iterator<ptrdiff_t>(shape[ndim-1]),
                  dp_keys_first,
                  divides<size_t>());

        // Copy data to the buffer.
        dp_buff_first = device_pointer_cast(static_cast<T*>(buff_start));
        dp_buff_last  = device_pointer_cast(static_cast<T*>(buff_start) + size);
        copy(dp_data_first, dp_data_last, dp_buff_first);

        // Sorting with back-to-back approach.

        stable_sort_by_key(dp_buff_first,
                           dp_buff_last,
                           dp_keys_first,
                           less<T>());

        stable_sort_by_key(dp_data_first,
                           dp_data_last,
                           dp_idx_first,
                           less<T>());

        stable_sort_by_key(dp_keys_first,
                           dp_keys_last,
                           dp_idx_first,
                           less<size_t>());
    }
}

template void cupy::thrust::_argsort<cpy_byte>(size_t *, void *, void *, void *, const std::vector<ptrdiff_t>& shape);
template void cupy::thrust::_argsort<cpy_ubyte>(size_t *, void *, void *, void *, const std::vector<ptrdiff_t>& shape);
template void cupy::thrust::_argsort<cpy_short>(size_t *, void *, void *, void *, const std::vector<ptrdiff_t>& shape);
template void cupy::thrust::_argsort<cpy_ushort>(size_t *, void *, void *, void *, const std::vector<ptrdiff_t>& shape);
template void cupy::thrust::_argsort<cpy_int>(size_t *, void *, void *, void *, const std::vector<ptrdiff_t>& shape);
template void cupy::thrust::_argsort<cpy_uint>(size_t *, void *, void *, void *, const std::vector<ptrdiff_t>& shape);
template void cupy::thrust::_argsort<cpy_long>(size_t *, void *, void *, void *, const std::vector<ptrdiff_t>& shape);
template void cupy::thrust::_argsort<cpy_ulong>(size_t *, void *, void *, void *, const std::vector<ptrdiff_t>& shape);
template void cupy::thrust::_argsort<cpy_float>(size_t *, void *, void *, void *, const std::vector<ptrdiff_t>& shape);
template void cupy::thrust::_argsort<cpy_double>(size_t *, void *, void *, void *, const std::vector<ptrdiff_t>& shape);
