#ifndef INCLUDE_GUARD_CUPY_CUDA_THRUST_H
#define INCLUDE_GUARD_CUPY_CUDA_THRUST_H

#ifndef CUPY_NO_CUDA

namespace cupy {

namespace thrust {

template <typename T>
void _sort(void *, size_t *, const std::vector<ptrdiff_t>&, size_t, void *);

template <typename T>
void _lexsort(size_t *, void *, size_t, size_t, size_t, void *);

template <typename T>
void _argsort(size_t *, void *, void *, const std::vector<ptrdiff_t>&, size_t,
              void *);

/*
   The functions with the suffix _fp16 are used only when certain conditions are met
*/
void _sort_fp16(void *, size_t *, const std::vector<ptrdiff_t>&, size_t, void *);

void _lexsort_fp16(size_t *, void *, size_t, size_t, size_t, void *);

void _argsort_fp16(size_t *, void *, void *, const std::vector<ptrdiff_t>&, size_t,
                   void *);

} // namespace thrust

} // namespace cupy

#else // CUPY_NO_CUDA

#include "cupy_common.h"

namespace cupy {

namespace thrust {

template <typename T>
void _sort(void *, size_t *, const std::vector<ptrdiff_t>&, size_t, void *) {
    return;
}

template <typename T>
void _lexsort(size_t *, void *, size_t, size_t, size_t, void *) {
    return;
}

template <typename T>
void _argsort(size_t *, void *, void *, const std::vector<ptrdiff_t>&, size_t,
              void *) {
    return;
}

void _sort_fp16(void *, size_t *, const std::vector<ptrdiff_t>&, size_t, void *) {
}

void _lexsort_fp16(size_t *, void *, size_t, size_t, size_t, void *) {
}

void _argsort_fp16(size_t *, void *, void *, const std::vector<ptrdiff_t>&, size_t,
                   void *) {
}

} // namespace thrust

} // namespace cupy

#endif // #ifndef CUPY_NO_CUDA

#endif // INCLUDE_GUARD_CUPY_CUDA_THRUST_H
