#ifndef INCLUDE_GUARD_CUPY_CUDA_THRUST_H
#define INCLUDE_GUARD_CUPY_CUDA_THRUST_H

#ifndef CUPY_NO_CUDA
#include <thrust/version.h>  // for THRUST_VERSION

void thrust_sort(int, void *, size_t *, const std::vector<ptrdiff_t>&, intptr_t, void *);
void thrust_lexsort(int, size_t *, void *, size_t, size_t, intptr_t, void *);
void thrust_argsort(int, size_t *, void *, void *, const std::vector<ptrdiff_t>&, intptr_t, void *);

#else // CUPY_NO_CUDA

#define THRUST_VERSION 0

void thrust_sort(...) {
}

void thrust_lexsort(...) {
}

void thrust_argsort(...) {
}

#endif // #ifndef CUPY_NO_CUDA

#endif // INCLUDE_GUARD_CUPY_CUDA_THRUST_H
