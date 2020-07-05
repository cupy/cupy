#ifndef INCLUDE_GUARD_CUPY_CUDA_THRUST_H
#define INCLUDE_GUARD_CUPY_CUDA_THRUST_H

#ifndef CUPY_NO_CUDA
#include <thrust/version.h>  // for THRUST_VERSION

namespace cupy {

namespace thrust {

void thrust_sort(int, void *, size_t *, const std::vector<ptrdiff_t>&, intptr_t, void *);
void thrust_lexsort(int, size_t *, void *, size_t, size_t, intptr_t, void *);
void thrust_argsort(int, size_t *, void *, void *, const std::vector<ptrdiff_t>&, intptr_t, void *);

// not exposed to Python
struct _sort;
struct _lexsort;
struct _argsort;

} // namespace thrust

} // namespace cupy

#else // CUPY_NO_CUDA

#include "cupy_common.h"
#define THRUST_VERSION 0

namespace cupy {

namespace thrust {

void thrust_sort(int, void *, size_t *, const std::vector<ptrdiff_t>&, intptr_t, void *) {
}

void thrust_lexsort(int, size_t *, void *, size_t, size_t, intptr_t, void *) {
}

void thrust_argsort(int, size_t *, void *, void *, const std::vector<ptrdiff_t>&, intptr_t, void *) {
}

} // namespace thrust

} // namespace cupy

#endif // #ifndef CUPY_NO_CUDA

#endif // INCLUDE_GUARD_CUPY_CUDA_THRUST_H
