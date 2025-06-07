#ifndef INCLUDE_GUARD_CUPY_CUDA_THRUST_H
#define INCLUDE_GUARD_CUPY_CUDA_THRUST_H

#ifndef CUPY_NO_CUDA
#include <vector>
#include <cstdint>

#ifndef CUPY_USE_HIP
#include <thrust/version.h>  // for THRUST_VERSION
#else
// WAR #9098:
// rocThrust 3.3.0 (ROCm 6.4.0) cannot be compiled by host compiler
#define THRUST_VERSION 0
#endif

void thrust_sort(int, void *, size_t *, const std::vector<ptrdiff_t>&, intptr_t, void *);
void thrust_lexsort(int, size_t *, void *, size_t, size_t, intptr_t, void *);
void thrust_argsort(int, size_t *, void *, void *, const std::vector<ptrdiff_t>&, intptr_t, void *);
void thrust_argsort2d(int, size_t *, void *, void *, const std::vector<ptrdiff_t>&, intptr_t, void *);

#if (defined(_MSC_VER) && (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ == 2))
  #define __builtin_unreachable() __assume(false)
#endif

#else // CUPY_NO_CUDA

#define THRUST_VERSION 0

void thrust_sort(...) {
}

void thrust_lexsort(...) {
}

void thrust_argsort(...) {
}

void thrust_argsort2d(...) {
}

#endif // #ifndef CUPY_NO_CUDA

#endif // INCLUDE_GUARD_CUPY_CUDA_THRUST_H
