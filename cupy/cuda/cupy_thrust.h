#ifndef INCLUDE_GUARD_CUPY_CUDA_THRUST_H
#define INCLUDE_GUARD_CUPY_CUDA_THRUST_H

#ifndef CUPY_NO_CUDA

namespace cupy {

namespace thrust {

template <typename T> void _sort(void *, ptrdiff_t);

template <typename T> void _lexsort(size_t *, void *, size_t, size_t);
   
template <typename T> void _argsort(size_t *, void *, size_t);

} // namespace thrust

} // namespace cupy

#else // CUPY_NO_CUDA

#include "cupy_common.h"

namespace cupy {

namespace thrust {

template <typename T> void _sort(void *, ptrdiff_t) { return; }

template <typename T> void _lexsort(size_t *, void *, size_t, size_t) { return; }

template <typename T> void _argsort(size_t *, void *, size_t) { return; }

} // namespace thrust

} // namespace cupy

#endif // #ifndef CUPY_NO_CUDA

#endif // INCLUDE_GUARD_CUPY_CUDA_THRUST_H
