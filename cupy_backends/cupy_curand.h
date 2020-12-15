#ifndef INCLUDE_GUARD_CUPY_CURAND_H
#define INCLUDE_GUARD_CUPY_CURAND_H

#if CUPY_USE_HIP

#include "hip/cupy_hiprand.h"

#elif !defined(CUPY_NO_CUDA)

#include <curand.h>

#else // #ifndef CUPY_NO_CUDA

#include "stub/cupy_curand.h"

#endif // #ifndef CUPY_NO_CUDA
#endif // #ifndef INCLUDE_GUARD_CUPY_CURAND_H
