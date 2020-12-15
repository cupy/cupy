#ifndef INCLUDE_GUARD_CUPY_CUBLAS_H
#define INCLUDE_GUARD_CUPY_CUBLAS_H

#if CUPY_USE_HIP

#include "hip/cupy_hipblas.h"

#elif !defined(CUPY_NO_CUDA)

#include "cuda/cupy_cublas.h"

#else // #ifndef CUPY_NO_CUDA

#include "stub/cupy_cublas.h"

#endif // #ifndef CUPY_NO_CUDA
#endif // #ifndef INCLUDE_GUARD_CUPY_CUBLAS_H
