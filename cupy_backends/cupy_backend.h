#ifndef INCLUDE_GUARD_CUPY_CUDA_H
#define INCLUDE_GUARD_CUPY_CUDA_H

#if CUPY_USE_HIP

#include "hip/cupy_hip.h"

#elif !defined(CUPY_NO_CUDA)

#include "cuda/cupy_cuda.h"

#else // #ifndef CUPY_NO_CUDA

#include "stub/cupy_cuda.h"

#endif // #ifndef CUPY_NO_CUDA
#endif // #ifndef INCLUDE_GUARD_CUPY_CUDA_H
