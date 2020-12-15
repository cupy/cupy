#ifndef INCLUDE_GUARD_CUPY_CUDA_RUNTIME_H
#define INCLUDE_GUARD_CUPY_CUDA_RUNTIME_H

#if CUPY_USE_HIP

#include "hip/cupy_hip_runtime.h"

#elif !defined(CUPY_NO_CUDA)

#include "cuda/cupy_cuda_runtime.h"

#else // #ifndef CUPY_NO_CUDA

#include "stub/cupy_cuda_runtime.h"

#endif // #ifndef CUPY_NO_CUDA
#endif // #ifndef INCLUDE_GUARD_CUPY_CUDA_RUNTIME_H
