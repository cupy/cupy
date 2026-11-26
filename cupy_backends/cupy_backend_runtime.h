#ifndef INCLUDE_GUARD_CUPY_CUDA_RUNTIME_H
#define INCLUDE_GUARD_CUPY_CUDA_RUNTIME_H

#if CUPY_USE_HIP

#include "hip/cupy_hip_runtime.h"

#else

#include "cuda/cupy_cuda_runtime.h"

#endif
#endif // #ifndef INCLUDE_GUARD_CUPY_CUDA_RUNTIME_H
