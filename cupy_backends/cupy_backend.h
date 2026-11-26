#ifndef INCLUDE_GUARD_CUPY_CUDA_H
#define INCLUDE_GUARD_CUPY_CUDA_H

#if CUPY_USE_HIP

#include "hip/cupy_hip.h"

#else

#include "cuda/cupy_cuda.h"

#endif
#endif // #ifndef INCLUDE_GUARD_CUPY_CUDA_H
