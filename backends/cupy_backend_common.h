#ifndef INCLUDE_GUARD_CUPY_CUDA_H
#define INCLUDE_GUARD_CUPY_CUDA_H

#if CUPY_USE_HIP

#include "hip/cupy_hip.h"

#elif CUPY_USE_ASCEND

#include "ascend/cupy_ascend_types.h"

#elif !defined(CUPY_NO_CUDA)

#include "cuda/cupy_cuda.h"

#else // #ifndef CUPY_NO_CUDA

#include "stub/cupy_stub_common.h"

#endif // #ifndef CUPY_NO_CUDA
#endif // #ifndef INCLUDE_GUARD_CUPY_CUDA_H
