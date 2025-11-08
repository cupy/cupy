#ifndef INCLUDE_GUARD_CUPY_PROFILER_H
#define INCLUDE_GUARD_CUPY_PROFILER_H

#if CUPY_USE_HIP

#include "hip/cupy_profiler.h"

#elif CUPY_USE_ASCEND

#include "ascend/cupy_ascend_profiler.h"

#elif !defined(CUPY_NO_CUDA)

#include "cuda/cupy_cuda_profiler_api.h"

#else // #ifndef CUPY_NO_CUDA

#include "stub/cupy_stub_profiler.h"

#endif // #ifndef CUPY_NO_CUDA
#endif // #ifndef INCLUDE_GUARD_CUPY_PROFILER_H
