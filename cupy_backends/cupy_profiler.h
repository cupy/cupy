#ifndef INCLUDE_GUARD_CUPY_PROFILER_H
#define INCLUDE_GUARD_CUPY_PROFILER_H

#if CUPY_USE_HIP

#include "hip/cupy_profiler.h"

#else

#include "cuda/cupy_cuda_profiler_api.h"

#endif
#endif // #ifndef INCLUDE_GUARD_CUPY_PROFILER_H
