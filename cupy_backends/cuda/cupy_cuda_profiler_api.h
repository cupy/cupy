#ifndef INCLUDE_GUARD_CUDA_CUPY_PROFILER_H
#define INCLUDE_GUARD_CUDA_CUPY_PROFILER_H

#include <cuda.h>
#include <cuda_profiler_api.h>

#if CUDA_VERSION >= 12000
// Functions removed in CUDA 12.0

enum cudaOutputMode_t {};

cudaError_t cudaProfilerInitialize(...) {
    return cudaErrorUnknown;
}

#endif // #if CUDA_VERSION >= 12

#endif // #ifndef INCLUDE_GUARD_CUDA_CUPY_PROFILER_H
