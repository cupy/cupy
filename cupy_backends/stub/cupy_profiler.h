// This file is a stub header file of cuda for Read the Docs.

#ifndef INCLUDE_GUARD_STUB_CUPY_PROFILER_H
#define INCLUDE_GUARD_STUB_CUPY_PROFILER_H

#include "cupy_cuda_common.h"

extern "C" {

typedef enum {} cudaOutputMode_t;

cudaError_t cudaProfilerInitialize(...) {
  return cudaSuccess;
}

cudaError_t cudaProfilerStart() {
  return cudaSuccess;
}

cudaError_t cudaProfilerStop() {
  return cudaSuccess;
}

} // extern "C"

#endif // #ifndef INCLUDE_GUARD_STUB_CUPY_PROFILER_H
