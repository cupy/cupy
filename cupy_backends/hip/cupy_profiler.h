#ifndef INCLUDE_GUARD_HIP_CUPY_PROFILER_H
#define INCLUDE_GUARD_HIP_CUPY_PROFILER_H

#include "cupy_hip_common.h"

extern "C" {

typedef enum {} cudaOutputMode_t;

cudaError_t cudaProfilerInitialize(...) {
  return cudaSuccess;
}

cudaError_t cudaProfilerStart() {
  return hipProfilerStart();
}

cudaError_t cudaProfilerStop() {
  return hipProfilerStop();
}

} // extern "C"

#endif // #ifndef INCLUDE_GUARD_HIP_CUPY_PROFILER_H
