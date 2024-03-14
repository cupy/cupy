#ifndef INCLUDE_GUARD_HIP_CUPY_PROFILER_H
#define INCLUDE_GUARD_HIP_CUPY_PROFILER_H

#include "cupy_hip_common.h"

extern "C" {

cudaError_t cudaProfilerStart() {
  return hipProfilerStart();
}

cudaError_t cudaProfilerStop() {
  return hipProfilerStop();
}

} // extern "C"

#endif // #ifndef INCLUDE_GUARD_HIP_CUPY_PROFILER_H
