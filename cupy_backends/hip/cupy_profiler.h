#ifndef INCLUDE_GUARD_HIP_CUPY_PROFILER_H
#define INCLUDE_GUARD_HIP_CUPY_PROFILER_H

#include "cupy_hip_common.h"
#ifndef CUPY_NO_HIP_PROFILER
// roctracer replaces the deprecated hipProfilerStart / hipProfilerStop.
#include <roctracer/roctracer_ext.h>
#endif

extern "C" {

cudaError_t cudaProfilerStart() {
#ifdef CUPY_NO_HIP_PROFILER
  return hipErrorNotSupported;
#else
  roctracer_start();
  return hipSuccess;
#endif
}

cudaError_t cudaProfilerStop() {
#ifdef CUPY_NO_HIP_PROFILER
  return hipErrorNotSupported;
#else
  roctracer_stop();
  return hipSuccess;
#endif
}

} // extern "C"

#endif // #ifndef INCLUDE_GUARD_HIP_CUPY_PROFILER_H
