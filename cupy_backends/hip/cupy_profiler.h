#ifndef INCLUDE_GUARD_HIP_CUPY_PROFILER_H
#define INCLUDE_GUARD_HIP_CUPY_PROFILER_H

#include "cupy_hip_common.h"
// roctracer_start / _stop replace deprecated hipProfilerStart / _Stop
// (libroctracer64 added to the HIP_cuda_nvtx_cusolver link
// group in install/cupy_builder/_features.py.
#include <roctracer/roctracer_ext.h>

extern "C" {

cudaError_t cudaProfilerStart() {
  roctracer_start();
  return hipSuccess;
}

cudaError_t cudaProfilerStop() {
  roctracer_stop();
  return hipSuccess;
}

} // extern "C"

#endif // #ifndef INCLUDE_GUARD_HIP_CUPY_PROFILER_H
