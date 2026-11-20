#ifndef INCLUDE_GUARD_HIP_CUPY_PROFILER_H
#define INCLUDE_GUARD_HIP_CUPY_PROFILER_H

#include "cupy_hip_common.h"
// roctracer_start / _stop replace deprecated hipProfilerStart / _Stop
// (libroctracer64 added to the HIP_cuda_nvtx_cusolver link
// group in install/cupy_builder/_features.py.
#ifndef CUPY_NO_ROCTRACER
#include <roctracer/roctracer_ext.h>
#endif

extern "C" {

// Without roctracer (e.g., ROCm on Windows), these succeed without effect, as
// cudaProfilerStart/Stop do when no profiler is attached.
cudaError_t cudaProfilerStart() {
#ifndef CUPY_NO_ROCTRACER
  roctracer_start();
#endif
  return hipSuccess;
}

cudaError_t cudaProfilerStop() {
#ifndef CUPY_NO_ROCTRACER
  roctracer_stop();
#endif
  return hipSuccess;
}

} // extern "C"

#endif // #ifndef INCLUDE_GUARD_HIP_CUPY_PROFILER_H
