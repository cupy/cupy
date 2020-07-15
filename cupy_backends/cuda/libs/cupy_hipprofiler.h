// This file is a stub header file of hip for Read the Docs.

#ifndef INCLUDE_GUARD_CUPY_HIPPROFILER_H
#define INCLUDE_GUARD_CUPY_HIPPROFILER_H

#include <hip/hip_runtime_api.h>
#include "../cupy_hip_common.h"

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

#endif // #ifndef INCLUDE_GUARD_CUPY_HIPPROFILER_H
