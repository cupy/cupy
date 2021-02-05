#ifndef INCLUDE_GUARD_CUDA_CUPY_CUDA_RUNTIME_H
#define INCLUDE_GUARD_CUDA_CUPY_CUDA_RUNTIME_H

#include <cuda.h>  // for CUDA_VERSION
#include <cuda_runtime.h>

extern "C" {

bool hip_environment = false;

#if CUDA_VERSION < 10010
const int cudaErrorContextIsDestroyed = 709;
#endif

} // extern "C"

#endif // #ifndef INCLUDE_GUARD_CUDA_CUPY_CUDA_RUNTIME_H
