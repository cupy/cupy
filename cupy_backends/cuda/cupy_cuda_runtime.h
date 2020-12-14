// This file is a stub header file of cuda for Read the Docs.

#ifndef INCLUDE_GUARD_CUPY_CUDA_RUNTIME_H
#define INCLUDE_GUARD_CUPY_CUDA_RUNTIME_H

#if CUPY_USE_HIP

extern "C" {

bool hip_environment = true;

} // extern "C"


#include "hip/cupy_runtime.h"

#elif !defined(CUPY_NO_CUDA)

#include <cuda_runtime.h>

extern "C" {

bool hip_environment = false;

} // extern "C"

#else // #ifndef CUPY_NO_CUDA

extern "C" {

bool hip_environment = false;

} // extern "C"

#include "stub/cupy_cuda_runtime.h"

#endif // #ifndef CUPY_NO_CUDA
#endif // #ifndef INCLUDE_GUARD_CUPY_CUDA_RUNTIME_H
