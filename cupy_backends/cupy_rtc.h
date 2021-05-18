#ifndef INCLUDE_GUARD_CUPY_NVRTC_H
#define INCLUDE_GUARD_CUPY_NVRTC_H

#ifdef CUPY_USE_HIP

#include "hip/cupy_hiprtc.h"

#elif !defined(CUPY_NO_CUDA)

#include "cuda/cupy_nvrtc.h"

#else

#include "stub/cupy_nvrtc.h"

#endif

#endif // #ifndef INCLUDE_GUARD_CUPY_NVRTC_H
