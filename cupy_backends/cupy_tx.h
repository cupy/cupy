#ifndef INCLUDE_GUARD_CUPY_NVTX_H
#define INCLUDE_GUARD_CUPY_NVTX_H

#if CUPY_USE_HIP

#include "hip/cupy_roctx.h"
#include "stub/cupy_nvtx.h"

#elif !defined(CUPY_NO_CUDA)

#include <nvToolsExt.h>

#else  // defined(CUPY_NO_CUDA)

#include "stub/cupy_nvtx.h"

#endif

#endif // #ifndef INCLUDE_GUARD_CUPY_NVTX_H
