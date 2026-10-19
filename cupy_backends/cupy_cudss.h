#ifndef INCLUDE_GUARD_CUPY_CUDSS_H
#define INCLUDE_GUARD_CUPY_CUDSS_H

#if defined(CUPY_USE_HIP) || defined(CUPY_NO_CUDA)
#include "hip/cupy_cudss.h"
#else
#include <cudss.h>
#endif

#endif  // INCLUDE_GUARD_CUPY_CUDSS_H
