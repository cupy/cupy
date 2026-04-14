#ifndef INCLUDE_GUARD_CUPY_CUTENSOR_H
#define INCLUDE_GUARD_CUPY_CUTENSOR_H

#ifdef CUPY_USE_HIP

#include "hip/cupy_hiptensor.h"

#elif !defined(CUPY_NO_CUDA)

#include "cuda/cupy_cutensor.h"

#else

#include "stub/cupy_cutensor.h"

#endif // #ifndef CUPY_NO_CUDA

#endif // #ifndef INCLUDE_GUARD_CUPY_CUTENSOR_H
