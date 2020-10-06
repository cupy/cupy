#ifndef INCLUDE_GUARD_CUPY_CUTENSOR_H
#define INCLUDE_GUARD_CUPY_CUTENSOR_H

#ifndef CUPY_NO_CUDA

#include <library_types.h>
#include <cutensor.h>

#else // #ifndef CUPY_NO_CUDA

#include "stub/cupy_cutensor.h"

#endif // #ifndef CUPY_NO_CUDA

#endif // #ifndef INCLUDE_GUARD_CUPY_CUTENSOR_H
