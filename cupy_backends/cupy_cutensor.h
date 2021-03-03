#ifndef INCLUDE_GUARD_CUPY_CUTENSOR_H
#define INCLUDE_GUARD_CUPY_CUTENSOR_H

#ifdef CUPY_USE_HIP

// Since ROCm/HIP does not have cuTENSOR, we simply include the stubs here
// to avoid code dup.
#include "stub/cupy_cutensor.h"

#elif !defined(CUPY_NO_CUDA)

#include <library_types.h>
#include <cutensor.h>

#else

#include "stub/cupy_cutensor.h"

#endif // #ifndef CUPY_NO_CUDA

#endif // #ifndef INCLUDE_GUARD_CUPY_CUTENSOR_H
