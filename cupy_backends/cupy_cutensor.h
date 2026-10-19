#ifndef INCLUDE_GUARD_CUPY_CUTENSOR_H
#define INCLUDE_GUARD_CUPY_CUTENSOR_H

#ifdef CUPY_USE_HIP

// Since ROCm/HIP does not have cuTENSOR, we simply include the stubs here
// to avoid code dup.
#include "hip/cupy_cutensor.h"

#else

#include "cuda/cupy_cutensor.h"

#endif

#endif // #ifndef INCLUDE_GUARD_CUPY_CUTENSOR_H
