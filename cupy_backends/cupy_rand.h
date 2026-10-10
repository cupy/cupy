#ifndef INCLUDE_GUARD_CUPY_CURAND_H
#define INCLUDE_GUARD_CUPY_CURAND_H

#if CUPY_USE_HIP

#include "hip/cupy_hiprand.h"

#else

#include <curand.h>

#endif
#endif // #ifndef INCLUDE_GUARD_CUPY_CURAND_H
