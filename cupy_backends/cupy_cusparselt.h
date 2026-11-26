#ifndef INCLUDE_GUARD_CUPY_CUSPARSELT_H
#define INCLUDE_GUARD_CUPY_CUSPARSELT_H

#ifdef CUPY_USE_HIP

#include "hip/cupy_cusparselt.h"

#else

#include <cusparseLt.h>

#endif

#endif // #ifndef INCLUDE_GUARD_CUPY_CUSPARSELT_H
