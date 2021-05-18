#ifndef INCLUDE_GUARD_CUPY_CUSPARSELT_H
#define INCLUDE_GUARD_CUPY_CUSPARSELT_H

#ifdef CUPY_USE_HIP

#include "stub/cupy_cusparselt.h"

#elif !defined(CUPY_NO_CUDA)

#include <library_types.h>
#include <cusparseLt.h>

#else

#include "stub/cupy_cusparselt.h"

#endif // #ifndef CUPY_NO_CUDA

#endif // #ifndef INCLUDE_GUARD_CUPY_CUSPARSELT_H
