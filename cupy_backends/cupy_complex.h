#ifndef INCLUDE_GUARD_CUPY_COMPLEX_H
#define INCLUDE_GUARD_CUPY_COMPLEX_H

#ifdef CUPY_USE_HIP

#include "hip/cupy_cuComplex.h"

#else

#include <cuComplex.h>

#endif
#endif // #ifndef INCLUDE_GUARD_CUPY_COMPLEX_H
