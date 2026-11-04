#ifndef INCLUDE_GUARD_CUPY_CUBLAS_H
#define INCLUDE_GUARD_CUPY_CUBLAS_H

#if CUPY_USE_HIP

#include "hip/cupy_hipblas.h"

#else

#include "cuda/cupy_cublas.h"

#endif
#endif // #ifndef INCLUDE_GUARD_CUPY_CUBLAS_H
