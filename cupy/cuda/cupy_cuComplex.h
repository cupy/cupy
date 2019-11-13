// This file is a stub header file of cuda for Read the Docs.

#ifndef INCLUDE_GUARD_CUPY_COMPLEX_H
#define INCLUDE_GUARD_CUPY_COMPLEX_H

#if !defined(CUPY_NO_CUDA) && !defined(CUPY_USE_HIP)

#include <cuComplex.h>

#else // #if !defined(CUPY_NO_CUDA) || !defined(CUPY_USE_HIP)

extern "C" {

///////////////////////////////////////////////////////////////////////////////
// cuComplex.h
///////////////////////////////////////////////////////////////////////////////

struct cuComplex{
    float x, y;
};

struct cuDoubleComplex{
    double x, y;
};

} // extern "C"

#endif // #ifndef CUPY_NO_CUDA
#endif // #ifndef INCLUDE_GUARD_CUPY_COMPLEX_H
