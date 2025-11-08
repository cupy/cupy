#ifndef INCLUDE_GUARD_HIP_CUPY_COMPLEX_H
#define INCLUDE_GUARD_HIP_CUPY_COMPLEX_H

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

#endif // #ifndef INCLUDE_GUARD_HIP_CUPY_COMPLEX_H
