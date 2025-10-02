#ifndef INCLUDE_GUARD_ASCEND_CUPY_COMPLEX_H
#define INCLUDE_GUARD_ASCEND_CUPY_COMPLEX_H

#include <complex>

extern "C" {

///////////////////////////////////////////////////////////////////////////////
// cuComplex.h
///////////////////////////////////////////////////////////////////////////////

using cuComplex = std::complex<float>;
using cuDoubleComplex = std::complex<double>;

// struct cuDoubleComplex{
//     double x, y;
// };

} // extern "C"

#endif // #ifndef INCLUDE_GUARD_ASCEND_CUPY_COMPLEX_H
