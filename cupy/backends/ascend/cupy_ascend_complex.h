#ifndef INCLUDE_GUARD_ASCEND_CUPY_COMPLEX_H
#define INCLUDE_GUARD_ASCEND_CUPY_COMPLEX_H

#include <complex>

extern "C" {

///////////////////////////////////////////////////////////////////////////////
// cupy_complex.h
///////////////////////////////////////////////////////////////////////////////

using cuComplex = std::complex<float>;
using cuDoubleComplex = std::complex<double>;

} // extern "C"

#endif // #ifndef INCLUDE_GUARD_ASCEND_CUPY_COMPLEX_H
