#pragma once

#include <cupy/complex/complex.h>

namespace thrust {

template <typename T>
__device__ inline complex<T> pow(const complex<T>& z,
                                 const complex<T>& exponent) {
  const T absz = abs(z);
  if (absz == 0) {
    return complex<T>(0, 0);
  }
  const T real = exponent.real();
  const T imag = exponent.imag();
  const T argz = arg(z);
  T r = ::pow(absz, real);
  T theta = real * argz;
  if (imag != 0) {
    r *= ::exp(-imag * argz);
    theta += imag * ::log(absz);
  }
  return complex<T>(r * cos(theta), r * sin(theta));
}

template <typename T>
__device__ inline complex<T> pow(const complex<T>& z, const T& exponent) {
  return pow(z, complex<T>(exponent));
}

template <typename T>
__device__ inline complex<T> pow(const T& x, const complex<T>& exponent) {
  return pow(complex<T>(x), exponent);
}

template <typename T, typename U>
__device__ inline complex<typename _select_greater_type<T, U>::type> pow(
    const complex<T>& z, const complex<T>& exponent) {
  typedef typename _select_greater_type<T, U>::type PromotedType;
  return pow(complex<PromotedType>(z), complex<PromotedType>(exponent));
}

template <typename T, typename U>
__device__ inline complex<typename _select_greater_type<T, U>::type> pow(
    const complex<T>& z, const U& exponent) {
  typedef typename _select_greater_type<T, U>::type PromotedType;
  return pow(complex<PromotedType>(z), PromotedType(exponent));
}

template <typename T, typename U>
__device__ inline complex<typename _select_greater_type<T, U>::type> pow(
    const T& x, const complex<U>& exponent) {
  typedef typename _select_greater_type<T, U>::type PromotedType;
  return pow(PromotedType(x), complex<PromotedType>(exponent));
}

}
