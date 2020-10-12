#pragma once

#include <cupy/complex/complex.h>

namespace thrust {

template <typename T>
__host__ __device__ inline complex<T> pow(const complex<T>& z,
                                          const complex<T>& exponent) {
  return exp(log(complex<T>(z)) * complex<T>(exponent));
}

template <typename T>
__host__ __device__ inline complex<T> pow(const complex<T>& z, const T& exponent) {
  return exp(log(complex<T>(z)) * T(exponent));
}

template <typename T>
__host__ __device__ inline complex<T> pow(const T& x, const complex<T>& exponent) {
  // Find `log` by ADL.
  using std::log;
  return exp(log(T(x)) * complex<T>(exponent));
}

template <typename T, typename U>
__host__ __device__ inline complex<typename _select_greater_type<T, U>::type> pow(
    const complex<T>& z, const complex<T>& exponent) {
  typedef typename _select_greater_type<T, U>::type PromotedType;
  return pow(complex<PromotedType>(z), complex<PromotedType>(exponent));
}

template <typename T, typename U>
__host__ __device__ inline complex<typename _select_greater_type<T, U>::type> pow(
    const complex<T>& z, const U& exponent) {
  typedef typename _select_greater_type<T, U>::type PromotedType;
  return pow(complex<PromotedType>(z), PromotedType(exponent));
}

template <typename T, typename U>
__host__ __device__ inline complex<typename _select_greater_type<T, U>::type> pow(
    const T& x, const complex<U>& exponent) {
  typedef typename _select_greater_type<T, U>::type PromotedType;
  return pow(PromotedType(x), complex<PromotedType>(exponent));
}

}
