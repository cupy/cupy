/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Copyright 2013 Filipe RNC Maia
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <cupy/complex/complex.h>
#include <cupy/complex/math_private.h>

namespace thrust {

/* --- Binary Arithmetic Operators --- */

template <typename T>
__host__ __device__ inline complex<T> operator+(const complex<T>& lhs,
                                                const complex<T>& rhs) {
  return complex<T>(lhs.real() + rhs.real(), lhs.imag() + rhs.imag());
}

template <typename T>
__host__ __device__ inline complex<T> operator+(const volatile complex<T>& lhs,
                                                const volatile complex<T>& rhs) {
  return complex<T>(lhs.real() + rhs.real(), lhs.imag() + rhs.imag());
}

template <typename T>
__host__ __device__ inline complex<T> operator+(const complex<T>& lhs,
                                                const T& rhs) {
  return complex<T>(lhs.real() + rhs, lhs.imag());
}

template <typename T>
__host__ __device__ inline complex<T> operator+(const T& lhs,
                                                const complex<T>& rhs) {
  return complex<T>(rhs.real() + lhs, rhs.imag());
}

// TODO(leofang): support operator+ for (complex<T0> x, complex<T1> y)

template <typename T>
__host__ __device__ inline complex<T> operator-(const complex<T>& lhs,
                                                const complex<T>& rhs) {
  return complex<T>(lhs.real() - rhs.real(), lhs.imag() - rhs.imag());
}

template <typename T>
__host__ __device__ inline complex<T> operator-(const complex<T>& lhs,
                                                const T& rhs) {
  return complex<T>(lhs.real() - rhs, lhs.imag());
}

template <typename T>
__host__ __device__ inline complex<T> operator-(const T& lhs,
                                                const complex<T>& rhs) {
  return complex<T>(lhs - rhs.real(), -rhs.imag());
}

// TODO(leofang): support operator- for (complex<T0> x, complex<T1> y)

template <typename T>
__host__ __device__ inline complex<T> operator*(const complex<T>& lhs,
                                                const complex<T>& rhs) {
  return complex<T>(lhs.real() * rhs.real() - lhs.imag() * rhs.imag(),
                            lhs.real() * rhs.imag() + lhs.imag() * rhs.real());
}

template <typename T>
__host__ __device__ inline complex<T> operator*(const complex<T>& lhs,
                                                const T& rhs) {
  return complex<T>(lhs.real() * rhs, lhs.imag() * rhs);
}

template <typename T>
__host__ __device__ inline complex<T> operator*(const T& lhs,
                                                const complex<T>& rhs) {
  return complex<T>(rhs.real() * lhs, rhs.imag() * lhs);
}

// TODO(leofang): support operator* for (complex<T0> x, complex<T1> y)

template <typename T>
__host__ __device__ inline complex<T> operator/(const complex<T>& lhs,
                                                const complex<T>& rhs) {
  T s = abs(rhs.real()) + abs(rhs.imag());
  T oos = T(1.0) / s;
  T ars = lhs.real() * oos;
  T ais = lhs.imag() * oos;
  T brs = rhs.real() * oos;
  T bis = rhs.imag() * oos;
  s = (brs * brs) + (bis * bis);
  oos = T(1.0) / s;
  complex<T> quot(((ars * brs) + (ais * bis)) * oos,
                  ((ais * brs) - (ars * bis)) * oos);
  return quot;
}

template <typename T>
__host__ __device__ inline complex<T> operator/(const complex<T>& lhs,
                                                const T& rhs) {
  return complex<T>(lhs.real() / rhs, lhs.imag() / rhs);
}

template <typename T>
__host__ __device__ inline complex<T> operator/(const T& lhs,
                                                const complex<T>& rhs) {
  return complex<T>(lhs) / rhs;
}

// TODO(leofang): support operator/ for (complex<T0> x, complex<T1> y)

/* --- Unary comparison with Numpy logic. This means that a + bi > c + di if either
 * a > c or a == c and b > d. --- */

template <typename T>
__host__ __device__ inline bool operator<(const complex<T>& lhs,
                                          const complex<T>& rhs) {
  if (lhs == rhs) {
      return false;
  } else if (lhs.real() < rhs.real()) {
      return true;
  } else if (lhs.real() == rhs.real()) {
      return lhs.imag() < rhs.imag();
  } else {
      return false;
  }
}

template <typename T>
__host__ __device__ inline bool operator<=(const complex<T>& lhs,
                                           const complex<T>& rhs) {
  if (lhs == rhs || lhs < rhs) {
      return true;
  } else {
      return false;
  }
}

template <typename T>
__host__ __device__ inline bool operator>(const complex<T>& lhs,
                                          const complex<T>& rhs) {
  if (lhs == rhs) {
      return false;
  } else if (lhs.real() > rhs.real()) {
      return true;
  } else if (lhs.real() == rhs.real()) {
      return lhs.imag() > rhs.imag();
  } else {
      return false;
  }
}

template <typename T>
__host__ __device__ inline bool operator>=(const complex<T>& lhs,
                                           const complex<T>& rhs) {
  if (lhs == rhs || lhs > rhs) {
      return true;
  } else {
      return false;
  }
}

template <typename T>
__host__ __device__ inline bool operator<(const T& lhs,
                                          const complex<T>& rhs) {
    return complex<T>(lhs) < rhs;
}

template <typename T>
__host__ __device__ inline bool operator>(const T& lhs,
                                          const complex<T>& rhs) {
    return complex<T>(lhs) > rhs;
}

template <typename T>
__host__ __device__ inline bool operator<(const complex<T>& lhs,
                                          const T& rhs) {
    return lhs < complex<T>(rhs);
}

template <typename T>
__host__ __device__ inline bool operator>(const complex<T>& lhs,
                                          const T& rhs) {
    return lhs > complex<T>(rhs);
}

template <typename T>
__host__ __device__ inline bool operator<=(const T& lhs,
                                           const complex<T>& rhs) {
    return complex<T>(lhs) <= rhs;
}

template <typename T>
__host__ __device__ inline bool operator>=(const T& lhs,
                                           const complex<T>& rhs) {
    return complex<T>(lhs) >= rhs;
}

template <typename T>
__host__ __device__ inline bool operator<=(const complex<T>& lhs,
                                           const T& rhs) {
    return lhs <= complex<T>(rhs);
}

template <typename T>
__host__ __device__ inline bool operator>=(const complex<T>& lhs,
                                           const T& rhs) {
    return lhs >= complex<T>(rhs);
}

/* --- Unary Arithmetic Operators --- */

template <typename T>
__host__ __device__ inline complex<T> operator+(const complex<T>& rhs) {
  return rhs;
}

template <typename T>
__host__ __device__ inline complex<T> operator-(const complex<T>& rhs) {
  return rhs * -T(1);
}

/* --- Other Basic Arithmetic Functions --- */

// As hypot is only C++11 we have to use the C interface
template <typename T>
__host__ __device__ inline T abs(const complex<T>& z) {
  return hypot(z.real(), z.imag());
}

namespace detail {
namespace complex {
__host__ __device__ inline float abs(const thrust::complex<float>& z) {
  return hypotf(z.real(), z.imag());
}

__host__ __device__ inline double abs(const thrust::complex<double>& z) {
  return hypot(z.real(), z.imag());
}
}
}

template <>
__host__ __device__ inline float abs(const complex<float>& z) {
  return detail::complex::abs(z);
}
template <>
__host__ __device__ inline double abs(const complex<double>& z) {
  return detail::complex::abs(z);
}

template <typename T>
__host__ __device__ inline T arg(const complex<T>& z) {
  return atan2(z.imag(), z.real());
}

template <typename T>
__host__ __device__ inline complex<T> conj(const complex<T>& z) {
  return complex<T>(z.real(), -z.imag());
}

template <typename T>
__host__ __device__ inline T real(const complex<T>& z) {
  return z.real();
}

template <typename T>
__host__ __device__ inline T imag(const complex<T>& z) {
  return z.imag();
}

template <typename T>
__host__ __device__ inline T norm(const complex<T>& z) {
  return z.real() * z.real() + z.imag() * z.imag();
}

template <>
__host__ __device__ inline float norm(const complex<float>& z) {
  if (abs(z.real()) < ::sqrtf(FLT_MIN) && abs(z.imag()) < ::sqrtf(FLT_MIN)) {
    float a = z.real() * 4.0f;
    float b = z.imag() * 4.0f;
    return (a * a + b * b) / 16.0f;
  }
  return z.real() * z.real() + z.imag() * z.imag();
}

template <>
__host__ __device__ inline double norm(const complex<double>& z) {
  if (abs(z.real()) < ::sqrt(DBL_MIN) && abs(z.imag()) < ::sqrt(DBL_MIN)) {
    double a = z.real() * 4.0;
    double b = z.imag() * 4.0;
    return (a * a + b * b) / 16.0;
  }
  return z.real() * z.real() + z.imag() * z.imag();
}

template <typename T>
__host__ __device__ inline complex<T> polar(const T& m,
                                           const T& theta) {
  return complex<T>(m * cos(theta), m * sin(theta));
}
}
