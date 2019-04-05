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

template <typename ValueType>
__device__ inline complex<ValueType> operator+(const complex<ValueType>& lhs,
                                               const complex<ValueType>& rhs) {
  return complex<ValueType>(lhs.real() + rhs.real(), lhs.imag() + rhs.imag());
}

template <typename ValueType>
__device__ inline complex<ValueType> operator+(
    const volatile complex<ValueType>& lhs,
    const volatile complex<ValueType>& rhs) {
  return complex<ValueType>(lhs.real() + rhs.real(), lhs.imag() + rhs.imag());
}

template <typename ValueType>
__device__ inline complex<ValueType> operator+(const complex<ValueType>& lhs,
                                               const ValueType& rhs) {
  return complex<ValueType>(lhs.real() + rhs, lhs.imag());
}

template <typename ValueType>
__device__ inline complex<ValueType> operator+(const ValueType& lhs,
                                               const complex<ValueType>& rhs) {
  return complex<ValueType>(rhs.real() + lhs, rhs.imag());
}

template <typename ValueType>
__device__ inline complex<ValueType> operator-(const complex<ValueType>& lhs,
                                               const complex<ValueType>& rhs) {
  return complex<ValueType>(lhs.real() - rhs.real(), lhs.imag() - rhs.imag());
}

template <typename ValueType>
__device__ inline complex<ValueType> operator-(const complex<ValueType>& lhs,
                                               const ValueType& rhs) {
  return complex<ValueType>(lhs.real() - rhs, lhs.imag());
}

template <typename ValueType>
__device__ inline complex<ValueType> operator-(const ValueType& lhs,
                                               const complex<ValueType>& rhs) {
  return complex<ValueType>(lhs - rhs.real(), -rhs.imag());
}

template <typename ValueType>
__device__ inline complex<ValueType> operator*(const complex<ValueType>& lhs,
                                               const complex<ValueType>& rhs) {
  return complex<ValueType>(lhs.real() * rhs.real() - lhs.imag() * rhs.imag(),
                            lhs.real() * rhs.imag() + lhs.imag() * rhs.real());
}

template <typename ValueType>
__device__ inline complex<ValueType> operator*(const complex<ValueType>& lhs,
                                               const ValueType& rhs) {
  return complex<ValueType>(lhs.real() * rhs, lhs.imag() * rhs);
}

template <typename ValueType>
__device__ inline complex<ValueType> operator*(const ValueType& lhs,
                                               const complex<ValueType>& rhs) {
  return complex<ValueType>(rhs.real() * lhs, rhs.imag() * lhs);
}

template <typename ValueType>
__device__ inline complex<ValueType> operator/(const complex<ValueType>& lhs,
                                               const complex<ValueType>& rhs) {
  ValueType s = abs(rhs.real()) + abs(rhs.imag());
  ValueType oos = ValueType(1.0) / s;
  ValueType ars = lhs.real() * oos;
  ValueType ais = lhs.imag() * oos;
  ValueType brs = rhs.real() * oos;
  ValueType bis = rhs.imag() * oos;
  s = (brs * brs) + (bis * bis);
  oos = ValueType(1.0) / s;
  complex<ValueType> quot(((ars * brs) + (ais * bis)) * oos,
                          ((ais * brs) - (ars * bis)) * oos);
  return quot;
}

template <typename ValueType>
__device__ inline complex<ValueType> operator/(const complex<ValueType>& lhs,
                                               const ValueType& rhs) {
  return complex<ValueType>(lhs.real() / rhs, lhs.imag() / rhs);
}

template <typename ValueType>
__device__ inline complex<ValueType> operator/(const ValueType& lhs,
                                               const complex<ValueType>& rhs) {
  return complex<ValueType>(lhs) / rhs;
}

/* --- Unary comparison with Numpy logic. This means that a + bi > c + di if either
 * a > c or a == c and b > d. --- */

template <typename ValueType>
__device__ inline bool operator<(const complex<ValueType>& lhs,
                                 const complex<ValueType>& rhs) {
  if (lhs == rhs)
  {
      return false;
  } else if (lhs.real() < rhs.real())
  {
      return true;
  } else if (lhs.real() == rhs.real())
  {
      return lhs.imag() < rhs.imag();
  } else
  {
      return false;
  }
}

template <typename ValueType>
__device__ inline bool operator<=(const complex<ValueType>& lhs,
                                  const complex<ValueType>& rhs) {
  if (lhs == rhs)
  {
      return true;
  } else if (lhs < rhs)
  {
      return true;
  } else
  {
      return false;
  }
}

template <typename ValueType>
__device__ inline bool operator>(const complex<ValueType>& lhs,
                                 const complex<ValueType>& rhs) {
  if (lhs == rhs)
  {
      return false;
  } else
  {
      return !(lhs < rhs);
  }
}

template <typename ValueType>
__device__ inline bool operator>=(const complex<ValueType>& lhs,
                                  const complex<ValueType>& rhs) {
  if (lhs == rhs || lhs > rhs)
  {
      return true;
  } else
  {
      return false;
  }
}

template <typename ValueType>
__device__ inline bool operator<(const ValueType& lhs,
                                 const complex<ValueType>& rhs) {
    return complex<ValueType>(lhs) < rhs;
}

template <typename ValueType>
__device__ inline bool operator>(const ValueType& lhs,
                                 const complex<ValueType>& rhs) {
    return complex<ValueType>(lhs) > rhs;
}

template <typename ValueType>
__device__ inline bool operator<(const complex<ValueType>& lhs,
                                 const ValueType& rhs) {
    return lhs < complex<ValueType>(rhs);
}

template <typename ValueType>
__device__ inline bool operator>(const complex<ValueType>& lhs,
                                 const ValueType& rhs) {
    return lhs > complex<ValueType>(rhs);
}

template <typename ValueType>
__device__ inline bool operator<=(const ValueType& lhs,
                                  const complex<ValueType>& rhs) {
    return complex<ValueType>(lhs) <= rhs;
}

template <typename ValueType>
__device__ inline bool operator>=(const ValueType& lhs,
                                  const complex<ValueType>& rhs) {
    return complex<ValueType>(lhs) >= rhs;
}

template <typename ValueType>
__device__ inline bool operator<=(const complex<ValueType>& lhs,
                                  const ValueType& rhs) {
    return lhs <= complex<ValueType>(rhs);
}

template <typename ValueType>
__device__ inline bool operator>=(const complex<ValueType>& lhs,
                                  const ValueType& rhs) {
    return lhs >= complex<ValueType>(rhs);
}

/* --- Unary Arithmetic Operators --- */

template <typename ValueType>
__device__ inline complex<ValueType> operator+(const complex<ValueType>& rhs) {
  return rhs;
}

template <typename ValueType>
__device__ inline complex<ValueType> operator-(const complex<ValueType>& rhs) {
  return rhs * -ValueType(1);
}

/* --- Other Basic Arithmetic Functions --- */

// As hypot is only C++11 we have to use the C interface
template <typename ValueType>
__device__ inline ValueType abs(const complex<ValueType>& z) {
  return hypot(z.real(), z.imag());
}

namespace detail {
namespace complex {
__device__ inline float abs(const thrust::complex<float>& z) {
  return hypotf(z.real(), z.imag());
}

__device__ inline double abs(const thrust::complex<double>& z) {
  return hypot(z.real(), z.imag());
}
}
}

template <>
__device__ inline float abs(const complex<float>& z) {
  return detail::complex::abs(z);
}
template <>
__device__ inline double abs(const complex<double>& z) {
  return detail::complex::abs(z);
}

template <typename ValueType>
__device__ inline ValueType arg(const complex<ValueType>& z) {
  return atan2(z.imag(), z.real());
}

template <typename ValueType>
__device__ inline complex<ValueType> conj(const complex<ValueType>& z) {
  return complex<ValueType>(z.real(), -z.imag());
}

template <typename ValueType>
__device__ inline ValueType norm(const complex<ValueType>& z) {
  return z.real() * z.real() + z.imag() * z.imag();
}

template <>
__device__ inline float norm(const complex<float>& z) {
  if (abs(z.real()) < ::sqrtf(FLT_MIN) && abs(z.imag()) < ::sqrtf(FLT_MIN)) {
    float a = z.real() * 4.0f;
    float b = z.imag() * 4.0f;
    return (a * a + b * b) / 16.0f;
  }
  return z.real() * z.real() + z.imag() * z.imag();
}

template <>
__device__ inline double norm(const complex<double>& z) {
  if (abs(z.real()) < ::sqrt(DBL_MIN) && abs(z.imag()) < ::sqrt(DBL_MIN)) {
    double a = z.real() * 4.0;
    double b = z.imag() * 4.0;
    return (a * a + b * b) / 16.0;
  }
  return z.real() * z.real() + z.imag() * z.imag();
}

template <typename ValueType>
__device__ inline complex<ValueType> polar(const ValueType& m,
                                           const ValueType& theta) {
  return complex<ValueType>(m * cos(theta), m * sin(theta));
}
}
