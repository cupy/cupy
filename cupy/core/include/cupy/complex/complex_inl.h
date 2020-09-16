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

namespace thrust {

/* --- Constructors --- */
template <typename T>
inline __host__ __device__ complex<T>::complex(const T& re) {
  real(re);
  imag(T());
}

template <typename T>
inline __host__ __device__ complex<T>::complex(const T& re, const T& im) {
  real(re);
  imag(im);
}

#if ((!defined(_MSC_VER) && __cplusplus < 201103L) || \
     (defined(_MSC_VER) && _MSC_VER < 1900))
template <typename T>
inline __host__ __device__ complex<T>::complex() {
  real(T());
  imag(T());
}

template <typename T>
inline __host__ __device__ complex<T>::complex(const complex<T>& z) {
  real(z.real());
  imag(z.imag());
}
#endif

template <typename T>
template <typename X>
inline __host__ __device__ complex<T>::complex(const complex<X>& z) {
  // The explicit T() is there no prevent Visual Studio from complaining
  // about potential loss of precision
  real(T(z.real()));
  imag(T(z.imag()));
}

/* --- Assignment Operators --- */

template <typename T>
inline __host__ __device__ complex<T>& complex<T>::operator=(const T& re) {
  real(re);
  imag(T());
  return *this;
}

template <typename T>
inline __host__ __device__ complex<T>& complex<T>::operator=(const complex<T>& z) {
  real(z.real());
  imag(z.imag());
  return *this;
}

template <typename T>
template <typename U>
inline __host__ __device__ complex<T>& complex<T>::operator=(const complex<U>& z) {
  real(T(z.real()));
  imag(T(z.imag()));
  return *this;
}

/* --- Compound Assignment Operators --- */
// TODO(leofang): support operators with argument of type T, see upstream

template <typename T>
__host__ __device__ inline complex<T>& complex<T>::operator+=(const complex<T> z) {
  *this = *this + z;
  return *this;
}

template <typename T>
__host__ __device__ inline complex<T>& complex<T>::operator-=(const complex<T> z) {
  *this = *this - z;
  return *this;
}

template <typename T>
__host__ __device__ inline complex<T>& complex<T>::operator*=(const complex<T> z) {
  *this = *this * z;
  return *this;
}

template <typename T>
__host__ __device__ inline complex<T>& complex<T>::operator/=(const complex<T> z) {
  *this = *this / z;
  return *this;
}

/* --- Equality Operators --- */

template <typename T>
__host__ __device__ inline bool operator==(const complex<T>& lhs,
                                           const complex<T>& rhs) {
  return lhs.real() == rhs.real() && lhs.imag() == rhs.imag();
}

template <typename T>
__host__ __device__ inline bool operator==(const T& lhs, const complex<T>& rhs) {
  return lhs == rhs.real() && rhs.imag() == 0;
}

template <typename T>
__host__ __device__ inline bool operator==(const complex<T>& lhs, const T& rhs) {
  return lhs.real() == rhs && lhs.imag() == 0;
}

template <typename T>
__host__ __device__ inline bool operator!=(const complex<T>& lhs,
                                           const complex<T>& rhs) {
  return !(lhs == rhs);
}

template <typename T>
__host__ __device__ inline bool operator!=(const T& lhs, const complex<T>& rhs) {
  return !(lhs == rhs);
}

template <typename T>
__host__ __device__ inline bool operator!=(const complex<T>& lhs, const T& rhs) {
  return !(lhs == rhs);
}
}


#include <cupy/complex/arithmetic.h>
#include <cupy/complex/cproj.h>
#include <cupy/complex/cexp.h>
#include <cupy/complex/cexpf.h>
#include <cupy/complex/clog.h>
#include <cupy/complex/clogf.h>
#include <cupy/complex/cpow.h>
#include <cupy/complex/ccosh.h>
#include <cupy/complex/ccoshf.h>
#include <cupy/complex/csinh.h>
#include <cupy/complex/csinhf.h>
#include <cupy/complex/ctanh.h>
#include <cupy/complex/ctanhf.h>
#include <cupy/complex/csqrt.h>
#include <cupy/complex/csqrtf.h>
#include <cupy/complex/catrig.h>
#include <cupy/complex/catrigf.h>
