/*  Copyright 2008-2013 NVIDIA Corporation
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

/*! \file complex.h
 *  \brief Complex numbers
 */

#pragma once

namespace thrust {

template <typename T, typename U, bool x>
struct _select_greater_type_impl {
  typedef T type;
};

template <typename T, typename U>
struct _select_greater_type_impl<T, U, false> {
  typedef U type;
};

template <typename T, typename U>
struct _select_greater_type
    : _select_greater_type_impl<T, U, (sizeof(T) > sizeof(U))> {};

/*
 *  Calls to the standard math library from inside the thrust namespace
 *  with real arguments require explicit scope otherwise they will fail
 *  to resolve as it will find the equivalent complex function but then
 *  fail to match the template, and give up looking for other scopes.
 */

/*! \addtogroup numerics
 *  \{
 */

/*! \addtogroup complex_numbers Complex Numbers
 *  \{
 */

/*! \p complex is the Thrust equivalent to <tt>std::complex</tt>. It is
 * functionally
 *  equivalent to it, but can also be used in device code which
 * <tt>std::complex</tt> currently cannot.
 *
 *  \tparam T The type used to hold the real and imaginary parts. Should be
 * <tt>float</tt>
 *  or <tt>double</tt>. Others types are not supported.
 *
 */
template <typename T>
#if defined(__CUDACC__)
struct __align__(sizeof(T)*2) complex {
#else
// ROCm (hipcc) does not support `__align__`
struct complex {
#endif
 public:
  /*! \p value_type is the type of \p complex's real and imaginary parts.
   */
  typedef T value_type;

  /* --- Constructors --- */

  /*! Construct a complex number with an imaginary part of 0.
   *
   *  \param re The real part of the number.
   */
  inline __host__ __device__ complex(const T& re);

  /*! Construct a complex number from its real and imaginary parts.
   *
   *  \param re The real part of the number.
   *  \param im The imaginary part of the number.
   */
  inline __host__ __device__ complex(const T& re, const T& im);

#if __cplusplus >= 201103L || (defined(_MSC_VER) && _MSC_VER >= 1900)
  /*! Default construct a complex number.
   */
  inline __host__ __device__ complex() = default;

  /*! This copy constructor copies from a \p complex with a type that is
   *  convertible to this \p complex's \c value_type.
   *
   *  \param z The \p complex to copy from.
   */
  inline __host__ __device__ complex(const complex<T>& z) = default;
#else
  /*! Default construct a complex number.
   */
  inline __host__ __device__ complex();

  /*! This copy constructor copies from a \p complex with a type that is
   *  convertible to this \p complex's \c value_type.
   *
   *  \param z The \p complex to copy from.
   */
  inline __host__ __device__ complex(const complex<T>& z);
#endif // c++11

  /*! This copy constructor copies from a \p complex with a type that
   *  is convertible to this \p complex \c value_type.
   *
   *  \param z The \p complex to copy from.
   *
   *  \tparam X is convertible to \c value_type.
   */
  template <typename X>
  inline __host__ __device__ complex(const complex<X>& z);

  /* --- Assignment Operators --- */

  /*! Assign `re` to the real part of this \p complex and set the imaginary part
   *  to 0.
   *
   *  \param re The real part of the number.
   */
  inline __host__ __device__ complex& operator=(const T& re);

  /*! Assign `z.real()` and `z.imag()` to the real and imaginary parts of this
   *  \p complex respectively.
   *
   *  \param z The \p complex to copy from.
   */
  inline __host__ __device__ complex& operator=(const complex<T>& z);

  /*! Assign `z.real()` and `z.imag()` to the real and imaginary parts of this
   *  \p complex respectively.
   *
   *  \param z The \p complex to copy from.
   *
   *  \tparam U is convertible to \c value_type.
   */
  template <typename U>
  inline __host__ __device__ complex& operator=(const complex<U>& z);

  /* --- Compound Assignment Operators --- */

  /*! Adds a \p complex to this \p complex and
   *  assigns the result to this \p complex.
   *
   *  \param z The \p complex to be Added.
   */
  __host__ __device__ inline complex<T>& operator+=(const complex<T> z);

  /*! Subtracts a \p complex from this \p complex and
   *  assigns the result to this \p complex.
   *
   *  \param z The \p complex to be subtracted.
   */
  __host__ __device__ inline complex<T>& operator-=(const complex<T> z);

  /*! Multiplies this \p complex by another \p complex and
   *  assigns the result to this \p complex.
   *
   *  \param z The \p complex to be multiplied.
   */
  __host__ __device__ inline complex<T>& operator*=(const complex<T> z);

  /*! Divides this \p complex by another \p complex and
   *  assigns the result to this \p complex.
   *
   *  \param z The \p complex to be divided.
   */
  __host__ __device__ inline complex<T>& operator/=(const complex<T> z);

  /* --- Getter functions ---
   * The volatile ones are there to help for example
   * with certain reductions optimizations
   */

  /*! Returns the real part of this \p complex.
   */
  __host__ __device__ inline T real() const volatile { return m_data[0]; }

  /*! Returns the imaginary part of this \p complex.
   */
  __host__ __device__ inline T imag() const volatile { return m_data[1]; }

  /*! Returns the real part of this \p complex.
   */
  __host__ __device__ inline T real() const { return m_data[0]; }

  /*! Returns the imaginary part of this \p complex.
   */
  __host__ __device__ inline T imag() const { return m_data[1]; }

  /* --- Setter functions ---
   * The volatile ones are there to help for example
   * with certain reductions optimizations
   */

  /*! Sets the real part of this \p complex.
   *
   *  \param re The new real part of this \p complex.
   */
  __host__ __device__ inline void real(T re) volatile { m_data[0] = re; }

  /*! Sets the imaginary part of this \p complex.
   *
   *  \param im The new imaginary part of this \p complex.e
   */
  __host__ __device__ inline void imag(T im) volatile { m_data[1] = im; }

  /*! Sets the real part of this \p complex.
   *
   *  \param re The new real part of this \p complex.
   */
  __host__ __device__ inline void real(T re) { m_data[0] = re; }

  /*! Sets the imaginary part of this \p complex.
   *
   *  \param im The new imaginary part of this \p complex.
   */
  __host__ __device__ inline void imag(T im) { m_data[1] = im; }

 private:
  T m_data[2];
};

/* --- General Functions --- */

/*! Returns the magnitude (also known as absolute value) of a \p complex.
 *
 *  \param z The \p complex from which to calculate the absolute value.
 */
template <typename T>
__host__ __device__ inline T abs(const complex<T>& z);

/*! Returns the phase angle (also known as argument) in radians of a \p complex.
 *
 *  \param z The \p complex from which to calculate the phase angle.
 */
template <typename T>
__host__ __device__ inline T arg(const complex<T>& z);

/*! Returns the square of the magnitude of a \p complex.
 *
 *  \param z The \p complex from which to calculate the norm.
 */
template <typename T>
__host__ __device__ inline T norm(const complex<T>& z);

/*! Returns the complex conjugate of a \p complex.
 *
 *  \param z The \p complex from which to calculate the complex conjugate.
 */
template <typename T>
__host__ __device__ inline complex<T> conj(const complex<T>& z);

/*! Returns the real part of a \p complex.
 *
 *  \param z The \p complex from which to return the real part
 */
template <typename T>
__host__ __device__ inline T real(const complex<T>& z);

/*! Returns the imaginary part of a \p complex.
 *
 *  \param z The \p complex from which to return the imaginary part
 */
template <typename T>
__host__ __device__ inline T imag(const complex<T>& z);

/*! Returns a \p complex with the specified magnitude and phase.
 *
 *  \param m The magnitude of the returned \p complex.
 *  \param theta The phase of the returned \p complex in radians.
 */
template <typename T>
__host__ __device__ inline complex<T> polar(const T& m, const T& theta = 0);

/*! Returns the projection of a \p complex on the Riemann sphere.
 *  For all finite \p complex it returns the argument. For \p complexs
 *  with a non finite part returns (INFINITY,+/-0) where the sign of
 *  the zero matches the sign of the imaginary part of the argument.
 *
 *  \param z The \p complex argument.
 */
template <typename T>
__host__ __device__ inline complex<T> proj(const T& z);

/* --- Binary Arithmetic operators --- */

/*! Multiplies two \p complex numbers.
 *
 *  \param lhs The first \p complex.
 *  \param rhs The second \p complex.
 */
template <typename T>
__host__ __device__ inline complex<T> operator*(const complex<T>& lhs,
                                       const complex<T>& rhs);

/*! Multiplies a \p complex number by a scalar.
 *
 *  \param lhs The \p complex.
 *  \param rhs The scalar.
 */
template <typename T>
__host__ __device__ inline complex<T> operator*(const complex<T>& lhs, const T& rhs);

/*! Multiplies a scalar by a \p complex number.
 *
 *  \param lhs The scalar.
 *  \param rhs The \p complex.
 */
template <typename T>
__host__ __device__ inline complex<T> operator*(const T& lhs, const complex<T>& rhs);

/*! Divides two \p complex numbers.
 *
 *  \param lhs The numerator (dividend).
 *  \param rhs The denomimator (divisor).
 */
template <typename T>
__host__ __device__ inline complex<T> operator/(const complex<T>& lhs,
                                       const complex<T>& rhs);

/*! Divides a \p complex number by a scalar.
 *
 *  \param lhs The complex numerator (dividend).
 *  \param rhs The scalar denomimator (divisor).
 */
template <typename T>
__host__ __device__ inline complex<T> operator/(const complex<T>& lhs, const T& rhs);

/*! Divides a scalar by a \p complex number.
 *
 *  \param lhs The scalar numerator (dividend).
 *  \param rhs The complex denomimator (divisor).
 */
template <typename T>
__host__ __device__ inline complex<T> operator/(const T& lhs, const complex<T>& rhs);

/*! Adds two \p complex numbers.
 *
 *  \param lhs The first \p complex.
 *  \param rhs The second \p complex.
 */
template <typename T>
__host__ __device__ inline complex<T> operator+(const complex<T>& lhs,
                                       const complex<T>& rhs);

/*! Adds a scalar to a \p complex number.
 *
 *  \param lhs The \p complex.
 *  \param rhs The scalar.
 */
template <typename T>
__host__ __device__ inline complex<T> operator+(const complex<T>& lhs, const T& rhs);

/*! Adds a \p complex number to a scalar.
 *
 *  \param lhs The scalar.
 *  \param rhs The \p complex.
 */
template <typename T>
__host__ __device__ inline complex<T> operator+(const T& lhs, const complex<T>& rhs);

/*! Subtracts two \p complex numbers.
 *
 *  \param lhs The first \p complex (minuend).
 *  \param rhs The second \p complex (subtrahend).
 */
template <typename T>
__host__ __device__ inline complex<T> operator-(const complex<T>& lhs,
                                       const complex<T>& rhs);

/*! Subtracts a scalar from a \p complex number.
 *
 *  \param lhs The \p complex (minuend).
 *  \param rhs The scalar (subtrahend).
 */
template <typename T>
__host__ __device__ inline complex<T> operator-(const complex<T>& lhs, const T& rhs);

/*! Subtracts a \p complex number from a scalar.
 *
 *  \param lhs The scalar (minuend).
 *  \param rhs The \p complex (subtrahend).
 */
template <typename T>
__host__ __device__ inline complex<T> operator-(const T& lhs, const complex<T>& rhs);

/* --- Unary Arithmetic operators --- */

/*! Unary plus, returns its \p complex argument.
 *
 *  \param rhs The \p complex argument.
 */
template <typename T>
__host__ __device__ inline complex<T> operator+(const complex<T>& rhs);

/*! Unary minus, returns the additive inverse (negation) of its \p complex
 * argument.
 *
 *  \param rhs The \p complex argument.
 */
template <typename T>
__host__ __device__ inline complex<T> operator-(const complex<T>& rhs);

/* --- Exponential Functions --- */

/*! Returns the complex exponential of a \p complex number.
 *
 *  \param z The \p complex argument.
 */
template <typename T>
__host__ __device__ complex<T> exp(const complex<T>& z);

/*! Returns the complex natural logarithm of a \p complex number.
 *
 *  \param z The \p complex argument.
 */
template <typename T>
__host__ __device__ complex<T> log(const complex<T>& z);

/*! Returns the complex base 10 logarithm of a \p complex number.
 *
 *  \param z The \p complex argument.
 */
template <typename T>
__host__ __device__ inline complex<T> log10(const complex<T>& z);

/* --- Power Functions --- */

/*! Returns a \p complex number raised to another.
 *
 *  \param x The base.
 *  \param y The exponent.
 */
template <typename T>
__host__ __device__ complex<T> pow(const complex<T>& x, const complex<T>& y);

/*! Returns a \p complex number raised to a scalar.
 *
 *  \param x The \p complex base.
 *  \param y The scalar exponent.
 */
template <typename T>
__host__ __device__ complex<T> pow(const complex<T>& x, const T& y);

/*! Returns a scalar raised to a \p complex number.
 *
 *  \param x The scalar base.
 *  \param y The \p complex exponent.
 */
template <typename T>
__host__ __device__ complex<T> pow(const T& x, const complex<T>& y);

/*! Returns a \p complex number raised to another. The types of the two \p
 * complex should be compatible
 * and the type of the returned \p complex is the promoted type of the two
 * arguments.
 *
 *  \param x The base.
 *  \param y The exponent.
 */
template <typename T, typename U>
__host__ __device__ complex<typename _select_greater_type<T, U>::type> pow(
    const complex<T>& x, const complex<U>& y);

/*! Returns a \p complex number raised to a scalar. The type of the \p complex
 * should be compatible with the scalar
 * and the type of the returned \p complex is the promoted type of the two
 * arguments.
 *
 *  \param x The base.
 *  \param y The exponent.
 */
template <typename T, typename U>
__host__ __device__ complex<typename _select_greater_type<T, U>::type> pow(
    const complex<T>& x, const U& y);

/*! Returns a scalar raised to a \p complex number. The type of the \p complex
 * should be compatible with the scalar
 * and the type of the returned \p complex is the promoted type of the two
 * arguments.
 *
 *  \param x The base.
 *  \param y The exponent.
 */
template <typename T, typename U>
__host__ __device__ complex<typename _select_greater_type<T, U>::type> pow(
    const T& x, const complex<U>& y);

/*! Returns the complex square root of a \p complex number.
 *
 *  \param z The \p complex argument.
 */
template <typename T>
__host__ __device__ complex<T> sqrt(const complex<T>& z);

/* --- Trigonometric Functions --- */

/*! Returns the complex cosine of a \p complex number.
 *
 *  \param z The \p complex argument.
 */
template <typename T>
__host__ __device__ complex<T> cos(const complex<T>& z);

/*! Returns the complex sine of a \p complex number.
 *
 *  \param z The \p complex argument.
 */
template <typename T>
__host__ __device__ complex<T> sin(const complex<T>& z);

/*! Returns the complex tangent of a \p complex number.
 *
 *  \param z The \p complex argument.
 */
template <typename T>
__host__ __device__ complex<T> tan(const complex<T>& z);

/* --- Hyperbolic Functions --- */

/*! Returns the complex hyperbolic cosine of a \p complex number.
 *
 *  \param z The \p complex argument.
 */
template <typename T>
__host__ __device__ complex<T> cosh(const complex<T>& z);

/*! Returns the complex hyperbolic sine of a \p complex number.
 *
 *  \param z The \p complex argument.
 */
template <typename T>
__host__ __device__ complex<T> sinh(const complex<T>& z);

/*! Returns the complex hyperbolic tangent of a \p complex number.
 *
 *  \param z The \p complex argument.
 */
template <typename T>
__host__ __device__ complex<T> tanh(const complex<T>& z);

/* --- Inverse Trigonometric Functions --- */

/*! Returns the complex arc cosine of a \p complex number.
 *
 *  The range of the real part of the result is [0, Pi] and
 *  the range of the imaginary part is [-inf, +inf]
 *
 *  \param z The \p complex argument.
 */
template <typename T>
__host__ __device__ complex<T> acos(const complex<T>& z);

/*! Returns the complex arc sine of a \p complex number.
 *
 *  The range of the real part of the result is [-Pi/2, Pi/2] and
 *  the range of the imaginary part is [-inf, +inf]
 *
 *  \param z The \p complex argument.
 */
template <typename T>
__host__ __device__ complex<T> asin(const complex<T>& z);

/*! Returns the complex arc tangent of a \p complex number.
 *
 *  The range of the real part of the result is [-Pi/2, Pi/2] and
 *  the range of the imaginary part is [-inf, +inf]
 *
 *  \param z The \p complex argument.
 */
template <typename T>
__host__ __device__ complex<T> atan(const complex<T>& z);

/* --- Inverse Hyperbolic Functions --- */

/*! Returns the complex inverse hyperbolic cosine of a \p complex number.
 *
 *  The range of the real part of the result is [0, +inf] and
 *  the range of the imaginary part is [-Pi, Pi]
 *
 *  \param z The \p complex argument.
 */
template <typename T>
__host__ __device__ complex<T> acosh(const complex<T>& z);

/*! Returns the complex inverse hyperbolic sine of a \p complex number.
 *
 *  The range of the real part of the result is [-inf, +inf] and
 *  the range of the imaginary part is [-Pi/2, Pi/2]
 *
 *  \param z The \p complex argument.
 */
template <typename T>
__host__ __device__ complex<T> asinh(const complex<T>& z);

/*! Returns the complex inverse hyperbolic tangent of a \p complex number.
 *
 *  The range of the real part of the result is [-inf, +inf] and
 *  the range of the imaginary part is [-Pi/2, Pi/2]
 *
 *  \param z The \p complex argument.
 */
template <typename T>
__host__ __device__ complex<T> atanh(const complex<T>& z);

/* --- Equality Operators --- */

/*! Returns true if two \p complex numbers are equal and false otherwise.
 *
 *  \param lhs The first \p complex.
 *  \param rhs The second \p complex.
 */
template <typename T>
__host__ __device__ inline bool operator==(const complex<T>& lhs, const complex<T>& rhs);

/*! Returns true if the imaginary part of the  \p complex number is zero and the
 * real part is equal to the scalar. Returns false otherwise.
 *
 *  \param lhs The scalar.
 *  \param rhs The \p complex.
 */
template <typename T>
__host__ __device__ inline bool operator==(const T& lhs, const complex<T>& rhs);

/*! Returns true if the imaginary part of the  \p complex number is zero and the
 * real part is equal to the scalar. Returns false otherwise.
 *
 *  \param lhs The \p complex.
 *  \param rhs The scalar.
 */
template <typename T>
__host__ __device__ inline bool operator==(const complex<T>& lhs, const T& rhs);

/*! Returns true if two \p complex numbers are different and false otherwise.
 *
 *  \param lhs The first \p complex.
 *  \param rhs The second \p complex.
 */
template <typename T>
__host__ __device__ inline bool operator!=(const complex<T>& lhs, const complex<T>& rhs);

/*! Returns true if the imaginary part of the  \p complex number is not zero or
 * the real part is different from the scalar. Returns false otherwise.
 *
 *  \param lhs The scalar.
 *  \param rhs The \p complex.
 */
template <typename T>
__host__ __device__ inline bool operator!=(const T& lhs, const complex<T>& rhs);

/*! Returns true if the imaginary part of the \p complex number is not zero or
 * the real part is different from the scalar. Returns false otherwise.
 *
 *  \param lhs The \p complex.
 *  \param rhs The scalar.
 */
template <typename T>
__host__ __device__ inline bool operator!=(const complex<T>& lhs, const T& rhs);

}  // end namespace thrust

#include <cupy/complex/complex_inl.h>
