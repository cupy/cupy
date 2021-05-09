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

/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */

/* adapted from FreeBSD:
 *    lib/msun/src/math_private.h
 */

#pragma once

namespace thrust {

const float FLT_MIN = 1.17549435e-38F;
const float FLT_MAX = 3.40282347e+38F;
const float FLT_EPSILON = 1.19209290e-07F;
const int FLT_MAX_EXP = 128;
const int FLT_MANT_DIG = 24;

const double DBL_MIN = 2.2250738585072014e-308;
const double DBL_MAX = 1.7976931348623157e+308;
const double DBL_EPSILON = 2.2204460492503131e-16;
const int DBL_MAX_EXP = 1024;
const int DBL_MANT_DIG = 53;

namespace detail {
namespace complex {

typedef int int32_t;
typedef unsigned int uint32_t;
typedef long long int64_t;
typedef unsigned long long uint64_t;

typedef union {
  float value;
  uint32_t word;
} ieee_float_shape_type;

__host__ __device__ inline void get_float_word(uint32_t& i, float d) {
  ieee_float_shape_type gf_u;
  gf_u.value = (d);
  (i) = gf_u.word;
}

__host__ __device__ inline void get_float_word(int32_t& i, float d) {
  ieee_float_shape_type gf_u;
  gf_u.value = (d);
  (i) = gf_u.word;
}

__host__ __device__ inline void set_float_word(float& d, uint32_t i) {
  ieee_float_shape_type sf_u;
  sf_u.word = (i);
  (d) = sf_u.value;
}

// Assumes little endian ordering
typedef union {
  double value;
  struct {
    uint32_t lsw;
    uint32_t msw;
  } parts;
  struct {
    uint64_t w;
  } xparts;
} ieee_double_shape_type;

__host__ __device__ inline void get_high_word(uint32_t& i, double d) {
  ieee_double_shape_type gh_u;
  gh_u.value = (d);
  (i) = gh_u.parts.msw;
}

/* Set the more significant 32 bits of a double from an int.  */
__host__ __device__ inline void set_high_word(double& d, uint32_t v) {
  ieee_double_shape_type sh_u;
  sh_u.value = (d);
  sh_u.parts.msw = (v);
  (d) = sh_u.value;
}

__host__ __device__ inline void insert_words(double& d, uint32_t ix0, uint32_t ix1) {
  ieee_double_shape_type iw_u;
  iw_u.parts.msw = (ix0);
  iw_u.parts.lsw = (ix1);
  (d) = iw_u.value;
}

/* Get two 32 bit ints from a double.  */
__host__ __device__ inline void extract_words(uint32_t& ix0, uint32_t& ix1, double d) {
  ieee_double_shape_type ew_u;
  ew_u.value = (d);
  (ix0) = ew_u.parts.msw;
  (ix1) = ew_u.parts.lsw;
}

/* Get two 32 bit ints from a double.  */
__host__ __device__ inline void extract_words(int32_t& ix0, int32_t& ix1, double d) {
  ieee_double_shape_type ew_u;
  ew_u.value = (d);
  (ix0) = ew_u.parts.msw;
  (ix1) = ew_u.parts.lsw;
}

template <typename T>
inline __host__ __device__ T infinity();

template <>
inline __host__ __device__ float infinity<float>() {
  float res;
  set_float_word(res, 0x7f800000);
  return res;
}

template <>
inline __host__ __device__ double infinity<double>() {
  double res;
  insert_words(res, 0x7ff00000, 0);
  return res;
}

using ::abs;
using ::log;
using ::acos;
using ::asin;
using ::sqrt;
using ::sinh;
using ::tan;
using ::cos;
using ::sin;
using ::exp;
using ::cosh;
using ::atan;
using ::atanh;
using ::isinf;
using ::isnan;
using ::signbit;
using ::isfinite;

}  // namespace complex

}  // namespace detail

using ::abs;
using ::log;
using ::acos;
using ::asin;
using ::sqrt;
using ::sinh;
using ::tan;
using ::cos;
using ::sin;
using ::exp;
using ::cosh;
using ::atan;
using ::atanh;
using ::isinf;
using ::isnan;
using ::signbit;
using ::isfinite;

}  // namespace thrust
