/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <string>

namespace tc {
namespace code {

//
// Various strings and code that get included during JIT compilation.
// We distinguish between:
//   1. the actual implementation of the C declarations that are visible to
//      NVRTC, mostly for common mathematical functions,
//   2. C++ code that is passed to NVRTC and makes heavy uses of templates for
//      specialization.
//
namespace c {

constexpr auto types = R"C(
// Halide type handling
typedef int int32;
typedef long int64;
typedef float float32;
typedef double float64;

)C";

constexpr auto mathFunctionDecl = R"C(

// BEGIN MATH FUNCTIONS FROM CUDA
float acosf ( float  x );
float acoshf ( float  x );
float asinf ( float  x );
float asinhf ( float  x );
float atan2f ( float  y, float  x );
float atanf ( float  x );
float atanhf ( float  x );
float cbrtf ( float  x );
float ceilf ( float  x );
float copysignf ( float  x, float  y );
float cosf ( float  x );
float coshf ( float  x );
float cospif ( float  x );
float cyl_bessel_i0f ( float  x );
float cyl_bessel_i1f ( float  x );
float erfcf ( float  x );
float erfcinvf ( float  y );
float erfcxf ( float  x );
float erff ( float  x );
float erfinvf ( float  y );
float exp10f ( float  x );
float exp2f ( float  x );
float expf ( float  x );
float expm1f ( float  x );
float fabsf ( float  x );
float fdimf ( float  x, float  y );
float fdividef ( float  x, float  y );
float floorf ( float  x );
float fmaf ( float  x, float  y, float  z );
float fmaxf ( float  x, float  y );
float fminf ( float  x, float  y );
float fmodf ( float  x, float  y );
//float frexpf ( float  x, int* nptr );
float hypotf ( float  x, float  y );
//int ilogbf ( float  x );
//__RETURN_TYPE 	isfinite ( float  a );
//__RETURN_TYPE 	isinf ( float  a );
//__RETURN_TYPE 	isnan ( float  a );
float j0f ( float  x );
float j1f ( float  x );
//float jnf ( int  n, float  x );
//float ldexpf ( float  x, int  exp );
float lgammaf ( float  x );
//long long int 	llrintf ( float  x );
//long long int 	llroundf ( float  x );
float log10f ( float  x );
float log1pf ( float  x );
float log2f ( float  x );
float logbf ( float  x );
float logf ( float  x );
//long int lrintf ( float  x );
//long int lroundf ( float  x );
//float modff ( float  x, float* iptr );
//float nanf ( const char* tagp );
//float nearbyintf ( float  x );
float nextafterf ( float  x, float  y );
float norm3df ( float  a, float  b, float  c );
float norm4df ( float  a, float  b, float  c, float  d );
float normcdff ( float  y );
float normcdfinvf ( float  y );
//float normf ( int  dim, const float* a );
float powf ( float  x, float  y );
float rcbrtf ( float  x );
float remainderf ( float  x, float  y );
//float remquof ( float  x, float  y, int* quo );
float rhypotf ( float  x, float  y );
//float rintf ( float  x );
float rnorm3df ( float  a, float  b, float  c );
float rnorm4df ( float  a, float  b, float  c, float  d );
//float rnormf ( int  dim, const float* a );
float roundf ( float  x );
float rsqrtf ( float  x );
//float scalblnf ( float  x, long int  n );
//float scalbnf ( float  x, int  n );
//__RETURN_TYPE 	signbit ( float  a );
//void sincosf ( float  x, float* sptr, float* cptr );
//void sincospif ( float  x, float* sptr, float* cptr );
float sinf ( float  x );
float sinhf ( float  x );
float sinpif ( float  x );
float sqrtf ( float  x );
float tanf ( float  x );
float tanhf ( float  x );
float tgammaf ( float  x );
float truncf ( float  x );
float y0f ( float  x );
float y1f ( float  x );
//float ynf ( int  n, float  x );
// END MATH FUNCTIONS FROM CUDA
)C";

} // namespace c

namespace cpp {
constexpr auto boundsAsTemplate = R"C(
template<typename T> inline __device__ T floord(T n, T d) {
  return n < 0 ? - (-n + d - 1)/d : n / d;
}
#define if_then_else(cond,a,b) (cond) ? (a) : (b);
)C";
} // namespace cpp

namespace cuda {
constexpr auto common = R"CUDA(

namespace __tc {

// Re-implementing bits of type_traits because nvrtc no likes std includes
template <typename T, typename TT>
struct is_same {
  static constexpr bool value = false;
};

template <typename T>
struct is_same<T, T> {
  static constexpr bool value = true;
};

template <typename T>
struct numeric_limits {
};

template <>
struct numeric_limits<float> {
  static inline __device__ float max() {
    return 3.40282e+38;
  }
  static inline __device__ float min() {
    return -3.40282e+38;
  }
};

template <>
struct numeric_limits<int> {
  static inline __device__ int max() {
    return 0x7FFFFFFF;
  }
  static inline __device__ int min() {
    return 0xFFFFFFFF;
  }
};

enum class ReductionOp : int { Sum = 0, Prod = 1, Min = 2, Max = 3};

// Partial specialization is only allowed for classes...
template <typename T, ReductionOp R>
struct Reducer {
};

template <typename T>
struct Reducer<T, ReductionOp::Sum> {
  typedef T value_type;

  template<typename CubReduce>
  static inline __device__ T reduce(CubReduce red, T val) {
    return red.Sum(val);
  }
  static inline __device__ T reduce(T red, T val) {
    return red + val;
  }
  static constexpr T neutral = T(0);
};

template <typename T>
struct Reducer<T, ReductionOp::Prod> {
  template<typename CubReduce>
  static inline __device__ T reduce(CubReduce red, T val) {
    return red.Prod(val);
  }
  static inline __device__ T reduce(T red, T val) {
    return red * val;
  }
  static constexpr T neutral = T(1);
};

template <typename T>
struct Reducer<T, ReductionOp::Min> {
  template<typename CubReduce>
  static inline __device__ T reduce(CubReduce red, T val) {
    return red.Min(val);
  }
  static inline __device__ T reduce(T red, T val) {
    return red < val ? red : val;
  }
  static constexpr T neutral = numeric_limits<T>::max();
};

template <typename T>
struct Reducer<T, ReductionOp::Max> {
  template<typename CubReduce>
  static inline __device__ T reduce(CubReduce red, T val) {
    return red.Max(val);
  }
  static inline __device__ T reduce(T red, T val) {
    return red > val ? red : val;
  }
  static constexpr T neutral = numeric_limits<T>::min();
};

template <ReductionOp R, typename T>
__inline__ __device__ T warpReduce(T val) {
  for (int i = warpSize / 2; i >= 1; i /= 2) {
    val = Reducer<T, R>::reduce(val, __shfl_down(val, i));
  }
  return val;
}

template <typename Reducer>
struct WithBool {
  WithBool() : val(Reducer::neutral), b(false) {}
  WithBool(typename Reducer::value_type v_, bool b_) : val(v_), b(b_) {}
  typename Reducer::value_type  val;
  bool b;
};

template<typename Reducer>
struct SegmentedReducer {
  __device__ WithBool<Reducer> operator()(
      const WithBool<Reducer>& a, const WithBool<Reducer>& b) {
    return WithBool<Reducer>(
      b.b ? b.val : Reducer::reduce(a.val, b.val),
      a.b || b.b);
  }
};

} // namespace __tc
)CUDA";

constexpr auto cubBlockReduce = R"CUDA(

#include "cub/nvrtc_cub.cuh"

namespace __tc {

#define WARP_SIZE 32

template <int REDUCTION_SIZE, int BLOCKDIMY, int BLOCKDIMZ, ReductionOp R, typename T>
inline __device__ void CubReduceAlongXPowerOf2(T* dest, T val) {
  assert(REDUCTION_SIZE == blockDim.x && "blockDim.x size mismatch");

  using CubReduce = cub::BlockReduce<T, REDUCTION_SIZE>;
  __shared__ typename CubReduce::TempStorage temp_storage[BLOCKDIMY][BLOCKDIMZ];
  T aggregate = Reducer<T, R>::reduce(
    CubReduce(temp_storage[threadIdx.y][threadIdx.z]), val);
  __syncthreads();
  if (threadIdx.x == 0) {
    *dest = Reducer<T, R>::reduce(*dest, aggregate);
  }
  __syncthreads();
}

#define POWEROF2(X)                             \
  ((X) & ((X) - 1) == 0)

template <int REDUCTION_SIZE, int BLOCKDIMY, int BLOCKDIMZ, ReductionOp R, typename T>
inline __device__ void CubReduceAlongX(T* dest, T val) {
  __syncthreads();

  assert(REDUCTION_SIZE == blockDim.x && "blockDim.x size mismatch");

  // Except when blockDim.y == blockDim.z == 1 which seems fine
  bool allowCubReduce = ((blockDim.y == 1) and (blockDim.z == 1));
  if (allowCubReduce or POWEROF2(REDUCTION_SIZE)) {
    CubReduceAlongXPowerOf2<REDUCTION_SIZE, BLOCKDIMY, BLOCKDIMZ, R, T>(dest, val);
    return;
  }

  // CUB reductions do not allow general partial-block reductions.
  // Consider a case where threads(x,y,z) = (11, 12, 13); we want to perform
  // 12x13 parallel 11-wide reductions.
  // A workaround is to perform a full-block prefix-sum that is 11x12x13-wide
  // with a segmented reduction operator.
  using CubScan = cub::BlockScan<
    WithBool<Reducer<T, R>>,
    REDUCTION_SIZE,
    cub::BLOCK_SCAN_RAKING,
    BLOCKDIMY,
    BLOCKDIMZ>;

  __shared__ typename CubScan::TempStorage temp_storage;

  using SegmentedReducerType = SegmentedReducer<Reducer<T, R>>;
  SegmentedReducerType segmentedReducer;

  WithBool<Reducer<T, R>> res;
  // Head of the segment -> true
  WithBool<Reducer<T, R>> v(val, threadIdx.x == 0);
  CubScan(temp_storage).InclusiveScan(v, res, segmentedReducer);
  if (threadIdx.x == REDUCTION_SIZE - 1) {
    *dest = Reducer<T, R>::reduce(*dest, res.val);
  }
}

} // namespace __tc
)CUDA";

const static std::string kCUBReductionName = "__tc::CubReduceAlongX";

} // namespace cuda
} // namespace code
} // namespace tc
