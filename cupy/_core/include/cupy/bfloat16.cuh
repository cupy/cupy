#pragma once

#define CUPY_BFLOAT16_CUH_

#include "cupy/carray.cuh"
#include "cupy/cuda_workaround.h"

#ifdef __HIPCC__

#include <hip/hip_bf16.h>
#define __nv_bfloat16 __hip_bfloat16;

#else

#include <cuda_bf16.h>

#endif


// Wrapper class that inherits from __nv_bfloat16 to get all operators
class bfloat16 {
private: 
  __nv_bfloat16 data_;
public:
  // Default constructor
  __device__ bfloat16() : data_() {}
  __device__ bfloat16(const __nv_bfloat16 &v) : data_(v) {}
  __device__ bfloat16(float v) : data_(v) {}

  // Explicit constructors from other numeric types
  explicit __device__ bfloat16(bool v) : data_(float(v)) {}
  explicit __device__ bfloat16(double v) : data_(float(v)) {}
  explicit __device__ bfloat16(int v) : data_(float(v)) {}
  explicit __device__ bfloat16(unsigned int v) : data_(float(v)) {}
  explicit __device__ bfloat16(long long v) : data_(float(v)) {}
  explicit __device__ bfloat16(unsigned long long v) : data_(float(v)) {}

  __device__ operator float() const {return float(data_);}

  // From float16: using template so it's ok if float16 is undefined
  template<typename T, 
    typename = typename cupy::type_traits::enable_if<
      cuda::std::is_same_v<T, float16>>::type>
  explicit __device__ bfloat16(T v) : data_(float(v)) {}

  // Generally allow assignments if explicit conversion is available
  // We may want a better solution for this (e.g. fix kernels to explicitly
  // convert)
  bfloat16& operator=(const bfloat16&) = default;

  template <typename T>
  __device__ bfloat16& operator=(const T &rhs) {
    *this = static_cast<bfloat16>(rhs);
    return *this;
  }

  // Special value checking methods
  __device__ int iszero() const {
    return data_ == __nv_bfloat16(0.0f);
  }

  __device__ int isnan() const {
    return __hisnan(data_);
  }

  __device__ int isinf() const {
    return __hisinf(data_);
  }

  __device__ int isfinite() const {
    return !__hisnan(data_) && !__hisinf(data_);
  }

  friend __device__ int signbit(bfloat16 x);
  friend __device__ bfloat16 nextafter(bfloat16 x, bfloat16 y);
};

// Min/max functions
__device__ bfloat16 min(bfloat16 x, bfloat16 y) {
  return bfloat16(min(float(x), float(y)));
}

__device__ bfloat16 max(bfloat16 x, bfloat16 y) {
  return bfloat16(max(float(x), float(y)));
}

__device__ bfloat16 fmin(bfloat16 x, bfloat16 y) {
  return bfloat16(fmin(float(x), float(y)));
}

__device__ bfloat16 fmax(bfloat16 x, bfloat16 y) {
  return bfloat16(fmax(float(x), float(y)));
}

// Special value checking functions (free functions for compatibility)
__device__ int iszero(bfloat16 x) {
  return x.iszero();
}

__device__ int isnan(bfloat16 x) {
  return x.isnan();
}

__device__ int isinf(bfloat16 x) {
  return x.isinf();
}

__device__ int isfinite(bfloat16 x) {
  return x.isfinite();
}

// TODO(seberg): There should be an easy direct definition?
__device__ int signbit(bfloat16 x) {
  return (__bfloat16_as_ushort(x.data_) & 0x8000u) != 0;
}

__device__ bfloat16 _floor_divide(bfloat16 x, bfloat16 y) {
  // Used for floor_divide and remainder ufuncs (a bit unclear what
  // the computation type should be for this and the rest/return).
  return bfloat16(floor(float(x) / float(y)));
}

__device__ bfloat16 nextafter(bfloat16 x, bfloat16 y) {
  unsigned short x_raw = __bfloat16_as_ushort(x.data_);
  unsigned short y_raw = __bfloat16_as_ushort(y.data_);
  unsigned short ret_raw;
  
  if (x.isnan() || y.isnan()) {
    ret_raw = 0x7fc0u;  // NaN
  } else if (x == y) {
    ret_raw = x_raw;
  } else if (isinf(x)) {
    // maximum finite value plus sign bit
    ret_raw = 0x7f7fu | (x_raw & 0x8000u);
  } else if (x == 0) {
    ret_raw = (y_raw & 0x8000u) + 1;
  } else if (!(x_raw & 0x8000u)) {
    if (static_cast<signed short>(x_raw) > static_cast<signed short>(y_raw)) {
      ret_raw = x_raw - 1;
    } else {
      ret_raw = x_raw + 1;
    }
  } else if (!(y_raw & 0x8000u) || (x_raw & 0x7fffu) > (y_raw & 0x7fffu)) {
    ret_raw = x_raw - 1;
  } else {
    ret_raw = x_raw + 1;
  }
  
  return *reinterpret_cast<bfloat16*>(&ret_raw);
}