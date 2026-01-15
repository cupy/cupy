#pragma once

#include "cupy/carray.cuh"  // for float16
#ifdef __HIPCC__

#include <hip/hip_bf16.h>
#define __nv_bfloat16 __hip_bfloat16;

#else

#include <cuda_bf16.h>

#endif

// Wrapper class that inherits from __nv_bfloat16 to get all operators
class bfloat16 : public __nv_bfloat16 {
public:
  // Default constructor
  __device__ bfloat16() : __nv_bfloat16() {}
  
  // Implicit constructor from float (most common use case)
  __device__ bfloat16(float v) : __nv_bfloat16(v) {}
  
  // Explicit constructors from other numeric types
  explicit __device__ bfloat16(bool v) : __nv_bfloat16(float(v)) {}
  explicit __device__ bfloat16(double v) : __nv_bfloat16(float(v)) {}
  explicit __device__ bfloat16(int v) : __nv_bfloat16(float(v)) {}
  explicit __device__ bfloat16(unsigned int v) : __nv_bfloat16(float(v)) {}
  explicit __device__ bfloat16(long long v) : __nv_bfloat16(float(v)) {}
  explicit __device__ bfloat16(unsigned long long v) : __nv_bfloat16(float(v)) {}
  
  // Constructor from base type
  __device__ bfloat16(const __nv_bfloat16 &v) : __nv_bfloat16(v) {}
  
  // Implicit conversion from float16
  __device__ bfloat16(const float16 &v);
  
  // Special value checking methods
  __device__ int iszero() const {
    return *this == __nv_bfloat16(0.0f);
  }
  
  __device__ int isnan() const {
    return __hisnan(*this);
  }
  
  __device__ int isinf() const {
    return __hisinf(*this);
  }
  
  __device__ int isfinite() const {
    return !__hisnan(*this) && !__hisinf(*this);
  }
  
  // Note: All arithmetic and comparison operators are inherited from __nv_bfloat16
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


// bfloat16 <- float16 conversion
inline __device__ bfloat16::bfloat16(const float16 &v) : __nv_bfloat16(float(v)) {}

// float16 <- bfloat16 conversion
inline __device__ float16::float16(const bfloat16 &v) : data_(float(v)) {}

// float16 -> bfloat16 helper (can't modify float16 class, so provide conversion function)
inline __device__ bfloat16 to_bfloat16(const float16 &v) {
  return bfloat16(float(v));
}

// bfloat16 -> float16 helper
inline __device__ float16 to_float16(const bfloat16 &v) {
  return float16(float(v));
}

