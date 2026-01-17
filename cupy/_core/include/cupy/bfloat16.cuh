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

  // Conversion with float16
  __device__ bfloat16(float16 v) : __nv_bfloat16(float(v)) {}
  __device__ operator float16() { return float16(float(*this)); }

  // Constructor from base type
  __device__ bfloat16(const __nv_bfloat16 &v) : __nv_bfloat16(v) {}
  
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

__device__ bfloat16 _floor_divide(bfloat16 x, bfloat16 y) {
  // Used for floor_divide and remainder ufuncs (a bit unclear what
  // the computation type should be for this and the rest/return).
  return floor(float(x) / float(y));
}

