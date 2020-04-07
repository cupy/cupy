#pragma once

// math
#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif

#ifdef __HIPCC__

#include <hip/hip_fp16.h>

#elif __CUDACC_VER_MAJOR__ >= 9

#include <cuda_fp16.h>

#else  // #if __CUDACC_VER_MAJOR__ >= 9

struct __half_raw {
  unsigned short x;
};

struct half {
private:
  unsigned short data_;
public:
  __device__ half() {}
  __device__ half(const half &v) : data_(v.data_) {}
  __device__ half(float v) : data_(__float2half_rn(v)) {}

  explicit __device__ half(const __half_raw &v) : data_(v.x) {}
  explicit __device__ half(bool v) : data_(__float2half_rn(float(v))) {}
  explicit __device__ half(double v) : data_(__float2half_rn(float(v))) {}
  explicit __device__ half(int v) : data_(__float2half_rn(float(v))) {}
  explicit __device__ half(unsigned int v) : data_(__float2half_rn(float(v))) {}
  explicit __device__ half(long long v) : data_(__float2half_rn(float(v))) {}
  explicit __device__ half(unsigned long long v) : data_(__float2half_rn(float(v))) {}

  __device__ operator float() const {return __half2float(data_);}
  __device__ operator __half_raw() const {__half_raw ret = {data_}; return ret;}
};

#endif  // #if __CUDACC_VER_MAJOR__ >= 9

class float16 {
private:
  half  data_;
public:
  __device__ float16() {}
  __device__ float16(float v) : data_(v) {}

  explicit __device__ float16(bool v) : data_(float(v)) {}
  explicit __device__ float16(double v) : data_(v) {}
  explicit __device__ float16(int v) : data_(v) {}
  explicit __device__ float16(unsigned int v) : data_(v) {}
  explicit __device__ float16(long long v) : data_(v) {}
  explicit __device__ float16(unsigned long long v) : data_(v) {}

  explicit __device__ float16(const half &v): data_(v) {}
  explicit __device__ float16(const __half_raw &v): data_(v) {}

  __device__ operator float() const {return float(data_);}

  static const unsigned short nan = 0x7e00u;

  __device__ int iszero() const {
    return (__half_raw(data_).x & 0x7fffu) == 0;
  }

  __device__ int isnan() const {
    __half_raw raw_ = __half_raw(data_);
    return (raw_.x & 0x7c00u) == 0x7c00u && (raw_.x & 0x03ffu) != 0x0000u;
  }

  __device__ int isinf() const {
    return (__half_raw(data_).x & 0x7fffu) == 0x7c00u;
  }

  __device__ int isfinite() const {
    return (__half_raw(data_).x & 0x7c00u) != 0x7c00u;
  }

  __device__ int signbit() const {
    return (__half_raw(data_).x & 0x8000u) != 0;
  }

  template<typename T>
  inline __device__ float16& operator+=(const T& rhs) {
    *this = *this + rhs;
    return *this;
  }

  template<typename T>
  inline __device__ float16& operator-=(const T& rhs) {
    *this = *this - rhs;
    return *this;
  }

  template<typename T>
  inline __device__ float16& operator*=(const T& rhs) {
    *this = *this * rhs;
    return *this;
  }

  template<typename T>
  inline __device__ float16& operator/=(const T& rhs) {
    *this = *this / rhs;
    return *this;
  }

  friend __device__ float16 copysign(float16 x, float16 y) {
    __half_raw x_raw_ = __half_raw(x.data_);
    __half_raw y_raw_ = __half_raw(y.data_);
    __half_raw ret_raw_;
    ret_raw_.x = (x_raw_.x & 0x7fffu) | (y_raw_.x & 0x8000u);
    return float16(ret_raw_);
  }

  friend __device__ float16 nextafter(float16 x, float16 y) {
    __half_raw x_raw_ = __half_raw(x.data_);
    __half_raw y_raw_ = __half_raw(y.data_);
    __half_raw ret_raw_;
    if (!x.isfinite() || y.isnan()) {
      ret_raw_.x = nan;
    } else if (eq_nonan(x, y)) {
      ret_raw_.x = x_raw_.x;
    } else if (x.iszero()) {
      ret_raw_.x = (y_raw_.x & 0x8000u) + 1;
    } else if (!(x_raw_.x & 0x8000u)) {
      if (static_cast<signed short>(x_raw_.x) > static_cast<signed short>(y_raw_.x)) {
        ret_raw_.x = x_raw_.x - 1;
      } else {
        ret_raw_.x = x_raw_.x + 1;
      }
    } else if(!(y_raw_.x & 0x8000u) || (x_raw_.x & 0x7fffu) > (y_raw_.x & 0x7fffu)) {
      ret_raw_.x = x_raw_.x - 1;
    } else {
      ret_raw_.x = x_raw_.x + 1;
    }
    return float16(ret_raw_);
  }

private:
  static __device__ int eq_nonan(const float16 x, const float16 y) {
    __half_raw x_raw_ = __half_raw(x.data_);
    __half_raw y_raw_ = __half_raw(y.data_);
    return (x_raw_.x == y_raw_.x || ((x_raw_.x | y_raw_.x) & 0x7fff) == 0);
  }
};


__device__ float16 min(float16 x, float16 y) {
  return float16(min(float(x), float(y)));
}
__device__ float16 max(float16 x, float16 y) {
  return float16(max(float(x), float(y)));
}
__device__ float16 fmin(float16 x, float16 y) {
  return float16(fmin(float(x), float(y)));
}
__device__ float16 fmax(float16 x, float16 y) {
  return float16(fmax(float(x), float(y)));
}
__device__ int iszero(float16 x) {return x.iszero();}
__device__ int isnan(float16 x) {return x.isnan();}
__device__ int isinf(float16 x) {return x.isinf();}
__device__ int isfinite(float16 x) {return x.isfinite();}
__device__ int signbit(float16 x) {return x.signbit();}

// CArray
#define CUPY_FOR(i, n) \
    for (ptrdiff_t i = \
            static_cast<ptrdiff_t>(blockIdx.x) * blockDim.x + threadIdx.x; \
         i < (n); \
         i += static_cast<ptrdiff_t>(blockDim.x) * gridDim.x)

template <typename T, int _ndim, bool _c_contiguous=false>
class CArray {
public:
  static const int ndim = _ndim;
  static const bool c_contiguous = _c_contiguous;
private:
  T* data_;
  ptrdiff_t size_;
  ptrdiff_t shape_[ndim];
  ptrdiff_t strides_[ndim];

public:
  __device__ ptrdiff_t size() const {
    return size_;
  }

  __device__ const ptrdiff_t* shape() const {
    return shape_;
  }

  __device__ const ptrdiff_t* strides() const {
    return strides_;
  }

  template <typename Int>
  __device__ T& operator[](const Int (&idx)[ndim]) {
    return const_cast<T&>(const_cast<const CArray&>(*this)[idx]);
  }

  template <typename Int>
  __device__ const T& operator[](const Int (&idx)[ndim]) const {
    const char* ptr = reinterpret_cast<const char*>(data_);
    for (int dim = 0; dim < ndim; ++dim) {
      ptr += static_cast<ptrdiff_t>(strides_[dim]) * idx[dim];
    }
    return *reinterpret_cast<const T*>(ptr);
  }

  __device__ T& operator[](ptrdiff_t i) {
    return const_cast<T&>(const_cast<const CArray&>(*this)[i]);
  }

  __device__ const T& operator[](ptrdiff_t i) const {
    if (c_contiguous) {
      // contiguous arrays can be directly addressed by the
      // numeric value, avoiding expensive 64 bit operations in cuda
      return data_[i];
    }
    const char* ptr = reinterpret_cast<const char*>(data_);
    for (int dim = ndim; --dim > 0; ) {
      ptr += static_cast<ptrdiff_t>(strides_[dim]) * (i % shape_[dim]);
      i /= shape_[dim];
    }
    if (ndim > 0) {
      ptr += static_cast<ptrdiff_t>(strides_[0]) * i;
    }

    return *reinterpret_cast<const T*>(ptr);
  }
};

template <typename T>
class CArray<T, 0, true> {
private:
  T* data_;
  ptrdiff_t size_;

public:
  static const int ndim = 0;

  __device__ ptrdiff_t size() const {
    return size_;
  }

  __device__ const ptrdiff_t* shape() const {
    return NULL;
  }

  __device__ const ptrdiff_t* strides() const {
    return NULL;
  }

  template <typename U>
  __device__ T& operator[](const U&) {
    return *data_;
  }

  template <typename U>
  __device__ T operator[](const U&) const {
    return *data_;
  }
};

template <int _ndim>
class CIndexer {
public:
  static const int ndim = _ndim;
private:
  ptrdiff_t size_;
  ptrdiff_t shape_[ndim];
  ptrdiff_t index_[ndim];

  typedef ptrdiff_t index_t[ndim];

public:
  __device__ ptrdiff_t size() const {
    return size_;
  }

  __device__ void set(ptrdiff_t i) {
    // ndim == 0 case uses partial template specialization
    if (ndim == 1) {
      index_[0] = i;
      return;
    }
    if (size_ > 1LL << 31) {
      // 64-bit division is very slow on GPU
      size_t a = static_cast<size_t>(i);
      for (int dim = ndim; --dim > 0; ) {
        size_t s = static_cast<size_t>(shape_[dim]);
        if (s & (s - 1)) {
          size_t t = a / s;
          index_[dim] = a - t * s;
          a = t;
        } else { // exp of 2
          index_[dim] = a & (s - 1);
          a >>= __popcll(s - 1);
        }
      }
      index_[0] = a;
    } else {
      unsigned int a = static_cast<unsigned int>(i);
      for (int dim = ndim; --dim > 0; ) {
        unsigned int s = static_cast<unsigned int>(shape_[dim]);
        if (s & (s - 1)) {
          unsigned int t = a / s;
          index_[dim] = a - t * s;
          a = t;
        } else { // exp of 2
          index_[dim] = a & (s - 1);
          a >>= __popc(s - 1);
        }
      }
      index_[0] = a;
    }
  }

  __device__ const index_t& get() const {
    return index_;
  }
};

template <>
class CIndexer<0> {
private:
  ptrdiff_t size_;

public:
  static const int ndim = 0;

  __device__ ptrdiff_t size() const {
    return size_;
  }

  __device__ void set(ptrdiff_t i) {
  }

  __device__ const ptrdiff_t* get() const {
    return NULL;
  }
};

__device__ int _floor_divide(int x, int y) {
  if (y == 0) return 0;
  int q = x / y;
  return q - (((x < 0) != (y < 0)) && q * y != x);
}

__device__ long long _floor_divide(long long x, long long y) {
  if (y == 0) return 0;
  long long q = x / y;
  return q - (((x < 0) != (y < 0)) && q * y != x);
}

__device__ unsigned _floor_divide(unsigned x, unsigned y) {
  if (y == 0) return 0;
  return x / y;
}

__device__ unsigned long long _floor_divide(
    unsigned long long x, unsigned long long y) {
  if (y == 0) return 0;
  return x / y;
}

__device__ float _floor_divide(float x, float y) {
  return floor(x / y);
}

__device__ double _floor_divide(double x, double y) {
  return floor(x / y);
}
