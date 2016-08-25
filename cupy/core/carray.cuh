// math
#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif

// float16
class float16
{
private:
  unsigned short data_;
public:
  __device__ float16() {}
  __device__ float16(float v) {
    data_ = __float2half_rn(v);
  }
  explicit __device__ float16(bool v) {
    data_ = __float2half_rn(static_cast<float>(v));
  }
  explicit __device__ float16(double v) {
    data_ = __float2half_rn(static_cast<float>(v));
  }
  explicit __device__ float16(int v) {
    data_ = __float2half_rn(static_cast<float>(v));
  }
  explicit __device__ float16(unsigned int v) {
    data_ = __float2half_rn(static_cast<float>(v));
  }
  explicit __device__ float16(long long v) {
    data_ = __float2half_rn(static_cast<float>(v));
  }
  explicit __device__ float16(unsigned long long v) {
    data_ = __float2half_rn(static_cast<float>(v));
  }

  __device__ operator float() const {return __half2float(data_);}

  static const unsigned short nan = 0x7e00u;

  __device__ int iszero() const {
    return (data_ & 0x7fff) == 0;
  }

  __device__ int isnan() const {
    return (data_ & 0x7c00u) == 0x7c00u && (data_ & 0x03ffu) != 0x0000u;
  }

  __device__ int isinf() const {
    return (data_ & 0x7fffu) == 0x7c00u;
  }

  __device__ int isfinite() const {
    return (data_ & 0x7c00u) != 0x7c00u;
  }

  __device__ int signbit() const
  {
    return (data_ & 0x8000u) != 0;
  }

  template<typename T>
  inline __device__ float16& operator+=(const T& rhs)
  {
    *this = *this + rhs;
    return *this;
  }

  template<typename T>
  inline __device__ float16& operator-=(const T& rhs)
  {
    *this = *this - rhs;
    return *this;
  }

  template<typename T>
  inline __device__ float16& operator*=(const T& rhs)
  {
    *this = *this * rhs;
    return *this;
  }

  template<typename T>
  inline __device__ float16& operator/=(const T& rhs)
  {
    *this = *this / rhs;
    return *this;
  }

  static __device__ float16 copysign(float16 x, float16 y) {
    float16 ret;
    ret.data_ = (x.data_ & 0x7fffu) | (y.data_ & 0x8000u);
    return ret;
  }

  static __device__ int eq_nonan(const float16 x, const float16 y) {
    return (x.data_ == y.data_ || ((x.data_ | y.data_) & 0x7fff) == 0);
  }

  static __device__ float16 nextafter(float16 x, float16 y) {
    float16 ret;
    if (!x.isfinite() || y.isnan()) {
      ret.data_ = nan;
    } else if (eq_nonan(x, y)) {
      ret = x;
    } else if (x.iszero()) {
      ret.data_ = (y.data_ & 0x8000u) + 1;
    } else if (!(x.data_ & 0x8000u)) {
      if (static_cast<short>(x.data_) > static_cast<short>(y.data_)) {
        ret.data_ = x.data_ - 1;
      } else {
        ret.data_ = x.data_ + 1;
      }
    } else if(!(y.data_ & 0x8000u) || (x.data_ & 0x7fffu) > (y.data_ & 0x7fffu)) {
      ret.data_ = x.data_ - 1;
    } else {
      ret.data_ = x.data_ + 1;
    }
    return ret;
  }
};

__device__ float16 min(float16 x, float16 y) {
  return float16(min(static_cast<float>(x), static_cast<float>(y)));
}
__device__ float16 max(float16 x, float16 y) {
  return float16(max(static_cast<float>(x), static_cast<float>(y)));
}
__device__ int iszero(float16 x) {return x.iszero();}
__device__ int isnan(float16 x) {return x.isnan();}
__device__ int isinf(float16 x) {return x.isinf();}
__device__ int isfinite(float16 x) {return x.isfinite();}
__device__ int signbit(float16 x) {return x.signbit();}
__device__ float16 copysign(float16 x, float16 y) {return float16::copysign(x, y);}
__device__ float16 nextafter(float16 x, float16 y) {return float16::nextafter(x, y);}


// CArray
#define CUPY_FOR(i, n) \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
         i < (n); \
         i += blockDim.x * gridDim.x)

template <typename T, int ndim>
class CArray {
private:
  T* data_;
  int size_;
  int shape_[ndim];
  int strides_[ndim];

public:
  __device__ int size() const {
    return size_;
  }

  __device__ const int* shape() const {
    return shape_;
  }

  __device__ const int* strides() const {
    return strides_;
  }

  __device__ T& operator[](const int* idx) {
    char* ptr = reinterpret_cast<char*>(data_);
    for (int dim = 0; dim < ndim; ++dim) {
      ptr += strides_[dim] * idx[dim];
    }
    return *reinterpret_cast<T*>(ptr);
  }

  __device__ T operator[](const int* idx) const {
    return (*const_cast<CArray<T, ndim>*>(this))[idx];
  }

  __device__ T& operator[](int i) {
    char* ptr = reinterpret_cast<char*>(data_);
    for (int dim = ndim; --dim > 0; ) {
      ptr += strides_[dim] * (i % shape_[dim]);
      i /= shape_[dim];
    }
    if (ndim > 0) {
      ptr += strides_[0] * i;
    }

    return *reinterpret_cast<T*>(ptr);
  }

  __device__ T operator[](int i) const {
    return (*const_cast<CArray<T, ndim>*>(this))[i];
  }
};

template <typename T>
class CArray<T, 0> {
private:
  T* data_;
  int size_;

public:
  __device__ int size() const {
    return size_;
  }

  __device__ T& operator[](const int* idx) {
    return *reinterpret_cast<T*>(data_);
  }

  __device__ T operator[](const int* idx) const {
    return (*const_cast<CArray<T, 0>*>(this))[idx];
  }

  __device__ T& operator[](int i) {
    return *reinterpret_cast<T*>(data_);
  }

  __device__ T operator[](int i) const {
    return (*const_cast<CArray<T, 0>*>(this))[i];
  }
};

template <int ndim>
class CIndexer {
private:
  int size_;
  int shape_[ndim];
  int index_[ndim];

public:
  __device__ int size() const {
    return size_;
  }

  __device__ void set(int i) {
    unsigned int a = i;
    for (int dim = ndim; --dim > 0; ) {
      unsigned int s = shape_[dim];
      index_[dim] = (a % s);
      a /= s;
    }
    if (ndim > 0) {
      index_[0] = a;
    }
  }

  __device__ const int* get() const {
    return index_;
  }
};

template <>
class CIndexer<0> {
private:
  int size_;

public:
  __device__ int size() const {
    return size_;
  }

  __device__ void set(int i) {
  }

  __device__ const int* get() const {
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
