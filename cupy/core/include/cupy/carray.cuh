#pragma once

// math
#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif

// CArray
#define CUPY_FOR(i, n) \
    for (ptrdiff_t i = \
            static_cast<ptrdiff_t>(blockIdx.x) * blockDim.x + threadIdx.x; \
         i < (n); \
         i += static_cast<ptrdiff_t>(blockDim.x) * gridDim.x)

template <typename T, int _ndim, bool _c_contiguous=false, bool _use_32bit_indexing=false>
class CArray {
public:
  static const int ndim = _ndim;
  static const bool c_contiguous = _c_contiguous;
  static const bool use_32bit_indexing = _use_32bit_indexing;
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
    if (use_32bit_indexing) {
      for (int dim = 0; dim < ndim; ++dim) {
        int i = static_cast<int>(idx[dim]);
        ptr += static_cast<int>(strides_[dim]) * i;
      }
    } else {
      for (int dim = 0; dim < ndim; ++dim) {
        ptr += static_cast<ptrdiff_t>(strides_[dim]) * idx[dim];
      }
    }
    return *reinterpret_cast<const T*>(ptr);
  }

  __device__ T& operator[](ptrdiff_t i) {
    return const_cast<T&>(const_cast<const CArray&>(*this)[i]);
  }

  __device__ const T& operator[](ptrdiff_t idx) const {
    if (c_contiguous) {
      // contiguous arrays can be directly addressed by the
      // numeric value, avoiding expensive 64 bit operations in cuda
      return data_[idx];
    }
    // 64-bit mults and divs are pretty expensive and can 
    // lead to severe perforamance degradation in computation bound
    // kernels
    const char* ptr = reinterpret_cast<const char*>(data_);
    if (use_32bit_indexing) {
      int i = static_cast<int>(idx);
      for (int dim = ndim; --dim > 0; ) {
        int shape_dim = static_cast<int>(shape_[dim]);
        ptr += static_cast<int>(strides_[dim]) * (i % shape_dim);
        i /= shape_dim;
      }
      if (ndim > 0) {
        ptr += static_cast<int>(strides_[0]) * i;
      }
    } else {
      ptrdiff_t i = idx;
      for (int dim = ndim; --dim > 0; ) {
        ptr += static_cast<ptrdiff_t>(strides_[dim]) * (i % shape_[dim]);
        i /= shape_[dim];
      }
      if (ndim > 0) {
        ptr += static_cast<ptrdiff_t>(strides_[0]) * i;
      }
    }
    return *reinterpret_cast<const T*>(ptr);
  }
};

template <typename T>

class CArray<T, 0, true, true> {
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
