#pragma once

#if __cplusplus >= 201103 || (defined(_MSC_VER) && _MSC_VER >= 1900)
#ifndef __CUDACC_RTC__
// in NVRTC std:initializer_list is pre-defined (no need to include it)
#include <initializer_list>
#endif
#endif

// Basic implementation of std::type_traits
// We use this regardless when C++ is requested, as NVRTC by default lacks many
// C++ features like this. We need to wrap in a namespace in case Jitify kicks
// in and/or users provide custom definitions.
namespace cupy {
  namespace type_traits {
    template<bool B, class T, class F>
    struct conditional { typedef T type; };
    template<class T, class F>
    struct conditional<false, T, F> { typedef F type; };

    template<bool B, class T = void>
    struct enable_if {};
    template<class T>
    struct enable_if<true, T> { typedef T type; };
  }
}

// math
#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif

#ifdef __HIPCC_RTC__

#include <hip/hip_version.h>
#if HIP_VERSION >= 40400000
// HIP runtime headers can be no longer explicitly included since ROCm 4.5 so
// we only include necessary standard headers.
#include <cassert>
#include <cstddef>

// Confirmed to AMD, ROCm 5.0 doesn't recognize __forceinline__ and
// __noinline__.
#define __noinline__ __attribute__((noinline))
#define __forceinline__ inline __attribute__((always_inline))

#else
#include <hip/hip_fp16.h>
#endif  // #if HIP_VERSION >= 40400000

#elif __HIPCC__

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

#ifdef __HIPCC__

__device__ float16 operator-() {
  return float16(-data_);
}

#endif

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

#ifdef CUPY_JIT_MODE
#ifdef CUPY_JIT_NVCC
#include <thrust/swap.h>
#include <thrust/tuple.h>
#include <thrust/pair.h>
#else
#include <cupy/swap.cuh>
#include <cupy/tuple.cuh>
#include <cupy/pair.cuh>
#endif  // CUPY_JIT_NVCC
#endif  // CUPY_JIT_MODE

#ifdef CUPY_JIT_MODE
namespace cupy {

/*
 * param ndim: the size of the returned tuple
 * param T: the type of the tuple elements
 */
template<int ndim, typename T>
struct as_tuple {

    template <int _ndim, typename... Args>
    struct as_tuple_impl {
        using ChildClass = as_tuple_impl<_ndim - 1, T, Args...>;
        using type = typename ChildClass::type;

        template <typename Ints>
        __device__ static type call(Ints ints, Args... args) {
            return ChildClass::call(ints, ints[_ndim - 1], args...);
        }
    };

    template <typename... Args>
    struct as_tuple_impl<0, Args...> {
        using type = thrust::tuple<Args...>;

        template <typename Ints>
        __device__ static type call(Ints ints, Args... args) {
            return thrust::make_tuple(args...);
        }
    };

    using type = typename as_tuple_impl<ndim>::type;

    template <typename Ints, typename... Args>
    __device__ static type call(Ints ints, Args... args) {
        return as_tuple_impl<ndim>::call(ints, args...);
    }

};

}  // namespace cupy

template <int dim>
struct Dim {
  __device__ Dim() {}
};
#endif  // CUPY_JIT_MODE

template <typename T, typename index_t>
class CArrayIterator {
public:
  typedef ptrdiff_t difference_type;
  typedef T value_type;
  typedef T* pointer;
  typedef T& reference;
#ifdef CUPY_JIT_NVCC
  typedef std::random_access_iterator_tag iterator_category;
#endif  // CUPY_JIT_NVCC

private:
  T* head_;
  index_t step_;
public:
  __host__ __device__ CArrayIterator(T* head, index_t step) {
    this->head_ = head;
    this->step_ = step;
  }
  __host__ __device__ CArrayIterator(const CArrayIterator& itr) {
    this->head_ = itr.head_;
    this->step_ = itr.step_;
  }
  __host__ __device__ bool operator==(const CArrayIterator& itr) const {
    return (this->head_ == itr.head_) && (this->step_ == itr.step_);
  }
  __host__ __device__ bool operator!=(const CArrayIterator& itr) const {
    return !(*this == itr);
  }
  __host__ __device__ T& operator*() const {
    return *(this->head_);
  }
  __host__ __device__ const T* operator->() const {
    return this->head_;
  }
  __host__ __device__ CArrayIterator& operator++() {
    this->head_ += this->step_;
    return *this;
  }
  __host__ __device__ CArrayIterator operator++(int) {
    CArrayIterator tmp = *this;
    this->head_ += this->step_;
    return tmp;
  }
  __host__ __device__ CArrayIterator& operator--() {
    this->head_ -= this->step_;
    return *this;
  }
  __host__ __device__ CArrayIterator operator--(int) {
    CArrayIterator tmp = *this;
    this->head_ -= this->step_;
    return tmp;
  }
  __host__ __device__ CArrayIterator operator+(ptrdiff_t n) const {
    CArrayIterator out = *this;
    out.head_ += out.step_ * n;
    return out;
  }
  __host__ __device__ difference_type operator-(const CArrayIterator& itr) const {
    return (this->head_ - itr.head_) / this->step_;
  }
  __host__ __device__ CArrayIterator operator-(ptrdiff_t n) const {
    CArrayIterator out = *this;
    out.head_ -= out.step_ * n;
    return out;
  }
  __host__ __device__ CArrayIterator& operator+=(ptrdiff_t n) {
    this->head_ += this->step_ * n;
    return *this;
  }
  __host__ __device__ CArrayIterator& operator-=(ptrdiff_t n) {
    this->head_ -= this->step_ * n;
    return *this;
  }
  __host__ __device__ T& operator[](index_t n) const {
    return *(this->head_ + this->step_ * n);
  }
  __host__ __device__ bool operator<(const CArrayIterator& itr) const {
    return this->head_ < itr.head_;
  }
  __host__ __device__ bool operator>(const CArrayIterator& itr) const {
    return this->head_ > itr.head_;
  }
  __host__ __device__ bool operator<=(const CArrayIterator& itr) const {
    return !(*this > itr);
  }
  __host__ __device__ bool operator>=(const CArrayIterator& itr) const {
    return !(*this < itr);
  }
};

template <typename T, int _ndim, bool _c_contiguous=false, bool _use_32bit_indexing=false>
class CArray {
public:
  static const int ndim = _ndim;
  static const bool c_contiguous = _c_contiguous;
  typedef typename cupy::type_traits::conditional<_use_32bit_indexing, int, ptrdiff_t>::type index_t;
  typedef typename cupy::type_traits::conditional<_c_contiguous, T*, CArrayIterator<T, index_t> >::type iterator;

private:
  T* data_;
  ptrdiff_t size_;
  ptrdiff_t shape_[ndim];
  ptrdiff_t strides_[ndim];

public:
  // Constructor supports pointers or initializer lists and strides is optional
  // as long as _c_contiguous=true.
  //    CArray<T, 3> ca(data, shape, strides);
  //    CArray<T, 3, true> ca(data, shape);
  //    CArray<T, 3> ca(data, {1, 2, 3}, {48, 24, 8});
  //    CArray<T, 3, true> ca(data, {1, 2, 3});
  // Initializer lists and optional strides are only supported with -std=c++11
  // or higher.

  template <typename Int1, typename Int2>
  __device__ CArray(T* data, const Int1* shape, const Int2* strides)
      : data_(data), size_(1)
  {
    if (_c_contiguous) {
      assert(strides[_ndim-1] == sizeof(T));
      for (int i = _ndim-1; i > 0; i--) {
        assert(strides[i-1] == shape[i] * strides[i]);
      }
    }
    for (int i = 0; i < _ndim; i++) {
      this->size_ *= shape[i];
      this->shape_[i] = shape[i];
      this->strides_[i] = strides[i];
    }
  } 

#if __cplusplus >= 201103 || (defined(_MSC_VER) && _MSC_VER >= 1900)
  template <typename Int, typename U=T>
  __device__ CArray(typename cupy::type_traits::enable_if<_c_contiguous, U>::type* data,
                    const Int* shape)
      : data_(data), size_(1)
  {
    for (int i = 0; i < _ndim; i++) {
      this->size_ *= shape[i];
      this->shape_[i] = shape[i];
    }
    this->strides_[_ndim-1] = sizeof(T);
    for (int i = _ndim-1; i > 0; i--) {
      this->strides_[i-1] = shape[i] * this->strides_[i];
    }
  }

  template <typename Int, typename U=T>
  __device__ CArray(typename cupy::type_traits::enable_if<_c_contiguous, U>::type* data,
                    const std::initializer_list<Int> shape)
      : CArray(data, shape.begin())
  {
    assert(shape.size() == _ndim);
  }

  template <typename Int1, typename Int2>
  __device__ CArray(T* data,
                    const std::initializer_list<Int1> shape,
                    const std::initializer_list<Int2> strides)
      : CArray(data, shape.begin(), strides.begin())
  {
    assert(shape.size() == _ndim);
    assert(strides.size() == _ndim);
  }
#endif

  __device__ CArray() : data_(NULL), size_(1)
  {
    memset(this->shape_, 0, sizeof(this->shape_));
    memset(this->strides_, 0, sizeof(this->strides_));
  }

  __device__ ptrdiff_t size() const {
    return size_;
  }

  __device__ const ptrdiff_t* shape() const {
    return shape_;
  }

  __device__ const ptrdiff_t* strides() const {
    return strides_;
  }

#ifdef CUPY_JIT_MODE
  __device__ typename cupy::as_tuple<_ndim, ptrdiff_t>::type get_shape() const {
    return cupy::as_tuple<_ndim, ptrdiff_t>::call(shape_);
  }

  __device__ typename cupy::as_tuple<_ndim, ptrdiff_t>::type get_strides() const {
    return cupy::as_tuple<_ndim, ptrdiff_t>::call(strides_);
  }
#endif  // CUPY_JIT_MODE

#if __cplusplus >= 201103 || (defined(_MSC_VER) && _MSC_VER >= 1900)
  template <typename Int>
  __device__ T& operator[](const std::initializer_list<Int> idx_) {
    assert(idx_.size() == _ndim);
    Int idx[ndim];
    memcpy(idx, idx_.begin(), ndim*sizeof(Int));
    return this->operator[](idx);
  }

  template <typename Int>
  __device__ const T& operator[](const std::initializer_list<Int> idx_) const {
    assert(idx_.size() == _ndim);
    Int idx[ndim];
    memcpy(idx, idx_.begin(), ndim*sizeof(Int));
    return this->operator[](idx);
  }
#endif

  template <typename Int>
  __device__ T& operator[](const Int (&idx)[ndim]) {
    return const_cast<T&>(const_cast<const CArray&>(*this)[idx]);
  }

  template <typename Int>
  __device__ const T& operator[](const Int (&idx)[ndim]) const {
    index_t diff = 0;
    for (int dim = 0; dim < ndim; ++dim) {
      diff += static_cast<index_t>(strides_[dim]) * static_cast<index_t>(idx[dim]);
    }
    const char* ptr = reinterpret_cast<const char*>(data_);
    return *reinterpret_cast<const T*>(ptr + diff);
  }

  __device__ T& operator[](ptrdiff_t i) {
    return const_cast<T&>(const_cast<const CArray&>(*this)[i]);
  }

#ifdef CUPY_JIT_MODE
  __forceinline__ __device__ iterator begin_ptr() const {
    return reinterpret_cast<T*>(data_);
  }

  __forceinline__ __device__ iterator end_ptr() const {
    return reinterpret_cast<T*>(data_) + size_;
  }

  __forceinline__ __device__ iterator begin() const {
    return iterator(data_, strides_[0] / sizeof(T));
  }

  __forceinline__ __device__ iterator end() const {
    return iterator(data_ + size_ * strides_[0] / sizeof(T), strides_[0] / sizeof(T));
  }

  template <typename Tuple, int dim>
  __forceinline__ __device__ const T& _indexing(const Tuple &idx, Dim<dim>, const char* ptr) const {
    index_t i = static_cast<index_t>(thrust::get<dim>(idx));
    ptr += static_cast<index_t>(strides_[dim]) * i;
    return _indexing(idx, Dim<dim + 1>(), ptr);
  }

  template <typename Tuple>
  __forceinline__ __device__ const T& _indexing(const Tuple &idx, Dim<_ndim>, const char* ptr) const {
    return *reinterpret_cast<const T*>(ptr);
  }

  template <typename Tuple>
  __forceinline__ __device__ const T& _indexing(const Tuple &idx) const {
    const char* ptr = reinterpret_cast<const char*>(data_);
    return _indexing(idx, Dim<0>(), ptr);
  }

  template <typename Tuple>
  __forceinline__ __device__ T& _indexing(const Tuple &idx) {
    return const_cast<T&>(const_cast<const CArray&>(*this)._indexing(idx));
  }

  template <typename Tuple, int dim, int dimreduce>
  __forceinline__ __device__ char* _slicing(const Tuple &idx, char* new_head_ptr, Dim<dim>, Dim<dimreduce>) const {
    index_t i = static_cast<index_t>(thrust::get<dim>(idx));
    new_head_ptr += static_cast<index_t>(strides_[dim]) * i;
    return _slicing(idx, new_head_ptr, Dim<dim+1>(), Dim<dimreduce>());
  }

  template <typename Tuple, int dimreduce>
  __forceinline__ __device__ char* _slicing(const Tuple &idx, char* new_head_ptr, Dim<dimreduce>, Dim<dimreduce>) const {
    return new_head_ptr;
  }

  template <typename Tuple, int dimreduce>
  __forceinline__ __device__ CArray<T, _ndim - dimreduce, _c_contiguous, _use_32bit_indexing> _slicing(const Tuple &idx, Dim<dimreduce>) {
    char* new_head_ptr = reinterpret_cast<char*>(data_);
    new_head_ptr = _slicing(idx, new_head_ptr, Dim<0>(), Dim<dimreduce>());
    T* new_head = reinterpret_cast<T*>(new_head_ptr);
    return CArray<T, _ndim - dimreduce, _c_contiguous, _use_32bit_indexing>(new_head, shape_ + dimreduce, strides_ + dimreduce);
  }

  __forceinline__ __device__ CArray<T, _ndim - 1, _c_contiguous, _use_32bit_indexing> _slicing(const int idx) {
    char* new_head_ptr = reinterpret_cast<char*>(data_);
    index_t i = static_cast<index_t>(idx);
    new_head_ptr += static_cast<index_t>(strides_[0]) * i;
    T* new_head = reinterpret_cast<T*>(new_head_ptr);
    return CArray<T, _ndim - 1, _c_contiguous, _use_32bit_indexing>(new_head, shape_ + 1, strides_ + 1);
  }
#endif  // CUPY_JIT_MODE

  __device__ const T& operator[](ptrdiff_t idx) const {
    if (c_contiguous) {
      // contiguous arrays can be directly addressed by the
      // numeric value, avoiding expensive 64 bit operations in cuda
      return data_[idx];
    } else {
      // 64-bit mults and divs are pretty expensive and can lead to severe
      // performance degradation in computation bound kernels
      index_t diff = 0;
      index_t i = static_cast<index_t>(idx);
      for (int dim = ndim; --dim > 0; ) {
          index_t shape_dim = static_cast<index_t>(shape_[dim]);
          diff += static_cast<index_t>(strides_[dim]) * (i % shape_dim);
          i /= shape_dim;
      }
      diff += static_cast<index_t>(strides_[0]) * i;
      const char* ptr = reinterpret_cast<const char*>(data_);
      return *reinterpret_cast<const T*>(ptr + diff);
    }
  }
};

template <typename T, bool _use_32bit_indexing>

class CArray<T, 0, true, _use_32bit_indexing> {
private:
  T* data_;
  ptrdiff_t size_;

public:
  static const int ndim = 0;

  __device__ CArray() : data_(NULL), size_(1) { }
  
  __device__ explicit CArray(T* data) : data_(data), size_(1) { }

  template <typename Int>
  __device__ CArray(T* data, Int size) : data_(data), size_(size) { }

  // These constructors are just to match the non-0-dim constructors
  template <typename Int1, typename Int2>
  __device__ CArray(T* data, const Int1* shape, const Int2* strides)
      : data_(data), size_(1) { }

#if __cplusplus >= 201103 || (defined(_MSC_VER) && _MSC_VER >= 1900)
  __device__ CArray(T* data,
                    const std::initializer_list<int> shape,
                    const std::initializer_list<int> strides)
      : data_(data), size_(1)
  {
    assert(shape.size() == 0);
    assert(strides.size() == 0);
  }
#endif


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

template <int _ndim, bool _use_32bit_indexing=false>
class CIndexer {
public:
  static const int ndim = _ndim;
private:
  ptrdiff_t size_;
  ptrdiff_t shape_[ndim];
  ptrdiff_t index_[ndim];

  typedef ptrdiff_t index_t[ndim];

public:
  // Constructor supports pointers or initializer lists and index is optional
  //    CIndexer<3> ca(shape, index);
  //    CIndexer<3> ca({1, 2, 3}, {0, 1, 1});
  //    CIndexer<3> ca = {1, 2, 3};
  // Initializer lists are only supported with -std=c++11 or higher.

  template <typename Int>
  __device__ explicit CIndexer(const Int* shape)
      : size_(1) {
    for (int i = 0; i < _ndim; i++) {
      this->size_ *= shape[i];
      this->shape_[i] = shape[i];
      this->index_[i] = 0;
    }
  }

  template <typename Int1, typename Int2>
  __device__ CIndexer(const Int1* shape, const Int2* index)
      : size_(1) {
    for (int i = 0; i < _ndim; i++) {
      this->size_ *= shape[i];
      this->shape_[i] = shape[i];
      this->index_[i] = index[i];
    }
  }

#if __cplusplus >= 201103 || (defined(_MSC_VER) && _MSC_VER >= 1900)
  template <typename Int>
  __device__ CIndexer(const std::initializer_list<Int> shape)
      : CIndexer(shape.begin())
  {
    assert(shape.size() == _ndim);
  }

  template <typename Int1, typename Int2>
  __device__ CIndexer(const std::initializer_list<Int1> shape,
                      const std::initializer_list<Int2> index)
      : CIndexer(shape.begin(), index.begin())
  {
    assert(shape.size() == _ndim);
    assert(index.size() == _ndim);
  }
#endif

  __device__ CIndexer() : size_(1) 
  {
    memset(this->shape_, 0, sizeof(this->shape_));
    memset(this->index_, 0, sizeof(this->index_));
  }

  __device__ ptrdiff_t size() const {
    return size_;
  }

  __device__ const index_t& get() const {
    return index_;
  }

  __device__ void set(ptrdiff_t i) {
    // ndim == 0 case uses partial template specialization
    if (ndim == 1) {
      index_[0] = i;
    } else if (!_use_32bit_indexing && size_ > 1LL << 31) {
      // 64-bit division is very slow on GPU
      this->_set(static_cast<unsigned long long int>(i));
    } else {
      this->_set(static_cast<unsigned int>(i));
    }
  }

private:
  template<typename index_t>
  __device__ void _set(index_t i) {
      for (int dim = ndim; --dim > 0; ) {
        index_t s = static_cast<index_t>(shape_[dim]);
        if (s & (s - 1)) {
          index_t t = i / s;
          index_[dim] = i - t * s;
          i = t;
        } else { // exp of 2
          index_[dim] = i & (s - 1);
          i >>= _log2(s);
        }
      }
      index_[0] = i;
  }

  // can also be implemented as __ffs(x)-1 or 31-__clz(x)
  static unsigned int __device__ _log2(unsigned int x) { return __popc(x-1); }
  static unsigned long long int __device__ _log2(unsigned long long int x) { return __popcll(x-1); }
};

template <bool _use_32bit_indexing>
class CIndexer<0, _use_32bit_indexing> {
private:
  ptrdiff_t size_;

public:
  static const int ndim = 0;

  __device__ CIndexer() : size_(1) { }

  template <typename Int>
  __device__ explicit CIndexer(Int size) : size_(size) { }

  // These constructors are just to match the non-0-dim constructors
  template <typename Int>
  __device__ explicit CIndexer(const Int* shape) : size_(1) { }

  template <typename Int1, typename Int2>
  __device__ CIndexer(const Int1* shape, const Int2* strides)
      : size_(1) { }

#if __cplusplus >= 201103 || (defined(_MSC_VER) && _MSC_VER >= 1900)
  __device__ CIndexer(const std::initializer_list<int> shape)
      : size_(1)
  {
    assert(shape.size() == 0);
  }

  __device__ CIndexer(const std::initializer_list<int> shape,
                      const std::initializer_list<int> index)
      : size_(1)
  {
    assert(shape.size() == 0);
    assert(index.size() == 0);
  }
#endif

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
