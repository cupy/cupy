#pragma once

#include "cupy/carray.cuh"

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)

__device__ double atomicAdd(double *address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull;
    unsigned long long int assumed;
    do {
	assumed = old;
	old = atomicCAS(address_as_ull, assumed,
			__double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

#endif // #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)


// Templated to defer in case float16/bfloat16 are not fully defined.
template<typename T, 
typename = typename cupy::type_traits::enable_if<
    cupy::type_traits::is_same<T, float16>::value ||
    cupy::type_traits::is_same<T, bfloat16>::value>::type>
__device__ T atomicAdd(T* address, T val) {
  unsigned int *aligned = (unsigned int*)((size_t)address - ((size_t)address & 2));
  unsigned int old = *aligned;
  unsigned int assumed;
  unsigned short old_as_us;
  do {
    assumed = old;
    old_as_us = (unsigned short)((size_t)address & 2 ? old >> 16 : old & 0xffff);

    T sum = T::from_bits(old_as_us) + val;
    unsigned short sum_as_us = sum.to_bits();

    unsigned int sum_as_ui = (size_t)address & 2 ? (sum_as_us << 16) | (old & 0xffff)
                                                 : (old & 0xffff0000) | sum_as_us;
    old = atomicCAS(aligned, assumed, sum_as_ui);
  } while (assumed != old);

  return T::from_bits(old_as_us);
};


__device__ long long atomicAdd(long long *address, long long val) {
    return atomicAdd(reinterpret_cast<unsigned long long*>(address),
                     static_cast<unsigned long long>(val));
}


#if __HIPCC__
#include <hip/hip_version.h>
#endif  // #if __HIPCC__

// Skip if ROCm 4.5+ as it implements the following atomic functions.
#if !defined(__HIPCC__) || HIP_VERSION < 40400000

__device__ float atomicMax(float* address, float val) {
  int* address_as_i = reinterpret_cast<int*>(address);
  int old = *address_as_i, assumed;
  do {
    assumed = old;
    old = atomicCAS(
        reinterpret_cast<int*>(address), assumed,
        __float_as_int(fmaxf(val, __int_as_float(assumed))));
  } while (assumed != old);
  return __int_as_float(old);
}


__device__ double atomicMax(double* address, double val) {
  unsigned long long* address_as_i =
    reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *address_as_i, assumed;
  do {
    assumed = old;
    const long long result = __double_as_longlong(
      fmaxf(val, __longlong_as_double(reinterpret_cast<long long&>(assumed))));
    old = atomicCAS(
      address_as_i, assumed,
      reinterpret_cast<const unsigned long long&>(result));
  } while (assumed != old);
  return __longlong_as_double(reinterpret_cast<long long&>(old));
}


__device__ float atomicMin(float* address, float val) {
  int* address_as_i = reinterpret_cast<int*>(address);
  int old = *address_as_i, assumed;
  do {
    assumed = old;
    old = atomicCAS(
        reinterpret_cast<int*>(address), assumed,
        __float_as_int(fminf(val, __int_as_float(assumed))));
  } while (assumed != old);
  return __int_as_float(old);
}


__device__ double atomicMin(double* address, double val) {
  unsigned long long* address_as_i =
    reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *address_as_i, assumed;
  do {
    assumed = old;
    const long long result = __double_as_longlong(
      fminf(val, __longlong_as_double(reinterpret_cast<long long&>(assumed))));
    old = atomicCAS(
      address_as_i, assumed,
      reinterpret_cast<const unsigned long long&>(result));
  } while (assumed != old);
  return __longlong_as_double(reinterpret_cast<long long&>(old));
}

#endif  // #if !defined(__HIPCC__) || HIP_VERSION < 40400000
