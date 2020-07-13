#pragma once

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

__device__ float16 atomicAdd(float16* address, float16 val) {
  unsigned int *aligned = (unsigned int*)((size_t)address - ((size_t)address & 2));
  unsigned int old = *aligned;
  unsigned int assumed;
  unsigned short old_as_us;
  do {
    assumed = old;
    old_as_us = (unsigned short)((size_t)address & 2 ? old >> 16 : old & 0xffff);
#if __CUDACC_VER_MAJOR__ >= 9
    half sum = __float2half_rn(__half2float(__ushort_as_half(old_as_us)) + float(val));
    unsigned short sum_as_us = __half_as_ushort(sum);
#else
    unsigned short sum_as_us = __float2half_rn(__half2float(old_as_us) + float(val));
#endif
    unsigned int sum_as_ui = (size_t)address & 2 ? (sum_as_us << 16) | (old & 0xffff)
                                                 : (old & 0xffff0000) | sum_as_us;
    old = atomicCAS(aligned, assumed, sum_as_ui);
  } while(assumed != old);
  __half_raw raw;
  raw.x = old_as_us;
  return float16(raw);
};


__device__ long long atomicAdd(long long *address, long long val) {
    return atomicAdd(reinterpret_cast<unsigned long long*>(address),
                     static_cast<unsigned long long>(val));
}


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
