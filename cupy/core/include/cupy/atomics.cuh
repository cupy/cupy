#pragma once

#if __CUDA_ARCH__ < 600

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

#endif

#if __CUDACC_VER_MAJOR__ >= 9
#include <device_functions_decls.h>
#endif

__device__ float16 atomicAdd(float16* address, float16 val) {
  unsigned int *aligned = (unsigned int*)((size_t)address - ((size_t)address & 2));
  unsigned int old = *aligned;
  unsigned int assumed;
  unsigned short old_as_us;
  do {
    assumed = old;
    old_as_us = (unsigned short)((size_t)address & 2 ? old >> 16 : old & 0xffff);
#if __CUDACC_VER_MAJOR__ >= 9
    unsigned short sum_as_us = __nv_float2half_rn(__nv_half2float(old_as_us) + float(val));
#else
    unsigned short sum_as_us = __float2half_rn(__half2float(old_as_us) + float(val));
#endif
    unsigned int sum_as_ui = (size_t)address & 2 ? (sum_as_us << 16) | (old & 0xffff)
                                                 : (old & 0xffff0000) | sum_as_us;
    old = atomicCAS(aligned, assumed, sum_as_ui);
  } while(assumed != old);
  return float16(old_as_us);
}
