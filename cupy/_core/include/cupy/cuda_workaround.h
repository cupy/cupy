#pragma once

#ifdef __CUDACC_RTC__
// cudaDeviceSynchronize() is no longer supported by nvrtc in device code on
// H100 GPUs or any GPUs in CUDA 12.x. cudaDeviceSynchronize() is used in CUB
// bundled with CuPy, resulting in comiplation error when GPU is H100 or later,
// or CUDA version is 12 or later.
#if __CUDA_ARCH__ >= 900
cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }
#elif __CUDACC_VER_MAJOR__ >= 12
cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }
#endif
#endif
