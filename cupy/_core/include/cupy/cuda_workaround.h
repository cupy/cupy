#pragma once

// nvrtc does not support cudaDeviceSynchronize() in device code on H100 GPUs.
// cudaDeviceSynchronize() is used in CUB bundled with CuPy, resulting in
// comiplation error on H100.
#ifdef __CUDACC_RTC__
#if __CUDA_ARCH__ >= 900
cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }
#endif
#endif
