#ifndef INCLUDE_GUARD_CUDA_CUPY_NVRTC_H
#define INCLUDE_GUARD_CUDA_CUPY_NVRTC_H

#include <cuda.h>  // for CUDA_VERSION
#include <nvrtc.h>

extern "C" {

#if CUDA_VERSION < 11010
// functions added in CUDA 11.1
nvrtcResult nvrtcGetCUBINSize(...) {
    return NVRTC_ERROR_COMPILATION;
}

nvrtcResult nvrtcGetCUBIN(...) {
    return NVRTC_ERROR_COMPILATION;
}
#endif

}

#endif // #ifndef INCLUDE_GUARD_CUDA_CUPY_NVRTC_H
