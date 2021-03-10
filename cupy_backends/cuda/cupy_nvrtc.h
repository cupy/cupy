#ifndef INCLUDE_GUARD_CUDA_CUPY_NVRTC_H
#define INCLUDE_GUARD_CUDA_CUPY_NVRTC_H

#include <cuda.h>  // for CUDA_VERSION
#include <nvrtc.h>

extern "C" {

#if CUDA_VERSION < 11020
// functions added in CUDA 11.2
nvrtcResult nvrtcGetNumSupportedArchs(...) {
    return NVRTC_ERROR_INTERNAL_ERROR;
}

nvrtcResult nvrtcGetSupportedArchs(...) {
    return NVRTC_ERROR_INTERNAL_ERROR;
}
#endif
}

#endif // #ifndef INCLUDE_GUARD_CUDA_CUPY_NVRTC_H
