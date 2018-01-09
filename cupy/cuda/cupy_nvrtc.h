// This file is a stub header file of nvrtc for Read the Docs.

#ifndef INCLUDE_GUARD_CUPY_NVRTC_H
#define INCLUDE_GUARD_CUPY_NVRTC_H

#if !defined(CUPY_NO_CUDA) && !defined(CUPY_USE_HIP)

#include <nvrtc.h>

#else // #ifndef #if !defined(CUPY_NO_CUDA) && !defined(CUPY_USE_HIP)

extern "C" {

typedef enum {
    NVRTC_SUCCESS = 0,
} nvrtcResult;

typedef struct _nvrtcProgram *nvrtcProgram;

const char *nvrtcGetErrorString(...) {
    return NULL;
}

nvrtcResult nvrtcVersion(...) {
    return NVRTC_SUCCESS;
}

nvrtcResult nvrtcCreateProgram(...) {
    return NVRTC_SUCCESS;
}

nvrtcResult nvrtcDestroyProgram(...) {
    return NVRTC_SUCCESS;
}

nvrtcResult nvrtcCompileProgram(...) {
    return NVRTC_SUCCESS;
}

nvrtcResult nvrtcGetPTXSize(...) {
    return NVRTC_SUCCESS;
}

nvrtcResult nvrtcGetPTX(...) {
    return NVRTC_SUCCESS;
}

nvrtcResult nvrtcGetProgramLogSize(...) {
    return NVRTC_SUCCESS;
}

nvrtcResult nvrtcGetProgramLog(...) {
    return NVRTC_SUCCESS;
}

}

#endif // #ifndef #if !defined(CUPY_NO_CUDA) && !defined(CUPY_USE_HIP)

#endif // #ifndef INCLUDE_GUARD_CUPY_NVRTC_H
