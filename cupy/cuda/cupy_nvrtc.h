// This file is a stub header file of nvrtc for Read the Docs.

#ifndef INCLUDE_GUARD_CUPY_NVRTC_H
#define INCLUDE_GUARD_CUPY_NVRTC_H

#ifdef CUPY_USE_HIP

#include "cupy_hiprtc.h"

#elif !defined(CUPY_NO_CUDA)

#include <nvrtc.h>

#else

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

#endif

#endif // #ifndef INCLUDE_GUARD_CUPY_NVRTC_H
