// This file is a stub header file of cudnn for Read the Docs.

#ifndef INCLUDE_GUARD_CUPY_NVRTC_H
#define INCLUDE_GUARD_CUPY_NVRTC_H

#ifndef CUPY_NO_CUDA

#include <nvrtc.h>

#else // #ifndef CUPY_NO_CUDA

extern "C" {

typedef enum {
    NVRTC_SUCCESS = 0,
} nvrtcResult;

typedef struct _nvrtcProgram *nvrtcProgram;

const char *nvrtcGetErrorString(...)
nvrtcResult nvrtcVersion(...)
nvrtcResult nvrtcCreateProgram(...)
nvrtcResult nvrtcDestroyProgram(...)
nvrtcResult nvrtcCompileProgram(...)
nvrtcResult nvrtcGetPTXSize(...)
nvrtcResult nvrtcGetPTX(...)
nvrtcResult nvrtcGetProgramLogSize(...)
nvrtcResult nvrtcGetProgramLog(...)
}

#endif // #ifndef CUPY_NO_CUDA

#endif // #ifndef INCLUDE_GUARD_CUPY_NVRTC_H
