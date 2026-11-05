#ifndef INCLUDE_GUARD_HIP_CUPY_ROCTX_H
#define INCLUDE_GUARD_HIP_CUPY_ROCTX_H

#ifndef CUPY_NO_NVTX
#include <roctx.h>

// this is to ensure we use non-"Ex" APIs like roctxMarkA etc
#define NVTX_VERSION (100 * ROCTX_VERSION_MAJOR + 10 * ROCTX_VERSION_MINOR)

extern "C" {

///////////////////////////////////////////////////////////////////////////////
// roctx
///////////////////////////////////////////////////////////////////////////////

void nvtxMarkA(const char* message) {
    roctxMarkA(message);
}

int nvtxRangePushA(const char* message) {
    return roctxRangePushA(message);
}

int nvtxRangePop() {
    return roctxRangePop();
}

} // extern "C"

#else // #ifndef CUPY_NO_NVTX

// roctx is unavailable (e.g., ROCm on Windows); markers and ranges are no-ops.
#define NVTX_VERSION 0

extern "C" {

void nvtxMarkA(const char* message) {
}

int nvtxRangePushA(const char* message) {
    return 0;
}

int nvtxRangePop() {
    return 0;
}

} // extern "C"

#endif // #ifndef CUPY_NO_NVTX

#endif // #ifndef INCLUDE_GUARD_HIP_CUPY_ROCTX_H
