#ifndef INCLUDE_GUARD_HIP_CUPY_ROCTX_H
#define INCLUDE_GUARD_HIP_CUPY_ROCTX_H

#ifndef CUPY_NO_NVTX
#include <roctx.h>
#endif // #ifndef CUPY_NO_NVTX

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

#endif // #ifndef INCLUDE_GUARD_HIP_CUPY_ROCTX_H
