#ifndef INCLUDE_GUARD_CUPY_TX_H
#define INCLUDE_GUARD_CUPY_TX_H

#if CUPY_USE_HIP

#include "hip/cupy_roctx.h"
#include "stub/cupy_nvtx.h"

#elif !defined(CUPY_NO_CUDA)

#include <nvToolsExt.h>

#else  // defined(CUPY_NO_CUDA)

#define NVTX_VERSION 1

extern "C" {

void nvtxMarkA(...) {
}

int nvtxRangePushA(...) {
    return 0;
}

int nvtxRangePop() {
    return 0;
}

}

#include "stub/cupy_nvtx.h"

#endif

#endif // #ifndef INCLUDE_GUARD_CUPY_TX_H
