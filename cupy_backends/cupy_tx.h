#ifndef INCLUDE_GUARD_CUPY_TX_H
#define INCLUDE_GUARD_CUPY_TX_H

#if CUPY_USE_HIP

#include "hip/cupy_roctx.h"
#include "hip/cupy_nvtx.h"

#else

#define NVTX_EXPORT_API
#include <nvtx3/nvToolsExt.h>

#endif

#endif // #ifndef INCLUDE_GUARD_CUPY_TX_H
