#ifndef INCLUDE_GUARD_CUPY_CUSPARSE_H
#define INCLUDE_GUARD_CUPY_CUSPARSE_H

#ifdef CUPY_USE_HIP

#include "hip/cupy_hip_common.h"
#include "hip/cupy_hipsparse.h"

#else

#include "cuda/cupy_cusparse.h"

#endif

#endif  // INCLUDE_GUARD_CUPY_CUSPARSE_H
