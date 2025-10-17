#ifndef INCLUDE_GUARD_CUPY_CUSPARSE_H
#define INCLUDE_GUARD_CUPY_CUSPARSE_H

#if !defined(CUPY_NO_CUDA) && !defined(CUPY_USE_HIP)

#include "cuda/cupy_cusparse.h"

#elif defined(CUPY_USE_HIP)

#include "hip/cupy_hip_common.h"
#include "hip/cupy_hipsparse.h"

#elif CUPY_USE_ASCEND
// CANN does not yet support sparse blas
#include "stub/cupy_cuda_common.h"
#include "stub/cupy_cusparse.h"

#else

#include "stub/cupy_cuda_common.h"
#include "stub/cupy_cusparse.h"

#endif  // #if defined(CUPY_NO_CUDA) || defined(CUPY_USE_HIP)


#endif  // INCLUDE_GUARD_CUPY_CUSPARSE_H
