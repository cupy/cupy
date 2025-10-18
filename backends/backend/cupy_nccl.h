#ifndef INCLUDE_GUARD_CUDA_CUPY_NCCL_H
#define INCLUDE_GUARD_CUDA_CUPY_NCCL_H

#include <nccl.h>

#ifndef NCCL_MAJOR
#ifndef CUDA_HAS_HALF
#define ncclHalf ((ncclDataType_t)2)
#endif
#endif

#endif
