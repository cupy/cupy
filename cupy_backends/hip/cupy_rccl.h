#ifndef INCLUDE_GUARD_HIP_CUPY_RCCL_H
#define INCLUDE_GUARD_HIP_CUPY_RCCL_H
#include <hip/hip_version.h>
#if HIP_VERSION >= 50530600 
#include <rccl/rccl.h>
#else
#include <rccl.h>
#endif
typedef hipStream_t cudaStream_t;

#endif
