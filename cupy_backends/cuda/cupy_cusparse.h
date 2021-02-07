#ifndef INCLUDE_GUARD_CUDA_CUPY_CUSPARSE_H
#define INCLUDE_GUARD_CUDA_CUPY_CUSPARSE_H

#include <cuda.h>
#include <cusparse.h>

#if !defined(CUSPARSE_VERSION)
#if CUDA_VERSION < 10000
#define CUSPARSE_VERSION CUDA_VERSION // CUDA_VERSION used instead
#else
#define CUSPARSE_VERSION 10000
#endif
#endif // #if !defined(CUSPARSE_VERSION)

#endif  // INCLUDE_GUARD_CUDA_CUPY_CUSPARSE_H
