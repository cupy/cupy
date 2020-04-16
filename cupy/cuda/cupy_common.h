#ifndef INCLUDE_GUARD_CUPY_COMMON_H
#define INCLUDE_GUARD_CUPY_COMMON_H

#include "cupy_cuComplex.h"

typedef char cpy_byte;
typedef unsigned char cpy_ubyte;
typedef short cpy_short;
typedef unsigned short cpy_ushort;
typedef int cpy_int;
typedef unsigned int cpy_uint;
typedef long long cpy_long;
typedef unsigned long long cpy_ulong;
typedef float cpy_float;
typedef double cpy_double;
typedef cuComplex cpy_complex64;
typedef cuDoubleComplex cpy_complex128;
typedef bool cpy_bool;

#ifndef CUPY_NO_CUDA
#if (__CUDACC_VER_MAJOR__ > 9 || (__CUDACC_VER_MAJOR__ == 9 && __CUDACC_VER_MINOR__ == 2)) \
    && (__CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__))
#include <cuda_fp16.h>
#endif
#endif // CUPY_NO_CUDA

typedef struct __half cpy_half;

#endif // INCLUDE_GUARD_CUPY_COMMON_H
