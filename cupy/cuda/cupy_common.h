#ifndef INCLUDE_GUARD_CUPY_COMMON_H
#define INCLUDE_GUARD_CUPY_COMMON_H

#include "cupy_cuComplex.h"
/*
   cuda_fp16.h should only be included in C++ codes that actually
   need __half. For bookkeeping purposes, in Cython codes we simply
   use cupy.float16.
*/

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


#endif // INCLUDE_GUARD_CUPY_COMMON_H
