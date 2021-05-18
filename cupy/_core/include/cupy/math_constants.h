#ifndef CUPY_MATH_CONSTANTS_H
#define CUPY_MATH_CONSTANTS_H

/*
   We bundle some constants found in math_constants.h, so that code can be
   JIT compiled in environments without a full CUDA installation. The constants
   are added as needed to avoid copying the entire header.
*/


/* single precision constants */
#define CUDART_INF_F            __int_as_float(0x7f800000)
#define CUDART_NAN_F            __int_as_float(0x7fffffff)

/* double precision constants */
#define CUDART_INF              __longlong_as_double(0x7ff0000000000000ULL)
#define CUDART_NAN              __longlong_as_double(0xfff8000000000000ULL)


#endif  // CUPY_MATH_CONSTANTS_H
