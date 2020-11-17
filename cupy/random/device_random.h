#ifndef _CUPY_TEST_H
#define _CUPY_TEST_H

#ifndef CUPY_NO_CUDA

#include <curand_kernel.h>

# else 

typedef struct {} curandState;
typedef struct {} curandStateMRG32k3a;
typedef struct {} curandStatePhilox4_32_10_t;

#endif
#endif
