#ifndef _CUPY_TEST_H
#define _CUPY_TEST_H

#if CUPY_USE_HIP

#include <hiprand/hiprand_kernel.h>

typedef hiprandState curandState;
typedef hiprandStateMRG32k3a {} curandStateMRG32k3a;
typedef hiprandStatePhilox4_32_10_t {} curandStatePhilox4_32_10_t;

# else

#ifndef CUPY_NO_CUDA

#include <curand_kernel.h>

# else 

typedef struct {} curandState;
typedef struct {} curandStateMRG32k3a;
typedef struct {} curandStatePhilox4_32_10_t;

#endif
#endif
#endif
