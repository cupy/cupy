#ifndef _CUPY_TEST_H
#define _CUPY_TEST_H


// This enum holds the generators, we can't fully templatize the generators
// because the dynamic design of BitGenerators in the python side does not allow us
// to determine the correct type at compile time
enum RandGenerators{
   CURAND_XOR_WOW,
   CURAND_MRG32k3a,
   CURAND_PHILOX_4x32_10
};


#ifndef CUPY_NO_CUDA
#include <curand_kernel.h>

void init_curand_generator(int generator, intptr_t state_ptr, uint64_t seed, ssize_t size, intptr_t stream);
void interval_32(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream, uint32_t mx, uint32_t mask);
void interval_64(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream, uint64_t mx, uint64_t mask);
void beta(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream, double a, double b);
void exponential(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream);

# else 

typedef struct {} curandState;
typedef struct {} curandStateMRG32k3a;
typedef struct {} curandStatePhilox4_32_10_t;

void init_curand_generator(int generator, ...) {}
void interval_32(int generator, ...) {}
void interval_64(int generator, ...) {}
void beta(int generator, ...) {}
void exponential(int generator, ...) {}

#endif
#endif
