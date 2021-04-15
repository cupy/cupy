#ifndef _CUPY_TEST_H
#define _CUPY_TEST_H


#if defined(_MSC_VER)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif


// This enum holds the generators, we can't fully templatize the generators
// because the dynamic design of BitGenerators in the python side does not allow us
// to determine the correct type at compile time
// We could use Jitify to avoid this code, but cuRAND EULA does not allow us to 
// redistribute cuRAND header
enum RandGenerators{
   CURAND_XOR_WOW,
   CURAND_MRG32k3a,
   CURAND_PHILOX_4x32_10
};


#if !defined(CUPY_NO_CUDA) && !defined(CUPY_USE_HIP)
#include <curand_kernel.h>

void init_curand_generator(int generator, intptr_t state_ptr, uint64_t seed, ssize_t size, intptr_t stream);
void random_uniform(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream);
void raw(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream);
void interval_32(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream, int32_t mx, int32_t mask);
void interval_64(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream, int64_t mx, int64_t mask);
void beta(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream, double a, double b);
void exponential(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream);
void poisson(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream, double lam);
void standard_normal(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream);
void standard_normal_float(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream);
void standard_gamma(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream, intptr_t shape);

# else 

typedef struct {} curandState;
typedef struct {} curandStateMRG32k3a;
typedef struct {} curandStatePhilox4_32_10_t;

//Travis doesn't like variadic templates in these functions
void init_curand_generator(int generator, intptr_t state_ptr, uint64_t seed, ssize_t size, intptr_t stream) {}
void random_uniform(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream) {}
void raw(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream) {}
void interval_32(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream, int32_t mx, int32_t mask) {}
void interval_64(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream, int64_t mx, int64_t mask) {}
void beta(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream, double a, double b) {}
void exponential(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream) {}
void poisson(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream, double lam) {}
void standard_normal(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream) {}
void standard_normal_float(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream){}
void standard_gamma(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream, intptr_t shape) {}

#endif
#endif
