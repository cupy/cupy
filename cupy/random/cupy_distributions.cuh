#ifndef _CUPY_RANDOM_H
#define _CUPY_RANDOM_H


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

struct rk_binomial_state {
    int initialized;
    int nsave, m;
    double psave, r, q, fm, p1, xm, xl, xr, c, laml, lamr, p2, p3, p4;
};

#ifdef CUPY_USE_HIP
#include <hip/hip_version.h>
#if HIP_VERSION >= 403
#define COMPILE_FOR_HIP
#endif

// When compiling cython extensions with hip 4.0
// gcc will be used, but the hiprand_kernel can only be compiled with llvm
// so we need to explicitly declare stubs for the functions
#if HIP_VERSION > 400
#include <hiprand_kernel.h>
#else
#include <hiprand.h>
typedef struct {} hiprandState;
typedef struct {} hiprandStateMRG32k3a;
typedef struct {} hiprandStatePhilox4_32_10_t;
#endif
#define cudaStream_t hipStream_t
#define curandState hiprandState
#define curandStateMRG32k3a hiprandStateMRG32k3a
#define curandStatePhilox4_32_10_t hiprandStatePhilox4_32_10_t
#define curand_init hiprand_init
#define curand hiprand
#define curand_uniform hiprand_uniform
#define curand_normal_double hiprand_normal_double
#define curand_normal hiprand_normal

#endif

#if !defined(CUPY_NO_CUDA) && !defined(CUPY_USE_HIP)
#include <curand_kernel.h>
#endif

#if !defined(CUPY_NO_CUDA)
void init_curand_generator(int generator, intptr_t state_ptr, uint64_t seed, ssize_t size, intptr_t stream);
void random_uniform(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream);
void raw(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream);
void interval_32(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream, int32_t mx, int32_t mask);
void interval_64(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream, int64_t mx, int64_t mask);
void beta(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream, intptr_t a, intptr_t b);
void exponential(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream);
void geometric(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream, intptr_t p);
void hypergeometric(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream, intptr_t ngood, intptr_t nbad, intptr_t nsample);
void logseries(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream, intptr_t p);
void poisson(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream, intptr_t lam);
void standard_normal(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream);
void standard_normal_float(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream);
void standard_gamma(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream, intptr_t shape);
void binomial(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream, intptr_t n, intptr_t p, intptr_t binomial_state);

#else
// --no -cuda will not compile the .cu file, so the definition needs to be done here explicitly
typedef struct {} curandState;
typedef struct {} curandStateMRG32k3a;
typedef struct {} curandStatePhilox4_32_10_t;
void init_curand_generator(int generator, intptr_t state_ptr, uint64_t seed, ssize_t size, intptr_t stream) {}
void random_uniform(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream) {}
void raw(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream) {}
void interval_32(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream, int32_t mx, int32_t mask) {}
void interval_64(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream, int64_t mx, int64_t mask) {}
void beta(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream, intptr_t a, intptr_t b) {}
void exponential(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream) {}
void geometric(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream, intptr_t p) {}
void hypergeometric(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream, intptr_t ngood, intptr_t nbad, intptr_t nsample) {}
void logseries(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream, intptr_t p) {}
void poisson(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream, intptr_t lam) {}
void standard_normal(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream) {}
void standard_normal_float(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream) {}
void standard_gamma(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream, intptr_t shape) {}
void binomial(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream, intptr_t n, intptr_t p, intptr_t binomial_state) {}

#endif

#endif
