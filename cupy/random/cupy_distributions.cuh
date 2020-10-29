#ifndef _CUPY_TEST_H
#define _CUPY_TEST_H
#include <cstdint>
#include <curand_kernel.h>

void init_xor_generator(intptr_t state_ptr, uint64_t seed, ssize_t size, intptr_t stream);
void interval_32(intptr_t state, intptr_t out, ssize_t size, intptr_t stream, int mx, int mask);
void interval_64(intptr_t state, intptr_t out, ssize_t size, intptr_t stream, uint64_t mx, uint64_t mask);
void beta(intptr_t state, intptr_t out, ssize_t size, intptr_t stream, double a, double b);
void standard_exponential(intptr_t state, intptr_t out, ssize_t size, intptr_t stream);

void exponential(intptr_t state, intptr_t out, ssize_t size, intptr_t stream);
// This enum holds the generators, we can't fully templatize the generators
// because the dynamic design of BitGenerators in the python side does not allow us
// to determine the correct type at compile time
enum RandGenerators{
   CURAND_XOR_WOW,
};

#endif
