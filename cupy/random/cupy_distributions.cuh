#ifndef _CUPY_TEST_H
#define _CUPY_TEST_H
#include <cstdint>
#include <curand_kernel.h>

void init_xor_generator(intptr_t state_ptr, uint64_t seed, ssize_t size);
void interval_32(intptr_t state, int mx, int mask, intptr_t out, ssize_t size);
void interval_64(intptr_t state, uint64_t mx, uint64_t mask, intptr_t out, ssize_t size);
void beta(intptr_t state, double a, double b, intptr_t out, ssize_t size);
void standard_exponential(intptr_t state, intptr_t out, ssize_t size);
#endif
