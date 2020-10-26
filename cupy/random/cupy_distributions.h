#ifndef _CUPY_TEST_H
#define _CUPY_TEST_H
#include <cstdint>

void interval_32(intptr_t param, int mx, int mask, void* out, ssize_t size);
void interval_64(intptr_t param, uint64_t mx, uint64_t mask, void* out, ssize_t size);
void beta(intptr_t  param, double a, double b, void* out, ssize_t size);
void standard_exponential(intptr_t param, void* out, ssize_t size);
#endif
