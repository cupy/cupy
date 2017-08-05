#pragma once

#include <cupy/complex/complex.h>

using thrust::complex;
using thrust::conj;
using thrust::arg;

using thrust::exp;
using thrust::log;
using thrust::log10;
using thrust::sin;
using thrust::cos;
using thrust::tan;
using thrust::sinh;
using thrust::cosh;
using thrust::tanh;
using thrust::asinh;
using thrust::acosh;
using thrust::atanh;
using thrust::asin;
using thrust::acos;
using thrust::atan;

template<typename T> __device__ bool isnan(complex<T> x) {
    return isnan(x.real()) || isnan(x.imag());
}
template<typename T> __device__ bool isinf(complex<T> x) {
    return isinf(x.real()) || isinf(x.imag());
}
template<typename T> __device__ bool isfinite(complex<T> x) {
    return isfinite(x.real()) && isfinite(x.imag());
}

// ToDo: assignment operator for complex<T> = T2 for T2 all types
