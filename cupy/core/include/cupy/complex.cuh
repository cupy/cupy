#pragma once

#include <cupy/complex/complex.h>

using thrust::complex;
using thrust::conj;
using thrust::real;
using thrust::imag;
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

template<typename T>
__host__ __device__ bool isnan(complex<T> x) {
    return isnan(x.real()) || isnan(x.imag());
}

template<typename T>
__host__ __device__ bool isinf(complex<T> x) {
    return isinf(x.real()) || isinf(x.imag());
}

template<typename T>
__host__ __device__ bool isfinite(complex<T> x) {
    return isfinite(x.real()) && isfinite(x.imag());
}

template<typename T>
__host__ __device__ complex<T> log1p(complex<T> x) {
    x += 1;
    return log(x);
}

template<typename T>
__host__ __device__ complex<T> log2(complex<T> x) {
    complex<T> y = log(x);
    y /= log(T(2));
    return y;
}

template<typename T>
__host__ __device__ complex<T> expm1(complex<T> x) {
    complex<T> y = exp(x);
    y -= 1;
    return y;
}

template<typename T>
__host__ __device__ complex<T> min(complex<T> x, complex<T> y) {
    if (isnan(x)) {
        return y;
    } else if (isnan(y)) {
        return x;
    } else if (x.real() < y.real()) {
        return x;
    } else if (x.real() > y.real()) {
        return y;
    } else if (x.imag() < y.imag()) {
        return x;
    } else {
        return y;
    }
}

template<typename T>
__host__ __device__ complex<T> max(complex<T> x, complex<T> y) {
    if (isnan(x)) {
        return y;
    } else if (isnan(y)) {
        return x;
    } else if (x.real() < y.real()) {
        return y;
    } else if (x.real() > y.real()) {
        return x;
    } else if (x.imag() < y.imag()) {
        return y;
    } else {
        return x;
    }
}

template<typename T>
__host__ __device__ complex<T> rint(complex<T> x) {
    return complex<T>(rint(x.real()), rint(x.imag()));
}

// ToDo: assignment operator for complex<T> = T2 for T2 all types
