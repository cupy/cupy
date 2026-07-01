// SPDX-License-Identifier: BSD-3-Clause
// HIP-only bridge: provide thrust::complex overloads for the xsf functions
// CuPy ufuncs call with complex args (xsf accepts std::complex; CuPy passes
// thrust::complex). Re-includes the source xsf headers so the std::complex
// overloads are guaranteed visible at overload-resolution time.

#pragma once

#if defined(__HIPCC__) || defined(__HIP__) || defined(__HIPCC_RTC__)

#include <complex>
#include <cupy/xsf/digamma.h>
#include <cupy/xsf/sici.h>
#include <cupy/xsf/lambertw.h>

namespace xsf {

// digamma

inline __host__ __device__ thrust::complex<float>
digamma(thrust::complex<float> z) {
    std::complex<float> sz(z.real(), z.imag());
    std::complex<float> r = ::xsf::digamma(sz);
    return thrust::complex<float>(r.real(), r.imag());
}

inline __host__ __device__ thrust::complex<double>
digamma(thrust::complex<double> z) {
    std::complex<double> sz(z.real(), z.imag());
    std::complex<double> r = ::xsf::digamma(sz);
    return thrust::complex<double>(r.real(), r.imag());
}

// sici / shichi (return int status; results via reference)

inline __host__ __device__ int
sici(thrust::complex<float> z, thrust::complex<float>& si,
     thrust::complex<float>& ci) {
    std::complex<float> sz(z.real(), z.imag());
    std::complex<float> ssi, sci;
    int rc = ::xsf::sici(sz, ssi, sci);
    si = thrust::complex<float>(ssi.real(), ssi.imag());
    ci = thrust::complex<float>(sci.real(), sci.imag());
    return rc;
}

inline __host__ __device__ int
sici(thrust::complex<double> z, thrust::complex<double>& si,
     thrust::complex<double>& ci) {
    std::complex<double> sz(z.real(), z.imag());
    std::complex<double> ssi, sci;
    int rc = ::xsf::sici(sz, ssi, sci);
    si = thrust::complex<double>(ssi.real(), ssi.imag());
    ci = thrust::complex<double>(sci.real(), sci.imag());
    return rc;
}

inline __host__ __device__ int
shichi(thrust::complex<float> z, thrust::complex<float>& shi,
       thrust::complex<float>& chi) {
    std::complex<float> sz(z.real(), z.imag());
    std::complex<float> sshi, schi;
    int rc = ::xsf::shichi(sz, sshi, schi);
    shi = thrust::complex<float>(sshi.real(), sshi.imag());
    chi = thrust::complex<float>(schi.real(), schi.imag());
    return rc;
}

inline __host__ __device__ int
shichi(thrust::complex<double> z, thrust::complex<double>& shi,
       thrust::complex<double>& chi) {
    std::complex<double> sz(z.real(), z.imag());
    std::complex<double> sshi, schi;
    int rc = ::xsf::shichi(sz, sshi, schi);
    shi = thrust::complex<double>(sshi.real(), sshi.imag());
    chi = thrust::complex<double>(schi.real(), schi.imag());
    return rc;
}

// lambertw

inline __host__ __device__ thrust::complex<double>
lambertw(thrust::complex<double> z, long k, double tol) {
    std::complex<double> sz(z.real(), z.imag());
    std::complex<double> r = ::xsf::lambertw(sz, k, tol);
    return thrust::complex<double>(r.real(), r.imag());
}

inline __host__ __device__ thrust::complex<float>
lambertw(thrust::complex<float> z, long k, float tol) {
    std::complex<float> sz(z.real(), z.imag());
    std::complex<float> r = ::xsf::lambertw(sz, k, tol);
    return thrust::complex<float>(r.real(), r.imag());
}

}  // namespace xsf

// libgcc compiler-rt shims for C99 _Complex arithmetic on AMDGPU.
// libstdc++'s std::complex<T> mul/div lowers to __muldc3/__divdc3 etc.
// from libgcc; AMDGPU has no such runtime. Inline implementations
// follow libgcc/libgcc2.c (textbook formulas + inf/NaN recovery per
// C99 -- needed for e.g. digamma(inf+0j) to return inf+0j instead of
// NaN+NaNj).

#include <cmath>

#ifndef INFINITY
#define INFINITY (__builtin_inff())
#endif

extern "C" {

__device__ inline __attribute__((always_inline))
double _Complex __muldc3(double a, double b, double c, double d) {
    double ac = a * c, bd = b * d, ad = a * d, bc = b * c;
    double x = ac - bd, y = ad + bc;
    if (isnan(x) && isnan(y)) {
        bool recalc = false;
        if (isinf(a) || isinf(b)) {
            a = copysign(isinf(a) ? 1.0 : 0.0, a);
            b = copysign(isinf(b) ? 1.0 : 0.0, b);
            if (isnan(c)) c = copysign(0.0, c);
            if (isnan(d)) d = copysign(0.0, d);
            recalc = true;
        }
        if (isinf(c) || isinf(d)) {
            c = copysign(isinf(c) ? 1.0 : 0.0, c);
            d = copysign(isinf(d) ? 1.0 : 0.0, d);
            if (isnan(a)) a = copysign(0.0, a);
            if (isnan(b)) b = copysign(0.0, b);
            recalc = true;
        }
        if (!recalc && (isinf(ac) || isinf(bd) || isinf(ad) || isinf(bc))) {
            if (isnan(a)) a = copysign(0.0, a);
            if (isnan(b)) b = copysign(0.0, b);
            if (isnan(c)) c = copysign(0.0, c);
            if (isnan(d)) d = copysign(0.0, d);
            recalc = true;
        }
        if (recalc) {
            x = INFINITY * (a * c - b * d);
            y = INFINITY * (a * d + b * c);
        }
    }
    return __builtin_complex(x, y);
}

__device__ inline __attribute__((always_inline))
float _Complex __mulsc3(float a, float b, float c, float d) {
    float ac = a * c, bd = b * d, ad = a * d, bc = b * c;
    float x = ac - bd, y = ad + bc;
    if (isnan(x) && isnan(y)) {
        bool recalc = false;
        if (isinf(a) || isinf(b)) {
            a = copysignf(isinf(a) ? 1.0f : 0.0f, a);
            b = copysignf(isinf(b) ? 1.0f : 0.0f, b);
            if (isnan(c)) c = copysignf(0.0f, c);
            if (isnan(d)) d = copysignf(0.0f, d);
            recalc = true;
        }
        if (isinf(c) || isinf(d)) {
            c = copysignf(isinf(c) ? 1.0f : 0.0f, c);
            d = copysignf(isinf(d) ? 1.0f : 0.0f, d);
            if (isnan(a)) a = copysignf(0.0f, a);
            if (isnan(b)) b = copysignf(0.0f, b);
            recalc = true;
        }
        if (!recalc && (isinf(ac) || isinf(bd) || isinf(ad) || isinf(bc))) {
            if (isnan(a)) a = copysignf(0.0f, a);
            if (isnan(b)) b = copysignf(0.0f, b);
            if (isnan(c)) c = copysignf(0.0f, c);
            if (isnan(d)) d = copysignf(0.0f, d);
            recalc = true;
        }
        if (recalc) {
            x = ((float)INFINITY) * (a * c - b * d);
            y = ((float)INFINITY) * (a * d + b * c);
        }
    }
    return __builtin_complex(x, y);
}

__device__ inline __attribute__((always_inline))
double _Complex __divdc3(double a, double b, double c, double d) {
    double denom, ratio, x, y;
    if (fabs(c) < fabs(d)) {
        ratio = c / d;  denom = (c * ratio) + d;
        x = ((a * ratio) + b) / denom;
        y = ((b * ratio) - a) / denom;
    } else {
        ratio = d / c;  denom = (d * ratio) + c;
        x = ((b * ratio) + a) / denom;
        y = (b - (a * ratio)) / denom;
    }
    if (isnan(x) && isnan(y)) {
        if (c == 0.0 && d == 0.0 && (!isnan(a) || !isnan(b))) {
            x = copysign(INFINITY, c) * a;
            y = copysign(INFINITY, c) * b;
        } else if ((isinf(a) || isinf(b)) && isfinite(c) && isfinite(d)) {
            a = copysign(isinf(a) ? 1.0 : 0.0, a);
            b = copysign(isinf(b) ? 1.0 : 0.0, b);
            x = INFINITY * (a * c + b * d);
            y = INFINITY * (b * c - a * d);
        } else if ((isinf(c) || isinf(d)) && isfinite(a) && isfinite(b)) {
            c = copysign(isinf(c) ? 1.0 : 0.0, c);
            d = copysign(isinf(d) ? 1.0 : 0.0, d);
            x = 0.0 * (a * c + b * d);
            y = 0.0 * (b * c - a * d);
        }
    }
    return __builtin_complex(x, y);
}

__device__ inline __attribute__((always_inline))
float _Complex __divsc3(float a, float b, float c, float d) {
    float denom, ratio, x, y;
    if (fabsf(c) < fabsf(d)) {
        ratio = c / d;  denom = (c * ratio) + d;
        x = ((a * ratio) + b) / denom;
        y = ((b * ratio) - a) / denom;
    } else {
        ratio = d / c;  denom = (d * ratio) + c;
        x = ((b * ratio) + a) / denom;
        y = (b - (a * ratio)) / denom;
    }
    if (isnan(x) && isnan(y)) {
        if (c == 0.0f && d == 0.0f && (!isnan(a) || !isnan(b))) {
            x = copysignf((float)INFINITY, c) * a;
            y = copysignf((float)INFINITY, c) * b;
        } else if ((isinf(a) || isinf(b)) && isfinite(c) && isfinite(d)) {
            a = copysignf(isinf(a) ? 1.0f : 0.0f, a);
            b = copysignf(isinf(b) ? 1.0f : 0.0f, b);
            x = ((float)INFINITY) * (a * c + b * d);
            y = ((float)INFINITY) * (b * c - a * d);
        } else if ((isinf(c) || isinf(d)) && isfinite(a) && isfinite(b)) {
            c = copysignf(isinf(c) ? 1.0f : 0.0f, c);
            d = copysignf(isinf(d) ? 1.0f : 0.0f, d);
            x = 0.0f * (a * c + b * d);
            y = 0.0f * (b * c - a * d);
        }
    }
    return __builtin_complex(x, y);
}

}  // extern "C"

#endif  // __HIPCC__ || __HIP__ || __HIPCC_RTC__
