#ifndef INCLUDE_GUARD_CUPY_HIP_WORKAROUND_H
#define INCLUDE_GUARD_CUPY_HIP_WORKAROUND_H

#ifdef __HIP_DEVICE_COMPILE__

// Determine whether native __shfl_*_sync functions are available.
// (defined in /opt/rocm/include/hip/amd_detail/amd_warp_sync_functions.h)
//
// The native builtins require a 64-bit mask and were introduced in ROCm 6.2
// behind an opt-in macro:
//   ROCm < 6.2   — not available at all
//   ROCm 6.2–6.4 — available only if HIP_ENABLE_WARP_SYNC_BUILTINS is defined
//   ROCm 7.0+    — available by default, disabled if HIP_DISABLE_WARP_SYNC_BUILTINS is defined
//
// When the native builtins are not available, we define compatibility macros
// that rewrite __shfl_*_sync(mask, ...) to __shfl_*(...), stripping the mask.
// This is safe because HIP wavefronts execute in lock-step.
#if !defined(HIP_VERSION) || HIP_VERSION < 60200000
  // ROCm < 6.2: no native builtins
  #define CUPY_HIP_SHFL_WORKAROUND
#elif HIP_VERSION < 70000000
  // ROCm 6.2–6.4: native builtins only if user opted in
  #if !defined(HIP_ENABLE_WARP_SYNC_BUILTINS)
    #define CUPY_HIP_SHFL_WORKAROUND
  #endif
#else
  // ROCm 7.0+: native builtins unless user opted out
  #if defined(HIP_DISABLE_WARP_SYNC_BUILTINS)
    #define CUPY_HIP_SHFL_WORKAROUND
  #endif
#endif

#ifdef CUPY_HIP_SHFL_WORKAROUND
  // ignore mask
  #define __shfl_sync(mask, ...) __shfl(__VA_ARGS__)
  #define __shfl_up_sync(mask, ...) __shfl_up(__VA_ARGS__)
  #define __shfl_down_sync(mask, ...) __shfl_down(__VA_ARGS__)
  #define __shfl_xor_sync(mask, ...) __shfl_xor(__VA_ARGS__)
  #undef CUPY_HIP_SHFL_WORKAROUND
#endif

// In ROCm, threads in a warp march in lock-step, so we don't need to
// synchronize the threads. But it doesn't guarantee the memory order,
// which still make us use memory fences.
// https://rocmdocs.amd.com/en/latest/Programming_Guides/Kernel_language.html#warp-cross-lane-functions
#define __syncwarp() { __threadfence_block(); }

#if defined(__HIPCC_RTC__) && defined(_MSC_VER)
// hiprtc on Windows compiles against the MSVC STL, which needs two things
// that hiprtc does not provide on the device.
//
// <complex> calls std::signbit(long double), but hiprtc only declares the
// float and double overloads. long double is double on the device.
static __device__ inline bool signbit(long double __x) {
  return ::__signbit(static_cast<double>(__x));
}
namespace std { using ::signbit; }

// <complex> computes exp, cosh and sinh through the C runtime helpers _Exp,
// _Cosh and _Sinh (and their float forms), which have no device definition.
// These return y * exp(x) * 2^eoff, y * cosh(x) and y * sinh(x) without
// overflowing early, and follow the C runtime's handling of special values.
__device__ short __cupy_msvc_Exp(double*, double, short) __asm__("_Exp");
__device__ __attribute__((weak)) short __cupy_msvc_Exp(double* px, double y, short eoff) {
  if (y == 0.0) {
    *px = y;
    return 0;
  }
  double x = *px;
  int n = 0;
  if (::fabs(x) > 700.0) {
    // Split off a power of two so that exp(x) cannot overflow on its own.
    n = static_cast<int>(::rint(::fmin(::fmax(x, -4000.0), 4000.0) *
                                1.4426950408889634));
    x = ::fma(-n, 6.93147180369123816490e-01, x);
    x = ::fma(-n, 1.90821492927058770002e-10, x);
  }
  double r = ::ldexp(y * ::exp(x), n + eoff);
  *px = r;
  return r == 0.0 ? 0 : __builtin_isinf(r) ? 1 : -1;
}

__device__ short __cupy_msvc_FExp(float*, float, short) __asm__("_FExp");
__device__ __attribute__((weak)) short __cupy_msvc_FExp(float* px, float y, short eoff) {
  double x = *px;
  __cupy_msvc_Exp(&x, y, eoff);
  *px = static_cast<float>(x);
  return *px == 0.0f ? 0 : __builtin_isinf(*px) ? 1 : -1;
}

__device__ double __cupy_msvc_Cosh(double, double) __asm__("_Cosh");
__device__ __attribute__((weak)) double __cupy_msvc_Cosh(double x, double y) {
  if (__builtin_isnan(x) || __builtin_isinf(x)) {
    return x;
  }
  if (x == 0.0 || y == 0.0) {
    return y;
  }
  x = ::fabs(x);
  if (x < 22.0) {
    return y * ::cosh(x);
  }
  __cupy_msvc_Exp(&x, y, -1);
  return x;
}

__device__ float __cupy_msvc_FCosh(float, float) __asm__("_FCosh");
__device__ __attribute__((weak)) float __cupy_msvc_FCosh(float x, float y) {
  return static_cast<float>(__cupy_msvc_Cosh(x, y));
}

__device__ double __cupy_msvc_Sinh(double, double) __asm__("_Sinh");
__device__ __attribute__((weak)) double __cupy_msvc_Sinh(double x, double y) {
  if (__builtin_isnan(x)) {
    return x;
  }
  if (__builtin_isinf(x)) {
    return y != 0.0 ? x : (__builtin_signbit(x) ? -y : y);
  }
  if (x == 0.0) {
    return x * y;
  }
  if (y == 0.0) {
    return x < 0.0 ? -y : y;
  }
  bool neg = x < 0.0;
  x = ::fabs(x);
  if (x < 22.0) {
    x = y * ::sinh(x);
  } else {
    __cupy_msvc_Exp(&x, y, -1);
  }
  return neg ? -x : x;
}

__device__ float __cupy_msvc_FSinh(float, float) __asm__("_FSinh");
__device__ __attribute__((weak)) float __cupy_msvc_FSinh(float x, float y) {
  return static_cast<float>(__cupy_msvc_Sinh(x, y));
}
#endif

#endif  // __HIP_DEVICE_COMPILE__

#endif  // INCLUDE_GUARD_CUPY_HIP_WORKAROUND_H
