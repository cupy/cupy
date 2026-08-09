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

#endif  // __HIP_DEVICE_COMPILE__

#endif  // INCLUDE_GUARD_CUPY_HIP_WORKAROUND_H
