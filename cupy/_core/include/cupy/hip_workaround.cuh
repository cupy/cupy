#ifndef INCLUDE_GUARD_CUPY_HIP_WORKAROUND_H
#define INCLUDE_GUARD_CUPY_HIP_WORKAROUND_H

#ifdef __HIP_DEVICE_COMPILE__

// ignore mask
#define __shfl_sync(mask, ...) __shfl(__VA_ARGS__)
#define __shfl_up_sync(mask, ...) __shfl_up(__VA_ARGS__)
#define __shfl_down_sync(mask, ...) __shfl_down(__VA_ARGS__)
#define __shfl_xor_sync(mask, ...) __shfl_xor(__VA_ARGS__)

// In ROCm, threads in a warp march in lock-step, so we don't need to
// synchronize the threads. But it doesn't guarantee the memory order,
// which still make us use memory fences.
// https://rocmdocs.amd.com/en/latest/Programming_Guides/Kernel_language.html#warp-cross-lane-functions
#define __syncwarp() { __threadfence_block(); }

#endif  // __HIP_DEVICE_COMPILE__

#endif  // INCLUDE_GUARD_CUPY_HIP_WORKAROUND_H
