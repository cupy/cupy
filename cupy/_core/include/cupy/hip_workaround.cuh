#ifndef INCLUDE_GUARD_CUPY_HIP_WORKAROUND_H
#define INCLUDE_GUARD_CUPY_HIP_WORKAROUND_H

#ifdef __HIP_DEVICE_COMPILE__

// ignore mask
#define __shfl_sync(mask, ...) __shfl(__VA_ARGS__)
#define __shfl_up_sync(mask, ...) __shfl_up(__VA_ARGS__)
#define __shfl_down_sync(mask, ...) __shfl_down(__VA_ARGS__)
#define __shfl_xor_sync(mask, ...) __shfl_xor(__VA_ARGS__)

// It is guaranteed to be safe on AMD's hardware, see
// https://rocmdocs.amd.com/en/latest/Programming_Guides/HIP-GUIDE.html#warp-cross-lane-functions
#define __syncwarp() {}

#endif  // __HIP_DEVICE_COMPILE__

#endif  // INCLUDE_GUARD_CUPY_HIP_WORKAROUND_H
