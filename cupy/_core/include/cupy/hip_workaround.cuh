#ifndef INCLUDE_GUARD_CUPY_HIP_WORKAROUND_H
#define INCLUDE_GUARD_CUPY_HIP_WORKAROUND_H

#ifdef __HIP_DEVICE_COMPILE__

// ignore mask
#define __shfl_sync(m, x, y, z) __shfl(x, y, z)
#define __shfl_up_sync(m, x, y, z) __shfl_up(x, y, z)
#define __shfl_down_sync(m, x, y, z) __shfl_down(x, y, z)
#define __shfl_xor_sync(m, x, y, z) __shfl_xor(x, y, z)

// It is guaranteed to be safe on AMD's hardware, see
// https://rocmdocs.amd.com/en/latest/Programming_Guides/HIP-GUIDE.html#warp-cross-lane-functions
#define __syncwarp() {}

#endif  // __HIP_DEVICE_COMPILE__

#endif  // INCLUDE_GUARD_CUPY_HIP_WORKAROUND_H
