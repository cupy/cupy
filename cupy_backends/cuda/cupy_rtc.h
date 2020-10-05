#ifndef INCLUDE_GUARD_CUPY_RTC_H
#define INCLUDE_GUARD_CUPY_RTC_H

#if CUPY_USE_HIP
#include "hip/cupy_hiprtc.h"
#else
#include "cupy_nvrtc.h"
#endif
#endif

