#include "cupy_cufft.h"


cufftResult inline setCallback(cufftHandle plan, void **callbackRoutine, cufftXtCallbackType type, void **callerInfo) {
    return cufftXtSetCallback(plan, callbackRoutine, type, callerInfo);
}
