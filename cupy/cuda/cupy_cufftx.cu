#include "cupy_cufft.h"
#include <cstdint>


/// Device-globals to keep function pointers
/// These need to be set on the device, copied to host,
/// and then passed to the cuFFT plan.

__device__ cufftCallbackLoadC d_loadCallbackCPtr; 
//__device__ cufftCallbackLoadZ d_loadCallbackZPtr; 
//__device__ cufftCallbackLoadR d_loadCallbackRPtr; 
//__device__ cufftCallbackLoadD d_loadCallbackDPtr; 
__device__ cufftCallbackStoreC d_storeCallbackCPtr;
//__device__ cufftCallbackStoreZ d_storeCallbackZPtr;
//__device__ cufftCallbackStoreR d_storeCallbackRPtr;
//__device__ cufftCallbackStoreD d_storeCallbackDPtr;


__global__ void setLoadCallbackCPtr(intptr_t dev_ptr) {
    d_loadCallbackCPtr = (cufftCallbackLoadC)dev_ptr;
}

__global__ void setStoreCallbackCPtr(intptr_t dev_ptr) {
    d_storeCallbackCPtr = (cufftCallbackStoreC)dev_ptr;
}


//cufftResult inline setCallback(cufftHandle plan, void **callbackRoutine, cufftXtCallbackType type, void **callerInfo) {
//    return cufftXtSetCallback(plan, callbackRoutine, type, callerInfo);
//}
