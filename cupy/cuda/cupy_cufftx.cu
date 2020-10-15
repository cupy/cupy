#include "cupy_cufftx.h"


// this defines d_loadCallbackPtr
${dev_load_callback_ker}

// this defines  d_storeCallbackPtr
${dev_store_callback_ker}

//
///// Device-globals to keep function pointers
///// These need to be set on the device, copied to host,
///// and then passed to the cuFFT plan.
//
//__device__ cufftCallbackLoadC d_loadCallbackCPtr; 
////__device__ cufftCallbackLoadZ d_loadCallbackZPtr; 
////__device__ cufftCallbackLoadR d_loadCallbackRPtr; 
////__device__ cufftCallbackLoadD d_loadCallbackDPtr; 
//__device__ cufftCallbackStoreC d_storeCallbackCPtr;
////__device__ cufftCallbackStoreZ d_storeCallbackZPtr;
////__device__ cufftCallbackStoreR d_storeCallbackRPtr;
////__device__ cufftCallbackStoreD d_storeCallbackDPtr;
//
//
//__global__ void setLoadCallbackCPtr(intptr_t dev_ptr) {
//    d_loadCallbackCPtr = (cufftCallbackLoadC)dev_ptr;
//}
//
//__global__ void setStoreCallbackCPtr(intptr_t dev_ptr) {
//    d_storeCallbackCPtr = (cufftCallbackStoreC)dev_ptr;
//}


cufftResult set_callback(cufftHandle plan, cufftXtCallbackType type, bool cb_load) {
    if (cb_load) {  // for load callback
        switch (type) {
            case CUFFT_CB_LD_COMPLEX: {
                cufftCallbackLoadC h_ptr;
                cudaMemcpyFromSymbol(&h_ptr, d_loadCallbackPtr, sizeof(h_ptr));
                return cufftXtSetCallback(plan, (void**)&h_ptr, type, NULL);
            }
            default: {
                throw std::runtime_error("unrecognized callback");
            }
        }
    } else {  // for store callback
        switch (type) {
            default: {
                throw std::runtime_error("unrecognized callback");
            }
        }
    }
}
