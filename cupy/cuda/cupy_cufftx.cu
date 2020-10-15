#include "cupy_cufftx.h"


// this must define d_loadCallbackPtr
${dev_load_callback_ker}

// this must define d_storeCallbackPtr
${dev_store_callback_ker}

cufftResult set_callback(cufftHandle plan, cufftXtCallbackType type, bool cb_load) {
    if (cb_load) {  // for load callback
        switch (type) {
            case CUFFT_CB_LD_COMPLEX: {
                cufftCallbackLoadC h_ptr;
                cudaMemcpyFromSymbol(&h_ptr, d_loadCallbackPtr, sizeof(h_ptr));
                return cufftXtSetCallback(plan, (void**)&h_ptr, type, NULL);
            }
            case CUFFT_CB_LD_COMPLEX_DOUBLE: {
                cufftCallbackLoadZ h_ptr;
                cudaMemcpyFromSymbol(&h_ptr, d_loadCallbackPtr, sizeof(h_ptr));
                return cufftXtSetCallback(plan, (void**)&h_ptr, type, NULL);
            }
            case CUFFT_CB_LD_REAL: {
                cufftCallbackLoadR h_ptr;
                cudaMemcpyFromSymbol(&h_ptr, d_loadCallbackPtr, sizeof(h_ptr));
                return cufftXtSetCallback(plan, (void**)&h_ptr, type, NULL);
            }
            case CUFFT_CB_LD_REAL_DOUBLE: {
                cufftCallbackLoadD h_ptr;
                cudaMemcpyFromSymbol(&h_ptr, d_loadCallbackPtr, sizeof(h_ptr));
                return cufftXtSetCallback(plan, (void**)&h_ptr, type, NULL);
            }
            default: {
                throw std::runtime_error("unrecognized callback");
            }
        }
    } else {  // for store callback
        switch (type) {
            case CUFFT_CB_ST_COMPLEX: {
                cufftCallbackStoreC h_ptr;
                cudaMemcpyFromSymbol(&h_ptr, d_loadCallbackPtr, sizeof(h_ptr));
                return cufftXtSetCallback(plan, (void**)&h_ptr, type, NULL);
            }
            case CUFFT_CB_ST_COMPLEX_DOUBLE: {
                cufftCallbackStoreZ h_ptr;
                cudaMemcpyFromSymbol(&h_ptr, d_loadCallbackPtr, sizeof(h_ptr));
                return cufftXtSetCallback(plan, (void**)&h_ptr, type, NULL);
            }
            case CUFFT_CB_ST_REAL: {
                cufftCallbackStoreR h_ptr;
                cudaMemcpyFromSymbol(&h_ptr, d_loadCallbackPtr, sizeof(h_ptr));
                return cufftXtSetCallback(plan, (void**)&h_ptr, type, NULL);
            }
            case CUFFT_CB_ST_REAL_DOUBLE: {
                cufftCallbackStoreD h_ptr;
                cudaMemcpyFromSymbol(&h_ptr, d_loadCallbackPtr, sizeof(h_ptr));
                return cufftXtSetCallback(plan, (void**)&h_ptr, type, NULL);
            }
            default: {
                throw std::runtime_error("unrecognized callback");
            }
        }
    }
}
