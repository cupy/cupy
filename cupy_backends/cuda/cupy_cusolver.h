#ifndef INCLUDE_GUARD_CUDA_CUPY_CUSOLVER_H
#define INCLUDE_GUARD_CUDA_CUPY_CUSOLVER_H

#include <cuda.h>
#include <cusolverDn.h>
#include <cusolverSp.h>

extern "C" {

#if CUSOLVER_VERSION < 11700
// Functions added in cuSOLVER 11.7 (CUDA 12.6.2)
cusolverStatus_t cusolverDnXgeev_bufferSize(...) {
    return CUSOLVER_STATUS_NOT_SUPPORTED;
}

cusolverStatus_t cusolverDnXgeev(...) {
    return CUSOLVER_STATUS_NOT_SUPPORTED;
}
#endif // #if CUSOLVER_VERSION < 11700

} // extern "C"

#endif // #ifndef INCLUDE_GUARD_CUDA_CUPY_CUSOLVER_H
