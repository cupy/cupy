#ifndef INCLUDE_GUARD_CUPY_CUSOLVER_H
#define INCLUDE_GUARD_CUPY_CUSOLVER_H

#if !defined(CUPY_NO_CUDA) && !defined(CUPY_USE_HIP)

#include <cuda.h>
#include <cusolverDn.h>
#include <cusolverSp.h>

extern "C" {

#if CUDA_VERSION < 9000
// Data types and functions added in CUDA 9.0
typedef void* gesvdjInfo_t;

cusolverStatus_t cusolverDnCreateGesvdjInfo(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDestroyGesvdjInfo(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnXgesvdjSetTolerance(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnXgesvdjSetMaxSweeps(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnXgesvdjSetSortEig(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnXgesvdjGetResidual(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnXgesvdjGetSweeps(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSgesvdj_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDgesvdj_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnCgesvdj_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnZgesvdj_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSgesvdj(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDgesvdj(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnCgesvdj(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnZgesvdj(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSgesvdjBatched_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDgesvdjBatched_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnCgesvdjBatched_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnZgesvdjBatched_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSgesvdjBatched(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDgesvdjBatched(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnCgesvdjBatched(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnZgesvdjBatched(...) {
    return CUSOLVER_STATUS_SUCCESS;
}
#endif // #if CUDA_VERSION < 9000

#if CUDA_VERSION < 9010
// Functions added in CUDA 9.1
cusolverStatus_t cusolverDnSpotrfBatched(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDpotrfBatched(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnCpotrfBatched(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnZpotrfBatched(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSpotrsBatched(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDpotrsBatched(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnCpotrsBatched(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnZpotrsBatched(...) {
    return CUSOLVER_STATUS_SUCCESS;
}
#endif // #if CUDA_VERSION < 9010

#if CUDA_VERSION < 10010
// Functions added in CUDA 10.1
cusolverStatus_t cusolverDnSgesvdaStridedBatched_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDgesvdaStridedBatched_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnCgesvdaStridedBatched_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnZgesvdaStridedBatched_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSgesvdaStridedBatched(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDgesvdaStridedBatched(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnCgesvdaStridedBatched(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnZgesvdaStridedBatched(...) {
    return CUSOLVER_STATUS_SUCCESS;
}
#endif // #if CUDA_VERSION < 10010

#if CUDA_VERSION < 10020
// Functions added in CUDA 10.2
cusolverStatus_t cusolverDnZZgesv_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnZCgesv_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnZKgesv_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnCCgesv_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnCKgesv_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnDDgesv_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnDSgesv_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnDHgesv_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnSSgesv_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnSHgesv_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnZZgesv(...) {
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnZCgesv(...) {
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnZKgesv(...) {
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnCCgesv(...) {
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnCKgesv(...) {
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnDDgesv(...) {
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnDSgesv(...) {
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnDHgesv(...) {
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnSSgesv(...) {
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnSHgesv(...) {
    return CUSOLVER_STATUS_SUCCESS;
}
#endif // #if CUDA_VERSION < 10020

#if CUDA_VERSION < 11000
// Functions added in CUDA 11.0
cusolverStatus_t cusolverDnZYgesv_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnCYgesv_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnDXgesv_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnSXgesv_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnZYgesv(...) {
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnCYgesv(...) {
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnDXgesv(...) {
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnSXgesv(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnZZgels_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnZCgels_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnZYgels_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnZKgels_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnCCgels_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnCYgels_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnCKgels_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnDDgels_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnDSgels_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnDXgels_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnDHgels_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnSSgels_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnSXgels_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnSHgels_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnZZgels(...) {
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnZCgels(...) {
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnZYgels(...) {
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnZKgels(...) {
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnCCgels(...) {
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnCYgels(...) {
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnCKgels(...) {
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnDDgels(...) {
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnDSgels(...) {
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnDXgels(...) {
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnDHgels(...) {
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnSSgels(...) {
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnSXgels(...) {
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnSHgels(...) {
    return CUSOLVER_STATUS_SUCCESS;
}
#endif // #if CUDA_VERSION < 11000

} // extern "C"

#elif defined(CUPY_USE_HIP) // #if !defined(CUPY_NO_CUDA) && !defined(CUPY_USE_HIP)

#include "hip/cupy_hipsolver.h"


#else // #if !defined(CUPY_NO_CUDA) && !defined(CUPY_USE_HIP)

#include "stub/cupy_cusolver.h"

#endif // #if !defined(CUPY_NO_CUDA) && !defined(CUPY_USE_HIP)
#endif // #ifndef INCLUDE_GUARD_CUPY_CUSOLVER_H
