// This file is a stub header file of cudnn for Read the Docs.

#ifndef INCLUDE_GUARD_CUPY_CUSOLVER_H
#define INCLUDE_GUARD_CUPY_CUSOLVER_H

#if !defined(CUPY_NO_CUDA) && !defined(CUPY_USE_HIP)

#include <cusolverDn.h>
#include <cusolverSp.h>

#else // #if !defined(CUPY_NO_CUDA) && !defined(CUPY_USE_HIP)

#ifdef CUPY_USE_HIP
#include "cupy_hip_common.h"
#else // #ifdef CUPY_USE_HIP
#include "cupy_cuda_common.h"
#endif // #ifdef CUPY_USE_HIP

extern "C" {

typedef enum {
    CUSOLVER_STATUS_SUCCESS = 0,
} cusolverStatus_t;

typedef enum{} cusolverEigType_t;
typedef enum{} cusolverEigMode_t;

typedef void* cusolverDnHandle_t;
typedef void* cusolverSpHandle_t;
typedef void* cusparseMatDescr_t;

cusolverStatus_t cusolverDnCreate(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDestroy(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnGetStream(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverSpGetStream(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSetStream(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverSpSetStream(...) {
    return CUSOLVER_STATUS_SUCCESS;
}


cusolverStatus_t cusolverGetProperty(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSpotrf_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDpotrf_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnCpotrf_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnZpotrf_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSpotrf(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDpotrf(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnCpotrf(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnZpotrf(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSpotrs(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDpotrs(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnCpotrs(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnZpotrs(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSgetrf_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDgetrf_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnCgetrf_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnZgetrf_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSgetrf(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDgetrf(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnCgetrf(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnZgetrf(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSgetrs(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDgetrs(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnCgetrs(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnZgetrs(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSgeqrf_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDgeqrf_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnCgeqrf_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnZgeqrf_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSgeqrf(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDgeqrf(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnCgeqrf(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnZgeqrf(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSorgqr_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDorgqr_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnCungqr_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnZungqr_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSorgqr(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDorgqr(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnCungqr(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnZungqr(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSormqr_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDormqr_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnCunmqr_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnZunmqr_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSormqr(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDormqr(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnCunmqr(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnZunmqr(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSsytrf_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDsytrf_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnCsytrf_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnZsytrf_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSsytrf(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDsytrf(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnCsytrf(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnZsytrf(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSgebrd_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDgebrd_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnCgebrd_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnZgebrd_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSgebrd(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDgebrd(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnCgebrd(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnZgebrd(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSgesvd_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDgesvd_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnCgesvd_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnZgesvd_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSgesvd(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDgesvd(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnCgesvd(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnZgesvd(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSsyevd_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDsyevd_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnCheevd_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnZheevd_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSsyevd(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDsyevd(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnCheevd(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnZheevd(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverSpCreate(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverSpDestroy(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverSpScsrlsvqr(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverSpDcsrlsvqr(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverSpCcsrlsvqr(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverSpZcsrlsvqr(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverSpScsrlsvchol(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverSpDcsrlsvchol(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverSpCcsrlsvchol(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverSpZcsrlsvchol(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverSpScsreigvsi(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverSpDcsreigvsi(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverSpCcsreigvsi(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverSpZcsreigvsi(...) {
    return CUSOLVER_STATUS_SUCCESS;
}


} // extern "C"

#endif // #if !defined(CUPY_NO_CUDA) && !defined(CUPY_USE_HIP)
#endif // #ifndef INCLUDE_GUARD_CUPY_CUSOLVER_H
