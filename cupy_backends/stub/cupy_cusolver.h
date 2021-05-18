// This file is a stub header file of cudnn for Read the Docs.

#ifndef INCLUDE_GUARD_STUB_CUPY_CUSOLVER_H
#define INCLUDE_GUARD_STUB_CUPY_CUSOLVER_H

#include "cupy_cuda_common.h"

extern "C" {

typedef enum {
    CUSOLVER_STATUS_SUCCESS = 0,
} cusolverStatus_t;

typedef enum{} cusolverEigType_t;
typedef enum{} cusolverEigMode_t;

typedef void* cusolverDnHandle_t;
typedef void* cusolverSpHandle_t;
typedef void* cusparseMatDescr_t;
typedef void* gesvdjInfo_t;
typedef void* syevjInfo_t;

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

// The function signatures are explicitly spelled out because we need to
// fetch the function pointers.
cusolverStatus_t cusolverDnSgesvd(cusolverDnHandle_t handle,
                                  signed char jobu,
                                  signed char jobvt,
                                  int m,
                                  int n,
                                  float *A,
                                  int lda,
                                  float *S,
                                  float *U,
                                  int ldu,
                                  float *VT,
                                  int ldvt,
                                  float *work,
                                  int lwork,
                                  float *rwork,
                                  int *info) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDgesvd(cusolverDnHandle_t handle,
                                  signed char jobu,
                                  signed char jobvt,
                                  int m,
                                  int n,
                                  double *A,
                                  int lda,
                                  double *S,
                                  double *U,
                                  int ldu,
                                  double *VT,
                                  int ldvt,
                                  double *work,
                                  int lwork,
                                  double *rwork,
                                  int *info) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnCgesvd(cusolverDnHandle_t handle,
                                  signed char jobu,
                                  signed char jobvt,
                                  int m,
                                  int n,
                                  cuComplex *A,
                                  int lda,
                                  float *S,
                                  cuComplex *U,
                                  int ldu,
                                  cuComplex *VT,
                                  int ldvt,
                                  cuComplex *work,
                                  int lwork,
                                  float *rwork,
                                  int *info) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnZgesvd(cusolverDnHandle_t handle,
                                  signed char jobu,
                                  signed char jobvt,
                                  int m,
                                  int n,
                                  cuDoubleComplex *A,
                                  int lda,
                                  double *S,
                                  cuDoubleComplex *U,
                                  int ldu,
                                  cuDoubleComplex *VT,
                                  int ldvt,
                                  cuDoubleComplex *work,
                                  int lwork,
                                  double *rwork,
                                  int *info) {
    return CUSOLVER_STATUS_SUCCESS;
}

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

cusolverStatus_t cusolverDnCreateSyevjInfo(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDestroySyevjInfo(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnXsyevjSetTolerance(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnXsyevjSetMaxSweeps(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnXsyevjSetSortEig(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnXsyevjGetResidual(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnXsyevjGetSweeps(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSsyevj_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDsyevj_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnCheevj_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnZheevj_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSsyevj(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDsyevj(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnCheevj(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnZheevj(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSsyevjBatched_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDsyevjBatched_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnCheevjBatched_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnZheevjBatched_bufferSize(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSsyevjBatched(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDsyevjBatched(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnCheevjBatched(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnZheevjBatched(...) {
    return CUSOLVER_STATUS_SUCCESS;
}

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

#endif // #ifndef INCLUDE_GUARD_STUB_CUPY_CUSOLVER_H
