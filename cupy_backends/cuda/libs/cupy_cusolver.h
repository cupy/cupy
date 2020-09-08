// This file is a stub header file of cudnn for Read the Docs.

#ifndef INCLUDE_GUARD_CUPY_CUSOLVER_H
#define INCLUDE_GUARD_CUPY_CUSOLVER_H

#if !defined(CUPY_NO_CUDA) && !defined(CUPY_USE_HIP)

#include <cuda.h>
#include <cusolverDn.h>
#include <cusolverSp.h>

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

#elif defined(CUPY_USE_HIP) // #if !defined(CUPY_NO_CUDA) && !defined(CUPY_USE_HIP)

//#include "../cupy_hip.h"
#include "../cupy_hip_common.h"
#include <rocsolver.h>


extern "C" {

typedef rocblas_status cusolverStatus_t;
typedef rocblas_handle cusolverDnHandle_t;


typedef enum{} cusolverEigType_t;
typedef enum{} cusolverEigMode_t;
typedef void* cusolverSpHandle_t;
typedef void* cusparseMatDescr_t;
typedef void* gesvdjInfo_t;
typedef void* syevjInfo_t;


cusolverStatus_t cusolverDnCreate(cusolverDnHandle_t *handle) {
    return rocblas_create_handle(handle);
}

cusolverStatus_t cusolverDnDestroy(cusolverDnHandle_t handle) {
    return rocblas_destroy_handle(handle);
}

cusolverStatus_t cusolverDnGetStream(cusolverDnHandle_t handle,
                                     cudaStream_t *streamId) {
    return rocblas_get_stream(handle, streamId);
}

cusolverStatus_t cusolverSpGetStream(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnSetStream (cusolverDnHandle_t handle,
                                      cudaStream_t streamId) {
    return rocblas_set_stream(handle, streamId);
}

cusolverStatus_t cusolverSpSetStream(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverGetProperty(...) {
    return rocblas_status_success;
}

static rocblas_fill convert_rocblas_fill(cublasFillMode_t mode) {
    switch(static_cast<int>(mode)) {
        case 0 /* CUBLAS_FILL_MODE_LOWER */: return rocblas_fill_lower;
        case 1 /* CUBLAS_FILL_MODE_UPPER */: return rocblas_fill_upper;
        default: throw std::runtime_error("unrecognized mode");
    }
}

static rocblas_operation convert_rocblas_operation(cublasOperation_t op) {
    return static_cast<rocblas_operation>(static_cast<int>(op) + 111);
}


cusolverStatus_t cusolverDnSpotrf_bufferSize(...) {
    // this needs to return 0 because rocSolver does not rely on it
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnDpotrf_bufferSize(...) {
    // this needs to return 0 because rocSolver does not rely on it
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnCpotrf_bufferSize(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnZpotrf_bufferSize(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnSpotrf(cusolverDnHandle_t handle,
                                  cublasFillMode_t uplo,
                                  int n,
                                  float *A,
                                  int lda,
                                  float *Workspace,
                                  int Lwork,
                                  int *devInfo) {
    // ignore Workspace and Lwork as rocSOLVER does not need them
    return rocsolver_spotrf(handle, convert_rocblas_fill(uplo),
                            n, A, lda, devInfo);
}

cusolverStatus_t cusolverDnDpotrf(cusolverDnHandle_t handle,
                                  cublasFillMode_t uplo,
                                  int n,
                                  double *A,
                                  int lda,
                                  double *Workspace,
                                  int Lwork,
                                  int *devInfo ) {
    // ignore Workspace and Lwork as rocSOLVER does not need them
    return rocsolver_dpotrf(handle, convert_rocblas_fill(uplo),
                            n, A, lda, devInfo);
}

cusolverStatus_t cusolverDnCpotrf(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnZpotrf(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnSpotrfBatched(cusolverDnHandle_t handle,
                                         cublasFillMode_t uplo,
                                         int n,
                                         float *Aarray[],
                                         int lda,
                                         int *infoArray,
                                         int batchSize) {
    return rocsolver_spotrf_batched(handle, convert_rocblas_fill(uplo),
                                    n, Aarray, lda, infoArray, batchSize);
}

cusolverStatus_t cusolverDnDpotrfBatched(cusolverDnHandle_t handle,
                                         cublasFillMode_t uplo,
                                         int n,
                                         double *Aarray[],
                                         int lda,
                                         int *infoArray,
                                         int batchSize) {
    return rocsolver_dpotrf_batched(handle, convert_rocblas_fill(uplo),
                                    n, Aarray, lda, infoArray, batchSize);
}

cusolverStatus_t cusolverDnCpotrfBatched(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnZpotrfBatched(...) {
    return rocblas_status_not_implemented;
}


cusolverStatus_t cusolverDnSpotrs(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnDpotrs(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnCpotrs(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnZpotrs(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnSpotrsBatched(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnDpotrsBatched(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnCpotrsBatched(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnZpotrsBatched(...) {
    return rocblas_status_not_implemented;
}


cusolverStatus_t cusolverDnSgetrf_bufferSize(...) {
    // this needs to return 0 because rocSolver does not rely on it
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnDgetrf_bufferSize(...) {
    // this needs to return 0 because rocSolver does not rely on it
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnCgetrf_bufferSize(...) {
    // this needs to return 0 because rocSolver does not rely on it
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnZgetrf_bufferSize(...) {
    // this needs to return 0 because rocSolver does not rely on it
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnSgetrf(cusolverDnHandle_t handle,
                                  int m,
                                  int n,
                                  float *A,
                                  int lda,
                                  float *Workspace,
                                  int *devIpiv,
                                  int *devInfo) {
    // ignore Workspace as rocSOLVER does not need it
    return rocsolver_sgetrf(handle, m, n, A, lda, devIpiv, devInfo);
}

cusolverStatus_t cusolverDnDgetrf(cusolverDnHandle_t handle,
                                  int m,
                                  int n,
                                  double *A,
                                  int lda,
                                  double *Workspace,
                                  int *devIpiv,
                                  int *devInfo) {
    // ignore Workspace as rocSOLVER does not need it
    return rocsolver_dgetrf(handle, m, n, A, lda, devIpiv, devInfo);
}

cusolverStatus_t cusolverDnCgetrf(cusolverDnHandle_t handle,
                                  int m,
                                  int n,
                                  cuComplex *A,
                                  int lda,
                                  cuComplex *Workspace,
                                  int *devIpiv,
                                  int *devInfo) {
    // ignore Workspace as rocSOLVER does not need it
    return rocsolver_cgetrf(handle, m, n,
                            reinterpret_cast<rocblas_float_complex*>(A), lda,
                            devIpiv, devInfo);
}

cusolverStatus_t cusolverDnZgetrf(cusolverDnHandle_t handle,
                                  int m,
                                  int n,
                                  cuDoubleComplex *A,
                                  int lda,
                                  cuDoubleComplex *Workspace,
                                  int *devIpiv,
                                  int *devInfo) {
    // ignore Workspace as rocSOLVER does not need it
    return rocsolver_zgetrf(handle, m, n,
                            reinterpret_cast<rocblas_double_complex*>(A), lda,
                            devIpiv, devInfo);
}


cusolverStatus_t cusolverDnSgetrs(cusolverDnHandle_t handle,
                                  cublasOperation_t trans,
                                  int n,
                                  int nrhs,
                                  const float *A,
                                  int lda,
                                  const int *devIpiv,
                                  float *B,
                                  int ldb,
                                  int *devInfo) {
    // ignore devInfo as rocSOLVER does not need it
    return rocsolver_sgetrs(handle,
                            convert_rocblas_operation(trans),
                            n, nrhs, const_cast<float*>(A), lda, devIpiv, B, ldb);
}

cusolverStatus_t cusolverDnDgetrs(cusolverDnHandle_t handle,
                                  cublasOperation_t trans,
                                  int n,
                                  int nrhs,
                                  const double *A,
                                  int lda,
                                  const int *devIpiv,
                                  double *B,
                                  int ldb,
                                  int *devInfo) {
    // ignore devInfo as rocSOLVER does not need it
    return rocsolver_dgetrs(handle,
                            convert_rocblas_operation(trans),
                            n, nrhs, const_cast<double*>(A), lda, devIpiv, B, ldb);
}

cusolverStatus_t cusolverDnCgetrs(cusolverDnHandle_t handle,
                                  cublasOperation_t trans,
                                  int n,
                                  int nrhs,
                                  const cuComplex *A,
                                  int lda,
                                  const int *devIpiv,
                                  cuComplex *B,
                                  int ldb,
                                  int *devInfo) {
    // ignore devInfo as rocSOLVER does not need it
    return rocsolver_cgetrs(handle,
                            convert_rocblas_operation(trans),
                            n, nrhs,
                            (rocblas_float_complex*)(A), lda,
                            devIpiv,
                            reinterpret_cast<rocblas_float_complex*>(B), ldb);
}

cusolverStatus_t cusolverDnZgetrs(cusolverDnHandle_t handle,
                                  cublasOperation_t trans,
                                  int n,
                                  int nrhs,
                                  const cuDoubleComplex *A,
                                  int lda,
                                  const int *devIpiv,
                                  cuDoubleComplex *B,
                                  int ldb,
                                  int *devInfo) {
    // ignore devInfo as rocSOLVER does not need it
    return rocsolver_zgetrs(handle,
                            convert_rocblas_operation(trans),
                            n, nrhs,
                            (rocblas_double_complex*)(A), lda,
                            devIpiv,
                            reinterpret_cast<rocblas_double_complex*>(B), ldb);
}


cusolverStatus_t cusolverDnSgeqrf_bufferSize(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnDgeqrf_bufferSize(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnCgeqrf_bufferSize(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnZgeqrf_bufferSize(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnSgeqrf(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnDgeqrf(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnCgeqrf(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnZgeqrf(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnSorgqr_bufferSize(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnDorgqr_bufferSize(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnCungqr_bufferSize(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnZungqr_bufferSize(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnSorgqr(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnDorgqr(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnCungqr(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnZungqr(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnSormqr_bufferSize(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnDormqr_bufferSize(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnCunmqr_bufferSize(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnZunmqr_bufferSize(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnSormqr(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnDormqr(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnCunmqr(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnZunmqr(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnSsytrf_bufferSize(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnDsytrf_bufferSize(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnCsytrf_bufferSize(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnZsytrf_bufferSize(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnSsytrf(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnDsytrf(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnCsytrf(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnZsytrf(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnSgebrd_bufferSize(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnDgebrd_bufferSize(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnCgebrd_bufferSize(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnZgebrd_bufferSize(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnSgebrd(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnDgebrd(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnCgebrd(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnZgebrd(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnSgesvd_bufferSize(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnDgesvd_bufferSize(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnCgesvd_bufferSize(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnZgesvd_bufferSize(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnSgesvd(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnDgesvd(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnCgesvd(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnZgesvd(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnCreateGesvdjInfo(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnDestroyGesvdjInfo(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnXgesvdjSetTolerance(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnXgesvdjSetMaxSweeps(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnXgesvdjSetSortEig(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnXgesvdjGetResidual(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnXgesvdjGetSweeps(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnSgesvdj_bufferSize(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnDgesvdj_bufferSize(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnCgesvdj_bufferSize(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnZgesvdj_bufferSize(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnSgesvdj(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnDgesvdj(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnCgesvdj(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnZgesvdj(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnSgesvdjBatched_bufferSize(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnDgesvdjBatched_bufferSize(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnCgesvdjBatched_bufferSize(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnZgesvdjBatched_bufferSize(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnSgesvdjBatched(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnDgesvdjBatched(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnCgesvdjBatched(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnZgesvdjBatched(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnSgesvdaStridedBatched_bufferSize(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnDgesvdaStridedBatched_bufferSize(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnCgesvdaStridedBatched_bufferSize(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnZgesvdaStridedBatched_bufferSize(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnSgesvdaStridedBatched(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnDgesvdaStridedBatched(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnCgesvdaStridedBatched(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnZgesvdaStridedBatched(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnSsyevd_bufferSize(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnDsyevd_bufferSize(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnCheevd_bufferSize(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnZheevd_bufferSize(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnSsyevd(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnDsyevd(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnCheevd(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnZheevd(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnCreateSyevjInfo(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnDestroySyevjInfo(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnXsyevjSetTolerance(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnXsyevjSetMaxSweeps(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnXsyevjSetSortEig(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnXsyevjGetResidual(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnXsyevjGetSweeps(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnSsyevj_bufferSize(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnDsyevj_bufferSize(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnCheevj_bufferSize(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnZheevj_bufferSize(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnSsyevj(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnDsyevj(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnCheevj(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnZheevj(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnSsyevjBatched_bufferSize(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnDsyevjBatched_bufferSize(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnCheevjBatched_bufferSize(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnZheevjBatched_bufferSize(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnSsyevjBatched(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnDsyevjBatched(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnCheevjBatched(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnZheevjBatched(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverSpCreate(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverSpDestroy(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverSpScsrlsvqr(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverSpDcsrlsvqr(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverSpCcsrlsvqr(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverSpZcsrlsvqr(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverSpScsrlsvchol(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverSpDcsrlsvchol(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverSpCcsrlsvchol(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverSpZcsrlsvchol(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverSpScsreigvsi(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverSpDcsreigvsi(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverSpCcsreigvsi(...) {
    return rocblas_status_success;
}

cusolverStatus_t cusolverSpZcsreigvsi(...) {
    return rocblas_status_success;
}


} // extern "C"

#else // #if !defined(CUPY_NO_CUDA) && !defined(CUPY_USE_HIP)

#include "../cupy_cuda_common.h"

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
