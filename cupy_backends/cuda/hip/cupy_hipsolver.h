#ifndef INCLUDE_GUARD_HIP_CUPY_HIPSOLVER_H
#define INCLUDE_GUARD_HIP_CUPY_HIPSOLVER_H

#include "cupy_cuda.h"
#include "cupy_hipblas.h"
#include <hipblas.h>

extern "C" {
/* ---------- helpers ---------- */
// TODO(leofang): perhaps these should be merged with the support of hipBLAS?
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

static rocblas_side convert_rocblas_side(cublasSideMode_t mode) {
    return static_cast<rocblas_side>(static_cast<int>(mode) + 141);
}


// rocSOLVER
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

cusolverStatus_t cusolverDnSetStream (cusolverDnHandle_t handle,
                                      cudaStream_t streamId) {
    return rocblas_set_stream(handle, streamId);
}

cusolverStatus_t cusolverGetProperty(libraryPropertyType type, int* val) {
    switch(type) {
        case MAJOR_VERSION: { *val = ROCSOLVER_VERSION_MAJOR; break; }
        case MINOR_VERSION: { *val = ROCSOLVER_VERSION_MINOR; break; }
        case PATCH_LEVEL:   { *val = ROCSOLVER_VERSION_PATCH; break; }
        default: throw std::runtime_error("invalid type");
    }
    return rocblas_status_success;
}


/* ---------- potrf ---------- */
cusolverStatus_t cusolverDnSpotrf_bufferSize(cusolverDnHandle_t handle,
                                             cublasFillMode_t uplo,
                                             int n,
                                             float *A,
                                             int lda,
                                             int *Lwork) {
    // this needs to return 0 because rocSolver does not rely on it
    *Lwork = 0;
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnDpotrf_bufferSize(cusolverDnHandle_t handle,
                                             cublasFillMode_t uplo,
                                             int n,
                                             double *A,
                                             int lda,
                                             int *Lwork) {
    // this needs to return 0 because rocSolver does not rely on it
    *Lwork = 0;
    return rocblas_status_success;
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


/* ---------- getrf ---------- */
cusolverStatus_t cusolverDnSgetrf_bufferSize(cusolverDnHandle_t handle,
                                             int m,
                                             int n,
                                             float *A,
                                             int lda,
                                             int *Lwork) {
    // this needs to return 0 because rocSolver does not rely on it
    *Lwork = 0;
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnDgetrf_bufferSize(cusolverDnHandle_t handle,
                                             int m,
                                             int n,
                                             double *A,
                                             int lda,
                                             int *Lwork) {
    // this needs to return 0 because rocSolver does not rely on it
    *Lwork = 0;
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnCgetrf_bufferSize(cusolverDnHandle_t handle,
                                             int m,
                                             int n,
                                             cuComplex *A,
                                             int lda,
                                             int *Lwork) {
    // this needs to return 0 because rocSolver does not rely on it
    *Lwork = 0;
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnZgetrf_bufferSize(cusolverDnHandle_t handle,
                                             int m,
                                             int n,
                                             cuDoubleComplex *A,
                                             int lda,
                                             int *Lwork) {
    // this needs to return 0 because rocSolver does not rely on it
    *Lwork = 0;
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


/* ---------- getrs ---------- */
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


/* ---------- geqrf ---------- */
cusolverStatus_t cusolverDnSgeqrf_bufferSize(cusolverDnHandle_t handle,
                                             int m,
                                             int n,
                                             float *A,
                                             int lda,
                                             int *lwork) {
    // this needs to return 0 because rocSolver does not rely on it
    *lwork = 0;
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnDgeqrf_bufferSize(cusolverDnHandle_t handle,
                                             int m,
                                             int n,
                                             double *A,
                                             int lda,
                                             int *lwork) {
    // this needs to return 0 because rocSolver does not rely on it
    *lwork = 0;
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnCgeqrf_bufferSize(cusolverDnHandle_t handle,
                                             int m,
                                             int n,
                                             cuComplex *A,
                                             int lda,
                                             int *lwork) {
    // this needs to return 0 because rocSolver does not rely on it
    *lwork = 0;
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnZgeqrf_bufferSize(cusolverDnHandle_t handle,
                                             int m,
                                             int n,
                                             cuDoubleComplex *A,
                                             int lda,
                                             int *lwork) {
    // this needs to return 0 because rocSolver does not rely on it
    *lwork = 0;
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnSgeqrf(cusolverDnHandle_t handle,
                                  int m,
                                  int n,
                                  float *A,
                                  int lda,
                                  float *TAU,
                                  float *Workspace,
                                  int Lwork,
                                  int *devInfo) {
    // ignore Workspace, Lwork and devInfo as rocSOLVER does not need them
    return rocsolver_sgeqrf(handle, m, n, A, lda, TAU);
}

cusolverStatus_t cusolverDnDgeqrf(cusolverDnHandle_t handle,
                                  int m,
                                  int n,
                                  double *A,
                                  int lda,
                                  double *TAU,
                                  double *Workspace,
                                  int Lwork,
                                  int *devInfo) {
    // ignore Workspace, Lwork and devInfo as rocSOLVER does not need them
    return rocsolver_dgeqrf(handle, m, n, A, lda, TAU);
}

cusolverStatus_t cusolverDnCgeqrf(cusolverDnHandle_t handle,
                                  int m,
                                  int n,
                                  cuComplex *A,
                                  int lda,
                                  cuComplex *TAU,
                                  cuComplex *Workspace,
                                  int Lwork,
                                  int *devInfo) {
    // ignore Workspace, Lwork and devInfo as rocSOLVER does not need them
    return rocsolver_cgeqrf(handle, m, n,
                            reinterpret_cast<rocblas_float_complex*>(A), lda,
                            reinterpret_cast<rocblas_float_complex*>(TAU));
}

cusolverStatus_t cusolverDnZgeqrf(cusolverDnHandle_t handle,
                                  int m,
                                  int n,
                                  cuDoubleComplex *A,
                                  int lda,
                                  cuDoubleComplex *TAU,
                                  cuDoubleComplex *Workspace,
                                  int Lwork,
                                  int *devInfo) {
    // ignore Workspace, Lwork and devInfo as rocSOLVER does not need them
    return rocsolver_zgeqrf(handle, m, n,
                            reinterpret_cast<rocblas_double_complex*>(A), lda,
                            reinterpret_cast<rocblas_double_complex*>(TAU));
}


/* ---------- orgqr ---------- */
cusolverStatus_t cusolverDnSorgqr_bufferSize(cusolverDnHandle_t handle,
                                             int m,
                                             int n,
                                             int k,
                                             const float *A,
                                             int lda,
                                             const float *tau,
                                             int *lwork) {
    // this needs to return 0 because rocSolver does not rely on it
    *lwork = 0;
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnDorgqr_bufferSize(cusolverDnHandle_t handle,
                                             int m,
                                             int n,
                                             int k,
                                             const double *A,
                                             int lda,
                                             const double *tau,
                                             int *lwork) {
    // this needs to return 0 because rocSolver does not rely on it
    *lwork = 0;
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnSorgqr(cusolverDnHandle_t handle,
                                  int m,
                                  int n,
                                  int k,
                                  float *A,
                                  int lda,
                                  const float *tau,
                                  float *work,
                                  int lwork,
                                  int *info) {
    // ignore work, lwork and info as rocSOLVER does not need them
    return rocsolver_sorgqr(handle, m, n, k, A, lda, const_cast<float*>(tau));
}

cusolverStatus_t cusolverDnDorgqr(cusolverDnHandle_t handle,
                                  int m,
                                  int n,
                                  int k,
                                  double *A,
                                  int lda,
                                  const double *tau,
                                  double *work,
                                  int lwork,
                                  int *info) {
    // ignore work, lwork and info as rocSOLVER does not need them
    return rocsolver_dorgqr(handle, m, n, k, A, lda, const_cast<double*>(tau));
}


/* ---------- ormqr ---------- */
cusolverStatus_t cusolverDnSormqr_bufferSize(cusolverDnHandle_t handle,
                                             cublasSideMode_t side,
                                             cublasOperation_t trans,
                                             int m,
                                             int n,
                                             int k,
                                             const float *A,
                                             int lda,
                                             const float *tau,
                                             const float *C,
                                             int ldc,
                                             int *lwork) {
    // this needs to return 0 because rocSolver does not rely on it
    *lwork = 0;
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnDormqr_bufferSize(cusolverDnHandle_t handle,
                                             cublasSideMode_t side,
                                             cublasOperation_t trans,
                                             int m,
                                             int n,
                                             int k,
                                             const double *A,
                                             int lda,
                                             const double *tau,
                                             const double *C,
                                             int ldc,
                                             int *lwork) {
    // this needs to return 0 because rocSolver does not rely on it
    *lwork = 0;
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnSormqr(cusolverDnHandle_t handle,
                                  cublasSideMode_t side,
                                  cublasOperation_t trans,
                                  int m,
                                  int n,
                                  int k,
                                  const float *A,
                                  int lda,
                                  const float *tau,
                                  float *C,
                                  int ldc,
                                  float *work,
                                  int lwork,
                                  int *devInfo) {
    // ignore work, lwork and devInfo as rocSOLVER does not need them
    return rocsolver_sormqr(handle,
                            convert_rocblas_side(side),
                            convert_rocblas_operation(trans),
                            m, n, k,
                            const_cast<float*>(A), lda,
                            const_cast<float*>(tau),
                            C, ldc);
}

cusolverStatus_t cusolverDnDormqr(cusolverDnHandle_t handle,
                                  cublasSideMode_t side,
                                  cublasOperation_t trans,
                                  int m,
                                  int n,
                                  int k,
                                  const double *A,
                                  int lda,
                                  const double *tau,
                                  double *C,
                                  int ldc,
                                  double *work,
                                  int lwork,
                                  int *devInfo) {
    // ignore work, lwork and devInfo as rocSOLVER does not need them
    return rocsolver_dormqr(handle,
                            convert_rocblas_side(side),
                            convert_rocblas_operation(trans),
                            m, n, k,
                            const_cast<double*>(A), lda,
                            const_cast<double*>(tau),
                            C, ldc);
}


/* all of the stubs here are unsupported functions; the supported ones are in cupy_hip.h */

typedef enum{} cusolverEigType_t;
typedef enum{} cusolverEigMode_t;
typedef void* cusolverSpHandle_t;
typedef void* cusparseMatDescr_t;
typedef void* gesvdjInfo_t;
typedef void* syevjInfo_t;

cusolverStatus_t cusolverSpGetStream(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverSpSetStream(...) {
    return rocblas_status_not_implemented;
}


/* ---------- potrf ---------- */
cusolverStatus_t cusolverDnCpotrf_bufferSize(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnZpotrf_bufferSize(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnCpotrf(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnZpotrf(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnCpotrfBatched(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnZpotrfBatched(...) {
    return rocblas_status_not_implemented;
}


/* ---------- potrs ---------- */
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


/* ---------- ungqr ---------- */
cusolverStatus_t cusolverDnCungqr_bufferSize(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnZungqr_bufferSize(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnCungqr(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnZungqr(...) {
    return rocblas_status_not_implemented;
}


/* ---------- unmqr ---------- */
cusolverStatus_t cusolverDnCunmqr_bufferSize(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnZunmqr_bufferSize(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnCunmqr(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnZunmqr(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnSsytrf_bufferSize(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnDsytrf_bufferSize(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnCsytrf_bufferSize(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnZsytrf_bufferSize(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnSsytrf(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnDsytrf(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnCsytrf(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnZsytrf(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnSgebrd_bufferSize(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnDgebrd_bufferSize(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnCgebrd_bufferSize(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnZgebrd_bufferSize(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnSgebrd(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnDgebrd(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnCgebrd(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnZgebrd(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnSgesvd_bufferSize(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnDgesvd_bufferSize(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnCgesvd_bufferSize(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnZgesvd_bufferSize(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnSgesvd(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnDgesvd(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnCgesvd(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnZgesvd(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnCreateGesvdjInfo(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnDestroyGesvdjInfo(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnXgesvdjSetTolerance(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnXgesvdjSetMaxSweeps(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnXgesvdjSetSortEig(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnXgesvdjGetResidual(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnXgesvdjGetSweeps(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnSgesvdj_bufferSize(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnDgesvdj_bufferSize(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnCgesvdj_bufferSize(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnZgesvdj_bufferSize(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnSgesvdj(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnDgesvdj(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnCgesvdj(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnZgesvdj(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnSgesvdjBatched_bufferSize(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnDgesvdjBatched_bufferSize(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnCgesvdjBatched_bufferSize(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnZgesvdjBatched_bufferSize(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnSgesvdjBatched(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnDgesvdjBatched(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnCgesvdjBatched(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnZgesvdjBatched(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnSgesvdaStridedBatched_bufferSize(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnDgesvdaStridedBatched_bufferSize(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnCgesvdaStridedBatched_bufferSize(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnZgesvdaStridedBatched_bufferSize(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnSgesvdaStridedBatched(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnDgesvdaStridedBatched(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnCgesvdaStridedBatched(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnZgesvdaStridedBatched(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnZZgels_bufferSize(...) {
    return rocblas_status_not_implemented;
}
cusolverStatus_t cusolverDnZCgels_bufferSize(...) {
    return rocblas_status_not_implemented;
}
cusolverStatus_t cusolverDnZYgels_bufferSize(...) {
    return rocblas_status_not_implemented;
}
cusolverStatus_t cusolverDnZKgels_bufferSize(...) {
    return rocblas_status_not_implemented;
}
cusolverStatus_t cusolverDnCCgels_bufferSize(...) {
    return rocblas_status_not_implemented;
}
cusolverStatus_t cusolverDnCYgels_bufferSize(...) {
    return rocblas_status_not_implemented;
}
cusolverStatus_t cusolverDnCKgels_bufferSize(...) {
    return rocblas_status_not_implemented;
}
cusolverStatus_t cusolverDnDDgels_bufferSize(...) {
    return rocblas_status_not_implemented;
}
cusolverStatus_t cusolverDnDSgels_bufferSize(...) {
    return rocblas_status_not_implemented;
}
cusolverStatus_t cusolverDnDXgels_bufferSize(...) {
    return rocblas_status_not_implemented;
}
cusolverStatus_t cusolverDnDHgels_bufferSize(...) {
    return rocblas_status_not_implemented;
}
cusolverStatus_t cusolverDnSSgels_bufferSize(...) {
    return rocblas_status_not_implemented;
}
cusolverStatus_t cusolverDnSXgels_bufferSize(...) {
    return rocblas_status_not_implemented;
}
cusolverStatus_t cusolverDnSHgels_bufferSize(...) {
    return rocblas_status_not_implemented;
}
cusolverStatus_t cusolverDnZZgels(...) {
    return rocblas_status_not_implemented;
}
cusolverStatus_t cusolverDnZCgels(...) {
    return rocblas_status_not_implemented;
}
cusolverStatus_t cusolverDnZYgels(...) {
    return rocblas_status_not_implemented;
}
cusolverStatus_t cusolverDnZKgels(...) {
    return rocblas_status_not_implemented;
}
cusolverStatus_t cusolverDnCCgels(...) {
    return rocblas_status_not_implemented;
}
cusolverStatus_t cusolverDnCYgels(...) {
    return rocblas_status_not_implemented;
}
cusolverStatus_t cusolverDnCKgels(...) {
    return rocblas_status_not_implemented;
}
cusolverStatus_t cusolverDnDDgels(...) {
    return rocblas_status_not_implemented;
}
cusolverStatus_t cusolverDnDSgels(...) {
    return rocblas_status_not_implemented;
}
cusolverStatus_t cusolverDnDXgels(...) {
    return rocblas_status_not_implemented;
}
cusolverStatus_t cusolverDnDHgels(...) {
    return rocblas_status_not_implemented;
}
cusolverStatus_t cusolverDnSSgels(...) {
    return rocblas_status_not_implemented;
}
cusolverStatus_t cusolverDnSXgels(...) {
    return rocblas_status_not_implemented;
}
cusolverStatus_t cusolverDnSHgels(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnSsyevd_bufferSize(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnDsyevd_bufferSize(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnCheevd_bufferSize(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnZheevd_bufferSize(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnSsyevd(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnDsyevd(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnCheevd(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnZheevd(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnCreateSyevjInfo(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnDestroySyevjInfo(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnXsyevjSetTolerance(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnXsyevjSetMaxSweeps(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnXsyevjSetSortEig(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnXsyevjGetResidual(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnXsyevjGetSweeps(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnSsyevj_bufferSize(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnDsyevj_bufferSize(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnCheevj_bufferSize(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnZheevj_bufferSize(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnSsyevj(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnDsyevj(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnCheevj(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnZheevj(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnSsyevjBatched_bufferSize(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnDsyevjBatched_bufferSize(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnCheevjBatched_bufferSize(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnZheevjBatched_bufferSize(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnSsyevjBatched(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnDsyevjBatched(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnCheevjBatched(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnZheevjBatched(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverDnZZgesv_bufferSize(...) {
    return rocblas_status_not_implemented;
}
cusolverStatus_t cusolverDnZCgesv_bufferSize(...) {
    return rocblas_status_not_implemented;
}
cusolverStatus_t cusolverDnZYgesv_bufferSize(...) {
    return rocblas_status_not_implemented;
}
cusolverStatus_t cusolverDnZKgesv_bufferSize(...) {
    return rocblas_status_not_implemented;
}
cusolverStatus_t cusolverDnCCgesv_bufferSize(...) {
    return rocblas_status_not_implemented;
}
cusolverStatus_t cusolverDnCYgesv_bufferSize(...) {
    return rocblas_status_not_implemented;
}
cusolverStatus_t cusolverDnCKgesv_bufferSize(...) {
    return rocblas_status_not_implemented;
}
cusolverStatus_t cusolverDnDDgesv_bufferSize(...) {
    return rocblas_status_not_implemented;
}
cusolverStatus_t cusolverDnDSgesv_bufferSize(...) {
    return rocblas_status_not_implemented;
}
cusolverStatus_t cusolverDnDXgesv_bufferSize(...) {
    return rocblas_status_not_implemented;
}
cusolverStatus_t cusolverDnDHgesv_bufferSize(...) {
    return rocblas_status_not_implemented;
}
cusolverStatus_t cusolverDnSSgesv_bufferSize(...) {
    return rocblas_status_not_implemented;
}
cusolverStatus_t cusolverDnSXgesv_bufferSize(...) {
    return rocblas_status_not_implemented;
}
cusolverStatus_t cusolverDnSHgesv_bufferSize(...) {
    return rocblas_status_not_implemented;
}
cusolverStatus_t cusolverDnZZgesv(...) {
    return rocblas_status_not_implemented;
}
cusolverStatus_t cusolverDnZCgesv(...) {
    return rocblas_status_not_implemented;
}
cusolverStatus_t cusolverDnZYgesv(...) {
    return rocblas_status_not_implemented;
}
cusolverStatus_t cusolverDnZKgesv(...) {
    return rocblas_status_not_implemented;
}
cusolverStatus_t cusolverDnCCgesv(...) {
    return rocblas_status_not_implemented;
}
cusolverStatus_t cusolverDnCYgesv(...) {
    return rocblas_status_not_implemented;
}
cusolverStatus_t cusolverDnCKgesv(...) {
    return rocblas_status_not_implemented;
}
cusolverStatus_t cusolverDnDDgesv(...) {
    return rocblas_status_not_implemented;
}
cusolverStatus_t cusolverDnDSgesv(...) {
    return rocblas_status_not_implemented;
}
cusolverStatus_t cusolverDnDXgesv(...) {
    return rocblas_status_not_implemented;
}
cusolverStatus_t cusolverDnDHgesv(...) {
    return rocblas_status_not_implemented;
}
cusolverStatus_t cusolverDnSSgesv(...) {
    return rocblas_status_not_implemented;
}
cusolverStatus_t cusolverDnSXgesv(...) {
    return rocblas_status_not_implemented;
}
cusolverStatus_t cusolverDnSHgesv(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverSpCreate(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverSpDestroy(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverSpScsrlsvqr(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverSpDcsrlsvqr(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverSpCcsrlsvqr(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverSpZcsrlsvqr(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverSpScsrlsvchol(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverSpDcsrlsvchol(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverSpCcsrlsvchol(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverSpZcsrlsvchol(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverSpScsreigvsi(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverSpDcsreigvsi(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverSpCcsreigvsi(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverSpZcsreigvsi(...) {
    return rocblas_status_not_implemented;
}

} // extern "C" 

#endif // #ifdef INCLUDE_GUARD_HIP_CUPY_HIPSOLVER_H
