#ifndef INCLUDE_GUARD_HIP_CUPY_ROCSOLVER_H
#define INCLUDE_GUARD_HIP_CUPY_ROCSOLVER_H

#include "cupy_hip.h"
#include "cupy_hipblas.h"


extern "C" {
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

#if HIP_VERSION >= 309
static rocblas_svect convert_rocblas_svect(signed char mode) {
    switch(mode) {
        case 'A': return rocblas_svect_all;
        case 'S': return rocblas_svect_singular;
        case 'O': return rocblas_svect_overwrite;
        case 'N': return rocblas_svect_none;
        default: throw std::runtime_error("unrecognized mode");
    }
}
#endif


// rocSOLVER
/* ---------- helpers ---------- */

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

cusolverStatus_t cusolverDnCpotrf_bufferSize(cusolverDnHandle_t handle,
                                             cublasFillMode_t uplo,
                                             int n,
                                             cuComplex *A,
                                             int lda,
                                             int *Lwork) {
    // this needs to return 0 because rocSolver does not rely on it
    *Lwork = 0;
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnZpotrf_bufferSize(cusolverDnHandle_t handle,
                                             cublasFillMode_t uplo,
                                             int n,
                                             cuDoubleComplex *A,
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

cusolverStatus_t cusolverDnCpotrf(cusolverDnHandle_t handle,
                                  cublasFillMode_t uplo,
                                  int n,
                                  cuComplex *A,
                                  int lda,
                                  cuComplex *Workspace,
                                  int Lwork,
                                  int *devInfo) {
    #if HIP_VERSION < 306
    return rocblas_status_not_implemented;
    #else
    // ignore Workspace and Lwork as rocSOLVER does not need them
    return rocsolver_cpotrf(handle, convert_rocblas_fill(uplo), n,
                            reinterpret_cast<rocblas_float_complex*>(A), lda, devInfo);
    #endif
}

cusolverStatus_t cusolverDnZpotrf(cusolverDnHandle_t handle,
                                  cublasFillMode_t uplo,
                                  int n,
                                  cuDoubleComplex *A,
                                  int lda,
                                  cuDoubleComplex *Workspace,
                                  int Lwork,
                                  int *devInfo) {
    #if HIP_VERSION < 306
    return rocblas_status_not_implemented;
    #else
    // ignore Workspace and Lwork as rocSOLVER does not need them
    return rocsolver_zpotrf(handle, convert_rocblas_fill(uplo), n,
                            reinterpret_cast<rocblas_double_complex*>(A), lda, devInfo);
    #endif
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

cusolverStatus_t cusolverDnCpotrfBatched(cusolverDnHandle_t handle,
                                         cublasFillMode_t uplo,
                                         int n,
                                         cuComplex *Aarray[],
                                         int lda,
                                         int *infoArray,
                                         int batchSize) {
    #if HIP_VERSION < 306
    return rocblas_status_not_implemented;
    #else
    return rocsolver_cpotrf_batched(handle, convert_rocblas_fill(uplo), n,
                                    reinterpret_cast<rocblas_float_complex* const*>(Aarray), lda,
                                    infoArray, batchSize);
    #endif
}

cusolverStatus_t cusolverDnZpotrfBatched(cusolverDnHandle_t handle,
                                         cublasFillMode_t uplo,
                                         int n,
                                         cuDoubleComplex *Aarray[],
                                         int lda,
                                         int *infoArray,
                                         int batchSize) {
    #if HIP_VERSION < 306
    return rocblas_status_not_implemented;
    #else
    return rocsolver_zpotrf_batched(handle, convert_rocblas_fill(uplo), n,
                                    reinterpret_cast<rocblas_double_complex* const*>(Aarray), lda,
                                    infoArray, batchSize);
    #endif
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


/* ---------- ungqr ---------- */
cusolverStatus_t cusolverDnCungqr_bufferSize(cusolverDnHandle_t handle,
                                             int m,
                                             int n,
                                             int k,
                                             const cuComplex *A,
                                             int lda,
                                             const cuComplex *tau,
                                             int *lwork) {
    // this needs to return 0 because rocSolver does not rely on it
    *lwork = 0;
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnZungqr_bufferSize(cusolverDnHandle_t handle,
                                             int m,
                                             int n,
                                             int k,
                                             const cuDoubleComplex *A,
                                             int lda,
                                             const cuDoubleComplex *tau,
                                             int *lwork) {
    // this needs to return 0 because rocSolver does not rely on it
    *lwork = 0;
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnCungqr(cusolverDnHandle_t handle,
                                  int m,
                                  int n,
                                  int k,
                                  cuComplex *A,
                                  int lda,
                                  const cuComplex *tau,
                                  cuComplex *work,
                                  int lwork,
                                  int *info) {
    #if HIP_VERSION < 306
    return rocblas_status_not_implemented;
    #else
    // ignore work, lwork and info as rocSOLVER does not need them
    return rocsolver_cungqr(handle, m, n, k,
                            reinterpret_cast<rocblas_float_complex*>(A), lda,
                            reinterpret_cast<rocblas_float_complex*>(const_cast<cuComplex*>(tau)));
    #endif
}

cusolverStatus_t cusolverDnZungqr(cusolverDnHandle_t handle,
                                  int m,
                                  int n,
                                  int k,
                                  cuDoubleComplex *A,
                                  int lda,
                                  const cuDoubleComplex *tau,
                                  cuDoubleComplex *work,
                                  int lwork,
                                  int *info) {
    #if HIP_VERSION < 306
    return rocblas_status_not_implemented;
    #else
    // ignore work, lwork and info as rocSOLVER does not need them
    return rocsolver_zungqr(handle, m, n, k,
                            reinterpret_cast<rocblas_double_complex*>(A), lda,
                            reinterpret_cast<rocblas_double_complex*>(const_cast<cuDoubleComplex*>(tau)));
    #endif
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


/* ---------- unmqr ---------- */
cusolverStatus_t cusolverDnCunmqr_bufferSize(cusolverDnHandle_t handle,
                                             cublasSideMode_t side,
                                             cublasOperation_t trans,
                                             int m,
                                             int n,
                                             int k,
                                             const cuComplex *A,
                                             int lda,
                                             const cuComplex *tau,
                                             const cuComplex *C,
                                             int ldc,
                                             int *lwork) {
    // this needs to return 0 because rocSolver does not rely on it
    *lwork = 0;
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnZunmqr_bufferSize(cusolverDnHandle_t handle,
                                             cublasSideMode_t side,
                                             cublasOperation_t trans,
                                             int m,
                                             int n,
                                             int k,
                                             const cuDoubleComplex *A,
                                             int lda,
                                             const cuDoubleComplex *tau,
                                             const cuDoubleComplex *C,
                                             int ldc,
                                             int *lwork) {
    // this needs to return 0 because rocSolver does not rely on it
    *lwork = 0;
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnCunmqr(cusolverDnHandle_t handle,
                                  cublasSideMode_t side,
                                  cublasOperation_t trans,
                                  int m,
                                  int n,
                                  int k,
                                  const cuComplex *A,
                                  int lda,
                                  const cuComplex *tau,
                                  cuComplex *C,
                                  int ldc,
                                  cuComplex *work,
                                  int lwork,
                                  int *devInfo) {
    #if HIP_VERSION < 306
    return rocblas_status_not_implemented;
    #else
    // ignore work, lwork and devInfo as rocSOLVER does not need them
    return rocsolver_cunmqr(handle, convert_rocblas_side(side), convert_rocblas_operation(trans),
                            m, n, k, reinterpret_cast<rocblas_float_complex*>(const_cast<cuComplex*>(A)),
                            lda, reinterpret_cast<rocblas_float_complex*>(const_cast<cuComplex*>(tau)),
                            reinterpret_cast<rocblas_float_complex*>(C), ldc);
    #endif
}

cusolverStatus_t cusolverDnZunmqr(cusolverDnHandle_t handle,
                                  cublasSideMode_t side,
                                  cublasOperation_t trans,
                                  int m,
                                  int n,
                                  int k,
                                  const cuDoubleComplex *A,
                                  int lda,
                                  const cuDoubleComplex *tau,
                                  cuDoubleComplex *C,
                                  int ldc,
                                  cuDoubleComplex *work,
                                  int lwork,
                                  int *devInfo) {
    #if HIP_VERSION < 306
    return rocblas_status_not_implemented;
    #else
    // ignore work, lwork and devInfo as rocSOLVER does not need them
    return rocsolver_zunmqr(handle, convert_rocblas_side(side), convert_rocblas_operation(trans),
                            m, n, k, reinterpret_cast<rocblas_double_complex*>(const_cast<cuDoubleComplex*>(A)),
                            lda, reinterpret_cast<rocblas_double_complex*>(const_cast<cuDoubleComplex*>(tau)),
                            reinterpret_cast<rocblas_double_complex*>(C), ldc);
    #endif
}


/* ---------- gesvd ---------- */
cusolverStatus_t cusolverDnSgesvd_bufferSize(cusolverDnHandle_t handle,
                                             int m,
                                             int n,
                                             int *lwork) {
    // this needs to return 0 because rocSolver does not rely on it
    *lwork = 0;
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnDgesvd_bufferSize(cusolverDnHandle_t handle,
                                             int m,
                                             int n,
                                             int *lwork) {
    // this needs to return 0 because rocSolver does not rely on it
    *lwork = 0;
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnCgesvd_bufferSize(cusolverDnHandle_t handle,
                                             int m,
                                             int n,
                                             int *lwork) {
    // this needs to return 0 because rocSolver does not rely on it
    *lwork = 0;
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnZgesvd_bufferSize(cusolverDnHandle_t handle,
                                             int m,
                                             int n,
                                             int *lwork) {
    // this needs to return 0 because rocSolver does not rely on it
    *lwork = 0;
    return rocblas_status_success;
}

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
    #if HIP_VERSION < 309
    return rocblas_status_not_implemented;
    #else
    // ignore work and lwork as rocSOLVER does not need them
    return rocsolver_sgesvd(handle, convert_rocblas_svect(jobu), convert_rocblas_svect(jobvt),
                            m, n, A, lda, S, U, ldu, VT, ldvt, rwork, rocblas_outofplace,  // always out-of-place
                            info);
    #endif
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
    #if HIP_VERSION < 309
    return rocblas_status_not_implemented;
    #else
    // ignore work and lwork as rocSOLVER does not need them
    return rocsolver_dgesvd(handle, convert_rocblas_svect(jobu), convert_rocblas_svect(jobvt),
                            m, n, A, lda, S, U, ldu, VT, ldvt, rwork, rocblas_outofplace,  // always out-of-place
                            info);
    #endif
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
    #if HIP_VERSION < 309
    return rocblas_status_not_implemented;
    #else
    // ignore work and lwork as rocSOLVER does not need them
    return rocsolver_cgesvd(handle, convert_rocblas_svect(jobu), convert_rocblas_svect(jobvt),
                            m, n, reinterpret_cast<rocblas_float_complex*>(A), lda,
                            S, reinterpret_cast<rocblas_float_complex*>(U), ldu,
                            reinterpret_cast<rocblas_float_complex*>(VT), ldvt, rwork,
                            rocblas_outofplace,  // always out-of-place
                            info);
    #endif
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
    #if HIP_VERSION < 309
    return rocblas_status_not_implemented;
    #else
    // ignore work and lwork as rocSOLVER does not need them
    return rocsolver_zgesvd(handle, convert_rocblas_svect(jobu), convert_rocblas_svect(jobvt),
                            m, n, reinterpret_cast<rocblas_double_complex*>(A), lda,
                            S, reinterpret_cast<rocblas_double_complex*>(U), ldu,
                            reinterpret_cast<rocblas_double_complex*>(VT), ldvt, rwork,
                            rocblas_outofplace,  // always out-of-place
                            info);
    #endif
}


/* ---------- batched gesvd ---------- */
// Because rocSOLVER provides no counterpart for gesvdjBatched, we wrap its batched version directly.
typedef enum {
    CUSOLVER_EIG_MODE_NOVECTOR=0,
    CUSOLVER_EIG_MODE_VECTOR=1
} cusolverEigMode_t;
typedef void* gesvdjInfo_t;

cusolverStatus_t cusolverDnCreateGesvdjInfo(...) {
    // should always success as rocSOLVER does not need it
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnDestroyGesvdjInfo(...) {
    // should always success as rocSOLVER does not need it
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnSgesvdjBatched_bufferSize(
        cusolverDnHandle_t handle,
        cusolverEigMode_t jobz,
        int m,                
        int n,                
        const float *A,    
        int lda,           
        const float *S, 
        const float *U,   
        int ldu, 
        const float *V,
        int ldv,  
        int *lwork,
        gesvdjInfo_t params,
        int batchSize) {
    // rocSOLVER does not need extra workspace, but it needs to allocate memory for storing
    // the bidiagonal matrix B associated with A, which we don't need, so we use this workspace
    // to store it
    *lwork = batchSize * (m<n?m:n) * sizeof(float);
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnDgesvdjBatched_bufferSize(
        cusolverDnHandle_t handle,
        cusolverEigMode_t jobz,
        int m,
        int n,
        const double *A, 
        int lda,
        const double *S,
        const double *U,
        int ldu,
        const double *V,
        int ldv,
        int *lwork,
        gesvdjInfo_t params,
        int batchSize) {
    // rocSOLVER does not need extra workspace, but it needs to allocate memory for storing
    // the bidiagonal matrix B associated with A, which we don't need, so we use this workspace
    // to store it
    *lwork = batchSize * (m<n?m:n) * sizeof(double);
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnCgesvdjBatched_bufferSize(
        cusolverDnHandle_t handle,
        cusolverEigMode_t jobz,
        int m,
        int n,
        const cuComplex *A,
        int lda,
        const float *S,
        const cuComplex *U,
        int ldu,
        const cuComplex *V,
        int ldv,
        int *lwork,
        gesvdjInfo_t params,
        int batchSize) {
    // rocSOLVER does not need extra workspace, but it needs to allocate memory for storing
    // the bidiagonal matrix B associated with A, which we don't need, so we use this workspace
    // to store it
    *lwork = batchSize * (m<n?m:n) * sizeof(float);
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnZgesvdjBatched_bufferSize(
        cusolverDnHandle_t handle,
        cusolverEigMode_t jobz, 
        int m, 
        int n, 
        const cuDoubleComplex *A,
        int lda,
        const double *S,
        const cuDoubleComplex *U,
        int ldu, 
        const cuDoubleComplex *V,
        int ldv,
        int *lwork,
        gesvdjInfo_t params,
        int batchSize) {
    // rocSOLVER does not need extra workspace, but it needs to allocate memory for storing
    // the bidiagonal matrix B associated with A, which we don't need, so we use this workspace
    // to store it
    *lwork = batchSize * (m<n?m:n) * sizeof(double);
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnSgesvdjBatched(
        cusolverDnHandle_t handle,
        cusolverEigMode_t jobz, 
        int m, 
        int n, 
        float *A, 
        int lda, 
        float *S, 
        float *U,
        int ldu,
        float *V,
        int ldv, 
        float *work,
        int lwork,
        int *info,
        gesvdjInfo_t params,
        int batchSize) {
    #if HIP_VERSION < 309
    return rocblas_status_not_implemented;
    #else
    rocblas_svect leftv, rightv;
    rocblas_stride stU, stV;
    if (jobz == CUSOLVER_EIG_MODE_NOVECTOR) {
        leftv = rocblas_svect_none;
        rightv = rocblas_svect_none;
        stU = ldu * (m<n?m:n);
        stV = ldv * n;
    } else {  // CUSOLVER_EIG_MODE_VECTOR
        leftv = rocblas_svect_all;
        rightv = rocblas_svect_all;
        stU = ldu * m;
        stV = ldv * n;
    }
    return rocsolver_sgesvd_batched(handle, leftv, rightv,
                                    m, n, reinterpret_cast<float* const*>(A), lda,
                                    S, m<n?m:n,
                                    U, ldu, stU,
                                    V, ldv, stV,
                                    // since we can't pass in another array through the API, and work is unused,
                                    // we use it to store the temporary E array, to be discarded after calculation
                                    work, (m<n?m:n)-1,
                                    rocblas_outofplace, // always out-of-place
                                    info, batchSize);
    #endif
}

cusolverStatus_t cusolverDnDgesvdjBatched(
        cusolverDnHandle_t handle,
        cusolverEigMode_t jobz,
        int m,
        int n,
        double *A,
        int lda,
        double *S,
        double *U,
        int ldu,
        double *V,
        int ldv,
        double *work,
        int lwork,
        int *info,
        gesvdjInfo_t params,
        int batchSize) { 
    #if HIP_VERSION < 309
    return rocblas_status_not_implemented;
    #else
    rocblas_svect leftv, rightv;
    rocblas_stride stU, stV;
    if (jobz == CUSOLVER_EIG_MODE_NOVECTOR) {
        leftv = rocblas_svect_none;
        rightv = rocblas_svect_none;
        stU = ldu * (m<n?m:n);
        stV = ldv * n;
    } else {  // CUSOLVER_EIG_MODE_VECTOR
        leftv = rocblas_svect_all;
        rightv = rocblas_svect_all;
        stU = ldu * m;
        stV = ldv * n;
    }
    return rocsolver_dgesvd_batched(handle, leftv, rightv,
                                    m, n, reinterpret_cast<double* const*>(A), lda,
                                    S, m<n?m:n,
                                    U, ldu, stU,
                                    V, ldv, stV,
                                    // since we can't pass in another array through the API, and work is unused,
                                    // we use it to store the temporary E array, to be discarded after calculation
                                    work, (m<n?m:n)-1,
                                    rocblas_outofplace, // always out-of-place
                                    info, batchSize);
    #endif
}

cusolverStatus_t cusolverDnCgesvdjBatched(
        cusolverDnHandle_t handle,
        cusolverEigMode_t jobz,
        int m,
        int n,
        cuComplex *A,
        int lda,
        float *S,
        cuComplex *U,
        int ldu,
        cuComplex *V,
        int ldv,
        cuComplex *work,
        int lwork,
        int *info,
        gesvdjInfo_t params,
        int batchSize) {
    #if HIP_VERSION < 309
    return rocblas_status_not_implemented;
    #else
    rocblas_svect leftv, rightv;
    rocblas_stride stU, stV;
    if (jobz == CUSOLVER_EIG_MODE_NOVECTOR) {
        leftv = rocblas_svect_none;
        rightv = rocblas_svect_none;
        stU = ldu * (m<n?m:n);
        stV = ldv * n;
    } else {  // CUSOLVER_EIG_MODE_VECTOR
        leftv = rocblas_svect_all;
        rightv = rocblas_svect_all;
        stU = ldu * m;
        stV = ldv * n;
    }
    return rocsolver_cgesvd_batched(handle, leftv, rightv,
                                    m, n, reinterpret_cast<rocblas_float_complex* const*>(A), lda,
                                    S, m<n?m:n,
                                    reinterpret_cast<rocblas_float_complex*>(U), ldu, stU,
                                    reinterpret_cast<rocblas_float_complex*>(V), ldv, stV,
                                    // since we can't pass in another array through the API, and work is unused,
                                    // we use it to store the temporary E array, to be discarded after calculation
                                    reinterpret_cast<float*>(work), (m<n?m:n)-1,
                                    rocblas_outofplace, // always out-of-place
                                    info, batchSize);
    #endif
}

cusolverStatus_t cusolverDnZgesvdjBatched(
        cusolverDnHandle_t handle,
        cusolverEigMode_t jobz,
        int m,
        int n,
        cuDoubleComplex *A,
        int lda,
        double *S,
        cuDoubleComplex *U,
        int ldu,
        cuDoubleComplex *V,
        int ldv,
        cuDoubleComplex *work,
        int lwork,
        int *info,
        gesvdjInfo_t params,
        int batchSize) {
    #if HIP_VERSION < 309
    return rocblas_status_not_implemented;
    #else
    rocblas_svect leftv, rightv;
    rocblas_stride stU, stV;
    if (jobz == CUSOLVER_EIG_MODE_NOVECTOR) {
        leftv = rocblas_svect_none;
        rightv = rocblas_svect_none;
        stU = ldu * (m<n?m:n);
        stV = ldv * n;
    } else {  // CUSOLVER_EIG_MODE_VECTOR
        leftv = rocblas_svect_all;
        rightv = rocblas_svect_all;
        stU = ldu * m;
        stV = ldv * n;
    }
    return rocsolver_zgesvd_batched(handle, leftv, rightv,
                                    m, n, reinterpret_cast<rocblas_double_complex* const*>(A), lda,
                                    S, m<n?m:n,
                                    reinterpret_cast<rocblas_double_complex*>(U), ldu, stU,
                                    reinterpret_cast<rocblas_double_complex*>(V), ldv, stV,
                                    // since we can't pass in another array through the API, and work is unused,
                                    // we use it to store the temporary E array, to be discarded after calculation
                                    reinterpret_cast<double*>(work), (m<n?m:n)-1,
                                    rocblas_outofplace, // always out-of-place
                                    info, batchSize);
    #endif
}


/* ---------- gebrd ---------- */
cusolverStatus_t cusolverDnSgebrd_bufferSize(cusolverDnHandle_t handle,
                                             int m,
                                             int n,
                                             int *Lwork) {
    // this needs to return 0 because rocSolver does not rely on it
    *Lwork = 0;
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnDgebrd_bufferSize(cusolverDnHandle_t handle,
                                             int m,
                                             int n,
                                             int *Lwork) {
    // this needs to return 0 because rocSolver does not rely on it
    *Lwork = 0;
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnCgebrd_bufferSize(cusolverDnHandle_t handle,
                                             int m,
                                             int n,
                                             int *Lwork) {
    // this needs to return 0 because rocSolver does not rely on it
    *Lwork = 0;
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnZgebrd_bufferSize(cusolverDnHandle_t handle,
                                             int m,
                                             int n,
                                             int *Lwork) {
    // this needs to return 0 because rocSolver does not rely on it
    *Lwork = 0;
    return rocblas_status_success;
}

cusolverStatus_t cusolverDnSgebrd(cusolverDnHandle_t handle,
                                  int m,
                                  int n,
                                  float *A,
                                  int lda,
                                  float *D,
                                  float *E,
                                  float *TAUQ,
                                  float *TAUP,
                                  float *Work,
                                  int Lwork,
                                  int *devInfo) {
    #if HIP_VERSION < 306
    return rocblas_status_not_implemented;
    #else
    // ignore work, lwork and devinfo as rocSOLVER does not need them
    return rocsolver_sgebrd(handle, m, n, A, lda, D, E, TAUQ, TAUP);
    #endif
}

cusolverStatus_t cusolverDnDgebrd(cusolverDnHandle_t handle,
                                  int m,
                                  int n,
                                  double *A,
                                  int lda,
                                  double *D,
                                  double *E,
                                  double *TAUQ,
                                  double *TAUP,
                                  double *Work,
                                  int Lwork,
                                  int *devInfo) {
    #if HIP_VERSION < 306
    return rocblas_status_not_implemented;
    #else
    // ignore work, lwork and devinfo as rocSOLVER does not need them
    return rocsolver_dgebrd(handle, m, n, A, lda, D, E, TAUQ, TAUP);
    #endif
}

cusolverStatus_t cusolverDnCgebrd(cusolverDnHandle_t handle,
                                  int m,
                                  int n,
                                  cuComplex *A,
                                  int lda,
                                  float *D,
                                  float *E,
                                  cuComplex *TAUQ,
                                  cuComplex *TAUP,
                                  cuComplex *Work,
                                  int Lwork,
                                  int *devInfo) {
    #if HIP_VERSION < 306
    return rocblas_status_not_implemented;
    #else
    // ignore work, lwork and devinfo as rocSOLVER does not need them
    return rocsolver_cgebrd(handle, m, n, reinterpret_cast<rocblas_float_complex*>(A),
                            lda, D, E, reinterpret_cast<rocblas_float_complex*>(TAUQ),
                            reinterpret_cast<rocblas_float_complex*>(TAUP));
    #endif
}

cusolverStatus_t cusolverDnZgebrd(cusolverDnHandle_t handle,
                                  int m,
                                  int n,
                                  cuDoubleComplex *A,
                                  int lda,
                                  double *D,
                                  double *E,
                                  cuDoubleComplex *TAUQ,
                                  cuDoubleComplex *TAUP,
                                  cuDoubleComplex *Work,
                                  int Lwork,
                                  int *devInfo) {
    #if HIP_VERSION < 306
    return rocblas_status_not_implemented;
    #else
    // ignore work, lwork and devinfo as rocSOLVER does not need them
    return rocsolver_zgebrd(handle, m, n, reinterpret_cast<rocblas_double_complex*>(A),
                            lda, D, E, reinterpret_cast<rocblas_double_complex*>(TAUQ),
                            reinterpret_cast<rocblas_double_complex*>(TAUP));
    #endif
}


/* all of the stubs below are unsupported functions; the supported ones are moved to above */

typedef enum{} cusolverEigType_t;
typedef void* cusolverSpHandle_t;
typedef void* cusparseMatDescr_t;
typedef void* syevjInfo_t;

cusolverStatus_t cusolverSpGetStream(...) {
    return rocblas_status_not_implemented;
}

cusolverStatus_t cusolverSpSetStream(...) {
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


/* ---------- sytrf ---------- */
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

#endif // #ifdef INCLUDE_GUARD_HIP_CUPY_ROCSOLVER_H
