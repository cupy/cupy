# distutils: language = c++

"""Thin wrapper of CUSOLVER."""

cimport cython  # NOQA

from cupy_backends.cuda.api cimport driver
from cupy_backends.cuda.api cimport runtime
from cupy_backends.cuda cimport stream as stream_module


###############################################################################
# Extern
###############################################################################

cdef extern from '../../cupy_complex.h':
    ctypedef struct cuComplex 'cuComplex':
        float x, y

    ctypedef struct cuDoubleComplex 'cuDoubleComplex':
        double x, y

cdef extern from '../../cupy_lapack.h' nogil:
    # Context
    int cusolverDnCreate(Handle* handle)
    int cusolverSpCreate(SpHandle* handle)
    int cusolverDnDestroy(Handle handle)
    int cusolverSpDestroy(SpHandle handle)

    # Stream
    int cusolverDnGetStream(Handle handle, driver.Stream* streamId)
    int cusolverSpGetStream(SpHandle handle, driver.Stream* streamId)
    int cusolverDnSetStream(Handle handle, driver.Stream streamId)
    int cusolverSpSetStream(SpHandle handle, driver.Stream streamId)

    # Library Property
    int cusolverGetProperty(LibraryPropertyType type, int* value)

    # libraryPropertyType_t
    int MAJOR_VERSION
    int MINOR_VERSION
    int PATCH_LEVEL

    ###########################################################################
    # Dense LAPACK Functions (Linear Solver)
    ###########################################################################

    # Cholesky factorization
    int cusolverDnSpotrf_bufferSize(Handle handle, FillMode uplo, int n,
                                    float* A, int lda, int* lwork)
    int cusolverDnDpotrf_bufferSize(Handle handle, FillMode uplo, int n,
                                    double* A, int lda, int* lwork)
    int cusolverDnCpotrf_bufferSize(Handle handle, FillMode uplo, int n,
                                    cuComplex* A, int lda, int* lwork)
    int cusolverDnZpotrf_bufferSize(Handle handle, FillMode uplo, int n,
                                    cuDoubleComplex* A, int lda, int* lwork)

    int cusolverDnSpotrf(Handle handle, FillMode uplo, int n,
                         float* A, int lda,
                         float* work, int lwork, int* devInfo)
    int cusolverDnDpotrf(Handle handle, FillMode uplo, int n,
                         double* A, int lda,
                         double* work, int lwork, int* devInfo)
    int cusolverDnCpotrf(Handle handle, FillMode uplo, int n,
                         cuComplex* A, int lda,
                         cuComplex* work, int lwork, int* devInfo)
    int cusolverDnZpotrf(Handle handle, FillMode uplo, int n,
                         cuDoubleComplex* A, int lda,
                         cuDoubleComplex* work, int lwork, int* devInfo)

    int cusolverDnSpotrs(Handle handle, FillMode uplo, int n, int nrhs,
                         const float* A, int lda,
                         float* B, int ldb, int* devInfo)
    int cusolverDnDpotrs(Handle handle, FillMode uplo, int n, int nrhs,
                         const double* A, int lda,
                         double* B, int ldb, int* devInfo)
    int cusolverDnCpotrs(Handle handle, FillMode uplo, int n, int nrhs,
                         const cuComplex* A, int lda,
                         cuComplex* B, int ldb, int* devInfo)
    int cusolverDnZpotrs(Handle handle, FillMode uplo, int n, int nrhs,
                         const cuDoubleComplex* A, int lda,
                         cuDoubleComplex* B, int ldb, int* devInfo)

    int cusolverDnSpotrfBatched(Handle handle, FillMode uplo, int n,
                                float** Aarray, int lda,
                                int* infoArray, int batchSize)
    int cusolverDnDpotrfBatched(Handle handle, FillMode uplo, int n,
                                double** Aarray, int lda,
                                int* infoArray, int batchSize)
    int cusolverDnCpotrfBatched(Handle handle, FillMode uplo, int n,
                                cuComplex** Aarray, int lda,
                                int* infoArray, int batchSize)
    int cusolverDnZpotrfBatched(Handle handle, FillMode uplo, int n,
                                cuDoubleComplex** Aarray, int lda,
                                int* infoArray, int batchSize)

    int cusolverDnSpotrsBatched(Handle handle, FillMode uplo, int n,
                                int nrhs, float** Aarray, int lda,
                                float** Barray, int ldb,
                                int* devInfo, int batchSize)
    int cusolverDnDpotrsBatched(Handle handle, FillMode uplo, int n,
                                int nrhs, double** Aarray, int lda,
                                double** Barray, int ldb,
                                int* devInfo, int batchSize)
    int cusolverDnCpotrsBatched(Handle handle, FillMode uplo, int n,
                                int nrhs, cuComplex** Aarray, int lda,
                                cuComplex** Barray, int ldb,
                                int* devInfo, int batchSize)
    int cusolverDnZpotrsBatched(Handle handle, FillMode uplo, int n,
                                int nrhs, cuDoubleComplex** Aarray, int lda,
                                cuDoubleComplex** Barray, int ldb,
                                int* devInfo, int batchSize)

    # LU factorization
    int cusolverDnSgetrf_bufferSize(Handle handle, int m, int n,
                                    float* A, int lda, int* lwork)
    int cusolverDnDgetrf_bufferSize(Handle handle, int m, int n,
                                    double* A, int lda, int* lwork)
    int cusolverDnCgetrf_bufferSize(Handle handle, int m, int n,
                                    cuComplex* A, int lda, int* lwork)
    int cusolverDnZgetrf_bufferSize(Handle handle, int m, int n,
                                    cuDoubleComplex* A, int lda, int* lwork)

    int cusolverDnSgetrf(Handle handle, int m, int n,
                         float* A, int lda,
                         float* work, int* devIpiv, int* devInfo)
    int cusolverDnDgetrf(Handle handle, int m, int n,
                         double* A, int lda,
                         double* work, int* devIpiv, int* devInfo)
    int cusolverDnCgetrf(Handle handle, int m, int n,
                         cuComplex* A, int lda,
                         cuComplex* work, int* devIpiv, int* devInfo)
    int cusolverDnZgetrf(Handle handle, int m, int n,
                         cuDoubleComplex* A, int lda,
                         cuDoubleComplex* work, int* devIpiv, int* devInfo)

    # TODO(anaruse): laswp

    # LU solve
    int cusolverDnSgetrs(Handle handle, Operation trans, int n, int nrhs,
                         const float* A, int lda, const int* devIpiv,
                         float* B, int ldb, int* devInfo)
    int cusolverDnDgetrs(Handle handle, Operation trans, int n, int nrhs,
                         const double* A, int lda, const int* devIpiv,
                         double* B, int ldb, int* devInfo)
    int cusolverDnCgetrs(Handle handle, Operation trans, int n, int nrhs,
                         const cuComplex* A, int lda, const int* devIpiv,
                         cuComplex* B, int ldb, int* devInfo)
    int cusolverDnZgetrs(Handle handle, Operation trans, int n, int nrhs,
                         const cuDoubleComplex* A, int lda, const int* devIpiv,
                         cuDoubleComplex* B, int ldb, int* devInfo)

    # QR factorization
    int cusolverDnSgeqrf_bufferSize(Handle handle, int m, int n,
                                    float* A, int lda, int* lwork)
    int cusolverDnDgeqrf_bufferSize(Handle handle, int m, int n,
                                    double* A, int lda, int* lwork)
    int cusolverDnCgeqrf_bufferSize(Handle handle, int m, int n,
                                    cuComplex* A, int lda, int* lwork)
    int cusolverDnZgeqrf_bufferSize(Handle handle, int m, int n,
                                    cuDoubleComplex* A, int lda, int* lwork)

    int cusolverDnSgeqrf(Handle handle, int m, int n,
                         float* A, int lda, float* tau,
                         float* work, int lwork, int* devInfo)
    int cusolverDnDgeqrf(Handle handle, int m, int n,
                         double* A, int lda, double* tau,
                         double* work, int lwork, int* devInfo)
    int cusolverDnCgeqrf(Handle handle, int m, int n,
                         cuComplex* A, int lda, cuComplex* tau,
                         cuComplex* work, int lwork, int* devInfo)
    int cusolverDnZgeqrf(Handle handle, int m, int n,
                         cuDoubleComplex* A, int lda, cuDoubleComplex* tau,
                         cuDoubleComplex* work, int lwork, int* devInfo)

    # Generate unitary matrix Q from QR factorization.
    int cusolverDnSorgqr_bufferSize(Handle handle, int m, int n, int k,
                                    const float* A, int lda,
                                    const float* tau, int* lwork)
    int cusolverDnDorgqr_bufferSize(Handle handle, int m, int n, int k,
                                    const double* A, int lda,
                                    const double* tau, int* lwork)
    int cusolverDnCungqr_bufferSize(Handle handle, int m, int n, int k,
                                    const cuComplex* A, int lda,
                                    const cuComplex* tau, int* lwork)
    int cusolverDnZungqr_bufferSize(Handle handle, int m, int n, int k,
                                    const cuDoubleComplex* A, int lda,
                                    const cuDoubleComplex* tau, int* lwork)

    int cusolverDnSorgqr(Handle handle, int m, int n, int k,
                         float* A, int lda,
                         const float* tau,
                         float* work, int lwork, int* devInfo)
    int cusolverDnDorgqr(Handle handle, int m, int n, int k,
                         double* A, int lda,
                         const double* tau,
                         double* work, int lwork, int* devInfo)
    int cusolverDnCungqr(Handle handle, int m, int n, int k,
                         cuComplex* A, int lda,
                         const cuComplex* tau,
                         cuComplex* work, int lwork, int* devInfo)
    int cusolverDnZungqr(Handle handle, int m, int n, int k,
                         cuDoubleComplex* A, int lda,
                         const cuDoubleComplex* tau,
                         cuDoubleComplex* work, int lwork, int* devInfo)

    # Compute Q**T*b in solve min||A*x = b||
    int cusolverDnSormqr_bufferSize(Handle handle, SideMode side,
                                    Operation trans, int m, int n, int k,
                                    const float* A, int lda,
                                    const float* tau,
                                    const float* C, int ldc,
                                    int* lwork)
    int cusolverDnDormqr_bufferSize(Handle handle, SideMode side,
                                    Operation trans, int m, int n, int k,
                                    const double* A, int lda,
                                    const double* tau,
                                    const double* C, int ldc,
                                    int* lwork)
    int cusolverDnCunmqr_bufferSize(Handle handle, SideMode side,
                                    Operation trans, int m, int n, int k,
                                    const cuComplex* A, int lda,
                                    const cuComplex* tau,
                                    const cuComplex* C, int ldc,
                                    int* lwork)
    int cusolverDnZunmqr_bufferSize(Handle handle, SideMode side,
                                    Operation trans, int m, int n, int k,
                                    const cuDoubleComplex* A, int lda,
                                    const cuDoubleComplex* tau,
                                    const cuDoubleComplex* C, int ldc,
                                    int* lwork)

    int cusolverDnSormqr(Handle handle, SideMode side, Operation trans,
                         int m, int n, int k,
                         const float* A, int lda,
                         const float* tau,
                         float* C, int ldc, float* work,
                         int lwork, int* devInfo)
    int cusolverDnDormqr(Handle handle, SideMode side, Operation trans,
                         int m, int n, int k,
                         const double* A, int lda,
                         const double* tau,
                         double* C, int ldc, double* work,
                         int lwork, int* devInfo)
    int cusolverDnCunmqr(Handle handle, SideMode side, Operation trans,
                         int m, int n, int k,
                         const cuComplex* A, int lda,
                         const cuComplex* tau,
                         cuComplex* C, int ldc, cuComplex* work,
                         int lwork, int* devInfo)
    int cusolverDnZunmqr(Handle handle, SideMode side, Operation trans,
                         int m, int n, int k,
                         const cuDoubleComplex* A, int lda,
                         const cuDoubleComplex* tau,
                         cuDoubleComplex* C, int ldc, cuDoubleComplex* work,
                         int lwork, int* devInfo)

    # L*D*L**T,U*D*U**T factorization
    int cusolverDnSsytrf_bufferSize(Handle handle, int n,
                                    float* A, int lda, int* lwork)
    int cusolverDnDsytrf_bufferSize(Handle handle, int n,
                                    double* A, int lda, int* lwork)
    int cusolverDnCsytrf_bufferSize(Handle handle, int n,
                                    cuComplex* A, int lda, int* lwork)
    int cusolverDnZsytrf_bufferSize(Handle handle, int n,
                                    cuDoubleComplex* A, int lda, int* lwork)

    int cusolverDnSsytrf(Handle handle, FillMode uplo, int n,
                         float* A, int lda, int* ipiv,
                         float* work, int lwork, int* devInfo)
    int cusolverDnDsytrf(Handle handle, FillMode uplo, int n,
                         double* A, int lda, int* ipiv,
                         double* work, int lwork, int* devInfo)
    int cusolverDnCsytrf(Handle handle, FillMode uplo, int n,
                         cuComplex* A, int lda, int* ipiv,
                         cuComplex* work, int lwork, int* devInfo)
    int cusolverDnZsytrf(Handle handle, FillMode uplo, int n,
                         cuDoubleComplex* A, int lda, int* ipiv,
                         cuDoubleComplex* work, int lwork, int* devInfo)

    # Solve A * X = B using iterative refinement
    int cusolverDnZZgesv_bufferSize(Handle handle, int n, int nrhs,
                                    cuDoubleComplex *dA, int ldda, int *dipiv,
                                    cuDoubleComplex *dB, int lddb,
                                    cuDoubleComplex *dX, int lddx,
                                    void *dWorkspace, size_t *lwork_bytes)
    int cusolverDnZCgesv_bufferSize(Handle handle, int n, int nrhs,
                                    cuDoubleComplex *dA, int ldda, int *dipiv,
                                    cuDoubleComplex *dB, int lddb,
                                    cuDoubleComplex *dX, int lddx,
                                    void *dWorkspace, size_t *lwork_bytes)
    int cusolverDnZYgesv_bufferSize(Handle handle, int n, int nrhs,
                                    cuDoubleComplex *dA, int ldda, int *dipiv,
                                    cuDoubleComplex *dB, int lddb,
                                    cuDoubleComplex *dX, int lddx,
                                    void *dWorkspace, size_t *lwork_bytes)
    int cusolverDnZKgesv_bufferSize(Handle handle, int n, int nrhs,
                                    cuDoubleComplex *dA, int ldda, int *dipiv,
                                    cuDoubleComplex *dB, int lddb,
                                    cuDoubleComplex *dX, int lddx,
                                    void *dWorkspace, size_t *lwork_bytes)
    int cusolverDnCCgesv_bufferSize(Handle handle, int n, int nrhs,
                                    cuComplex *dA, int ldda, int *dipiv,
                                    cuComplex *dB, int lddb,
                                    cuComplex *dX, int lddx,
                                    void *dWorkspace, size_t *lwork_bytes)
    int cusolverDnCYgesv_bufferSize(Handle handle, int n, int nrhs,
                                    cuComplex *dA, int ldda, int *dipiv,
                                    cuComplex *dB, int lddb,
                                    cuComplex *dX, int lddx,
                                    void *dWorkspace, size_t *lwork_bytes)
    int cusolverDnCKgesv_bufferSize(Handle handle, int n, int nrhs,
                                    cuComplex *dA, int ldda, int *dipiv,
                                    cuComplex *dB, int lddb,
                                    cuComplex *dX, int lddx,
                                    void *dWorkspace, size_t *lwork_bytes)
    int cusolverDnDDgesv_bufferSize(Handle handle, int n, int nrhs,
                                    double *dA, int ldda, int *dipiv,
                                    double *dB, int lddb,
                                    double *dX, int lddx,
                                    void *dWorkspace, size_t *lwork_bytes)
    int cusolverDnDSgesv_bufferSize(Handle handle, int n, int nrhs,
                                    double *dA, int ldda, int *dipiv,
                                    double *dB, int lddb,
                                    double *dX, int lddx,
                                    void *dWorkspace, size_t *lwork_bytes)
    int cusolverDnDXgesv_bufferSize(Handle handle, int n, int nrhs,
                                    double *dA, int ldda, int *dipiv,
                                    double *dB, int lddb,
                                    double *dX, int lddx,
                                    void *dWorkspace, size_t *lwork_bytes)
    int cusolverDnDHgesv_bufferSize(Handle handle, int n, int nrhs,
                                    double *dA, int ldda, int *dipiv,
                                    double *dB, int lddb,
                                    double *dX, int lddx,
                                    void *dWorkspace, size_t *lwork_bytes)
    int cusolverDnSSgesv_bufferSize(Handle handle, int n, int nrhs,
                                    float *dA, int ldda, int *dipiv,
                                    float *dB, int lddb,
                                    float *dX, int lddx,
                                    void *dWorkspace, size_t *lwork_bytes)
    int cusolverDnSXgesv_bufferSize(Handle handle, int n, int nrhs,
                                    float *dA, int ldda, int *dipiv,
                                    float *dB, int lddb,
                                    float *dX, int lddx,
                                    void *dWorkspace, size_t *lwork_bytes)
    int cusolverDnSHgesv_bufferSize(Handle handle, int n, int nrhs,
                                    float *dA, int ldda, int *dipiv,
                                    float *dB, int lddb,
                                    float *dX, int lddx,
                                    void *dWorkspace, size_t *lwork_bytes)

    int cusolverDnZZgesv(Handle handle, int n, int nrhs,
                         cuDoubleComplex *dA, int ldda, int *dipiv,
                         cuDoubleComplex *dB, int lddb,
                         cuDoubleComplex *dX, int lddx,
                         void *dWorkspace, size_t lwork_bytes,
                         int *iter, int *dInfo)
    int cusolverDnZCgesv(Handle handle, int n, int nrhs,
                         cuDoubleComplex *dA, int ldda, int *dipiv,
                         cuDoubleComplex *dB, int lddb,
                         cuDoubleComplex *dX, int lddx,
                         void *dWorkspace, size_t lwork_bytes,
                         int *iter, int *dInfo)
    int cusolverDnZYgesv(Handle handle, int n, int nrhs,
                         cuDoubleComplex *dA, int ldda, int *dipiv,
                         cuDoubleComplex *dB, int lddb,
                         cuDoubleComplex *dX, int lddx,
                         void *dWorkspace, size_t lwork_bytes,
                         int *iter, int *dInfo)
    int cusolverDnZKgesv(Handle handle, int n, int nrhs,
                         cuDoubleComplex *dA, int ldda, int *dipiv,
                         cuDoubleComplex *dB, int lddb,
                         cuDoubleComplex *dX, int lddx,
                         void *dWorkspace, size_t lwork_bytes,
                         int *iter, int *dInfo)
    int cusolverDnCCgesv(Handle handle, int n, int nrhs,
                         cuComplex *dA, int ldda, int *dipiv,
                         cuComplex *dB, int lddb,
                         cuComplex *dX, int lddx,
                         void *dWorkspace, size_t lwork_bytes,
                         int *iter, int *dInfo)
    int cusolverDnCYgesv(Handle handle, int n, int nrhs,
                         cuComplex *dA, int ldda, int *dipiv,
                         cuComplex *dB, int lddb,
                         cuComplex *dX, int lddx,
                         void *dWorkspace, size_t lwork_bytes,
                         int *iter, int *dInfo)
    int cusolverDnCKgesv(Handle handle, int n, int nrhs,
                         cuComplex *dA, int ldda, int *dipiv,
                         cuComplex *dB, int lddb,
                         cuComplex *dX, int lddx,
                         void *dWorkspace, size_t lwork_bytes,
                         int *iter, int *dInfo)
    int cusolverDnDDgesv(Handle handle, int n, int nrhs,
                         double *dA, int ldda, int *dipiv,
                         double *dB, int lddb,
                         double *dX, int lddx,
                         void *dWorkspace, size_t lwork_bytes,
                         int *iter, int *dInfo)
    int cusolverDnDSgesv(Handle handle, int n, int nrhs,
                         double *dA, int ldda, int *dipiv,
                         double *dB, int lddb,
                         double *dX, int lddx,
                         void *dWorkspace, size_t lwork_bytes,
                         int *iter, int *dInfo)
    int cusolverDnDXgesv(Handle handle, int n, int nrhs,
                         double *dA, int ldda, int *dipiv,
                         double *dB, int lddb,
                         double *dX, int lddx,
                         void *dWorkspace, size_t lwork_bytes,
                         int *iter, int *dInfo)
    int cusolverDnDHgesv(Handle handle, int n, int nrhs,
                         double *dA, int ldda, int *dipiv,
                         double *dB, int lddb,
                         double *dX, int lddx,
                         void *dWorkspace, size_t lwork_bytes,
                         int *iter, int *dInfo)
    int cusolverDnSSgesv(Handle handle, int n, int nrhs,
                         float *dA, int ldda, int *dipiv,
                         float *dB, int lddb,
                         float *dX, int lddx,
                         void *dWorkspace, size_t lwork_bytes,
                         int *iter, int *dInfo)
    int cusolverDnSXgesv(Handle handle, int n, int nrhs,
                         float *dA, int ldda, int *dipiv,
                         float *dB, int lddb,
                         float *dX, int lddx,
                         void *dWorkspace, size_t lwork_bytes,
                         int *iter, int *dInfo)
    int cusolverDnSHgesv(Handle handle, int n, int nrhs,
                         float *dA, int ldda, int *dipiv,
                         float *dB, int lddb,
                         float *dX, int lddx,
                         void *dWorkspace, size_t lwork_bytes,
                         int *iter, int *dInfo)

    # Compute least square solution to A * X = B using iterative refinement
    int cusolverDnZZgels_bufferSize(Handle handle, int m, int n, int nrhs,
                                    cuDoubleComplex *dA, int ldda,
                                    cuDoubleComplex *dB, int lddb,
                                    cuDoubleComplex *dX, int lddx,
                                    void *dWorkspace, size_t *lwork_bytes)
    int cusolverDnZCgels_bufferSize(Handle handle, int m, int n, int nrhs,
                                    cuDoubleComplex *dA, int ldda,
                                    cuDoubleComplex *dB, int lddb,
                                    cuDoubleComplex *dX, int lddx,
                                    void *dWorkspace, size_t *lwork_bytes)
    int cusolverDnZYgels_bufferSize(Handle handle, int m, int n, int nrhs,
                                    cuDoubleComplex *dA, int ldda,
                                    cuDoubleComplex *dB, int lddb,
                                    cuDoubleComplex *dX, int lddx,
                                    void *dWorkspace, size_t *lwork_bytes)
    int cusolverDnZKgels_bufferSize(Handle handle, int m, int n, int nrhs,
                                    cuDoubleComplex *dA, int ldda,
                                    cuDoubleComplex *dB, int lddb,
                                    cuDoubleComplex *dX, int lddx,
                                    void *dWorkspace, size_t *lwork_bytes)
    int cusolverDnCCgels_bufferSize(Handle handle, int m, int n, int nrhs,
                                    cuComplex *dA, int ldda,
                                    cuComplex *dB, int lddb,
                                    cuComplex *dX, int lddx,
                                    void *dWorkspace, size_t *lwork_bytes)
    int cusolverDnCYgels_bufferSize(Handle handle, int m, int n, int nrhs,
                                    cuComplex *dA, int ldda,
                                    cuComplex *dB, int lddb,
                                    cuComplex *dX, int lddx,
                                    void *dWorkspace, size_t *lwork_bytes)
    int cusolverDnCKgels_bufferSize(Handle handle, int m, int n, int nrhs,
                                    cuComplex *dA, int ldda,
                                    cuComplex *dB, int lddb,
                                    cuComplex *dX, int lddx,
                                    void *dWorkspace, size_t *lwork_bytes)
    int cusolverDnDDgels_bufferSize(Handle handle, int m, int n, int nrhs,
                                    double *dA, int ldda,
                                    double *dB, int lddb,
                                    double *dX, int lddx,
                                    void *dWorkspace, size_t *lwork_bytes)
    int cusolverDnDSgels_bufferSize(Handle handle, int m, int n, int nrhs,
                                    double *dA, int ldda,
                                    double *dB, int lddb,
                                    double *dX, int lddx,
                                    void *dWorkspace, size_t *lwork_bytes)
    int cusolverDnDXgels_bufferSize(Handle handle, int m, int n, int nrhs,
                                    double *dA, int ldda,
                                    double *dB, int lddb,
                                    double *dX, int lddx,
                                    void *dWorkspace, size_t *lwork_bytes)
    int cusolverDnDHgels_bufferSize(Handle handle, int m, int n, int nrhs,
                                    double *dA, int ldda,
                                    double *dB, int lddb,
                                    double *dX, int lddx,
                                    void *dWorkspace, size_t *lwork_bytes)
    int cusolverDnSSgels_bufferSize(Handle handle, int m, int n, int nrhs,
                                    float *dA, int ldda,
                                    float *dB, int lddb,
                                    float *dX, int lddx,
                                    void *dWorkspace, size_t *lwork_bytes)
    int cusolverDnSXgels_bufferSize(Handle handle, int m, int n, int nrhs,
                                    float *dA, int ldda,
                                    float *dB, int lddb,
                                    float *dX, int lddx,
                                    void *dWorkspace, size_t *lwork_bytes)
    int cusolverDnSHgels_bufferSize(Handle handle, int m, int n, int nrhs,
                                    float *dA, int ldda,
                                    float *dB, int lddb,
                                    float *dX, int lddx,
                                    void *dWorkspace, size_t *lwork_bytes)

    int cusolverDnZZgels(Handle handle, int m, int n, int nrhs,
                         cuDoubleComplex *dA, int ldda,
                         cuDoubleComplex *dB, int lddb,
                         cuDoubleComplex *dX, int lddx,
                         void *dWorkspace, size_t lwork_bytes,
                         int *iter, int *dInfo)
    int cusolverDnZCgels(Handle handle, int m, int n, int nrhs,
                         cuDoubleComplex *dA, int ldda,
                         cuDoubleComplex *dB, int lddb,
                         cuDoubleComplex *dX, int lddx,
                         void *dWorkspace, size_t lwork_bytes,
                         int *iter, int *dInfo)
    int cusolverDnZYgels(Handle handle, int m, int n, int nrhs,
                         cuDoubleComplex *dA, int ldda,
                         cuDoubleComplex *dB, int lddb,
                         cuDoubleComplex *dX, int lddx,
                         void *dWorkspace, size_t lwork_bytes,
                         int *iter, int *dInfo)
    int cusolverDnZKgels(Handle handle, int m, int n, int nrhs,
                         cuDoubleComplex *dA, int ldda,
                         cuDoubleComplex *dB, int lddb,
                         cuDoubleComplex *dX, int lddx,
                         void *dWorkspace, size_t lwork_bytes,
                         int *iter, int *dInfo)
    int cusolverDnCCgels(Handle handle, int m, int n, int nrhs,
                         cuComplex *dA, int ldda,
                         cuComplex *dB, int lddb,
                         cuComplex *dX, int lddx,
                         void *dWorkspace, size_t lwork_bytes,
                         int *iter, int *dInfo)
    int cusolverDnCYgels(Handle handle, int m, int n, int nrhs,
                         cuComplex *dA, int ldda,
                         cuComplex *dB, int lddb,
                         cuComplex *dX, int lddx,
                         void *dWorkspace, size_t lwork_bytes,
                         int *iter, int *dInfo)
    int cusolverDnCKgels(Handle handle, int m, int n, int nrhs,
                         cuComplex *dA, int ldda,
                         cuComplex *dB, int lddb,
                         cuComplex *dX, int lddx,
                         void *dWorkspace, size_t lwork_bytes,
                         int *iter, int *dInfo)
    int cusolverDnDDgels(Handle handle, int m, int n, int nrhs,
                         double *dA, int ldda,
                         double *dB, int lddb,
                         double *dX, int lddx,
                         void *dWorkspace, size_t lwork_bytes,
                         int *iter, int *dInfo)
    int cusolverDnDSgels(Handle handle, int m, int n, int nrhs,
                         double *dA, int ldda,
                         double *dB, int lddb,
                         double *dX, int lddx,
                         void *dWorkspace, size_t lwork_bytes,
                         int *iter, int *dInfo)
    int cusolverDnDXgels(Handle handle, int m, int n, int nrhs,
                         double *dA, int ldda,
                         double *dB, int lddb,
                         double *dX, int lddx,
                         void *dWorkspace, size_t lwork_bytes,
                         int *iter, int *dInfo)
    int cusolverDnDHgels(Handle handle, int m, int n, int nrhs,
                         double *dA, int ldda,
                         double *dB, int lddb,
                         double *dX, int lddx,
                         void *dWorkspace, size_t lwork_bytes,
                         int *iter, int *dInfo)
    int cusolverDnSSgels(Handle handle, int m, int n, int nrhs,
                         float *dA, int ldda,
                         float *dB, int lddb,
                         float *dX, int lddx,
                         void *dWorkspace, size_t lwork_bytes,
                         int *iter, int *dInfo)
    int cusolverDnSXgels(Handle handle, int m, int n, int nrhs,
                         float *dA, int ldda,
                         float *dB, int lddb,
                         float *dX, int lddx,
                         void *dWorkspace, size_t lwork_bytes,
                         int *iter, int *dInfo)
    int cusolverDnSHgels(Handle handle, int m, int n, int nrhs,
                         float *dA, int ldda,
                         float *dB, int lddb,
                         float *dX, int lddx,
                         void *dWorkspace, size_t lwork_bytes,
                         int *iter, int *dInfo)

    ###########################################################################
    # Dense LAPACK Functions (Eigenvalue Solver)
    ###########################################################################

    # Bidiagonal factorization
    int cusolverDnSgebrd_bufferSize(Handle handle, int m, int n, int* lwork)
    int cusolverDnDgebrd_bufferSize(Handle handle, int m, int n, int* lwork)
    int cusolverDnCgebrd_bufferSize(Handle handle, int m, int n, int* lwork)
    int cusolverDnZgebrd_bufferSize(Handle handle, int m, int n, int* lwork)

    int cusolverDnSgebrd(Handle handle, int m, int n,
                         float* A, int lda,
                         float* D, float* E,
                         float* tauQ, float* tauP,
                         float* Work, int lwork, int* devInfo)
    int cusolverDnDgebrd(Handle handle, int m, int n,
                         double* A, int lda,
                         double* D, double* E,
                         double* tauQ, double* tauP,
                         double* Work, int lwork, int* devInfo)
    int cusolverDnCgebrd(Handle handle, int m, int n,
                         cuComplex* A, int lda,
                         float* D, float* E,
                         cuComplex* tauQ, cuComplex* tauP,
                         cuComplex* Work, int lwork, int* devInfo)
    int cusolverDnZgebrd(Handle handle, int m, int n,
                         cuDoubleComplex* A, int lda,
                         double* D, double* E,
                         cuDoubleComplex* tauQ, cuDoubleComplex* tauP,
                         cuDoubleComplex* Work, int lwork, int* devInfo)

    # Singular value decomposition, A = U * Sigma * V^H
    int cusolverDnSgesvd_bufferSize(Handle handle, int m, int n, int* lwork)
    int cusolverDnDgesvd_bufferSize(Handle handle, int m, int n, int* lwork)
    int cusolverDnCgesvd_bufferSize(Handle handle, int m, int n, int* lwork)
    int cusolverDnZgesvd_bufferSize(Handle handle, int m, int n, int* lwork)

    int cusolverDnSgesvd(Handle handle, char jobu, char jobvt, int m, int n,
                         float* A, int lda, float* S,
                         float* U, int ldu,
                         float* VT, int ldvt,
                         float* Work, int lwork,
                         float* rwork, int* devInfo)
    int cusolverDnDgesvd(Handle handle, char jobu, char jobvt, int m, int n,
                         double* A, int lda, double* S,
                         double* U, int ldu,
                         double* VT, int ldvt,
                         double* Work, int lwork,
                         double* rwork, int* devInfo)
    int cusolverDnCgesvd(Handle handle, char jobu, char jobvt, int m, int n,
                         cuComplex* A, int lda, float* S,
                         cuComplex* U, int ldu,
                         cuComplex* VT, int ldvt,
                         cuComplex* Work, int lwork,
                         float* rwork, int* devInfo)
    int cusolverDnZgesvd(Handle handle, char jobu, char jobvt, int m, int n,
                         cuDoubleComplex* A, int lda, double* S,
                         cuDoubleComplex* U, int ldu,
                         cuDoubleComplex* VT, int ldvt,
                         cuDoubleComplex* Work, int lwork,
                         double* rwork, int* devInfo)

    # gesvdj ... Singular value decomposition using Jacobi mathod
    int cusolverDnCreateGesvdjInfo(GesvdjInfo *info)
    int cusolverDnDestroyGesvdjInfo(GesvdjInfo info)

    int cusolverDnXgesvdjSetTolerance(GesvdjInfo info, double tolerance)
    int cusolverDnXgesvdjSetMaxSweeps(GesvdjInfo info, int max_sweeps)
    int cusolverDnXgesvdjSetSortEig(GesvdjInfo info, int sort_svd)
    int cusolverDnXgesvdjGetResidual(Handle handle, GesvdjInfo info,
                                     double* residual)
    int cusolverDnXgesvdjGetSweeps(Handle handle, GesvdjInfo info,
                                   int* executed_sweeps)

    int cusolverDnSgesvdj_bufferSize(Handle handle, EigMode jobz, int econ,
                                     int m, int n, const float* A, int lda,
                                     const float* S, const float* U, int ldu,
                                     const float* V, int ldv, int* lwork,
                                     GesvdjInfo params)
    int cusolverDnDgesvdj_bufferSize(Handle handle, EigMode jobz, int econ,
                                     int m, int n, const double* A, int lda,
                                     const double* S, const double* U, int ldu,
                                     const double* V, int ldv, int* lwork,
                                     GesvdjInfo params)
    int cusolverDnCgesvdj_bufferSize(Handle handle, EigMode jobz, int econ,
                                     int m, int n, const cuComplex* A, int lda,
                                     const float* S, const cuComplex* U,
                                     int ldu, const cuComplex* V, int ldv,
                                     int* lwork, GesvdjInfo params)
    int cusolverDnZgesvdj_bufferSize(Handle handle, EigMode jobz, int econ,
                                     int m, int n, const cuDoubleComplex* A,
                                     int lda, const double* S,
                                     const cuDoubleComplex* U, int ldu,
                                     const cuDoubleComplex* V, int ldv,
                                     int* lwork, GesvdjInfo params)

    int cusolverDnSgesvdj(Handle handle, EigMode jobz, int econ, int m, int n,
                          float *A, int lda, float *S, float *U, int ldu,
                          float *V, int ldv, float *work, int lwork, int *info,
                          GesvdjInfo params)
    int cusolverDnDgesvdj(Handle handle, EigMode jobz, int econ, int m, int n,
                          double *A, int lda, double *S, double *U, int ldu,
                          double *V, int ldv, double *work, int lwork,
                          int *info, GesvdjInfo params)
    int cusolverDnCgesvdj(Handle handle, EigMode jobz, int econ, int m, int n,
                          cuComplex *A, int lda, float *S, cuComplex *U,
                          int ldu, cuComplex *V, int ldv, cuComplex *work,
                          int lwork, int *info, GesvdjInfo params)
    int cusolverDnZgesvdj(Handle handle, EigMode jobz, int econ, int m, int n,
                          cuDoubleComplex *A, int lda, double *S,
                          cuDoubleComplex *U, int ldu, cuDoubleComplex *V,
                          int ldv, cuDoubleComplex *work, int lwork, int *info,
                          GesvdjInfo params)

    int cusolverDnSgesvdjBatched_bufferSize(
        Handle handle, EigMode jobz, int m, int n, float* A, int lda,
        float* S, float* U, int ldu, float* V, int ldv,
        int* lwork, GesvdjInfo params, int batchSize)
    int cusolverDnDgesvdjBatched_bufferSize(
        Handle handle, EigMode jobz, int m, int n, double* A, int lda,
        double* S, double* U, int ldu, double* V, int ldv,
        int* lwork, GesvdjInfo params, int batchSize)
    int cusolverDnCgesvdjBatched_bufferSize(
        Handle handle, EigMode jobz, int m, int n, cuComplex* A, int lda,
        float* S, cuComplex* U, int ldu, cuComplex* V, int ldv,
        int* lwork, GesvdjInfo params, int batchSize)
    int cusolverDnZgesvdjBatched_bufferSize(
        Handle handle, EigMode jobz, int m, int n, cuDoubleComplex* A, int lda,
        double* S, cuDoubleComplex* U, int ldu, cuDoubleComplex* V, int ldv,
        int* lwork, GesvdjInfo params, int batchSize)
    int cusolverDnSgesvdjBatched(
        Handle handle, EigMode jobz, int m, int n, float* A, int lda, float* S,
        float* U, int ldu, float* V, int ldv, float* work, int lwork,
        int* info, GesvdjInfo params, int batchSize)
    int cusolverDnDgesvdjBatched(
        Handle handle, EigMode jobz, int m, int n, double* A, int lda,
        double* S, double* U, int ldu, double* V, int ldv,
        double* work, int lwork,
        int* info, GesvdjInfo params, int batchSize)
    int cusolverDnCgesvdjBatched(
        Handle handle, EigMode jobz, int m, int n, cuComplex* A, int lda,
        float* S, cuComplex* U, int ldu, cuComplex* V, int ldv,
        cuComplex* work, int lwork,
        int* info, GesvdjInfo params, int batchSize)
    int cusolverDnZgesvdjBatched(
        Handle handle, EigMode jobz, int m, int n, cuDoubleComplex* A, int lda,
        double* S, cuDoubleComplex* U, int ldu, cuDoubleComplex* V, int ldv,
        cuDoubleComplex* work, int lwork,
        int* info, GesvdjInfo params, int batchSize)

    # gesvda ... Approximate singular value decomposition
    int cusolverDnSgesvdaStridedBatched_bufferSize(
        Handle handle, EigMode jobz, int rank, int m, int n, const float *d_A,
        int lda, long long int strideA, const float *d_S,
        long long int strideS, const float *d_U, int ldu,
        long long int strideU, const float *d_V, int ldv,
        long long int strideV, int *lwork, int batchSize)

    int cusolverDnDgesvdaStridedBatched_bufferSize(
        Handle handle, EigMode jobz, int rank, int m, int n, const double *d_A,
        int lda, long long int strideA, const double *d_S,
        long long int strideS, const double *d_U, int ldu,
        long long int strideU, const double *d_V, int ldv,
        long long int strideV, int *lwork, int batchSize)

    int cusolverDnCgesvdaStridedBatched_bufferSize(
        Handle handle, EigMode jobz, int rank, int m, int n,
        const cuComplex *d_A, int lda, long long int strideA, const float *d_S,
        long long int strideS, const cuComplex *d_U, int ldu,
        long long int strideU, const cuComplex *d_V, int ldv,
        long long int strideV, int *lwork, int batchSize)

    int cusolverDnZgesvdaStridedBatched_bufferSize(
        Handle handle, EigMode jobz, int rank, int m, int n,
        const cuDoubleComplex *d_A, int lda, long long int strideA,
        const double *d_S, long long int strideS, const cuDoubleComplex *d_U,
        int ldu, long long int strideU, const cuDoubleComplex *d_V, int ldv,
        long long int strideV, int *lwork, int batchSize)

    int cusolverDnSgesvdaStridedBatched(
        Handle handle, EigMode jobz, int rank, int m, int n, const float *d_A,
        int lda, long long int strideA, float *d_S, long long int strideS,
        float *d_U, int ldu, long long int strideU, float *d_V, int ldv,
        long long int strideV, float *d_work, int lwork, int *d_info,
        double *h_R_nrmF, int batchSize)

    int cusolverDnDgesvdaStridedBatched(
        Handle handle, EigMode jobz, int rank, int m, int n, const double *d_A,
        int lda, long long int strideA, double *d_S, long long int strideS,
        double *d_U, int ldu, long long int strideU, double *d_V, int ldv,
        long long int strideV, double *d_work, int lwork, int *d_info,
        double *h_R_nrmF, int batchSize)

    int cusolverDnCgesvdaStridedBatched(
        Handle handle, EigMode jobz, int rank, int m, int n,
        const cuComplex *d_A, int lda, long long int strideA, float *d_S,
        long long int strideS, cuComplex *d_U, int ldu, long long int strideU,
        cuComplex *d_V, int ldv, long long int strideV, cuComplex *d_work,
        int lwork, int *d_info, double *h_R_nrmF, int batchSize)

    int cusolverDnZgesvdaStridedBatched(
        Handle handle, EigMode jobz, int rank, int m, int n,
        const cuDoubleComplex *d_A, int lda, long long int strideA,
        double *d_S, long long int strideS, cuDoubleComplex *d_U, int ldu,
        long long int strideU, cuDoubleComplex *d_V, int ldv,
        long long int strideV, cuDoubleComplex *d_work, int lwork, int *d_info,
        double *h_R_nrmF, int batchSize)

    # Standard symmetric eigenvalue solver
    int cusolverDnSsyevd_bufferSize(Handle handle,
                                    EigMode jobz, FillMode uplo, int n,
                                    const float* A, int lda,
                                    const float* W, int* lwork)
    int cusolverDnDsyevd_bufferSize(Handle handle,
                                    EigMode jobz, FillMode uplo, int n,
                                    const double* A, int lda,
                                    const double* W, int* lwork)
    int cusolverDnCheevd_bufferSize(Handle handle,
                                    EigMode jobz, FillMode uplo, int n,
                                    const cuComplex* A, int lda,
                                    const float* W, int* lwork)
    int cusolverDnZheevd_bufferSize(Handle handle,
                                    EigMode jobz, FillMode uplo, int n,
                                    const cuDoubleComplex* A, int lda,
                                    const double* W, int* lwork)

    int cusolverDnSsyevd(Handle handle, EigMode jobz, FillMode uplo, int n,
                         float* A, int lda, float* W,
                         float* work, int lwork, int* info)
    int cusolverDnDsyevd(Handle handle, EigMode jobz, FillMode uplo, int n,
                         double* A, int lda, double* W,
                         double* work, int lwork, int* info)
    int cusolverDnCheevd(Handle handle, EigMode jobz, FillMode uplo, int n,
                         cuComplex* A, int lda, float* W,
                         cuComplex* work, int lwork, int* info)
    int cusolverDnZheevd(Handle handle, EigMode jobz, FillMode uplo, int n,
                         cuDoubleComplex* A, int lda, double* W,
                         cuDoubleComplex* work, int lwork, int* info)

    # Symmetric eigenvalue solver using Jacobi method
    int cusolverDnCreateSyevjInfo(SyevjInfo *info)
    int cusolverDnDestroySyevjInfo(SyevjInfo info)

    int cusolverDnXsyevjSetTolerance(SyevjInfo info, double tolerance)
    int cusolverDnXsyevjSetMaxSweeps(SyevjInfo info, int max_sweeps)
    int cusolverDnXsyevjSetSortEig(SyevjInfo info, int sort_eig)
    int cusolverDnXsyevjGetResidual(
        Handle handle, SyevjInfo info, double* residual)
    int cusolverDnXsyevjGetSweeps(
        Handle handle, SyevjInfo info, int* executed_sweeps)

    int cusolverDnSsyevj_bufferSize(
        Handle handle, EigMode jobz, FillMode uplo, int n,
        const float *A, int lda, const float *W, int *lwork,
        SyevjInfo params)
    int cusolverDnDsyevj_bufferSize(
        Handle handle, EigMode jobz, FillMode uplo, int n,
        const double *A, int lda, const double *W, int *lwork,
        SyevjInfo params)
    int cusolverDnCheevj_bufferSize(
        Handle handle, EigMode jobz, FillMode uplo, int n,
        const cuComplex *A, int lda, const float *W, int *lwork,
        SyevjInfo params)
    int cusolverDnZheevj_bufferSize(
        Handle handle, EigMode jobz, FillMode uplo, int n,
        const cuDoubleComplex *A, int lda, const double *W, int *lwork,
        SyevjInfo params)

    int cusolverDnSsyevj(
        Handle handle, EigMode jobz, FillMode uplo, int n,
        float *A, int lda, float *W, float *work,
        int lwork, int *info, SyevjInfo params)
    int cusolverDnDsyevj(
        Handle handle, EigMode jobz, FillMode uplo, int n,
        double *A, int lda, double *W, double *work,
        int lwork, int *info, SyevjInfo params)
    int cusolverDnCheevj(
        Handle handle, EigMode jobz, FillMode uplo, int n,
        cuComplex *A, int lda, float *W, cuComplex *work,
        int lwork, int *info, SyevjInfo params)
    int cusolverDnZheevj(
        Handle handle, EigMode jobz, FillMode uplo, int n,
        cuDoubleComplex *A, int lda, double *W, cuDoubleComplex *work,
        int lwork, int *info, SyevjInfo params)

    int cusolverDnSsyevjBatched_bufferSize(
        Handle handle, EigMode jobz, FillMode uplo, int n,
        const float *A, int lda, const float *W, int *lwork,
        SyevjInfo params, int batchSize)

    int cusolverDnDsyevjBatched_bufferSize(
        Handle handle, EigMode jobz, FillMode uplo, int n,
        const double *A, int lda, const double *W, int *lwork,
        SyevjInfo params, int batchSize)

    int cusolverDnCheevjBatched_bufferSize(
        Handle handle, EigMode jobz, FillMode uplo, int n,
        const cuComplex *A, int lda, const float *W, int *lwork,
        SyevjInfo params, int batchSize)

    int cusolverDnZheevjBatched_bufferSize(
        Handle handle, EigMode jobz, FillMode uplo, int n,
        const cuDoubleComplex *A, int lda, const double *W, int *lwork,
        SyevjInfo params, int batchSize)

    int cusolverDnSsyevjBatched(
        Handle handle, EigMode jobz, FillMode uplo, int n,
        float *A, int lda, float *W, float *work, int lwork,
        int *info, SyevjInfo params, int batchSize)

    int cusolverDnDsyevjBatched(
        Handle handle, EigMode jobz, FillMode uplo, int n,
        double *A, int lda, double *W, double *work, int lwork,
        int *info, SyevjInfo params, int batchSize)

    int cusolverDnCheevjBatched(
        Handle handle, EigMode jobz, FillMode uplo, int n,
        cuComplex *A, int lda, float *W, cuComplex *work, int lwork,
        int *info, SyevjInfo params, int batchSize)

    int cusolverDnZheevjBatched(
        Handle handle, EigMode jobz, FillMode uplo, int n,
        cuDoubleComplex *A, int lda, double *W, cuDoubleComplex *work,
        int lwork, int *info, SyevjInfo params, int batchSize)

    ###########################################################################
    # Sparse LAPACK Functions
    ###########################################################################

    int cusolverSpScsrlsvchol(
        SpHandle handle, int m, int nnz, const MatDescr descrA,
        const float* csrValA, const int* csrRowPtrA, const int* csrColIndA,
        const float* b, float tol, int reorder, float* x, int* singularity)
    int cusolverSpDcsrlsvchol(
        SpHandle handle, int m, int nnz, const MatDescr descrA,
        const double* csrValA, const int* csrRowPtrA, const int* csrColIndA,
        const double* b, double tol, int reorder, double* x, int* singularity)
    int cusolverSpCcsrlsvchol(
        SpHandle handle, int m, int nnz,
        const MatDescr descrA, const cuComplex *csrVal,
        const int *csrRowPtr, const int *csrColInd, const cuComplex *b,
        float tol, int reorder, cuComplex *x, int *singularity)
    int cusolverSpZcsrlsvchol(
        SpHandle handle, int m, int nnz,
        const MatDescr descrA, const cuDoubleComplex *csrVal,
        const int *csrRowPtr, const int *csrColInd, const cuDoubleComplex *b,
        double tol, int reorder, cuDoubleComplex *x, int *singularity)

    int cusolverSpScsrlsvqr(
        SpHandle handle, int m, int nnz, const MatDescr descrA,
        const float* csrValA, const int* csrRowPtrA, const int* csrColIndA,
        const float* b, float tol, int reorder, float* x, int* singularity)
    int cusolverSpDcsrlsvqr(
        SpHandle handle, int m, int nnz, const MatDescr descrA,
        const double* csrValA, const int* csrRowPtrA, const int* csrColIndA,
        const double* b, double tol, int reorder, double* x, int* singularity)
    int cusolverSpCcsrlsvqr(
        SpHandle handle, int m, int nnz,
        const MatDescr descrA, const cuComplex *csrVal,
        const int *csrRowPtr, const int *csrColInd, const cuComplex *b,
        float tol, int reorder, cuComplex *x, int *singularity)
    int cusolverSpZcsrlsvqr(
        SpHandle handle, int m, int nnz,
        const MatDescr descrA, const cuDoubleComplex *csrVal,
        const int *csrRowPtr, const int *csrColInd, const cuDoubleComplex *b,
        double tol, int reorder, cuDoubleComplex *x, int *singularity)

    int cusolverSpScsreigvsi(
        SpHandle handle, int m, int nnz,
        const MatDescr descrA, const float *csrValA,
        const int *csrRowPtrA, const int *csrColIndA, float mu0,
        const float *x0, int maxite, float eps, float *mu, float *x)
    int cusolverSpDcsreigvsi(
        SpHandle handle, int m, int nnz,
        const MatDescr descrA, const double *csrValA,
        const int *csrRowPtrA, const int *csrColIndA, double mu0,
        const double *x0, int maxite, double eps, double *mu, double *x)
    int cusolverSpCcsreigvsi(
        SpHandle handle, int m, int nnz,
        const MatDescr descrA, const cuComplex *csrValA,
        const int *csrRowPtrA, const int *csrColIndA, cuComplex mu0,
        const cuComplex *x0, int maxite, float eps, cuComplex *mu,
        cuComplex *x)
    int cusolverSpZcsreigvsi(
        SpHandle handle, int m, int nnz,
        const MatDescr descrA, const cuDoubleComplex *csrValA,
        const int *csrRowPtrA, const int *csrColIndA, cuDoubleComplex mu0,
        const cuDoubleComplex *x0, int maxite, double eps, cuDoubleComplex *mu,
        cuDoubleComplex *x)

###############################################################################
# Error handling
###############################################################################

cdef dict STATUS = {
    0: 'CUSOLVER_STATUS_SUCCESS',
    1: 'CUSOLVER_STATUS_NOT_INITIALIZED',
    2: 'CUSOLVER_STATUS_ALLOC_FAILED',
    3: 'CUSOLVER_STATUS_INVALID_VALUE',
    4: 'CUSOLVER_STATUS_ARCH_MISMATCH',
    5: 'CUSOLVER_STATUS_MAPPING_ERROR',
    6: 'CUSOLVER_STATUS_EXECUTION_FAILED',
    7: 'CUSOLVER_STATUS_INTERNAL_ERROR',
    8: 'CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED',
    9: 'CUSOLVER_STATUS_NOT_SUPPORTED',
    10: 'CUSOLVER_STATUS_ZERO_PIVOT',
    11: 'CUSOLVER_STATUS_INVALID_LICENSE',
    12: 'CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED',
    13: 'CUSOLVER_STATUS_IRS_PARAMS_INVALID',
    14: 'CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC',
    15: 'CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE',
    16: 'CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER',
    20: 'CUSOLVER_STATUS_IRS_INTERNAL_ERROR',
    21: 'CUSOLVER_STATUS_IRS_NOT_SUPPORTED',
    22: 'CUSOLVER_STATUS_IRS_OUT_OF_RANGE',
    23: 'CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES',
    25: 'CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED',
    26: 'CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED',
    30: 'CUSOLVER_STATUS_IRS_MATRIX_SINGULAR',
    31: 'CUSOLVER_STATUS_INVALID_WORKSPACE',
}

# for rocBLAS and rocSOLVER
cdef dict ROC_STATUS = {
    0: 'rocblas_status_success',
    1: 'rocblas_status_invalid_handle',
    2: 'rocblas_status_not_implemented',
    3: 'rocblas_status_invalid_pointer',
    4: 'rocblas_status_invalid_size',
    5: 'rocblas_status_memory_error',
    6: 'rocblas_status_internal_error',
    7: 'rocblas_status_perf_degraded',
    8: 'rocblas_status_size_query_mismatch',
    9: 'rocblas_status_size_increased',
    10: 'rocblas_status_size_unchanged',
    11: 'rocblas_status_invalid_value',
    12: 'rocblas_status_continue',
}


class CUSOLVERError(RuntimeError):

    def __init__(self, status):
        self.status = status
        if runtime._is_hip_environment:
            err = ROC_STATUS
        else:
            err = STATUS
        super(CUSOLVERError, self).__init__(err[status])

    def __reduce__(self):
        return (type(self), (self.status,))


@cython.profile(False)
cpdef inline check_status(int status):
    if status != 0:
        raise CUSOLVERError(status)


###############################################################################
# Library Attributes
###############################################################################

cpdef int getProperty(int type) except? -1:
    cdef int value
    with nogil:
        status = cusolverGetProperty(<LibraryPropertyType>type, &value)
    check_status(status)
    return value


cpdef tuple _getVersion():
    return (getProperty(MAJOR_VERSION),
            getProperty(MINOR_VERSION),
            getProperty(PATCH_LEVEL))


###############################################################################
# Context
###############################################################################

cpdef intptr_t create() except? 0:
    cdef Handle handle
    with nogil:
        status = cusolverDnCreate(&handle)
    check_status(status)
    return <intptr_t>handle


cpdef intptr_t spCreate() except? 0:
    cdef SpHandle handle
    with nogil:
        status = cusolverSpCreate(&handle)
    check_status(status)
    return <intptr_t>handle


cpdef destroy(intptr_t handle):
    with nogil:
        status = cusolverDnDestroy(<Handle>handle)
    check_status(status)


cpdef spDestroy(intptr_t handle):
    with nogil:
        status = cusolverSpDestroy(<SpHandle>handle)
    check_status(status)


###############################################################################
# Stream
###############################################################################

cpdef setStream(intptr_t handle, size_t stream):
    with nogil:
        status = cusolverDnSetStream(<Handle>handle, <driver.Stream>stream)
    check_status(status)


cpdef size_t getStream(intptr_t handle) except? 0:
    cdef driver.Stream stream
    with nogil:
        status = cusolverDnGetStream(<Handle>handle, &stream)
    check_status(status)
    return <size_t>stream


cpdef spSetStream(intptr_t handle, size_t stream):
    with nogil:
        status = cusolverSpSetStream(<SpHandle>handle, <driver.Stream>stream)
    check_status(status)


cpdef size_t spGetStream(intptr_t handle) except *:
    cdef driver.Stream stream
    with nogil:
        status = cusolverSpGetStream(<SpHandle>handle, &stream)
    check_status(status)
    return <size_t>stream


cdef _setStream(intptr_t handle):
    """Set current stream"""
    setStream(handle, stream_module.get_current_stream_ptr())


cdef _spSetStream(intptr_t handle):
    """Set current stream"""
    spSetStream(handle, stream_module.get_current_stream_ptr())

###########################################################################
# Dense LAPACK Functions (Linear Solver)
###########################################################################

# Cholesky factorization
cpdef int spotrf_bufferSize(intptr_t handle, int uplo,
                            int n, size_t A, int lda) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnSpotrf_bufferSize(
            <Handle>handle, <FillMode>uplo, n,
            <float*>A, <int>lda, &lwork)
    check_status(status)
    return lwork

cpdef int dpotrf_bufferSize(intptr_t handle, int uplo,
                            int n, size_t A, int lda) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnDpotrf_bufferSize(
            <Handle>handle, <FillMode>uplo, n,
            <double*>A, <int>lda, &lwork)
    check_status(status)
    return lwork

cpdef int cpotrf_bufferSize(intptr_t handle, int uplo,
                            int n, size_t A, int lda) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnCpotrf_bufferSize(
            <Handle>handle, <FillMode>uplo, n,
            <cuComplex*>A, <int>lda, &lwork)
    check_status(status)
    return lwork

cpdef int zpotrf_bufferSize(intptr_t handle, int uplo,
                            int n, size_t A, int lda) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnZpotrf_bufferSize(
            <Handle>handle, <FillMode>uplo, n,
            <cuDoubleComplex*>A, <int>lda, &lwork)
    check_status(status)
    return lwork

cpdef spotrf(intptr_t handle, int uplo, int n, size_t A, int lda,
             size_t work, int lwork, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = cusolverDnSpotrf(
            <Handle>handle, <FillMode>uplo, n, <float*>A,
            lda, <float*>work, lwork, <int*>devInfo)
    check_status(status)

cpdef dpotrf(intptr_t handle, int uplo, int n, size_t A, int lda,
             size_t work, int lwork, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = cusolverDnDpotrf(
            <Handle>handle, <FillMode>uplo, n, <double*>A,
            lda, <double*>work, lwork, <int*>devInfo)
    check_status(status)

cpdef cpotrf(intptr_t handle, int uplo, int n, size_t A, int lda,
             size_t work, int lwork, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = cusolverDnCpotrf(
            <Handle>handle, <FillMode>uplo, n, <cuComplex*>A,
            lda, <cuComplex*>work, lwork, <int*>devInfo)
    check_status(status)

cpdef zpotrf(intptr_t handle, int uplo, int n, size_t A, int lda,
             size_t work, int lwork, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = cusolverDnZpotrf(
            <Handle>handle, <FillMode>uplo, n, <cuDoubleComplex*>A,
            lda, <cuDoubleComplex*>work, lwork, <int*>devInfo)
    check_status(status)

cpdef spotrs(intptr_t handle, int uplo, int n, int nrhs,
             size_t A, int lda, size_t B, int ldb, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = cusolverDnSpotrs(
            <Handle>handle, <FillMode>uplo, n, nrhs,
            <const float*>A, lda, <float*>B, ldb,
            <int*>devInfo)
    check_status(status)

cpdef dpotrs(intptr_t handle, int uplo, int n, int nrhs,
             size_t A, int lda, size_t B, int ldb, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = cusolverDnDpotrs(
            <Handle>handle, <FillMode>uplo, n, nrhs,
            <const double*>A, lda, <double*>B, ldb,
            <int*>devInfo)
    check_status(status)

cpdef cpotrs(intptr_t handle, int uplo, int n, int nrhs,
             size_t A, int lda, size_t B, int ldb, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = cusolverDnCpotrs(
            <Handle>handle, <FillMode>uplo, n, nrhs,
            <const cuComplex*>A, lda, <cuComplex*>B, ldb,
            <int*>devInfo)
    check_status(status)

cpdef zpotrs(intptr_t handle, int uplo, int n, int nrhs,
             size_t A, int lda, size_t B, int ldb, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = cusolverDnZpotrs(
            <Handle>handle, <FillMode>uplo, n, nrhs,
            <const cuDoubleComplex*>A, lda, <cuDoubleComplex*>B, ldb,
            <int*>devInfo)
    check_status(status)

cpdef spotrfBatched(intptr_t handle, int uplo, int n, size_t Aarray, int lda,
                    size_t infoArray, int batchSize):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnSpotrfBatched(
            <Handle>handle, <FillMode>uplo, n, <float**>Aarray,
            lda, <int*>infoArray, batchSize)
    check_status(status)

cpdef dpotrfBatched(intptr_t handle, int uplo, int n, size_t Aarray, int lda,
                    size_t infoArray, int batchSize):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnDpotrfBatched(
            <Handle>handle, <FillMode>uplo, n, <double**>Aarray,
            lda, <int*>infoArray, batchSize)
    check_status(status)

cpdef cpotrfBatched(intptr_t handle, int uplo, int n, size_t Aarray, int lda,
                    size_t infoArray, int batchSize):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnCpotrfBatched(
            <Handle>handle, <FillMode>uplo, n, <cuComplex**>Aarray,
            lda, <int*>infoArray, batchSize)
    check_status(status)

cpdef zpotrfBatched(intptr_t handle, int uplo, int n, size_t Aarray, int lda,
                    size_t infoArray, int batchSize):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnZpotrfBatched(
            <Handle>handle, <FillMode>uplo, n, <cuDoubleComplex**>Aarray,
            lda, <int*>infoArray, batchSize)
    check_status(status)

cpdef spotrsBatched(intptr_t handle, int uplo, int n, int nrhs, size_t Aarray,
                    int lda, size_t Barray, int ldb, size_t devInfo,
                    int batchSize):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnSpotrsBatched(
            <Handle>handle, <FillMode>uplo, n, nrhs,
            <float**>Aarray, lda, <float**>Barray, ldb,
            <int*>devInfo, batchSize)
    check_status(status)

cpdef dpotrsBatched(intptr_t handle, int uplo, int n, int nrhs, size_t Aarray,
                    int lda, size_t Barray, int ldb, size_t devInfo,
                    int batchSize):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnDpotrsBatched(
            <Handle>handle, <FillMode>uplo, n, nrhs,
            <double**>Aarray, lda, <double**>Barray, ldb,
            <int*>devInfo, batchSize)
    check_status(status)

cpdef cpotrsBatched(intptr_t handle, int uplo, int n, int nrhs, size_t Aarray,
                    int lda, size_t Barray, int ldb, size_t devInfo,
                    int batchSize):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnCpotrsBatched(
            <Handle>handle, <FillMode>uplo, n, nrhs,
            <cuComplex**>Aarray, lda, <cuComplex**>Barray, ldb,
            <int*>devInfo, batchSize)
    check_status(status)

cpdef zpotrsBatched(intptr_t handle, int uplo, int n, int nrhs, size_t Aarray,
                    int lda, size_t Barray, int ldb, size_t devInfo,
                    int batchSize):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnZpotrsBatched(
            <Handle>handle, <FillMode>uplo, n, nrhs,
            <cuDoubleComplex**>Aarray, lda, <cuDoubleComplex**>Barray, ldb,
            <int*>devInfo, batchSize)
    check_status(status)

# LU factorization
cpdef int sgetrf_bufferSize(intptr_t handle, int m, int n,
                            size_t A, int lda) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnSgetrf_bufferSize(
            <Handle>handle, m, n, <float*>A, lda, &lwork)
    check_status(status)
    return lwork

cpdef int dgetrf_bufferSize(intptr_t handle, int m, int n,
                            size_t A, int lda) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnDgetrf_bufferSize(
            <Handle>handle, m, n, <double*>A, lda, &lwork)
    check_status(status)
    return lwork

cpdef int cgetrf_bufferSize(intptr_t handle, int m, int n,
                            size_t A, int lda) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnCgetrf_bufferSize(
            <Handle>handle, m, n, <cuComplex*>A, lda, &lwork)
    check_status(status)
    return lwork

cpdef int zgetrf_bufferSize(intptr_t handle, int m, int n,
                            size_t A, int lda) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnZgetrf_bufferSize(
            <Handle>handle, m, n, <cuDoubleComplex*>A, lda, &lwork)
    check_status(status)
    return lwork

cpdef sgetrf(intptr_t handle, int m, int n, size_t A, int lda,
             size_t work, size_t devIpiv, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = cusolverDnSgetrf(
            <Handle>handle, m, n, <float*>A, lda,
            <float*>work, <int*>devIpiv, <int*>devInfo)
    check_status(status)

cpdef dgetrf(intptr_t handle, int m, int n, size_t A, int lda,
             size_t work, size_t devIpiv, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = cusolverDnDgetrf(
            <Handle>handle, m, n, <double*>A, lda,
            <double*>work, <int*>devIpiv, <int*>devInfo)
    check_status(status)

cpdef cgetrf(intptr_t handle, int m, int n, size_t A, int lda,
             size_t work, size_t devIpiv, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = cusolverDnCgetrf(
            <Handle>handle, m, n, <cuComplex*>A, lda,
            <cuComplex*>work, <int*>devIpiv, <int*>devInfo)
    check_status(status)

cpdef zgetrf(intptr_t handle, int m, int n, size_t A, int lda,
             size_t work, size_t devIpiv, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = cusolverDnZgetrf(
            <Handle>handle, m, n, <cuDoubleComplex*>A, lda,
            <cuDoubleComplex*>work, <int*>devIpiv, <int*>devInfo)
    check_status(status)


# LU solve
cpdef sgetrs(intptr_t handle, int trans, int n, int nrhs,
             size_t A, int lda, size_t devIpiv,
             size_t B, int ldb, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = cusolverDnSgetrs(
            <Handle>handle, <Operation>trans, n, nrhs,
            <const float*> A, lda, <const int*>devIpiv,
            <float*>B, ldb, <int*> devInfo)
    check_status(status)

cpdef dgetrs(intptr_t handle, int trans, int n, int nrhs,
             size_t A, int lda, size_t devIpiv,
             size_t B, int ldb, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = cusolverDnDgetrs(
            <Handle>handle, <Operation>trans, n, nrhs,
            <const double*> A, lda, <const int*>devIpiv,
            <double*>B, ldb, <int*> devInfo)
    check_status(status)

cpdef cgetrs(intptr_t handle, int trans, int n, int nrhs,
             size_t A, int lda, size_t devIpiv,
             size_t B, int ldb, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = cusolverDnCgetrs(
            <Handle>handle, <Operation>trans, n, nrhs,
            <const cuComplex*> A, lda, <const int*>devIpiv,
            <cuComplex*>B, ldb, <int*> devInfo)
    check_status(status)

cpdef zgetrs(intptr_t handle, int trans, int n, int nrhs,
             size_t A, int lda, size_t devIpiv,
             size_t B, int ldb, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = cusolverDnZgetrs(
            <Handle>handle, <Operation>trans, n, nrhs,
            <const cuDoubleComplex*> A, lda, <const int*>devIpiv,
            <cuDoubleComplex*>B, ldb, <int*> devInfo)
    check_status(status)


# QR factorization
cpdef int sgeqrf_bufferSize(intptr_t handle, int m, int n,
                            size_t A, int lda) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnSgeqrf_bufferSize(
            <Handle>handle, m, n, <float*>A, lda, &lwork)
    check_status(status)
    return lwork

cpdef int dgeqrf_bufferSize(intptr_t handle, int m, int n,
                            size_t A, int lda) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnDgeqrf_bufferSize(
            <Handle>handle, m, n, <double*>A, lda, &lwork)
    check_status(status)
    return lwork

cpdef int cgeqrf_bufferSize(intptr_t handle, int m, int n,
                            size_t A, int lda) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnCgeqrf_bufferSize(
            <Handle>handle, m, n, <cuComplex*>A, lda, &lwork)
    check_status(status)
    return lwork

cpdef int zgeqrf_bufferSize(intptr_t handle, int m, int n,
                            size_t A, int lda) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnZgeqrf_bufferSize(
            <Handle>handle, m, n, <cuDoubleComplex*>A, lda, &lwork)
    check_status(status)
    return lwork

cpdef sgeqrf(intptr_t handle, int m, int n, size_t A, int lda,
             size_t tau, size_t work, int lwork, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = cusolverDnSgeqrf(
            <Handle>handle, m, n, <float*>A, lda,
            <float*>tau, <float*>work, lwork,
            <int*>devInfo)
    check_status(status)

cpdef dgeqrf(intptr_t handle, int m, int n, size_t A, int lda,
             size_t tau, size_t work, int lwork, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = cusolverDnDgeqrf(
            <Handle>handle, m, n, <double*>A, lda,
            <double*>tau, <double*>work, lwork,
            <int*>devInfo)
    check_status(status)

cpdef cgeqrf(intptr_t handle, int m, int n, size_t A, int lda,
             size_t tau, size_t work, int lwork, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = cusolverDnCgeqrf(
            <Handle>handle, m, n, <cuComplex*>A, lda,
            <cuComplex*>tau, <cuComplex*>work, lwork,
            <int*>devInfo)
    check_status(status)

cpdef zgeqrf(intptr_t handle, int m, int n, size_t A, int lda,
             size_t tau, size_t work, int lwork, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = cusolverDnZgeqrf(
            <Handle>handle, m, n, <cuDoubleComplex*>A, lda,
            <cuDoubleComplex*>tau, <cuDoubleComplex*>work, lwork,
            <int*>devInfo)
    check_status(status)


# Generate unitary matrix Q from QR factorization
cpdef int sorgqr_bufferSize(intptr_t handle, int m, int n, int k,
                            size_t A, int lda, size_t tau) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnSorgqr_bufferSize(
            <Handle>handle, m, n, k, <const float*>A, lda,
            <const float*>tau, &lwork)
    check_status(status)
    return lwork

cpdef int dorgqr_bufferSize(intptr_t handle, int m, int n, int k,
                            size_t A, int lda, size_t tau) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnDorgqr_bufferSize(
            <Handle>handle, m, n, k, <const double*>A, lda,
            <const double*>tau, &lwork)
    check_status(status)
    return lwork

cpdef int cungqr_bufferSize(intptr_t handle, int m, int n, int k,
                            size_t A, int lda, size_t tau) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnCungqr_bufferSize(
            <Handle>handle, m, n, k, <const cuComplex*>A, lda,
            <const cuComplex*>tau, &lwork)
    check_status(status)
    return lwork

cpdef int zungqr_bufferSize(intptr_t handle, int m, int n, int k,
                            size_t A, int lda, size_t tau) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnZungqr_bufferSize(
            <Handle>handle, m, n, k, <const cuDoubleComplex*>A, lda,
            <const cuDoubleComplex*>tau, &lwork)
    check_status(status)
    return lwork

cpdef sorgqr(intptr_t handle, int m, int n, int k, size_t A, int lda,
             size_t tau, size_t work, int lwork, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = cusolverDnSorgqr(
            <Handle>handle, m, n, k, <float*>A, lda,
            <const float*>tau, <float*>work, lwork,
            <int*>devInfo)
    check_status(status)

cpdef dorgqr(intptr_t handle, int m, int n, int k, size_t A, int lda,
             size_t tau, size_t work, int lwork, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = cusolverDnDorgqr(
            <Handle>handle, m, n, k, <double*>A, lda,
            <const double*>tau, <double*>work, lwork,
            <int*>devInfo)
    check_status(status)

cpdef cungqr(intptr_t handle, int m, int n, int k, size_t A, int lda,
             size_t tau, size_t work, int lwork, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = cusolverDnCungqr(
            <Handle>handle, m, n, k, <cuComplex*>A, lda,
            <const cuComplex*>tau, <cuComplex*>work, lwork,
            <int*>devInfo)
    check_status(status)

cpdef zungqr(intptr_t handle, int m, int n, int k, size_t A, int lda,
             size_t tau, size_t work, int lwork, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = cusolverDnZungqr(
            <Handle>handle, m, n, k, <cuDoubleComplex*>A, lda,
            <const cuDoubleComplex*>tau, <cuDoubleComplex*>work, lwork,
            <int*>devInfo)
    check_status(status)


# Compute Q**T*b in solve min||A*x = b||
cpdef int sormqr_bufferSize(intptr_t handle, int side, int trans,
                            int m, int n, int k, size_t A, int lda, size_t tau,
                            size_t C, int ldc) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnSormqr_bufferSize(
            <Handle>handle, <SideMode>side, <Operation>trans, m, n, k,
            <const float*>A, lda, <const float*>tau,
            <float*>C, ldc, &lwork)
    check_status(status)
    return lwork

cpdef int dormqr_bufferSize(intptr_t handle, int side, int trans,
                            int m, int n, int k, size_t A, int lda, size_t tau,
                            size_t C, int ldc) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnDormqr_bufferSize(
            <Handle>handle, <SideMode>side, <Operation>trans, m, n, k,
            <const double*>A, lda, <const double*>tau,
            <double*>C, ldc, &lwork)
    check_status(status)
    return lwork

cpdef int cunmqr_bufferSize(intptr_t handle, int side, int trans,
                            int m, int n, int k, size_t A, int lda, size_t tau,
                            size_t C, int ldc) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnCunmqr_bufferSize(
            <Handle>handle, <SideMode>side, <Operation>trans, m, n, k,
            <const cuComplex*>A, lda, <const cuComplex*>tau,
            <cuComplex*>C, ldc, &lwork)
    check_status(status)
    return lwork

cpdef int zunmqr_bufferSize(intptr_t handle, int side, int trans,
                            int m, int n, int k, size_t A, int lda, size_t tau,
                            size_t C, int ldc) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnZunmqr_bufferSize(
            <Handle>handle, <SideMode>side, <Operation>trans, m, n, k,
            <const cuDoubleComplex*>A, lda, <const cuDoubleComplex*>tau,
            <cuDoubleComplex*>C, ldc, &lwork)
    check_status(status)
    return lwork


cpdef sormqr(intptr_t handle, int side, int trans,
             int m, int n, int k, size_t A, int lda, size_t tau,
             size_t C, int ldc, size_t work, int lwork, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = cusolverDnSormqr(
            <Handle>handle, <SideMode>side, <Operation>trans, m, n, k,
            <const float*>A, lda, <const float*>tau,
            <float*>C, ldc,
            <float*>work, lwork, <int*>devInfo)
    check_status(status)

cpdef dormqr(intptr_t handle, int side, int trans,
             int m, int n, int k, size_t A, int lda, size_t tau,
             size_t C, int ldc, size_t work, int lwork, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = cusolverDnDormqr(
            <Handle>handle, <SideMode>side, <Operation>trans, m, n, k,
            <const double*>A, lda, <const double*>tau,
            <double*>C, ldc,
            <double*>work, lwork, <int*>devInfo)
    check_status(status)

cpdef cunmqr(intptr_t handle, int side, int trans,
             int m, int n, int k, size_t A, int lda, size_t tau,
             size_t C, int ldc, size_t work, int lwork, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = cusolverDnCunmqr(
            <Handle>handle, <SideMode>side, <Operation>trans, m, n, k,
            <const cuComplex*>A, lda, <const cuComplex*>tau,
            <cuComplex*>C, ldc,
            <cuComplex*>work, lwork, <int*>devInfo)
    check_status(status)

cpdef zunmqr(intptr_t handle, int side, int trans,
             int m, int n, int k, size_t A, int lda, size_t tau,
             size_t C, int ldc, size_t work, int lwork, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = cusolverDnZunmqr(
            <Handle>handle, <SideMode>side, <Operation>trans, m, n, k,
            <const cuDoubleComplex*>A, lda, <const cuDoubleComplex*>tau,
            <cuDoubleComplex*>C, ldc,
            <cuDoubleComplex*>work, lwork, <int*>devInfo)
    check_status(status)

# (obsoleted)
cpdef cormqr(intptr_t handle, int side, int trans,
             int m, int n, int k, size_t A, int lda, size_t tau,
             size_t C, int ldc, size_t work, int lwork, size_t devInfo):
    return cunmqr(handle, side, trans, m, n, k, A, lda, tau,
                  C, ldc, work, lwork, devInfo)

# (obsoleted)
cpdef zormqr(intptr_t handle, int side, int trans,
             int m, int n, int k, size_t A, int lda, size_t tau,
             size_t C, int ldc, size_t work, int lwork, size_t devInfo):
    return zunmqr(handle, side, trans, m, n, k, A, lda, tau,
                  C, ldc, work, lwork, devInfo)


# L*D*L**T,U*D*U**T factorization
cpdef int ssytrf_bufferSize(intptr_t handle, int n, size_t A,
                            int lda) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnSsytrf_bufferSize(
            <Handle>handle, n, <float*>A, lda, &lwork)
    check_status(status)
    return lwork

cpdef int dsytrf_bufferSize(intptr_t handle, int n, size_t A,
                            int lda) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnDsytrf_bufferSize(
            <Handle>handle, n, <double*>A, lda, &lwork)
    check_status(status)
    return lwork

cpdef int csytrf_bufferSize(intptr_t handle, int n, size_t A,
                            int lda) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnCsytrf_bufferSize(
            <Handle>handle, n, <cuComplex*>A, lda, &lwork)
    check_status(status)
    return lwork

cpdef int zsytrf_bufferSize(intptr_t handle, int n, size_t A,
                            int lda) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnZsytrf_bufferSize(
            <Handle>handle, n, <cuDoubleComplex*>A, lda, &lwork)
    check_status(status)
    return lwork

cpdef ssytrf(intptr_t handle, int uplo, int n, size_t A, int lda,
             size_t ipiv, size_t work, int lwork, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = cusolverDnSsytrf(
            <Handle>handle, <FillMode>uplo, n, <float*>A, lda,
            <int*>ipiv, <float*>work, lwork, <int*>devInfo)
    check_status(status)

cpdef dsytrf(intptr_t handle, int uplo, int n, size_t A, int lda,
             size_t ipiv, size_t work, int lwork, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = cusolverDnDsytrf(
            <Handle>handle, <FillMode>uplo, n, <double*>A, lda,
            <int*>ipiv, <double*>work, lwork, <int*>devInfo)
    check_status(status)

cpdef csytrf(intptr_t handle, int uplo, int n, size_t A, int lda,
             size_t ipiv, size_t work, int lwork, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = cusolverDnCsytrf(
            <Handle>handle, <FillMode>uplo, n, <cuComplex*>A, lda,
            <int*>ipiv, <cuComplex*>work, lwork, <int*>devInfo)
    check_status(status)

cpdef zsytrf(intptr_t handle, int uplo, int n, size_t A, int lda,
             size_t ipiv, size_t work, int lwork, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = cusolverDnZsytrf(
            <Handle>handle, <FillMode>uplo, n, <cuDoubleComplex*>A, lda,
            <int*>ipiv, <cuDoubleComplex*>work, lwork, <int*>devInfo)
    check_status(status)

cpdef size_t zzgesv_bufferSize(intptr_t handle, int n, int nrhs, size_t dA,
                               int ldda, size_t dipiv, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1:
    cdef size_t lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnZZgesv_bufferSize(
            <Handle>handle, n, nrhs, <cuDoubleComplex*>dA, ldda, <int*>dipiv,
            <cuDoubleComplex*>dB, lddb, <cuDoubleComplex*>dX, lddx,
            <void*>dwork, &lwork)
    check_status(status)
    return lwork

cpdef size_t zcgesv_bufferSize(intptr_t handle, int n, int nrhs, size_t dA,
                               int ldda, size_t dipiv, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1:
    cdef size_t lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnZCgesv_bufferSize(
            <Handle>handle, n, nrhs, <cuDoubleComplex*>dA, ldda, <int*>dipiv,
            <cuDoubleComplex*>dB, lddb, <cuDoubleComplex*>dX, lddx,
            <void*>dwork, &lwork)
    check_status(status)
    return lwork

cpdef size_t zygesv_bufferSize(intptr_t handle, int n, int nrhs, size_t dA,
                               int ldda, size_t dipiv, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1:
    cdef size_t lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnZYgesv_bufferSize(
            <Handle>handle, n, nrhs, <cuDoubleComplex*>dA, ldda, <int*>dipiv,
            <cuDoubleComplex*>dB, lddb, <cuDoubleComplex*>dX, lddx,
            <void*>dwork, &lwork)
    check_status(status)
    return lwork

cpdef size_t zkgesv_bufferSize(intptr_t handle, int n, int nrhs, size_t dA,
                               int ldda, size_t dipiv, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1:
    cdef size_t lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnZKgesv_bufferSize(
            <Handle>handle, n, nrhs, <cuDoubleComplex*>dA, ldda, <int*>dipiv,
            <cuDoubleComplex*>dB, lddb, <cuDoubleComplex*>dX, lddx,
            <void*>dwork, &lwork)
    check_status(status)
    return lwork

cpdef size_t ccgesv_bufferSize(intptr_t handle, int n, int nrhs, size_t dA,
                               int ldda, size_t dipiv, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1:
    cdef size_t lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnCCgesv_bufferSize(
            <Handle>handle, n, nrhs, <cuComplex*>dA, ldda, <int*>dipiv,
            <cuComplex*>dB, lddb, <cuComplex*>dX, lddx, <void*>dwork, &lwork)
    check_status(status)
    return lwork

cpdef size_t cygesv_bufferSize(intptr_t handle, int n, int nrhs, size_t dA,
                               int ldda, size_t dipiv, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1:
    cdef size_t lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnCYgesv_bufferSize(
            <Handle>handle, n, nrhs, <cuComplex*>dA, ldda, <int*>dipiv,
            <cuComplex*>dB, lddb, <cuComplex*>dX, lddx, <void*>dwork, &lwork)
    check_status(status)
    return lwork

cpdef size_t ckgesv_bufferSize(intptr_t handle, int n, int nrhs, size_t dA,
                               int ldda, size_t dipiv, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1:
    cdef size_t lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnCKgesv_bufferSize(
            <Handle>handle, n, nrhs, <cuComplex*>dA, ldda, <int*>dipiv,
            <cuComplex*>dB, lddb, <cuComplex*>dX, lddx, <void*>dwork, &lwork)
    check_status(status)
    return lwork

cpdef size_t ddgesv_bufferSize(intptr_t handle, int n, int nrhs, size_t dA,
                               int ldda, size_t dipiv, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1:
    cdef size_t lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnDDgesv_bufferSize(
            <Handle>handle, n, nrhs, <double*>dA, ldda, <int*>dipiv,
            <double*>dB, lddb, <double*>dX, lddx, <void*>dwork, &lwork)
    check_status(status)
    return lwork

cpdef size_t dsgesv_bufferSize(intptr_t handle, int n, int nrhs, size_t dA,
                               int ldda, size_t dipiv, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1:
    cdef size_t lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnDSgesv_bufferSize(
            <Handle>handle, n, nrhs, <double*>dA, ldda, <int*>dipiv,
            <double*>dB, lddb, <double*>dX, lddx, <void*>dwork, &lwork)
    check_status(status)
    return lwork

cpdef size_t dxgesv_bufferSize(intptr_t handle, int n, int nrhs, size_t dA,
                               int ldda, size_t dipiv, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1:
    cdef size_t lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnDXgesv_bufferSize(
            <Handle>handle, n, nrhs, <double*>dA, ldda, <int*>dipiv,
            <double*>dB, lddb, <double*>dX, lddx, <void*>dwork, &lwork)
    check_status(status)
    return lwork

cpdef size_t dhgesv_bufferSize(intptr_t handle, int n, int nrhs, size_t dA,
                               int ldda, size_t dipiv, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1:
    cdef size_t lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnDHgesv_bufferSize(
            <Handle>handle, n, nrhs, <double*>dA, ldda, <int*>dipiv,
            <double*>dB, lddb, <double*>dX, lddx, <void*>dwork, &lwork)
    check_status(status)
    return lwork

cpdef size_t ssgesv_bufferSize(intptr_t handle, int n, int nrhs, size_t dA,
                               int ldda, size_t dipiv, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1:
    cdef size_t lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnSSgesv_bufferSize(
            <Handle>handle, n, nrhs, <float*>dA, ldda, <int*>dipiv,
            <float*>dB, lddb, <float*>dX, lddx, <void*>dwork, &lwork)
    check_status(status)
    return lwork

cpdef size_t sxgesv_bufferSize(intptr_t handle, int n, int nrhs, size_t dA,
                               int ldda, size_t dipiv, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1:
    cdef size_t lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnSXgesv_bufferSize(
            <Handle>handle, n, nrhs, <float*>dA, ldda, <int*>dipiv,
            <float*>dB, lddb, <float*>dX, lddx, <void*>dwork, &lwork)
    check_status(status)
    return lwork

cpdef size_t shgesv_bufferSize(intptr_t handle, int n, int nrhs, size_t dA,
                               int ldda, size_t dipiv, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1:
    cdef size_t lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnSHgesv_bufferSize(
            <Handle>handle, n, nrhs, <float*>dA, ldda, <int*>dipiv,
            <float*>dB, lddb, <float*>dX, lddx, <void*>dwork, &lwork)
    check_status(status)
    return lwork

cpdef int zzgesv(intptr_t handle, int n, int nrhs, size_t dA, int ldda,
                 size_t dipiv, size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork, size_t dInfo):
    cdef int iter
    _setStream(handle)
    with nogil:
        status = cusolverDnZZgesv(
            <Handle>handle, n, nrhs, <cuDoubleComplex*>dA, ldda, <int*>dipiv,
            <cuDoubleComplex*>dB, lddb, <cuDoubleComplex*>dX, lddx,
            <void*>dwork, lwork, &iter, <int*>dInfo)
    check_status(status)
    return iter

cpdef int zcgesv(intptr_t handle, int n, int nrhs, size_t dA, int ldda,
                 size_t dipiv, size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork, size_t dInfo):
    cdef int iter
    _setStream(handle)
    with nogil:
        status = cusolverDnZCgesv(
            <Handle>handle, n, nrhs, <cuDoubleComplex*>dA, ldda, <int*>dipiv,
            <cuDoubleComplex*>dB, lddb, <cuDoubleComplex*>dX, lddx,
            <void*>dwork, lwork, &iter, <int*>dInfo)
    check_status(status)
    return iter

cpdef int zygesv(intptr_t handle, int n, int nrhs, size_t dA, int ldda,
                 size_t dipiv, size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork, size_t dInfo):
    cdef int iter
    _setStream(handle)
    with nogil:
        status = cusolverDnZYgesv(
            <Handle>handle, n, nrhs, <cuDoubleComplex*>dA, ldda, <int*>dipiv,
            <cuDoubleComplex*>dB, lddb, <cuDoubleComplex*>dX, lddx,
            <void*>dwork, lwork, &iter, <int*>dInfo)
    check_status(status)
    return iter

cpdef int zkgesv(intptr_t handle, int n, int nrhs, size_t dA, int ldda,
                 size_t dipiv, size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork, size_t dInfo):
    cdef int iter
    _setStream(handle)
    with nogil:
        status = cusolverDnZKgesv(
            <Handle>handle, n, nrhs, <cuDoubleComplex*>dA, ldda, <int*>dipiv,
            <cuDoubleComplex*>dB, lddb, <cuDoubleComplex*>dX, lddx,
            <void*>dwork, lwork, &iter, <int*>dInfo)
    check_status(status)
    return iter

cpdef int ccgesv(intptr_t handle, int n, int nrhs, size_t dA, int ldda,
                 size_t dipiv, size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork, size_t dInfo):
    cdef int iter
    _setStream(handle)
    with nogil:
        status = cusolverDnCCgesv(
            <Handle>handle, n, nrhs, <cuComplex*>dA, ldda, <int*>dipiv,
            <cuComplex*>dB, lddb, <cuComplex*>dX, lddx,
            <void*>dwork, lwork, &iter, <int*>dInfo)
    check_status(status)
    return iter

cpdef int cygesv(intptr_t handle, int n, int nrhs, size_t dA, int ldda,
                 size_t dipiv, size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork, size_t dInfo):
    cdef int iter
    _setStream(handle)
    with nogil:
        status = cusolverDnCYgesv(
            <Handle>handle, n, nrhs, <cuComplex*>dA, ldda, <int*>dipiv,
            <cuComplex*>dB, lddb, <cuComplex*>dX, lddx,
            <void*>dwork, lwork, &iter, <int*>dInfo)
    check_status(status)
    return iter

cpdef int ckgesv(intptr_t handle, int n, int nrhs, size_t dA, int ldda,
                 size_t dipiv, size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork, size_t dInfo):
    cdef int iter
    _setStream(handle)
    with nogil:
        status = cusolverDnCKgesv(
            <Handle>handle, n, nrhs, <cuComplex*>dA, ldda, <int*>dipiv,
            <cuComplex*>dB, lddb, <cuComplex*>dX, lddx,
            <void*>dwork, lwork, &iter, <int*>dInfo)
    check_status(status)
    return iter

cpdef int ddgesv(intptr_t handle, int n, int nrhs, size_t dA, int ldda,
                 size_t dipiv, size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork, size_t dInfo):
    cdef int iter
    _setStream(handle)
    with nogil:
        status = cusolverDnDDgesv(
            <Handle>handle, n, nrhs, <double*>dA, ldda, <int*>dipiv,
            <double*>dB, lddb, <double*>dX, lddx,
            <void*>dwork, lwork, &iter, <int*>dInfo)
    check_status(status)
    return iter

cpdef int dsgesv(intptr_t handle, int n, int nrhs, size_t dA, int ldda,
                 size_t dipiv, size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork, size_t dInfo):
    cdef int iter
    _setStream(handle)
    with nogil:
        status = cusolverDnDSgesv(
            <Handle>handle, n, nrhs, <double*>dA, ldda, <int*>dipiv,
            <double*>dB, lddb, <double*>dX, lddx,
            <void*>dwork, lwork, &iter, <int*>dInfo)
    check_status(status)
    return iter

cpdef int dxgesv(intptr_t handle, int n, int nrhs, size_t dA, int ldda,
                 size_t dipiv, size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork, size_t dInfo):
    cdef int iter
    _setStream(handle)
    with nogil:
        status = cusolverDnDXgesv(
            <Handle>handle, n, nrhs, <double*>dA, ldda, <int*>dipiv,
            <double*>dB, lddb, <double*>dX, lddx,
            <void*>dwork, lwork, &iter, <int*>dInfo)
    check_status(status)
    return iter

cpdef int dhgesv(intptr_t handle, int n, int nrhs, size_t dA, int ldda,
                 size_t dipiv, size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork, size_t dInfo):
    cdef int iter
    _setStream(handle)
    with nogil:
        status = cusolverDnDHgesv(
            <Handle>handle, n, nrhs, <double*>dA, ldda, <int*>dipiv,
            <double*>dB, lddb, <double*>dX, lddx,
            <void*>dwork, lwork, &iter, <int*>dInfo)
    check_status(status)
    return iter

cpdef int ssgesv(intptr_t handle, int n, int nrhs, size_t dA, int ldda,
                 size_t dipiv, size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork, size_t dInfo):
    cdef int iter
    _setStream(handle)
    with nogil:
        status = cusolverDnSSgesv(
            <Handle>handle, n, nrhs, <float*>dA, ldda, <int*>dipiv,
            <float*>dB, lddb, <float*>dX, lddx,
            <void*>dwork, lwork, &iter, <int*>dInfo)
    check_status(status)
    return iter

cpdef int sxgesv(intptr_t handle, int n, int nrhs, size_t dA, int ldda,
                 size_t dipiv, size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork, size_t dInfo):
    cdef int iter
    _setStream(handle)
    with nogil:
        status = cusolverDnSXgesv(
            <Handle>handle, n, nrhs, <float*>dA, ldda, <int*>dipiv,
            <float*>dB, lddb, <float*>dX, lddx,
            <void*>dwork, lwork, &iter, <int*>dInfo)
    check_status(status)
    return iter

cpdef int shgesv(intptr_t handle, int n, int nrhs, size_t dA, int ldda,
                 size_t dipiv, size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork, size_t dInfo):
    cdef int iter
    _setStream(handle)
    with nogil:
        status = cusolverDnSHgesv(
            <Handle>handle, n, nrhs, <float*>dA, ldda, <int*>dipiv,
            <float*>dB, lddb, <float*>dX, lddx,
            <void*>dwork, lwork, &iter, <int*>dInfo)
    check_status(status)
    return iter

cpdef size_t zzgels_bufferSize(intptr_t handle, int m, int n, int nrhs,
                               size_t dA, int ldda, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1:
    cdef size_t lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnZZgels_bufferSize(
            <Handle>handle, m, n, nrhs, <cuDoubleComplex*>dA, ldda,
            <cuDoubleComplex*>dB, lddb, <cuDoubleComplex*>dX, lddx,
            <void*>dwork, &lwork)
    check_status(status)
    return lwork

cpdef size_t zcgels_bufferSize(intptr_t handle, int m, int n, int nrhs,
                               size_t dA, int ldda, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1:
    cdef size_t lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnZCgels_bufferSize(
            <Handle>handle, m, n, nrhs, <cuDoubleComplex*>dA, ldda,
            <cuDoubleComplex*>dB, lddb, <cuDoubleComplex*>dX, lddx,
            <void*>dwork, &lwork)
    check_status(status)
    return lwork

cpdef size_t zygels_bufferSize(intptr_t handle, int m, int n, int nrhs,
                               size_t dA, int ldda, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1:
    cdef size_t lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnZYgels_bufferSize(
            <Handle>handle, m, n, nrhs, <cuDoubleComplex*>dA, ldda,
            <cuDoubleComplex*>dB, lddb, <cuDoubleComplex*>dX, lddx,
            <void*>dwork, &lwork)
    check_status(status)
    return lwork

cpdef size_t zkgels_bufferSize(intptr_t handle, int m, int n, int nrhs,
                               size_t dA, int ldda, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1:
    cdef size_t lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnZKgels_bufferSize(
            <Handle>handle, m, n, nrhs, <cuDoubleComplex*>dA, ldda,
            <cuDoubleComplex*>dB, lddb, <cuDoubleComplex*>dX, lddx,
            <void*>dwork, &lwork)
    check_status(status)
    return lwork

cpdef size_t ccgels_bufferSize(intptr_t handle, int m, int n, int nrhs,
                               size_t dA, int ldda, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1:
    cdef size_t lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnCCgels_bufferSize(
            <Handle>handle, m, n, nrhs, <cuComplex*>dA, ldda,
            <cuComplex*>dB, lddb, <cuComplex*>dX, lddx, <void*>dwork, &lwork)
    check_status(status)
    return lwork

cpdef size_t cygels_bufferSize(intptr_t handle, int m, int n, int nrhs,
                               size_t dA, int ldda, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1:
    cdef size_t lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnCYgels_bufferSize(
            <Handle>handle, m, n, nrhs, <cuComplex*>dA, ldda,
            <cuComplex*>dB, lddb, <cuComplex*>dX, lddx, <void*>dwork, &lwork)
    check_status(status)
    return lwork

cpdef size_t ckgels_bufferSize(intptr_t handle, int m, int n, int nrhs,
                               size_t dA, int ldda, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1:
    cdef size_t lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnCKgels_bufferSize(
            <Handle>handle, m, n, nrhs, <cuComplex*>dA, ldda,
            <cuComplex*>dB, lddb, <cuComplex*>dX, lddx, <void*>dwork, &lwork)
    check_status(status)
    return lwork

cpdef size_t ddgels_bufferSize(intptr_t handle, int m, int n, int nrhs,
                               size_t dA, int ldda, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1:
    cdef size_t lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnDDgels_bufferSize(
            <Handle>handle, m, n, nrhs, <double*>dA, ldda,
            <double*>dB, lddb, <double*>dX, lddx, <void*>dwork, &lwork)
    check_status(status)
    return lwork

cpdef size_t dsgels_bufferSize(intptr_t handle, int m, int n, int nrhs,
                               size_t dA, int ldda, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1:
    cdef size_t lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnDSgels_bufferSize(
            <Handle>handle, m, n, nrhs, <double*>dA, ldda,
            <double*>dB, lddb, <double*>dX, lddx, <void*>dwork, &lwork)
    check_status(status)
    return lwork

cpdef size_t dxgels_bufferSize(intptr_t handle, int m, int n, int nrhs,
                               size_t dA, int ldda, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1:
    cdef size_t lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnDXgels_bufferSize(
            <Handle>handle, m, n, nrhs, <double*>dA, ldda,
            <double*>dB, lddb, <double*>dX, lddx, <void*>dwork, &lwork)
    check_status(status)
    return lwork

cpdef size_t dhgels_bufferSize(intptr_t handle, int m, int n, int nrhs,
                               size_t dA, int ldda, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1:
    cdef size_t lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnDHgels_bufferSize(
            <Handle>handle, m, n, nrhs, <double*>dA, ldda,
            <double*>dB, lddb, <double*>dX, lddx, <void*>dwork, &lwork)
    check_status(status)
    return lwork

cpdef size_t ssgels_bufferSize(intptr_t handle, int m, int n, int nrhs,
                               size_t dA, int ldda, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1:
    cdef size_t lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnSSgels_bufferSize(
            <Handle>handle, m, n, nrhs, <float*>dA, ldda,
            <float*>dB, lddb, <float*>dX, lddx, <void*>dwork, &lwork)
    check_status(status)
    return lwork

cpdef size_t sxgels_bufferSize(intptr_t handle, int m, int n, int nrhs,
                               size_t dA, int ldda, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1:
    cdef size_t lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnSXgels_bufferSize(
            <Handle>handle, m, n, nrhs, <float*>dA, ldda,
            <float*>dB, lddb, <float*>dX, lddx, <void*>dwork, &lwork)
    check_status(status)
    return lwork

cpdef size_t shgels_bufferSize(intptr_t handle, int m, int n, int nrhs,
                               size_t dA, int ldda, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1:
    cdef size_t lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnSHgels_bufferSize(
            <Handle>handle, m, n, nrhs, <float*>dA, ldda,
            <float*>dB, lddb, <float*>dX, lddx, <void*>dwork, &lwork)
    check_status(status)
    return lwork

cpdef int zzgels(intptr_t handle, int m, int n, int nrhs, size_t dA, int ldda,
                 size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork, size_t dInfo):
    cdef int iter
    _setStream(handle)
    with nogil:
        status = cusolverDnZZgels(
            <Handle>handle, m, n, nrhs, <cuDoubleComplex*>dA, ldda,
            <cuDoubleComplex*>dB, lddb, <cuDoubleComplex*>dX, lddx,
            <void*>dwork, lwork, &iter, <int*>dInfo)
    check_status(status)
    return iter

cpdef int zcgels(intptr_t handle, int m, int n, int nrhs, size_t dA, int ldda,
                 size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork, size_t dInfo):
    cdef int iter
    _setStream(handle)
    with nogil:
        status = cusolverDnZCgels(
            <Handle>handle, m, n, nrhs, <cuDoubleComplex*>dA, ldda,
            <cuDoubleComplex*>dB, lddb, <cuDoubleComplex*>dX, lddx,
            <void*>dwork, lwork, &iter, <int*>dInfo)
    check_status(status)
    return iter

cpdef int zygels(intptr_t handle, int m, int n, int nrhs, size_t dA, int ldda,
                 size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork, size_t dInfo):
    cdef int iter
    _setStream(handle)
    with nogil:
        status = cusolverDnZYgels(
            <Handle>handle, m, n, nrhs, <cuDoubleComplex*>dA, ldda,
            <cuDoubleComplex*>dB, lddb, <cuDoubleComplex*>dX, lddx,
            <void*>dwork, lwork, &iter, <int*>dInfo)
    check_status(status)
    return iter

cpdef int zkgels(intptr_t handle, int m, int n, int nrhs, size_t dA, int ldda,
                 size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork, size_t dInfo):
    cdef int iter
    _setStream(handle)
    with nogil:
        status = cusolverDnZKgels(
            <Handle>handle, m, n, nrhs, <cuDoubleComplex*>dA, ldda,
            <cuDoubleComplex*>dB, lddb, <cuDoubleComplex*>dX, lddx,
            <void*>dwork, lwork, &iter, <int*>dInfo)
    check_status(status)
    return iter

cpdef int ccgels(intptr_t handle, int m, int n, int nrhs, size_t dA, int ldda,
                 size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork, size_t dInfo):
    cdef int iter
    _setStream(handle)
    with nogil:
        status = cusolverDnCCgels(
            <Handle>handle, m, n, nrhs, <cuComplex*>dA, ldda,
            <cuComplex*>dB, lddb, <cuComplex*>dX, lddx,
            <void*>dwork, lwork, &iter, <int*>dInfo)
    check_status(status)
    return iter

cpdef int cygels(intptr_t handle, int m, int n, int nrhs, size_t dA, int ldda,
                 size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork, size_t dInfo):
    cdef int iter
    _setStream(handle)
    with nogil:
        status = cusolverDnCYgels(
            <Handle>handle, m, n, nrhs, <cuComplex*>dA, ldda,
            <cuComplex*>dB, lddb, <cuComplex*>dX, lddx,
            <void*>dwork, lwork, &iter, <int*>dInfo)
    check_status(status)
    return iter

cpdef int ckgels(intptr_t handle, int m, int n, int nrhs, size_t dA, int ldda,
                 size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork, size_t dInfo):
    cdef int iter
    _setStream(handle)
    with nogil:
        status = cusolverDnCKgels(
            <Handle>handle, m, n, nrhs, <cuComplex*>dA, ldda,
            <cuComplex*>dB, lddb, <cuComplex*>dX, lddx,
            <void*>dwork, lwork, &iter, <int*>dInfo)
    check_status(status)
    return iter

cpdef int ddgels(intptr_t handle, int m, int n, int nrhs, size_t dA, int ldda,
                 size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork, size_t dInfo):
    cdef int iter
    _setStream(handle)
    with nogil:
        status = cusolverDnDDgels(
            <Handle>handle, m, n, nrhs, <double*>dA, ldda,
            <double*>dB, lddb, <double*>dX, lddx,
            <void*>dwork, lwork, &iter, <int*>dInfo)
    check_status(status)
    return iter

cpdef int dsgels(intptr_t handle, int m, int n, int nrhs, size_t dA, int ldda,
                 size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork, size_t dInfo):
    cdef int iter
    _setStream(handle)
    with nogil:
        status = cusolverDnDSgels(
            <Handle>handle, m, n, nrhs, <double*>dA, ldda,
            <double*>dB, lddb, <double*>dX, lddx,
            <void*>dwork, lwork, &iter, <int*>dInfo)
    check_status(status)
    return iter

cpdef int dxgels(intptr_t handle, int m, int n, int nrhs, size_t dA, int ldda,
                 size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork, size_t dInfo):
    cdef int iter
    _setStream(handle)
    with nogil:
        status = cusolverDnDXgels(
            <Handle>handle, m, n, nrhs, <double*>dA, ldda,
            <double*>dB, lddb, <double*>dX, lddx,
            <void*>dwork, lwork, &iter, <int*>dInfo)
    check_status(status)
    return iter

cpdef int dhgels(intptr_t handle, int m, int n, int nrhs, size_t dA, int ldda,
                 size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork, size_t dInfo):
    cdef int iter
    _setStream(handle)
    with nogil:
        status = cusolverDnDHgels(
            <Handle>handle, m, n, nrhs, <double*>dA, ldda,
            <double*>dB, lddb, <double*>dX, lddx,
            <void*>dwork, lwork, &iter, <int*>dInfo)
    check_status(status)
    return iter

cpdef int ssgels(intptr_t handle, int m, int n, int nrhs, size_t dA, int ldda,
                 size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork, size_t dInfo):
    cdef int iter
    _setStream(handle)
    with nogil:
        status = cusolverDnSSgels(
            <Handle>handle, m, n, nrhs, <float*>dA, ldda,
            <float*>dB, lddb, <float*>dX, lddx,
            <void*>dwork, lwork, &iter, <int*>dInfo)
    check_status(status)
    return iter

cpdef int sxgels(intptr_t handle, int m, int n, int nrhs, size_t dA, int ldda,
                 size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork, size_t dInfo):
    cdef int iter
    _setStream(handle)
    with nogil:
        status = cusolverDnSXgels(
            <Handle>handle, m, n, nrhs, <float*>dA, ldda,
            <float*>dB, lddb, <float*>dX, lddx,
            <void*>dwork, lwork, &iter, <int*>dInfo)
    check_status(status)
    return iter

cpdef int shgels(intptr_t handle, int m, int n, int nrhs, size_t dA, int ldda,
                 size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork, size_t dInfo):
    cdef int iter
    _setStream(handle)
    with nogil:
        status = cusolverDnSHgels(
            <Handle>handle, m, n, nrhs, <float*>dA, ldda,
            <float*>dB, lddb, <float*>dX, lddx,
            <void*>dwork, lwork, &iter, <int*>dInfo)
    check_status(status)
    return iter

###############################################################################
# Dense LAPACK Functions (Eigenvalue Solver)
###############################################################################

# Bidiagonal factorization
cpdef int sgebrd_bufferSize(intptr_t handle, int m, int n) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnSgebrd_bufferSize(<Handle>handle, m, n, &lwork)
    check_status(status)
    return lwork

cpdef int dgebrd_bufferSize(intptr_t handle, int m, int n) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnDgebrd_bufferSize(<Handle>handle, m, n, &lwork)
    check_status(status)
    return lwork

cpdef int cgebrd_bufferSize(intptr_t handle, int m, int n) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnCgebrd_bufferSize(<Handle>handle, m, n, &lwork)
    check_status(status)
    return lwork

cpdef int zgebrd_bufferSize(intptr_t handle, int m, int n) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnZgebrd_bufferSize(<Handle>handle, m, n, &lwork)
    check_status(status)
    return lwork

cpdef sgebrd(intptr_t handle, int m, int n, size_t A, int lda,
             size_t D, size_t E, size_t tauQ, size_t tauP,
             size_t Work, int lwork, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = cusolverDnSgebrd(
            <Handle>handle, m, n,
            <float*>A, lda,
            <float*>D, <float*>E,
            <float*>tauQ, <float*>tauP,
            <float*>Work, lwork, <int*>devInfo)
    check_status(status)

cpdef dgebrd(intptr_t handle, int m, int n, size_t A, int lda,
             size_t D, size_t E, size_t tauQ, size_t tauP,
             size_t Work, int lwork, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = cusolverDnDgebrd(
            <Handle>handle, m, n,
            <double*>A, lda,
            <double*>D, <double*>E,
            <double*>tauQ, <double*>tauP,
            <double*>Work, lwork, <int*>devInfo)
    check_status(status)

cpdef cgebrd(intptr_t handle, int m, int n, size_t A, int lda,
             size_t D, size_t E, size_t tauQ, size_t tauP,
             size_t Work, int lwork, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = cusolverDnCgebrd(
            <Handle>handle, m, n,
            <cuComplex*>A, lda,
            <float*>D, <float*>E,
            <cuComplex*>tauQ, <cuComplex*>tauP,
            <cuComplex*>Work, lwork, <int*>devInfo)
    check_status(status)

cpdef zgebrd(intptr_t handle, int m, int n, size_t A, int lda,
             size_t D, size_t E, size_t tauQ, size_t tauP,
             size_t Work, int lwork, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = cusolverDnZgebrd(
            <Handle>handle, m, n,
            <cuDoubleComplex*>A, lda,
            <double*>D, <double*>E,
            <cuDoubleComplex*>tauQ, <cuDoubleComplex*>tauP,
            <cuDoubleComplex*>Work, lwork, <int*>devInfo)
    check_status(status)


# Singular value decomposition, A = U * Sigma * V^H
cpdef int sgesvd_bufferSize(intptr_t handle, int m, int n) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnSgesvd_bufferSize(<Handle>handle, m, n, &lwork)
    check_status(status)
    return lwork

cpdef int dgesvd_bufferSize(intptr_t handle, int m, int n) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnDgesvd_bufferSize(<Handle>handle, m, n, &lwork)
    check_status(status)
    return lwork

cpdef int cgesvd_bufferSize(intptr_t handle, int m, int n) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnCgesvd_bufferSize(<Handle>handle, m, n, &lwork)
    check_status(status)
    return lwork

cpdef int zgesvd_bufferSize(intptr_t handle, int m, int n) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnZgesvd_bufferSize(<Handle>handle, m, n, &lwork)
    check_status(status)
    return lwork

cpdef sgesvd(intptr_t handle, char jobu, char jobvt, int m, int n, size_t A,
             int lda, size_t S, size_t U, int ldu, size_t VT, int ldvt,
             size_t Work, int lwork, size_t rwork, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = cusolverDnSgesvd(
            <Handle>handle, jobu, jobvt, m, n, <float*>A, lda,
            <float*>S, <float*>U, ldu, <float*>VT, ldvt,
            <float*>Work, lwork, <float*>rwork, <int*>devInfo)
    check_status(status)

cpdef dgesvd(intptr_t handle, char jobu, char jobvt, int m, int n, size_t A,
             int lda, size_t S, size_t U, int ldu, size_t VT, int ldvt,
             size_t Work, int lwork, size_t rwork, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = cusolverDnDgesvd(
            <Handle>handle, jobu, jobvt, m, n, <double*>A, lda,
            <double*>S, <double*>U, ldu, <double*>VT, ldvt,
            <double*>Work, lwork, <double*>rwork, <int*>devInfo)
    check_status(status)

cpdef cgesvd(intptr_t handle, char jobu, char jobvt, int m, int n, size_t A,
             int lda, size_t S, size_t U, int ldu, size_t VT, int ldvt,
             size_t Work, int lwork, size_t rwork, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = cusolverDnCgesvd(
            <Handle>handle, jobu, jobvt, m, n, <cuComplex*>A, lda,
            <float*>S, <cuComplex*>U, ldu, <cuComplex*>VT, ldvt,
            <cuComplex*>Work, lwork, <float*>rwork, <int*>devInfo)
    check_status(status)

cpdef zgesvd(intptr_t handle, char jobu, char jobvt, int m, int n, size_t A,
             int lda, size_t S, size_t U, int ldu, size_t VT, int ldvt,
             size_t Work, int lwork, size_t rwork, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = cusolverDnZgesvd(
            <Handle>handle, jobu, jobvt, m, n, <cuDoubleComplex*>A, lda,
            <double*>S, <cuDoubleComplex*>U, ldu, <cuDoubleComplex*>VT, ldvt,
            <cuDoubleComplex*>Work, lwork, <double*>rwork, <int*>devInfo)
    check_status(status)

# gesvdj ... Singular value decomposition using Jacobi mathod
cpdef intptr_t createGesvdjInfo() except? 0:
    cdef GesvdjInfo info
    status = cusolverDnCreateGesvdjInfo(&info)
    check_status(status)
    return <intptr_t>info

cpdef destroyGesvdjInfo(intptr_t info):
    status = cusolverDnDestroyGesvdjInfo(<GesvdjInfo>info)
    check_status(status)

cpdef xgesvdjSetTolerance(intptr_t info, double tolerance):
    status = cusolverDnXgesvdjSetTolerance(<GesvdjInfo>info, tolerance)
    check_status(status)

cpdef xgesvdjSetMaxSweeps(intptr_t info, int max_sweeps):
    status = cusolverDnXgesvdjSetMaxSweeps(<GesvdjInfo>info, max_sweeps)
    check_status(status)

cpdef xgesvdjSetSortEig(intptr_t info, int sort_svd):
    status = cusolverDnXgesvdjSetSortEig(<GesvdjInfo>info, sort_svd)
    check_status(status)

cpdef double xgesvdjGetResidual(intptr_t handle, intptr_t info):
    cdef double residual
    status = cusolverDnXgesvdjGetResidual(<Handle>handle, <GesvdjInfo>info,
                                          &residual)
    check_status(status)
    return residual

cpdef int xgesvdjGetSweeps(intptr_t handle, intptr_t info):
    cdef int executed_sweeps
    status = cusolverDnXgesvdjGetSweeps(<Handle>handle, <GesvdjInfo>info,
                                        &executed_sweeps)
    check_status(status)
    return executed_sweeps

cpdef int sgesvdj_bufferSize(intptr_t handle, int jobz, int econ, int m, int n,
                             intptr_t A, int lda, intptr_t S, intptr_t U,
                             int ldu, intptr_t V, int ldv, intptr_t params):
    cdef int lwork, status
    _setStream(handle)
    with nogil:
        status = cusolverDnSgesvdj_bufferSize(
            <Handle>handle, <EigMode>jobz, econ, m, n, <const float*>A, lda,
            <const float*>S, <const float*>U, ldu, <const float*>V, ldv,
            &lwork, <GesvdjInfo>params)
    check_status(status)
    return lwork

cpdef int dgesvdj_bufferSize(intptr_t handle, int jobz, int econ, int m, int n,
                             intptr_t A, int lda, intptr_t S, intptr_t U,
                             int ldu, intptr_t V, int ldv, intptr_t params):
    cdef int lwork, status
    _setStream(handle)
    with nogil:
        status = cusolverDnDgesvdj_bufferSize(
            <Handle>handle, <EigMode>jobz, econ, m, n, <const double*>A, lda,
            <const double*>S, <const double*>U, ldu, <const double*>V, ldv,
            &lwork, <GesvdjInfo>params)
    check_status(status)
    return lwork

cpdef int cgesvdj_bufferSize(intptr_t handle, int jobz, int econ, int m, int n,
                             intptr_t A, int lda, intptr_t S, intptr_t U,
                             int ldu, intptr_t V, int ldv, intptr_t params):
    cdef int lwork, status
    _setStream(handle)
    with nogil:
        status = cusolverDnCgesvdj_bufferSize(
            <Handle>handle, <EigMode>jobz, econ, m, n, <const cuComplex*>A,
            lda, <const float*>S, <const cuComplex*>U, ldu,
            <const cuComplex*>V, ldv, &lwork, <GesvdjInfo>params)
    check_status(status)
    return lwork

cpdef int zgesvdj_bufferSize(intptr_t handle, int jobz, int econ, int m, int n,
                             intptr_t A, int lda, intptr_t S, intptr_t U,
                             int ldu, intptr_t V, int ldv, intptr_t params):
    cdef int lwork, status
    _setStream(handle)
    with nogil:
        status = cusolverDnZgesvdj_bufferSize(
            <Handle>handle, <EigMode>jobz, econ, m, n,
            <const cuDoubleComplex*>A, lda, <const double*>S,
            <const cuDoubleComplex*>U, ldu, <const cuDoubleComplex*>V,
            ldv, &lwork, <GesvdjInfo>params)
    check_status(status)
    return lwork

cpdef sgesvdj(intptr_t handle, int jobz, int econ, int m, int n, intptr_t A,
              int lda, intptr_t S, intptr_t U, int ldu, intptr_t V, int ldv,
              intptr_t work, int lwork, intptr_t info, intptr_t params):
    _setStream(handle)
    with nogil:
        status = cusolverDnSgesvdj(<Handle>handle, <EigMode>jobz, econ, m, n,
                                   <float*>A, lda, <float*>S, <float*>U, ldu,
                                   <float*>V, ldv, <float*>work, lwork,
                                   <int*>info, <GesvdjInfo>params)
    check_status(status)

cpdef dgesvdj(intptr_t handle, int jobz, int econ, int m, int n, intptr_t A,
              int lda, intptr_t S, intptr_t U, int ldu, intptr_t V, int ldv,
              intptr_t work, int lwork, intptr_t info, intptr_t params):
    _setStream(handle)
    with nogil:
        status = cusolverDnDgesvdj(<Handle>handle, <EigMode>jobz, econ, m, n,
                                   <double*>A, lda, <double*>S, <double*>U,
                                   ldu, <double*>V, ldv, <double*>work, lwork,
                                   <int*>info, <GesvdjInfo>params)
    check_status(status)

cpdef cgesvdj(intptr_t handle, int jobz, int econ, int m, int n, intptr_t A,
              int lda, intptr_t S, intptr_t U, int ldu, intptr_t V, int ldv,
              intptr_t work, int lwork, intptr_t info, intptr_t params):
    _setStream(handle)
    with nogil:
        status = cusolverDnCgesvdj(
            <Handle>handle, <EigMode>jobz, econ, m, n, <cuComplex*>A, lda,
            <float*>S, <cuComplex*>U, ldu, <cuComplex*>V, ldv,
            <cuComplex*>work, lwork, <int*>info, <GesvdjInfo>params)
    check_status(status)

cpdef zgesvdj(intptr_t handle, int jobz, int econ, int m, int n, intptr_t A,
              int lda, intptr_t S, intptr_t U, int ldu, intptr_t V, int ldv,
              intptr_t work, int lwork, intptr_t info, intptr_t params):
    _setStream(handle)
    with nogil:
        status = cusolverDnZgesvdj(
            <Handle>handle, <EigMode>jobz, econ, m, n, <cuDoubleComplex*>A,
            lda, <double*>S, <cuDoubleComplex*>U, ldu, <cuDoubleComplex*>V,
            ldv, <cuDoubleComplex*>work, lwork, <int*>info, <GesvdjInfo>params)
    check_status(status)

cpdef int sgesvdjBatched_bufferSize(
        intptr_t handle, int jobz, int m, int n, intptr_t A,
        int lda, intptr_t S, intptr_t U, int ldu, intptr_t V, int ldv,
        intptr_t params, int batchSize) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnSgesvdjBatched_bufferSize(
            <Handle>handle, <EigMode>jobz, m, n, <float*>A, lda,
            <float*>S, <float*>U, ldu, <float*>V, ldv, &lwork,
            <GesvdjInfo>params, batchSize)
    check_status(status)
    return lwork

cpdef int dgesvdjBatched_bufferSize(
        intptr_t handle, int jobz, int m, int n, intptr_t A,
        int lda, intptr_t S, intptr_t U, int ldu, intptr_t V, int ldv,
        intptr_t params, int batchSize) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnDgesvdjBatched_bufferSize(
            <Handle>handle, <EigMode>jobz, m, n, <double*>A, lda,
            <double*>S, <double*>U, ldu, <double*>V, ldv, &lwork,
            <GesvdjInfo>params, batchSize)
    check_status(status)
    return lwork

cpdef int cgesvdjBatched_bufferSize(
        intptr_t handle, int jobz, int m, int n, intptr_t A,
        int lda, intptr_t S, intptr_t U, int ldu, intptr_t V, int ldv,
        intptr_t params, int batchSize) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnCgesvdjBatched_bufferSize(
            <Handle>handle, <EigMode>jobz, m, n, <cuComplex*>A, lda,
            <float*>S, <cuComplex*>U, ldu, <cuComplex*>V, ldv, &lwork,
            <GesvdjInfo>params, batchSize)
    check_status(status)
    return lwork

cpdef int zgesvdjBatched_bufferSize(
        intptr_t handle, int jobz, int m, int n, intptr_t A,
        int lda, intptr_t S, intptr_t U, int ldu, intptr_t V, int ldv,
        intptr_t params, int batchSize) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnZgesvdjBatched_bufferSize(
            <Handle>handle, <EigMode>jobz, m, n, <cuDoubleComplex*>A, lda,
            <double*>S, <cuDoubleComplex*>U, ldu, <cuDoubleComplex*>V, ldv,
            &lwork,
            <GesvdjInfo>params, batchSize)
    check_status(status)
    return lwork

cpdef sgesvdjBatched(
        intptr_t handle, int jobz, int m, int n, intptr_t A,
        int lda, intptr_t S, intptr_t U, int ldu, intptr_t V, int ldv,
        intptr_t work, int lwork, intptr_t info,
        intptr_t params, int batchSize):
    _setStream(handle)
    with nogil:
        status = cusolverDnSgesvdjBatched(
            <Handle>handle, <EigMode>jobz, m, n, <float*>A, lda,
            <float*>S, <float*>U, ldu, <float*>V, ldv,
            <float*>work, lwork, <int*>info,
            <GesvdjInfo>params, batchSize)
    check_status(status)

cpdef dgesvdjBatched(
        intptr_t handle, int jobz, int m, int n, intptr_t A,
        int lda, intptr_t S, intptr_t U, int ldu, intptr_t V, int ldv,
        intptr_t work, int lwork, intptr_t info,
        intptr_t params, int batchSize):
    _setStream(handle)
    with nogil:
        status = cusolverDnDgesvdjBatched(
            <Handle>handle, <EigMode>jobz, m, n, <double*>A, lda,
            <double*>S, <double*>U, ldu, <double*>V, ldv,
            <double*>work, lwork, <int*>info,
            <GesvdjInfo>params, batchSize)
    check_status(status)

cpdef cgesvdjBatched(
        intptr_t handle, int jobz, int m, int n, intptr_t A,
        int lda, intptr_t S, intptr_t U, int ldu, intptr_t V, int ldv,
        intptr_t work, int lwork, intptr_t info,
        intptr_t params, int batchSize):
    _setStream(handle)
    with nogil:
        status = cusolverDnCgesvdjBatched(
            <Handle>handle, <EigMode>jobz, m, n, <cuComplex*>A, lda,
            <float*>S, <cuComplex*>U, ldu, <cuComplex*>V, ldv,
            <cuComplex*>work, lwork, <int*>info,
            <GesvdjInfo>params, batchSize)
    check_status(status)

cpdef zgesvdjBatched(
        intptr_t handle, int jobz, int m, int n, intptr_t A,
        int lda, intptr_t S, intptr_t U, int ldu, intptr_t V, int ldv,
        intptr_t work, int lwork, intptr_t info,
        intptr_t params, int batchSize):
    _setStream(handle)
    with nogil:
        status = cusolverDnZgesvdjBatched(
            <Handle>handle, <EigMode>jobz, m, n, <cuDoubleComplex*>A, lda,
            <double*>S, <cuDoubleComplex*>U, ldu, <cuDoubleComplex*>V, ldv,
            <cuDoubleComplex*>work, lwork, <int*>info,
            <GesvdjInfo>params, batchSize)
    check_status(status)

# gesvda ... Approximate singular value decomposition
cpdef int sgesvdaStridedBatched_bufferSize(
        intptr_t handle, int jobz, int rank, int m, int n, intptr_t d_A,
        int lda, long long int strideA, intptr_t d_S, long long int strideS,
        intptr_t d_U, int ldu, long long int strideU, intptr_t d_V, int ldv,
        long long int strideV, int batchSize):
    cdef int lwork
    status = cusolverDnSgesvdaStridedBatched_bufferSize(
        <Handle>handle, <EigMode>jobz, rank, m, n, <const float*>d_A, lda,
        strideA, <const float*>d_S, strideS, <const float*>d_U, ldu, strideU,
        <const float*>d_V, ldv, strideV, &lwork, batchSize)
    check_status(status)
    return lwork

cpdef int dgesvdaStridedBatched_bufferSize(
        intptr_t handle, int jobz, int rank, int m, int n, intptr_t d_A,
        int lda, long long int strideA, intptr_t d_S, long long int strideS,
        intptr_t d_U, int ldu, long long int strideU, intptr_t d_V, int ldv,
        long long int strideV, int batchSize):
    cdef int lwork
    status = cusolverDnDgesvdaStridedBatched_bufferSize(
        <Handle>handle, <EigMode>jobz, rank, m, n, <const double*>d_A, lda,
        strideA, <const double*>d_S, strideS, <const double*>d_U, ldu, strideU,
        <const double*>d_V, ldv, strideV, &lwork, batchSize)
    check_status(status)
    return lwork

cpdef int cgesvdaStridedBatched_bufferSize(
        intptr_t handle, int jobz, int rank, int m, int n, intptr_t d_A,
        int lda, long long int strideA, intptr_t d_S, long long int strideS,
        intptr_t d_U, int ldu, long long int strideU, intptr_t d_V, int ldv,
        long long int strideV, int batchSize):
    cdef int lwork
    status = cusolverDnCgesvdaStridedBatched_bufferSize(
        <Handle>handle, <EigMode>jobz, rank, m, n, <const cuComplex*>d_A, lda,
        strideA, <const float*>d_S, strideS, <const cuComplex*>d_U, ldu,
        strideU, <const cuComplex*>d_V, ldv, strideV, &lwork, batchSize)
    check_status(status)
    return lwork

cpdef int zgesvdaStridedBatched_bufferSize(
        intptr_t handle, int jobz, int rank, int m, int n, intptr_t d_A,
        int lda, long long int strideA, intptr_t d_S, long long int strideS,
        intptr_t d_U, int ldu, long long int strideU, intptr_t d_V, int ldv,
        long long int strideV, int batchSize):
    cdef int lwork
    status = cusolverDnZgesvdaStridedBatched_bufferSize(
        <Handle>handle, <EigMode>jobz, rank, m, n, <const cuDoubleComplex*>d_A,
        lda, strideA, <const double*>d_S, strideS, <const cuDoubleComplex*>d_U,
        ldu, strideU, <const cuDoubleComplex*>d_V, ldv, strideV, &lwork,
        batchSize)
    check_status(status)
    return lwork

cpdef sgesvdaStridedBatched(
        intptr_t handle, int jobz, int rank, int m, int n, intptr_t d_A,
        int lda, long long int strideA, intptr_t d_S, long long int strideS,
        intptr_t d_U, int ldu, long long int strideU, intptr_t d_V, int ldv,
        long long int strideV, intptr_t d_work, int lwork, intptr_t d_info,
        intptr_t h_R_nrmF, int batchSize):
    _setStream(handle)
    with nogil:
        status = cusolverDnSgesvdaStridedBatched(
            <Handle>handle, <EigMode>jobz, rank, m, n, <const float*>d_A, lda,
            strideA, <float*>d_S, strideS, <float*>d_U, ldu, strideU,
            <float*>d_V, ldv, strideV, <float*>d_work, lwork, <int*>d_info,
            <double*>h_R_nrmF, batchSize)
    check_status(status)

cpdef dgesvdaStridedBatched(
        intptr_t handle, int jobz, int rank, int m, int n, intptr_t d_A,
        int lda, long long int strideA, intptr_t d_S, long long int strideS,
        intptr_t d_U, int ldu, long long int strideU, intptr_t d_V, int ldv,
        long long int strideV, intptr_t d_work, int lwork, intptr_t d_info,
        intptr_t h_R_nrmF, int batchSize):
    _setStream(handle)
    with nogil:
        status = cusolverDnDgesvdaStridedBatched(
            <Handle>handle, <EigMode>jobz, rank, m, n, <const double*>d_A, lda,
            strideA, <double*>d_S, strideS, <double*>d_U, ldu, strideU,
            <double*>d_V, ldv, strideV, <double*>d_work, lwork, <int*>d_info,
            <double*>h_R_nrmF, batchSize)
    check_status(status)

cpdef cgesvdaStridedBatched(
        intptr_t handle, int jobz, int rank, int m, int n, intptr_t d_A,
        int lda, long long int strideA, intptr_t d_S, long long int strideS,
        intptr_t d_U, int ldu, long long int strideU, intptr_t d_V, int ldv,
        long long int strideV, intptr_t d_work, int lwork, intptr_t d_info,
        intptr_t h_R_nrmF, int batchSize):
    _setStream(handle)
    with nogil:
        status = cusolverDnCgesvdaStridedBatched(
            <Handle>handle, <EigMode>jobz, rank, m, n, <const cuComplex*>d_A,
            lda, strideA, <float*>d_S, strideS, <cuComplex*>d_U, ldu, strideU,
            <cuComplex*>d_V, ldv, strideV, <cuComplex*>d_work, lwork,
            <int*>d_info, <double*>h_R_nrmF, batchSize)
    check_status(status)

cpdef zgesvdaStridedBatched(
        intptr_t handle, int jobz, int rank, int m, int n, intptr_t d_A,
        int lda, long long int strideA, intptr_t d_S, long long int strideS,
        intptr_t d_U, int ldu, long long int strideU, intptr_t d_V, int ldv,
        long long int strideV, intptr_t d_work, int lwork, intptr_t d_info,
        intptr_t h_R_nrmF, int batchSize):
    _setStream(handle)
    with nogil:
        status = cusolverDnZgesvdaStridedBatched(
            <Handle>handle, <EigMode>jobz, rank, m, n,
            <const cuDoubleComplex*>d_A, lda, strideA, <double*>d_S, strideS,
            <cuDoubleComplex*>d_U, ldu, strideU, <cuDoubleComplex*>d_V, ldv,
            strideV, <cuDoubleComplex*>d_work, lwork, <int*>d_info,
            <double*>h_R_nrmF, batchSize)
    check_status(status)

# Standard symmetric eigenvalue solver
cpdef int ssyevd_bufferSize(intptr_t handle, int jobz, int uplo, int n,
                            size_t A, int lda, size_t W) except? -1:
    cdef int lwork, status
    _setStream(handle)
    with nogil:
        status = cusolverDnSsyevd_bufferSize(
            <Handle>handle, <EigMode>jobz, <FillMode>uplo, n,
            <const float*>A,
            lda, <const float*>W, &lwork)
    check_status(status)
    return lwork

cpdef int dsyevd_bufferSize(intptr_t handle, int jobz, int uplo, int n,
                            size_t A, int lda, size_t W) except? -1:
    cdef int lwork, status
    _setStream(handle)
    with nogil:
        status = cusolverDnDsyevd_bufferSize(
            <Handle>handle, <EigMode>jobz, <FillMode>uplo, n,
            <const double*>A,
            lda, <const double*>W, &lwork)
    check_status(status)
    return lwork

cpdef int cheevd_bufferSize(intptr_t handle, int jobz, int uplo, int n,
                            size_t A, int lda, size_t W) except? -1:
    cdef int lwork, status
    _setStream(handle)
    with nogil:
        status = cusolverDnCheevd_bufferSize(
            <Handle>handle, <EigMode>jobz, <FillMode>uplo, n,
            <const cuComplex*>A,
            lda, <const float*>W, &lwork)
    check_status(status)
    return lwork

cpdef int zheevd_bufferSize(intptr_t handle, int jobz, int uplo, int n,
                            size_t A, int lda, size_t W) except? -1:
    cdef int lwork, status
    _setStream(handle)
    with nogil:
        status = cusolverDnZheevd_bufferSize(
            <Handle>handle, <EigMode>jobz, <FillMode>uplo, n,
            <const cuDoubleComplex*>A,
            lda, <const double*>W, &lwork)
    check_status(status)
    return lwork

cpdef ssyevd(intptr_t handle, int jobz, int uplo, int n, size_t A, int lda,
             size_t W, size_t work, int lwork, size_t info):
    cdef int status
    _setStream(handle)
    with nogil:
        status = cusolverDnSsyevd(
            <Handle>handle, <EigMode>jobz, <FillMode>uplo, n,
            <float*>A, lda, <float*>W,
            <float*>work, lwork, <int*>info)
    check_status(status)

cpdef dsyevd(intptr_t handle, int jobz, int uplo, int n, size_t A, int lda,
             size_t W, size_t work, int lwork, size_t info):
    cdef int status
    _setStream(handle)
    with nogil:
        status = cusolverDnDsyevd(
            <Handle>handle, <EigMode>jobz, <FillMode>uplo, n,
            <double*>A, lda, <double*>W,
            <double*>work, lwork, <int*>info)
    check_status(status)

cpdef cheevd(intptr_t handle, int jobz, int uplo, int n, size_t A, int lda,
             size_t W, size_t work, int lwork, size_t info):
    cdef int status
    _setStream(handle)
    with nogil:
        status = cusolverDnCheevd(
            <Handle>handle, <EigMode>jobz, <FillMode>uplo, n,
            <cuComplex*>A, lda, <float*>W,
            <cuComplex*>work, lwork, <int*>info)
    check_status(status)

cpdef zheevd(intptr_t handle, int jobz, int uplo, int n, size_t A, int lda,
             size_t W, size_t work, int lwork, size_t info):
    cdef int status
    _setStream(handle)
    with nogil:
        status = cusolverDnZheevd(
            <Handle>handle, <EigMode>jobz, <FillMode>uplo, n,
            <cuDoubleComplex*>A, lda, <double*>W,
            <cuDoubleComplex*>work, lwork, <int*>info)
    check_status(status)

# Symmetric eigenvalue solver via Jacobi method
cpdef intptr_t createSyevjInfo() except? 0:
    cdef SyevjInfo info
    status = cusolverDnCreateSyevjInfo(&info)
    check_status(status)
    return <intptr_t>info

cpdef destroySyevjInfo(intptr_t info):
    status = cusolverDnDestroySyevjInfo(<SyevjInfo>info)
    check_status(status)

cpdef xsyevjSetTolerance(intptr_t info, double tolerance):
    status = cusolverDnXsyevjSetTolerance(<SyevjInfo>info, tolerance)
    check_status(status)

cpdef xsyevjSetMaxSweeps(intptr_t info, int max_sweeps):
    status = cusolverDnXsyevjSetMaxSweeps(<SyevjInfo>info, max_sweeps)
    check_status(status)

cpdef xsyevjSetSortEig(intptr_t info, int sort_eig):
    status = cusolverDnXsyevjSetSortEig(<SyevjInfo>info, sort_eig)
    check_status(status)

cpdef double xsyevjGetResidual(intptr_t handle, intptr_t info):
    cdef double residual
    status = cusolverDnXsyevjGetResidual(
        <Handle>handle, <SyevjInfo>info, &residual)
    check_status(status)
    return residual

cpdef int xsyevjGetSweeps(intptr_t handle, intptr_t info):
    cdef int executed_sweeps
    status = cusolverDnXsyevjGetSweeps(
        <Handle>handle, <SyevjInfo>info, &executed_sweeps)
    check_status(status)
    return executed_sweeps

cpdef int ssyevj_bufferSize(intptr_t handle, int jobz, int uplo,
                            int n, size_t A, int lda, size_t W,
                            intptr_t params) except? -1:
    cdef int lwork, status
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnSsyevj_bufferSize(
            <Handle>handle, <EigMode>jobz, <FillMode>uplo, n,
            <const float*>A,
            lda, <const float*>W, &lwork, <SyevjInfo>params)
    check_status(status)
    return lwork

cpdef int dsyevj_bufferSize(intptr_t handle, int jobz, int uplo,
                            int n, size_t A, int lda, size_t W,
                            intptr_t params) except? -1:
    cdef int lwork, status
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnDsyevj_bufferSize(
            <Handle>handle, <EigMode>jobz, <FillMode>uplo, n,
            <const double*>A,
            lda, <const double*>W, &lwork, <SyevjInfo>params)
    check_status(status)
    return lwork

cpdef int cheevj_bufferSize(intptr_t handle, int jobz, int uplo,
                            int n, size_t A, int lda, size_t W,
                            intptr_t params) except? -1:
    cdef int lwork, status
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnCheevj_bufferSize(
            <Handle>handle, <EigMode>jobz, <FillMode>uplo, n,
            <const cuComplex*>A,
            lda, <const float*>W, &lwork, <SyevjInfo>params)
    check_status(status)
    return lwork

cpdef int zheevj_bufferSize(intptr_t handle, int jobz, int uplo,
                            int n, size_t A, int lda, size_t W,
                            intptr_t params) except? -1:
    cdef int lwork, status
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnZheevj_bufferSize(
            <Handle>handle, <EigMode>jobz, <FillMode>uplo, n,
            <const cuDoubleComplex*>A,
            lda, <const double*>W, &lwork, <SyevjInfo>params)
    check_status(status)
    return lwork

cpdef ssyevj(intptr_t handle, int jobz, int uplo, int n, size_t A, int lda,
             size_t W, size_t work, int lwork, size_t info, intptr_t params):
    cdef int status
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnSsyevj(
            <Handle>handle, <EigMode>jobz, <FillMode>uplo, n,
            <float*>A, lda, <float*>W,
            <float*>work, lwork, <int*>info, <SyevjInfo>params)
    check_status(status)

cpdef dsyevj(intptr_t handle, int jobz, int uplo, int n, size_t A, int lda,
             size_t W, size_t work, int lwork, size_t info, intptr_t params):
    cdef int status
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnDsyevj(
            <Handle>handle, <EigMode>jobz, <FillMode>uplo, n,
            <double*>A, lda, <double*>W,
            <double*>work, lwork, <int*>info, <SyevjInfo>params)
    check_status(status)

cpdef cheevj(intptr_t handle, int jobz, int uplo, int n, size_t A, int lda,
             size_t W, size_t work, int lwork, size_t info, intptr_t params):
    cdef int status
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnCheevj(
            <Handle>handle, <EigMode>jobz, <FillMode>uplo, n,
            <cuComplex*>A, lda, <float*>W,
            <cuComplex*>work, lwork, <int*>info, <SyevjInfo>params)
    check_status(status)

cpdef zheevj(intptr_t handle, int jobz, int uplo, int n, size_t A, int lda,
             size_t W, size_t work, int lwork, size_t info, intptr_t params):
    cdef int status
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnZheevj(
            <Handle>handle, <EigMode>jobz, <FillMode>uplo, n,
            <cuDoubleComplex*>A, lda, <double*>W,
            <cuDoubleComplex*>work, lwork, <int*>info, <SyevjInfo>params)
    check_status(status)

# Batched symmetric eigenvalue solver via Jacobi method

cpdef int ssyevjBatched_bufferSize(
        intptr_t handle, int jobz, int uplo, int n,
        size_t A, int lda, size_t W, intptr_t params,
        int batchSize) except? -1:
    cdef int lwork, status
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnSsyevjBatched_bufferSize(
            <Handle>handle, <EigMode>jobz, <FillMode>uplo, n,
            <const float *>A, lda, <const float *>W, &lwork,
            <SyevjInfo>params, batchSize)
    check_status(status)
    return lwork

cpdef int dsyevjBatched_bufferSize(
        intptr_t handle, int jobz, int uplo, int n,
        size_t A, int lda, size_t W, intptr_t params,
        int batchSize) except? -1:
    cdef int lwork, status
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnDsyevjBatched_bufferSize(
            <Handle>handle, <EigMode>jobz, <FillMode>uplo, n,
            <const double *>A, lda, <const double *>W, &lwork,
            <SyevjInfo>params, batchSize)
    check_status(status)
    return lwork

cpdef int cheevjBatched_bufferSize(
        intptr_t handle, int jobz, int uplo, int n,
        size_t A, int lda, size_t W, intptr_t params,
        int batchSize) except? -1:
    cdef int lwork, status
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnCheevjBatched_bufferSize(
            <Handle>handle, <EigMode>jobz, <FillMode>uplo, n,
            <const cuComplex *>A, lda, <const float *>W, &lwork,
            <SyevjInfo>params, batchSize)
    check_status(status)
    return lwork

cpdef int zheevjBatched_bufferSize(
        intptr_t handle, int jobz, int uplo, int n,
        size_t A, int lda, size_t W, intptr_t params,
        int batchSize) except? -1:
    cdef int lwork, status
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnZheevjBatched_bufferSize(
            <Handle>handle, <EigMode>jobz, <FillMode>uplo, n,
            <const cuDoubleComplex *>A, lda, <const double *>W, &lwork,
            <SyevjInfo>params, batchSize)
    check_status(status)
    return lwork

cpdef ssyevjBatched(intptr_t handle, int jobz, int uplo, int n,
                    size_t A, int lda, size_t W, size_t work, int lwork,
                    size_t info, intptr_t params, int batchSize):
    cdef int status
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnSsyevjBatched(
            <Handle>handle, <EigMode>jobz, <FillMode>uplo, n,
            <float*>A, lda, <float*>W,
            <float*>work, lwork, <int*>info, <SyevjInfo>params, batchSize)
    check_status(status)

cpdef dsyevjBatched(intptr_t handle, int jobz, int uplo, int n,
                    size_t A, int lda, size_t W, size_t work, int lwork,
                    size_t info, intptr_t params, int batchSize):
    cdef int status
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnDsyevjBatched(
            <Handle>handle, <EigMode>jobz, <FillMode>uplo, n,
            <double*>A, lda, <double*>W,
            <double*>work, lwork, <int*>info, <SyevjInfo>params, batchSize)
    check_status(status)

cpdef cheevjBatched(intptr_t handle, int jobz, int uplo, int n,
                    size_t A, int lda, size_t W, size_t work, int lwork,
                    size_t info, intptr_t params, int batchSize):
    cdef int status
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnCheevjBatched(
            <Handle>handle, <EigMode>jobz, <FillMode>uplo, n,
            <cuComplex*>A, lda, <float*>W,
            <cuComplex*>work, lwork, <int*>info, <SyevjInfo>params, batchSize)
    check_status(status)

cpdef zheevjBatched(intptr_t handle, int jobz, int uplo, int n,
                    size_t A, int lda, size_t W, size_t work, int lwork,
                    size_t info, intptr_t params, int batchSize):
    cdef int status
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnZheevjBatched(
            <Handle>handle, <EigMode>jobz, <FillMode>uplo, n,
            <cuDoubleComplex*>A, lda, <double*>W,
            <cuDoubleComplex*>work, lwork, <int*>info,
            <SyevjInfo>params, batchSize)
    check_status(status)

###############################################################################
# Sparse LAPACK Functions
###############################################################################
cpdef scsrlsvchol(intptr_t handle, int m, int nnz, size_t descrA,
                  size_t csrValA, size_t csrRowPtrA, size_t csrColIndA,
                  size_t b, float tol, int reorder, size_t x,
                  size_t singularity):
    cdef int status
    _spSetStream(handle)
    with nogil:
        status = cusolverSpScsrlsvchol(
            <SpHandle>handle, m, nnz, <const MatDescr> descrA,
            <const float*> csrValA, <const int*> csrRowPtrA,
            <const int*> csrColIndA, <const float*> b,
            tol, reorder, <float*> x, <int*> singularity)
    check_status(status)

cpdef dcsrlsvchol(intptr_t handle, int m, int nnz, size_t descrA,
                  size_t csrValA, size_t csrRowPtrA, size_t csrColIndA,
                  size_t b, double tol, int reorder, size_t x,
                  size_t singularity):
    cdef int status
    _spSetStream(handle)
    with nogil:
        status = cusolverSpDcsrlsvchol(
            <SpHandle>handle, m, nnz, <const MatDescr> descrA,
            <const double*> csrValA, <const int*> csrRowPtrA,
            <const int*> csrColIndA, <const double*> b,
            tol, reorder, <double*> x, <int*> singularity)
    check_status(status)

cpdef ccsrlsvchol(intptr_t handle, int m, int nnz, size_t descrA,
                  size_t csrVal, size_t csrRowPtr, size_t csrColInd, size_t b,
                  float tol, int reorder, size_t x, size_t singularity):
    cdef int status
    _spSetStream(handle)
    with nogil:
        status = cusolverSpCcsrlsvchol(
            <SpHandle>handle, m, nnz, <const MatDescr>descrA,
            <const cuComplex*>csrVal, <const int*>csrRowPtr,
            <const int*>csrColInd, <const cuComplex*>b, tol, reorder,
            <cuComplex*>x, <int*>singularity)
    check_status(status)

cpdef zcsrlsvchol(intptr_t handle, int m, int nnz, size_t descrA,
                  size_t csrVal, size_t csrRowPtr, size_t csrColInd, size_t b,
                  double tol, int reorder, size_t x, size_t singularity):
    cdef int status
    _spSetStream(handle)
    with nogil:
        status = cusolverSpZcsrlsvchol(
            <SpHandle>handle, m, nnz, <const MatDescr>descrA,
            <const cuDoubleComplex*>csrVal, <const int*>csrRowPtr,
            <const int*>csrColInd, <const cuDoubleComplex*>b, tol, reorder,
            <cuDoubleComplex*>x, <int*>singularity)
    check_status(status)

cpdef scsrlsvqr(intptr_t handle, int m, int nnz, size_t descrA, size_t csrValA,
                size_t csrRowPtrA, size_t csrColIndA, size_t b, float tol,
                int reorder, size_t x, size_t singularity):
    cdef int status
    _spSetStream(handle)
    with nogil:
        status = cusolverSpScsrlsvqr(
            <SpHandle>handle, m, nnz, <const MatDescr> descrA,
            <const float*> csrValA, <const int*> csrRowPtrA,
            <const int*> csrColIndA, <const float*> b,
            tol, reorder, <float*> x, <int*> singularity)
    check_status(status)

cpdef dcsrlsvqr(intptr_t handle, int m, int nnz, size_t descrA, size_t csrValA,
                size_t csrRowPtrA, size_t csrColIndA, size_t b, double tol,
                int reorder, size_t x, size_t singularity):
    cdef int status
    _spSetStream(handle)
    with nogil:
        status = cusolverSpDcsrlsvqr(
            <SpHandle>handle, m, nnz, <const MatDescr> descrA,
            <const double*> csrValA, <const int*> csrRowPtrA,
            <const int*> csrColIndA, <const double*> b,
            tol, reorder, <double*> x, <int*> singularity)
    check_status(status)

cpdef ccsrlsvqr(intptr_t handle, int m, int nnz, size_t descrA, size_t csrVal,
                size_t csrRowPtr, size_t csrColInd, size_t b, float tol,
                int reorder, size_t x, size_t singularity):
    cdef int status
    _spSetStream(handle)
    with nogil:
        status = cusolverSpCcsrlsvqr(
            <SpHandle>handle, m, nnz, <const MatDescr>descrA,
            <const cuComplex*>csrVal, <const int*>csrRowPtr,
            <const int*>csrColInd, <const cuComplex*>b, tol, reorder,
            <cuComplex*>x, <int*>singularity)
    check_status(status)

cpdef zcsrlsvqr(intptr_t handle, int m, int nnz, size_t descrA, size_t csrVal,
                size_t csrRowPtr, size_t csrColInd, size_t b, double tol,
                int reorder, size_t x, size_t singularity):
    cdef int status
    _spSetStream(handle)
    with nogil:
        status = cusolverSpZcsrlsvqr(
            <SpHandle>handle, m, nnz, <const MatDescr>descrA,
            <const cuDoubleComplex*>csrVal, <const int*>csrRowPtr,
            <const int*>csrColInd, <const cuDoubleComplex*>b, tol, reorder,
            <cuDoubleComplex*>x, <int*>singularity)
    check_status(status)

cpdef scsreigvsi(intptr_t handle, int m, int nnz, size_t descrA,
                 size_t csrValA, size_t csrRowPtrA, size_t csrColIndA,
                 float mu0, size_t x0, int maxite, float eps, size_t mu,
                 size_t x):
    cdef int status
    _spSetStream(handle)
    with nogil:
        status = cusolverSpScsreigvsi(
            <SpHandle>handle, m, nnz, <const MatDescr>descrA,
            <const float*>csrValA, <const int*>csrRowPtrA,
            <const int*>csrColIndA, mu0, <const float*>x0, maxite, eps,
            <float*>mu, <float*>x)
    check_status(status)

cpdef dcsreigvsi(intptr_t handle, int m, int nnz, size_t descrA,
                 size_t csrValA, size_t csrRowPtrA, size_t csrColIndA,
                 double mu0, size_t x0, int maxite, double eps, size_t mu,
                 size_t x):
    cdef int status
    _spSetStream(handle)
    with nogil:
        status = cusolverSpDcsreigvsi(
            <SpHandle>handle, m, nnz, <const MatDescr>descrA,
            <const double*>csrValA, <const int*>csrRowPtrA,
            <const int*>csrColIndA, mu0, <const double*>x0, maxite, eps,
            <double*>mu, <double*>x)
    check_status(status)

cpdef ccsreigvsi(intptr_t handle, int m, int nnz, size_t descrA,
                 size_t csrValA, size_t csrRowPtrA, size_t csrColIndA,
                 size_t mu0, size_t x0, int maxite, float eps, size_t mu,
                 size_t x):
    cdef int status
    _spSetStream(handle)
    with nogil:
        status = cusolverSpCcsreigvsi(
            <SpHandle>handle, m, nnz, <const MatDescr>descrA,
            <const cuComplex*>csrValA, <const int*>csrRowPtrA,
            <const int*>csrColIndA, (<cuComplex*>mu0)[0], <const cuComplex*>x0,
            maxite, eps, <cuComplex*>mu, <cuComplex*>x)
    check_status(status)

cpdef zcsreigvsi(intptr_t handle, int m, int nnz, size_t descrA,
                 size_t csrValA, size_t csrRowPtrA, size_t csrColIndA,
                 size_t mu0, size_t x0, int maxite, double eps, size_t mu,
                 size_t x):
    cdef int status
    _spSetStream(handle)
    with nogil:
        status = cusolverSpZcsreigvsi(
            <SpHandle>handle, m, nnz, <const MatDescr>descrA,
            <const cuDoubleComplex*>csrValA, <const int*>csrRowPtrA,
            <const int*>csrColIndA, (<cuDoubleComplex*>mu0)[0],
            <const cuDoubleComplex*>x0, maxite,
            eps, <cuDoubleComplex*>mu, <cuDoubleComplex*>x)
    check_status(status)
