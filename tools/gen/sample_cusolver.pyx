# This code was automatically generated. Do not modify it directly.

cimport cython  # NOQA

from cupy_backends.cuda.api cimport driver
from cupy_backends.cuda.api cimport runtime
from cupy_backends.cuda.libs.cublas cimport FillMode
from cupy_backends.cuda.libs.cublas cimport Operation
from cupy_backends.cuda.libs.cublas cimport SideMode
from cupy_backends.cuda.libs.cusparse cimport MatDescr
from cupy_backends.cuda cimport stream as stream_module


cdef extern from *:
    ctypedef void* LibraryPropertyType 'libraryPropertyType_t'

cdef extern from '../../cupy_complex.h':
    ctypedef struct cuComplex 'cuComplex':
        float x, y

    ctypedef struct cuDoubleComplex 'cuDoubleComplex':
        double x, y

cdef extern from '../../cupy_lapack.h' nogil:

    # Library Attributes
    Status cusolverGetProperty(LibraryPropertyType type, int* value)

    # cuSOLVER Dense LAPACK Function - Helper Function
    Status cusolverDnCreate(DnHandle* handle)
    Status cusolverDnDestroy(DnHandle handle)
    Status cusolverDnSetStream(DnHandle handle, driver.Stream streamId)
    Status cusolverDnGetStream(DnHandle handle, driver.Stream* streamId)
    Status cusolverDnCreateSyevjInfo(syevjInfo_t* info)
    Status cusolverDnDestroySyevjInfo(syevjInfo_t info)
    Status cusolverDnXsyevjSetTolerance(syevjInfo_t info, double tolerance)
    Status cusolverDnXsyevjSetMaxSweeps(syevjInfo_t info, int max_sweeps)
    Status cusolverDnXsyevjSetSortEig(syevjInfo_t info, int sort_eig)
    Status cusolverDnXsyevjGetResidual(DnHandle handle, syevjInfo_t info, double* residual)
    Status cusolverDnXsyevjGetSweeps(DnHandle handle, syevjInfo_t info, int* executed_sweeps)
    Status cusolverDnCreateGesvdjInfo(gesvdjInfo_t* info)
    Status cusolverDnDestroyGesvdjInfo(gesvdjInfo_t info)
    Status cusolverDnXgesvdjSetTolerance(gesvdjInfo_t info, double tolerance)
    Status cusolverDnXgesvdjSetMaxSweeps(gesvdjInfo_t info, int max_sweeps)
    Status cusolverDnXgesvdjSetSortEig(gesvdjInfo_t info, int sort_svd)
    Status cusolverDnXgesvdjGetResidual(DnHandle handle, gesvdjInfo_t info, double* residual)
    Status cusolverDnXgesvdjGetSweeps(DnHandle handle, gesvdjInfo_t info, int* executed_sweeps)

    # cuSOLVER Dense LAPACK Function - Dense Linear Solver
    Status cusolverDnSpotrf_bufferSize(DnHandle handle, FillMode uplo, int n, float* A, int lda, int* Lwork)
    Status cusolverDnDpotrf_bufferSize(DnHandle handle, FillMode uplo, int n, double* A, int lda, int* Lwork)
    Status cusolverDnCpotrf_bufferSize(DnHandle handle, FillMode uplo, int n, cuComplex* A, int lda, int* Lwork)
    Status cusolverDnZpotrf_bufferSize(DnHandle handle, FillMode uplo, int n, cuDoubleComplex* A, int lda, int* Lwork)
    Status cusolverDnSpotrf(DnHandle handle, FillMode uplo, int n, float* A, int lda, float* Workspace, int Lwork, int* devInfo)
    Status cusolverDnDpotrf(DnHandle handle, FillMode uplo, int n, double* A, int lda, double* Workspace, int Lwork, int* devInfo)
    Status cusolverDnCpotrf(DnHandle handle, FillMode uplo, int n, cuComplex* A, int lda, cuComplex* Workspace, int Lwork, int* devInfo)
    Status cusolverDnZpotrf(DnHandle handle, FillMode uplo, int n, cuDoubleComplex* A, int lda, cuDoubleComplex* Workspace, int Lwork, int* devInfo)
    Status cusolverDnSpotrs(DnHandle handle, FillMode uplo, int n, int nrhs, const float* A, int lda, float* B, int ldb, int* devInfo)
    Status cusolverDnDpotrs(DnHandle handle, FillMode uplo, int n, int nrhs, const double* A, int lda, double* B, int ldb, int* devInfo)
    Status cusolverDnCpotrs(DnHandle handle, FillMode uplo, int n, int nrhs, const cuComplex* A, int lda, cuComplex* B, int ldb, int* devInfo)
    Status cusolverDnZpotrs(DnHandle handle, FillMode uplo, int n, int nrhs, const cuDoubleComplex* A, int lda, cuDoubleComplex* B, int ldb, int* devInfo)
    Status cusolverDnSpotrfBatched(DnHandle handle, FillMode uplo, int n, float** Aarray, int lda, int* infoArray, int batchSize)
    Status cusolverDnDpotrfBatched(DnHandle handle, FillMode uplo, int n, double** Aarray, int lda, int* infoArray, int batchSize)
    Status cusolverDnCpotrfBatched(DnHandle handle, FillMode uplo, int n, cuComplex** Aarray, int lda, int* infoArray, int batchSize)
    Status cusolverDnZpotrfBatched(DnHandle handle, FillMode uplo, int n, cuDoubleComplex** Aarray, int lda, int* infoArray, int batchSize)
    Status cusolverDnSpotrsBatched(DnHandle handle, FillMode uplo, int n, int nrhs, float** A, int lda, float** B, int ldb, int* d_info, int batchSize)
    Status cusolverDnDpotrsBatched(DnHandle handle, FillMode uplo, int n, int nrhs, double** A, int lda, double** B, int ldb, int* d_info, int batchSize)
    Status cusolverDnCpotrsBatched(DnHandle handle, FillMode uplo, int n, int nrhs, cuComplex** A, int lda, cuComplex** B, int ldb, int* d_info, int batchSize)
    Status cusolverDnZpotrsBatched(DnHandle handle, FillMode uplo, int n, int nrhs, cuDoubleComplex** A, int lda, cuDoubleComplex** B, int ldb, int* d_info, int batchSize)
    Status cusolverDnSgetrf_bufferSize(DnHandle handle, int m, int n, float* A, int lda, int* Lwork)
    Status cusolverDnDgetrf_bufferSize(DnHandle handle, int m, int n, double* A, int lda, int* Lwork)
    Status cusolverDnCgetrf_bufferSize(DnHandle handle, int m, int n, cuComplex* A, int lda, int* Lwork)
    Status cusolverDnZgetrf_bufferSize(DnHandle handle, int m, int n, cuDoubleComplex* A, int lda, int* Lwork)
    Status cusolverDnSgetrf(DnHandle handle, int m, int n, float* A, int lda, float* Workspace, int* devIpiv, int* devInfo)
    Status cusolverDnDgetrf(DnHandle handle, int m, int n, double* A, int lda, double* Workspace, int* devIpiv, int* devInfo)
    Status cusolverDnCgetrf(DnHandle handle, int m, int n, cuComplex* A, int lda, cuComplex* Workspace, int* devIpiv, int* devInfo)
    Status cusolverDnZgetrf(DnHandle handle, int m, int n, cuDoubleComplex* A, int lda, cuDoubleComplex* Workspace, int* devIpiv, int* devInfo)
    Status cusolverDnSgetrs(DnHandle handle, Operation trans, int n, int nrhs, const float* A, int lda, const int* devIpiv, float* B, int ldb, int* devInfo)
    Status cusolverDnDgetrs(DnHandle handle, Operation trans, int n, int nrhs, const double* A, int lda, const int* devIpiv, double* B, int ldb, int* devInfo)
    Status cusolverDnCgetrs(DnHandle handle, Operation trans, int n, int nrhs, const cuComplex* A, int lda, const int* devIpiv, cuComplex* B, int ldb, int* devInfo)
    Status cusolverDnZgetrs(DnHandle handle, Operation trans, int n, int nrhs, const cuDoubleComplex* A, int lda, const int* devIpiv, cuDoubleComplex* B, int ldb, int* devInfo)
    Status cusolverDnSgeqrf_bufferSize(DnHandle handle, int m, int n, float* A, int lda, int* lwork)
    Status cusolverDnDgeqrf_bufferSize(DnHandle handle, int m, int n, double* A, int lda, int* lwork)
    Status cusolverDnCgeqrf_bufferSize(DnHandle handle, int m, int n, cuComplex* A, int lda, int* lwork)
    Status cusolverDnZgeqrf_bufferSize(DnHandle handle, int m, int n, cuDoubleComplex* A, int lda, int* lwork)
    Status cusolverDnSgeqrf(DnHandle handle, int m, int n, float* A, int lda, float* TAU, float* Workspace, int Lwork, int* devInfo)
    Status cusolverDnDgeqrf(DnHandle handle, int m, int n, double* A, int lda, double* TAU, double* Workspace, int Lwork, int* devInfo)
    Status cusolverDnCgeqrf(DnHandle handle, int m, int n, cuComplex* A, int lda, cuComplex* TAU, cuComplex* Workspace, int Lwork, int* devInfo)
    Status cusolverDnZgeqrf(DnHandle handle, int m, int n, cuDoubleComplex* A, int lda, cuDoubleComplex* TAU, cuDoubleComplex* Workspace, int Lwork, int* devInfo)
    Status cusolverDnSorgqr_bufferSize(DnHandle handle, int m, int n, int k, const float* A, int lda, const float* tau, int* lwork)
    Status cusolverDnDorgqr_bufferSize(DnHandle handle, int m, int n, int k, const double* A, int lda, const double* tau, int* lwork)
    Status cusolverDnCungqr_bufferSize(DnHandle handle, int m, int n, int k, const cuComplex* A, int lda, const cuComplex* tau, int* lwork)
    Status cusolverDnZungqr_bufferSize(DnHandle handle, int m, int n, int k, const cuDoubleComplex* A, int lda, const cuDoubleComplex* tau, int* lwork)
    Status cusolverDnSorgqr(DnHandle handle, int m, int n, int k, float* A, int lda, const float* tau, float* work, int lwork, int* info)
    Status cusolverDnDorgqr(DnHandle handle, int m, int n, int k, double* A, int lda, const double* tau, double* work, int lwork, int* info)
    Status cusolverDnCungqr(DnHandle handle, int m, int n, int k, cuComplex* A, int lda, const cuComplex* tau, cuComplex* work, int lwork, int* info)
    Status cusolverDnZungqr(DnHandle handle, int m, int n, int k, cuDoubleComplex* A, int lda, const cuDoubleComplex* tau, cuDoubleComplex* work, int lwork, int* info)
    Status cusolverDnSormqr_bufferSize(DnHandle handle, SideMode side, Operation trans, int m, int n, int k, const float* A, int lda, const float* tau, const float* C, int ldc, int* lwork)
    Status cusolverDnDormqr_bufferSize(DnHandle handle, SideMode side, Operation trans, int m, int n, int k, const double* A, int lda, const double* tau, const double* C, int ldc, int* lwork)
    Status cusolverDnCunmqr_bufferSize(DnHandle handle, SideMode side, Operation trans, int m, int n, int k, const cuComplex* A, int lda, const cuComplex* tau, const cuComplex* C, int ldc, int* lwork)
    Status cusolverDnZunmqr_bufferSize(DnHandle handle, SideMode side, Operation trans, int m, int n, int k, const cuDoubleComplex* A, int lda, const cuDoubleComplex* tau, const cuDoubleComplex* C, int ldc, int* lwork)
    Status cusolverDnSormqr(DnHandle handle, SideMode side, Operation trans, int m, int n, int k, const float* A, int lda, const float* tau, float* C, int ldc, float* work, int lwork, int* devInfo)
    Status cusolverDnDormqr(DnHandle handle, SideMode side, Operation trans, int m, int n, int k, const double* A, int lda, const double* tau, double* C, int ldc, double* work, int lwork, int* devInfo)
    Status cusolverDnCunmqr(DnHandle handle, SideMode side, Operation trans, int m, int n, int k, const cuComplex* A, int lda, const cuComplex* tau, cuComplex* C, int ldc, cuComplex* work, int lwork, int* devInfo)
    Status cusolverDnZunmqr(DnHandle handle, SideMode side, Operation trans, int m, int n, int k, const cuDoubleComplex* A, int lda, const cuDoubleComplex* tau, cuDoubleComplex* C, int ldc, cuDoubleComplex* work, int lwork, int* devInfo)
    Status cusolverDnSsytrf_bufferSize(DnHandle handle, int n, float* A, int lda, int* lwork)
    Status cusolverDnDsytrf_bufferSize(DnHandle handle, int n, double* A, int lda, int* lwork)
    Status cusolverDnCsytrf_bufferSize(DnHandle handle, int n, cuComplex* A, int lda, int* lwork)
    Status cusolverDnZsytrf_bufferSize(DnHandle handle, int n, cuDoubleComplex* A, int lda, int* lwork)
    Status cusolverDnSsytrf(DnHandle handle, FillMode uplo, int n, float* A, int lda, int* ipiv, float* work, int lwork, int* info)
    Status cusolverDnDsytrf(DnHandle handle, FillMode uplo, int n, double* A, int lda, int* ipiv, double* work, int lwork, int* info)
    Status cusolverDnCsytrf(DnHandle handle, FillMode uplo, int n, cuComplex* A, int lda, int* ipiv, cuComplex* work, int lwork, int* info)
    Status cusolverDnZsytrf(DnHandle handle, FillMode uplo, int n, cuDoubleComplex* A, int lda, int* ipiv, cuDoubleComplex* work, int lwork, int* info)
    Status cusolverDnZZgesv_bufferSize(DnHandle handle, int n, int nrhs, cuDoubleComplex* dA, int ldda, int* dipiv, cuDoubleComplex* dB, int lddb, cuDoubleComplex* dX, int lddx, void* dWorkspace, size_t* lwork_bytes)
    Status cusolverDnZCgesv_bufferSize(DnHandle handle, int n, int nrhs, cuDoubleComplex* dA, int ldda, int* dipiv, cuDoubleComplex* dB, int lddb, cuDoubleComplex* dX, int lddx, void* dWorkspace, size_t* lwork_bytes)
    Status cusolverDnZKgesv_bufferSize(DnHandle handle, int n, int nrhs, cuDoubleComplex* dA, int ldda, int* dipiv, cuDoubleComplex* dB, int lddb, cuDoubleComplex* dX, int lddx, void* dWorkspace, size_t* lwork_bytes)
    Status cusolverDnZEgesv_bufferSize(DnHandle handle, int n, int nrhs, cuDoubleComplex* dA, int ldda, int* dipiv, cuDoubleComplex* dB, int lddb, cuDoubleComplex* dX, int lddx, void* dWorkspace, size_t* lwork_bytes)
    Status cusolverDnZYgesv_bufferSize(DnHandle handle, int n, int nrhs, cuDoubleComplex* dA, int ldda, int* dipiv, cuDoubleComplex* dB, int lddb, cuDoubleComplex* dX, int lddx, void* dWorkspace, size_t* lwork_bytes)
    Status cusolverDnCCgesv_bufferSize(DnHandle handle, int n, int nrhs, cuComplex* dA, int ldda, int* dipiv, cuComplex* dB, int lddb, cuComplex* dX, int lddx, void* dWorkspace, size_t* lwork_bytes)
    Status cusolverDnCKgesv_bufferSize(DnHandle handle, int n, int nrhs, cuComplex* dA, int ldda, int* dipiv, cuComplex* dB, int lddb, cuComplex* dX, int lddx, void* dWorkspace, size_t* lwork_bytes)
    Status cusolverDnCEgesv_bufferSize(DnHandle handle, int n, int nrhs, cuComplex* dA, int ldda, int* dipiv, cuComplex* dB, int lddb, cuComplex* dX, int lddx, void* dWorkspace, size_t* lwork_bytes)
    Status cusolverDnCYgesv_bufferSize(DnHandle handle, int n, int nrhs, cuComplex* dA, int ldda, int* dipiv, cuComplex* dB, int lddb, cuComplex* dX, int lddx, void* dWorkspace, size_t* lwork_bytes)
    Status cusolverDnDDgesv_bufferSize(DnHandle handle, int n, int nrhs, double* dA, int ldda, int* dipiv, double* dB, int lddb, double* dX, int lddx, void* dWorkspace, size_t* lwork_bytes)
    Status cusolverDnDSgesv_bufferSize(DnHandle handle, int n, int nrhs, double* dA, int ldda, int* dipiv, double* dB, int lddb, double* dX, int lddx, void* dWorkspace, size_t* lwork_bytes)
    Status cusolverDnDHgesv_bufferSize(DnHandle handle, int n, int nrhs, double* dA, int ldda, int* dipiv, double* dB, int lddb, double* dX, int lddx, void* dWorkspace, size_t* lwork_bytes)
    Status cusolverDnDBgesv_bufferSize(DnHandle handle, int n, int nrhs, double* dA, int ldda, int* dipiv, double* dB, int lddb, double* dX, int lddx, void* dWorkspace, size_t* lwork_bytes)
    Status cusolverDnDXgesv_bufferSize(DnHandle handle, int n, int nrhs, double* dA, int ldda, int* dipiv, double* dB, int lddb, double* dX, int lddx, void* dWorkspace, size_t* lwork_bytes)
    Status cusolverDnSSgesv_bufferSize(DnHandle handle, int n, int nrhs, float* dA, int ldda, int* dipiv, float* dB, int lddb, float* dX, int lddx, void* dWorkspace, size_t* lwork_bytes)
    Status cusolverDnSHgesv_bufferSize(DnHandle handle, int n, int nrhs, float* dA, int ldda, int* dipiv, float* dB, int lddb, float* dX, int lddx, void* dWorkspace, size_t* lwork_bytes)
    Status cusolverDnSBgesv_bufferSize(DnHandle handle, int n, int nrhs, float* dA, int ldda, int* dipiv, float* dB, int lddb, float* dX, int lddx, void* dWorkspace, size_t* lwork_bytes)
    Status cusolverDnSXgesv_bufferSize(DnHandle handle, int n, int nrhs, float* dA, int ldda, int* dipiv, float* dB, int lddb, float* dX, int lddx, void* dWorkspace, size_t* lwork_bytes)
    Status cusolverDnZZgesv(DnHandle handle, int n, int nrhs, cuDoubleComplex* dA, int ldda, int* dipiv, cuDoubleComplex* dB, int lddb, cuDoubleComplex* dX, int lddx, void* dWorkspace, size_t lwork_bytes, int* iter, int* d_info)
    Status cusolverDnZCgesv(DnHandle handle, int n, int nrhs, cuDoubleComplex* dA, int ldda, int* dipiv, cuDoubleComplex* dB, int lddb, cuDoubleComplex* dX, int lddx, void* dWorkspace, size_t lwork_bytes, int* iter, int* d_info)
    Status cusolverDnZKgesv(DnHandle handle, int n, int nrhs, cuDoubleComplex* dA, int ldda, int* dipiv, cuDoubleComplex* dB, int lddb, cuDoubleComplex* dX, int lddx, void* dWorkspace, size_t lwork_bytes, int* iter, int* d_info)
    Status cusolverDnZEgesv(DnHandle handle, int n, int nrhs, cuDoubleComplex* dA, int ldda, int* dipiv, cuDoubleComplex* dB, int lddb, cuDoubleComplex* dX, int lddx, void* dWorkspace, size_t lwork_bytes, int* iter, int* d_info)
    Status cusolverDnZYgesv(DnHandle handle, int n, int nrhs, cuDoubleComplex* dA, int ldda, int* dipiv, cuDoubleComplex* dB, int lddb, cuDoubleComplex* dX, int lddx, void* dWorkspace, size_t lwork_bytes, int* iter, int* d_info)
    Status cusolverDnCCgesv(DnHandle handle, int n, int nrhs, cuComplex* dA, int ldda, int* dipiv, cuComplex* dB, int lddb, cuComplex* dX, int lddx, void* dWorkspace, size_t lwork_bytes, int* iter, int* d_info)
    Status cusolverDnCEgesv(DnHandle handle, int n, int nrhs, cuComplex* dA, int ldda, int* dipiv, cuComplex* dB, int lddb, cuComplex* dX, int lddx, void* dWorkspace, size_t lwork_bytes, int* iter, int* d_info)
    Status cusolverDnCKgesv(DnHandle handle, int n, int nrhs, cuComplex* dA, int ldda, int* dipiv, cuComplex* dB, int lddb, cuComplex* dX, int lddx, void* dWorkspace, size_t lwork_bytes, int* iter, int* d_info)
    Status cusolverDnCYgesv(DnHandle handle, int n, int nrhs, cuComplex* dA, int ldda, int* dipiv, cuComplex* dB, int lddb, cuComplex* dX, int lddx, void* dWorkspace, size_t lwork_bytes, int* iter, int* d_info)
    Status cusolverDnDDgesv(DnHandle handle, int n, int nrhs, double* dA, int ldda, int* dipiv, double* dB, int lddb, double* dX, int lddx, void* dWorkspace, size_t lwork_bytes, int* iter, int* d_info)
    Status cusolverDnDSgesv(DnHandle handle, int n, int nrhs, double* dA, int ldda, int* dipiv, double* dB, int lddb, double* dX, int lddx, void* dWorkspace, size_t lwork_bytes, int* iter, int* d_info)
    Status cusolverDnDHgesv(DnHandle handle, int n, int nrhs, double* dA, int ldda, int* dipiv, double* dB, int lddb, double* dX, int lddx, void* dWorkspace, size_t lwork_bytes, int* iter, int* d_info)
    Status cusolverDnDBgesv(DnHandle handle, int n, int nrhs, double* dA, int ldda, int* dipiv, double* dB, int lddb, double* dX, int lddx, void* dWorkspace, size_t lwork_bytes, int* iter, int* d_info)
    Status cusolverDnDXgesv(DnHandle handle, int n, int nrhs, double* dA, int ldda, int* dipiv, double* dB, int lddb, double* dX, int lddx, void* dWorkspace, size_t lwork_bytes, int* iter, int* d_info)
    Status cusolverDnSSgesv(DnHandle handle, int n, int nrhs, float* dA, int ldda, int* dipiv, float* dB, int lddb, float* dX, int lddx, void* dWorkspace, size_t lwork_bytes, int* iter, int* d_info)
    Status cusolverDnSHgesv(DnHandle handle, int n, int nrhs, float* dA, int ldda, int* dipiv, float* dB, int lddb, float* dX, int lddx, void* dWorkspace, size_t lwork_bytes, int* iter, int* d_info)
    Status cusolverDnSBgesv(DnHandle handle, int n, int nrhs, float* dA, int ldda, int* dipiv, float* dB, int lddb, float* dX, int lddx, void* dWorkspace, size_t lwork_bytes, int* iter, int* d_info)
    Status cusolverDnSXgesv(DnHandle handle, int n, int nrhs, float* dA, int ldda, int* dipiv, float* dB, int lddb, float* dX, int lddx, void* dWorkspace, size_t lwork_bytes, int* iter, int* d_info)
    Status cusolverDnZZgels_bufferSize(DnHandle handle, int m, int n, int nrhs, cuDoubleComplex* dA, int ldda, cuDoubleComplex* dB, int lddb, cuDoubleComplex* dX, int lddx, void* dWorkspace, size_t* lwork_bytes)
    Status cusolverDnZCgels_bufferSize(DnHandle handle, int m, int n, int nrhs, cuDoubleComplex* dA, int ldda, cuDoubleComplex* dB, int lddb, cuDoubleComplex* dX, int lddx, void* dWorkspace, size_t* lwork_bytes)
    Status cusolverDnZKgels_bufferSize(DnHandle handle, int m, int n, int nrhs, cuDoubleComplex* dA, int ldda, cuDoubleComplex* dB, int lddb, cuDoubleComplex* dX, int lddx, void* dWorkspace, size_t* lwork_bytes)
    Status cusolverDnZEgels_bufferSize(DnHandle handle, int m, int n, int nrhs, cuDoubleComplex* dA, int ldda, cuDoubleComplex* dB, int lddb, cuDoubleComplex* dX, int lddx, void* dWorkspace, size_t* lwork_bytes)
    Status cusolverDnZYgels_bufferSize(DnHandle handle, int m, int n, int nrhs, cuDoubleComplex* dA, int ldda, cuDoubleComplex* dB, int lddb, cuDoubleComplex* dX, int lddx, void* dWorkspace, size_t* lwork_bytes)
    Status cusolverDnCCgels_bufferSize(DnHandle handle, int m, int n, int nrhs, cuComplex* dA, int ldda, cuComplex* dB, int lddb, cuComplex* dX, int lddx, void* dWorkspace, size_t* lwork_bytes)
    Status cusolverDnCKgels_bufferSize(DnHandle handle, int m, int n, int nrhs, cuComplex* dA, int ldda, cuComplex* dB, int lddb, cuComplex* dX, int lddx, void* dWorkspace, size_t* lwork_bytes)
    Status cusolverDnCEgels_bufferSize(DnHandle handle, int m, int n, int nrhs, cuComplex* dA, int ldda, cuComplex* dB, int lddb, cuComplex* dX, int lddx, void* dWorkspace, size_t* lwork_bytes)
    Status cusolverDnCYgels_bufferSize(DnHandle handle, int m, int n, int nrhs, cuComplex* dA, int ldda, cuComplex* dB, int lddb, cuComplex* dX, int lddx, void* dWorkspace, size_t* lwork_bytes)
    Status cusolverDnDDgels_bufferSize(DnHandle handle, int m, int n, int nrhs, double* dA, int ldda, double* dB, int lddb, double* dX, int lddx, void* dWorkspace, size_t* lwork_bytes)
    Status cusolverDnDSgels_bufferSize(DnHandle handle, int m, int n, int nrhs, double* dA, int ldda, double* dB, int lddb, double* dX, int lddx, void* dWorkspace, size_t* lwork_bytes)
    Status cusolverDnDHgels_bufferSize(DnHandle handle, int m, int n, int nrhs, double* dA, int ldda, double* dB, int lddb, double* dX, int lddx, void* dWorkspace, size_t* lwork_bytes)
    Status cusolverDnDBgels_bufferSize(DnHandle handle, int m, int n, int nrhs, double* dA, int ldda, double* dB, int lddb, double* dX, int lddx, void* dWorkspace, size_t* lwork_bytes)
    Status cusolverDnDXgels_bufferSize(DnHandle handle, int m, int n, int nrhs, double* dA, int ldda, double* dB, int lddb, double* dX, int lddx, void* dWorkspace, size_t* lwork_bytes)
    Status cusolverDnSSgels_bufferSize(DnHandle handle, int m, int n, int nrhs, float* dA, int ldda, float* dB, int lddb, float* dX, int lddx, void* dWorkspace, size_t* lwork_bytes)
    Status cusolverDnSHgels_bufferSize(DnHandle handle, int m, int n, int nrhs, float* dA, int ldda, float* dB, int lddb, float* dX, int lddx, void* dWorkspace, size_t* lwork_bytes)
    Status cusolverDnSBgels_bufferSize(DnHandle handle, int m, int n, int nrhs, float* dA, int ldda, float* dB, int lddb, float* dX, int lddx, void* dWorkspace, size_t* lwork_bytes)
    Status cusolverDnSXgels_bufferSize(DnHandle handle, int m, int n, int nrhs, float* dA, int ldda, float* dB, int lddb, float* dX, int lddx, void* dWorkspace, size_t* lwork_bytes)
    Status cusolverDnZZgels(DnHandle handle, int m, int n, int nrhs, cuDoubleComplex* dA, int ldda, cuDoubleComplex* dB, int lddb, cuDoubleComplex* dX, int lddx, void* dWorkspace, size_t lwork_bytes, int* iter, int* d_info)
    Status cusolverDnZCgels(DnHandle handle, int m, int n, int nrhs, cuDoubleComplex* dA, int ldda, cuDoubleComplex* dB, int lddb, cuDoubleComplex* dX, int lddx, void* dWorkspace, size_t lwork_bytes, int* iter, int* d_info)
    Status cusolverDnZKgels(DnHandle handle, int m, int n, int nrhs, cuDoubleComplex* dA, int ldda, cuDoubleComplex* dB, int lddb, cuDoubleComplex* dX, int lddx, void* dWorkspace, size_t lwork_bytes, int* iter, int* d_info)
    Status cusolverDnZEgels(DnHandle handle, int m, int n, int nrhs, cuDoubleComplex* dA, int ldda, cuDoubleComplex* dB, int lddb, cuDoubleComplex* dX, int lddx, void* dWorkspace, size_t lwork_bytes, int* iter, int* d_info)
    Status cusolverDnZYgels(DnHandle handle, int m, int n, int nrhs, cuDoubleComplex* dA, int ldda, cuDoubleComplex* dB, int lddb, cuDoubleComplex* dX, int lddx, void* dWorkspace, size_t lwork_bytes, int* iter, int* d_info)
    Status cusolverDnCCgels(DnHandle handle, int m, int n, int nrhs, cuComplex* dA, int ldda, cuComplex* dB, int lddb, cuComplex* dX, int lddx, void* dWorkspace, size_t lwork_bytes, int* iter, int* d_info)
    Status cusolverDnCKgels(DnHandle handle, int m, int n, int nrhs, cuComplex* dA, int ldda, cuComplex* dB, int lddb, cuComplex* dX, int lddx, void* dWorkspace, size_t lwork_bytes, int* iter, int* d_info)
    Status cusolverDnCEgels(DnHandle handle, int m, int n, int nrhs, cuComplex* dA, int ldda, cuComplex* dB, int lddb, cuComplex* dX, int lddx, void* dWorkspace, size_t lwork_bytes, int* iter, int* d_info)
    Status cusolverDnCYgels(DnHandle handle, int m, int n, int nrhs, cuComplex* dA, int ldda, cuComplex* dB, int lddb, cuComplex* dX, int lddx, void* dWorkspace, size_t lwork_bytes, int* iter, int* d_info)
    Status cusolverDnDDgels(DnHandle handle, int m, int n, int nrhs, double* dA, int ldda, double* dB, int lddb, double* dX, int lddx, void* dWorkspace, size_t lwork_bytes, int* iter, int* d_info)
    Status cusolverDnDSgels(DnHandle handle, int m, int n, int nrhs, double* dA, int ldda, double* dB, int lddb, double* dX, int lddx, void* dWorkspace, size_t lwork_bytes, int* iter, int* d_info)
    Status cusolverDnDHgels(DnHandle handle, int m, int n, int nrhs, double* dA, int ldda, double* dB, int lddb, double* dX, int lddx, void* dWorkspace, size_t lwork_bytes, int* iter, int* d_info)
    Status cusolverDnDBgels(DnHandle handle, int m, int n, int nrhs, double* dA, int ldda, double* dB, int lddb, double* dX, int lddx, void* dWorkspace, size_t lwork_bytes, int* iter, int* d_info)
    Status cusolverDnDXgels(DnHandle handle, int m, int n, int nrhs, double* dA, int ldda, double* dB, int lddb, double* dX, int lddx, void* dWorkspace, size_t lwork_bytes, int* iter, int* d_info)
    Status cusolverDnSSgels(DnHandle handle, int m, int n, int nrhs, float* dA, int ldda, float* dB, int lddb, float* dX, int lddx, void* dWorkspace, size_t lwork_bytes, int* iter, int* d_info)
    Status cusolverDnSHgels(DnHandle handle, int m, int n, int nrhs, float* dA, int ldda, float* dB, int lddb, float* dX, int lddx, void* dWorkspace, size_t lwork_bytes, int* iter, int* d_info)
    Status cusolverDnSBgels(DnHandle handle, int m, int n, int nrhs, float* dA, int ldda, float* dB, int lddb, float* dX, int lddx, void* dWorkspace, size_t lwork_bytes, int* iter, int* d_info)
    Status cusolverDnSXgels(DnHandle handle, int m, int n, int nrhs, float* dA, int ldda, float* dB, int lddb, float* dX, int lddx, void* dWorkspace, size_t lwork_bytes, int* iter, int* d_info)

    # cuSOLVER Dense LAPACK Function - Dense Eigenvalue Solver
    Status cusolverDnSgebrd_bufferSize(DnHandle handle, int m, int n, int* Lwork)
    Status cusolverDnDgebrd_bufferSize(DnHandle handle, int m, int n, int* Lwork)
    Status cusolverDnCgebrd_bufferSize(DnHandle handle, int m, int n, int* Lwork)
    Status cusolverDnZgebrd_bufferSize(DnHandle handle, int m, int n, int* Lwork)
    Status cusolverDnSgebrd(DnHandle handle, int m, int n, float* A, int lda, float* D, float* E, float* TAUQ, float* TAUP, float* Work, int Lwork, int* devInfo)
    Status cusolverDnDgebrd(DnHandle handle, int m, int n, double* A, int lda, double* D, double* E, double* TAUQ, double* TAUP, double* Work, int Lwork, int* devInfo)
    Status cusolverDnCgebrd(DnHandle handle, int m, int n, cuComplex* A, int lda, float* D, float* E, cuComplex* TAUQ, cuComplex* TAUP, cuComplex* Work, int Lwork, int* devInfo)
    Status cusolverDnZgebrd(DnHandle handle, int m, int n, cuDoubleComplex* A, int lda, double* D, double* E, cuDoubleComplex* TAUQ, cuDoubleComplex* TAUP, cuDoubleComplex* Work, int Lwork, int* devInfo)
    Status cusolverDnSgesvd_bufferSize(DnHandle handle, int m, int n, int* lwork)
    Status cusolverDnDgesvd_bufferSize(DnHandle handle, int m, int n, int* lwork)
    Status cusolverDnCgesvd_bufferSize(DnHandle handle, int m, int n, int* lwork)
    Status cusolverDnZgesvd_bufferSize(DnHandle handle, int m, int n, int* lwork)
    Status cusolverDnSgesvd(DnHandle handle, signed char jobu, signed char jobvt, int m, int n, float* A, int lda, float* S, float* U, int ldu, float* VT, int ldvt, float* work, int lwork, float* rwork, int* info)
    Status cusolverDnDgesvd(DnHandle handle, signed char jobu, signed char jobvt, int m, int n, double* A, int lda, double* S, double* U, int ldu, double* VT, int ldvt, double* work, int lwork, double* rwork, int* info)
    Status cusolverDnCgesvd(DnHandle handle, signed char jobu, signed char jobvt, int m, int n, cuComplex* A, int lda, float* S, cuComplex* U, int ldu, cuComplex* VT, int ldvt, cuComplex* work, int lwork, float* rwork, int* info)
    Status cusolverDnZgesvd(DnHandle handle, signed char jobu, signed char jobvt, int m, int n, cuDoubleComplex* A, int lda, double* S, cuDoubleComplex* U, int ldu, cuDoubleComplex* VT, int ldvt, cuDoubleComplex* work, int lwork, double* rwork, int* info)
    Status cusolverDnSgesvdj_bufferSize(DnHandle handle, EigMode jobz, int econ, int m, int n, const float* A, int lda, const float* S, const float* U, int ldu, const float* V, int ldv, int* lwork, gesvdjInfo_t params)
    Status cusolverDnDgesvdj_bufferSize(DnHandle handle, EigMode jobz, int econ, int m, int n, const double* A, int lda, const double* S, const double* U, int ldu, const double* V, int ldv, int* lwork, gesvdjInfo_t params)
    Status cusolverDnCgesvdj_bufferSize(DnHandle handle, EigMode jobz, int econ, int m, int n, const cuComplex* A, int lda, const float* S, const cuComplex* U, int ldu, const cuComplex* V, int ldv, int* lwork, gesvdjInfo_t params)
    Status cusolverDnZgesvdj_bufferSize(DnHandle handle, EigMode jobz, int econ, int m, int n, const cuDoubleComplex* A, int lda, const double* S, const cuDoubleComplex* U, int ldu, const cuDoubleComplex* V, int ldv, int* lwork, gesvdjInfo_t params)
    Status cusolverDnSgesvdj(DnHandle handle, EigMode jobz, int econ, int m, int n, float* A, int lda, float* S, float* U, int ldu, float* V, int ldv, float* work, int lwork, int* info, gesvdjInfo_t params)
    Status cusolverDnDgesvdj(DnHandle handle, EigMode jobz, int econ, int m, int n, double* A, int lda, double* S, double* U, int ldu, double* V, int ldv, double* work, int lwork, int* info, gesvdjInfo_t params)
    Status cusolverDnCgesvdj(DnHandle handle, EigMode jobz, int econ, int m, int n, cuComplex* A, int lda, float* S, cuComplex* U, int ldu, cuComplex* V, int ldv, cuComplex* work, int lwork, int* info, gesvdjInfo_t params)
    Status cusolverDnZgesvdj(DnHandle handle, EigMode jobz, int econ, int m, int n, cuDoubleComplex* A, int lda, double* S, cuDoubleComplex* U, int ldu, cuDoubleComplex* V, int ldv, cuDoubleComplex* work, int lwork, int* info, gesvdjInfo_t params)
    Status cusolverDnSgesvdjBatched_bufferSize(DnHandle handle, EigMode jobz, int m, int n, const float* A, int lda, const float* S, const float* U, int ldu, const float* V, int ldv, int* lwork, gesvdjInfo_t params, int batchSize)
    Status cusolverDnDgesvdjBatched_bufferSize(DnHandle handle, EigMode jobz, int m, int n, const double* A, int lda, const double* S, const double* U, int ldu, const double* V, int ldv, int* lwork, gesvdjInfo_t params, int batchSize)
    Status cusolverDnCgesvdjBatched_bufferSize(DnHandle handle, EigMode jobz, int m, int n, const cuComplex* A, int lda, const float* S, const cuComplex* U, int ldu, const cuComplex* V, int ldv, int* lwork, gesvdjInfo_t params, int batchSize)
    Status cusolverDnZgesvdjBatched_bufferSize(DnHandle handle, EigMode jobz, int m, int n, const cuDoubleComplex* A, int lda, const double* S, const cuDoubleComplex* U, int ldu, const cuDoubleComplex* V, int ldv, int* lwork, gesvdjInfo_t params, int batchSize)
    Status cusolverDnSgesvdjBatched(DnHandle handle, EigMode jobz, int m, int n, float* A, int lda, float* S, float* U, int ldu, float* V, int ldv, float* work, int lwork, int* info, gesvdjInfo_t params, int batchSize)
    Status cusolverDnDgesvdjBatched(DnHandle handle, EigMode jobz, int m, int n, double* A, int lda, double* S, double* U, int ldu, double* V, int ldv, double* work, int lwork, int* info, gesvdjInfo_t params, int batchSize)
    Status cusolverDnCgesvdjBatched(DnHandle handle, EigMode jobz, int m, int n, cuComplex* A, int lda, float* S, cuComplex* U, int ldu, cuComplex* V, int ldv, cuComplex* work, int lwork, int* info, gesvdjInfo_t params, int batchSize)
    Status cusolverDnZgesvdjBatched(DnHandle handle, EigMode jobz, int m, int n, cuDoubleComplex* A, int lda, double* S, cuDoubleComplex* U, int ldu, cuDoubleComplex* V, int ldv, cuDoubleComplex* work, int lwork, int* info, gesvdjInfo_t params, int batchSize)
    Status cusolverDnSgesvdaStridedBatched_bufferSize(DnHandle handle, EigMode jobz, int rank, int m, int n, const float* d_A, int lda, long long int strideA, const float* d_S, long long int strideS, const float* d_U, int ldu, long long int strideU, const float* d_V, int ldv, long long int strideV, int* lwork, int batchSize)
    Status cusolverDnDgesvdaStridedBatched_bufferSize(DnHandle handle, EigMode jobz, int rank, int m, int n, const double* d_A, int lda, long long int strideA, const double* d_S, long long int strideS, const double* d_U, int ldu, long long int strideU, const double* d_V, int ldv, long long int strideV, int* lwork, int batchSize)
    Status cusolverDnCgesvdaStridedBatched_bufferSize(DnHandle handle, EigMode jobz, int rank, int m, int n, const cuComplex* d_A, int lda, long long int strideA, const float* d_S, long long int strideS, const cuComplex* d_U, int ldu, long long int strideU, const cuComplex* d_V, int ldv, long long int strideV, int* lwork, int batchSize)
    Status cusolverDnZgesvdaStridedBatched_bufferSize(DnHandle handle, EigMode jobz, int rank, int m, int n, const cuDoubleComplex* d_A, int lda, long long int strideA, const double* d_S, long long int strideS, const cuDoubleComplex* d_U, int ldu, long long int strideU, const cuDoubleComplex* d_V, int ldv, long long int strideV, int* lwork, int batchSize)
    Status cusolverDnSgesvdaStridedBatched(DnHandle handle, EigMode jobz, int rank, int m, int n, const float* d_A, int lda, long long int strideA, float* d_S, long long int strideS, float* d_U, int ldu, long long int strideU, float* d_V, int ldv, long long int strideV, float* d_work, int lwork, int* d_info, double* h_R_nrmF, int batchSize)
    Status cusolverDnDgesvdaStridedBatched(DnHandle handle, EigMode jobz, int rank, int m, int n, const double* d_A, int lda, long long int strideA, double* d_S, long long int strideS, double* d_U, int ldu, long long int strideU, double* d_V, int ldv, long long int strideV, double* d_work, int lwork, int* d_info, double* h_R_nrmF, int batchSize)
    Status cusolverDnCgesvdaStridedBatched(DnHandle handle, EigMode jobz, int rank, int m, int n, const cuComplex* d_A, int lda, long long int strideA, float* d_S, long long int strideS, cuComplex* d_U, int ldu, long long int strideU, cuComplex* d_V, int ldv, long long int strideV, cuComplex* d_work, int lwork, int* d_info, double* h_R_nrmF, int batchSize)
    Status cusolverDnZgesvdaStridedBatched(DnHandle handle, EigMode jobz, int rank, int m, int n, const cuDoubleComplex* d_A, int lda, long long int strideA, double* d_S, long long int strideS, cuDoubleComplex* d_U, int ldu, long long int strideU, cuDoubleComplex* d_V, int ldv, long long int strideV, cuDoubleComplex* d_work, int lwork, int* d_info, double* h_R_nrmF, int batchSize)
    Status cusolverDnSsyevd_bufferSize(DnHandle handle, EigMode jobz, FillMode uplo, int n, const float* A, int lda, const float* W, int* lwork)
    Status cusolverDnDsyevd_bufferSize(DnHandle handle, EigMode jobz, FillMode uplo, int n, const double* A, int lda, const double* W, int* lwork)
    Status cusolverDnCheevd_bufferSize(DnHandle handle, EigMode jobz, FillMode uplo, int n, const cuComplex* A, int lda, const float* W, int* lwork)
    Status cusolverDnZheevd_bufferSize(DnHandle handle, EigMode jobz, FillMode uplo, int n, const cuDoubleComplex* A, int lda, const double* W, int* lwork)
    Status cusolverDnSsyevd(DnHandle handle, EigMode jobz, FillMode uplo, int n, float* A, int lda, float* W, float* work, int lwork, int* info)
    Status cusolverDnDsyevd(DnHandle handle, EigMode jobz, FillMode uplo, int n, double* A, int lda, double* W, double* work, int lwork, int* info)
    Status cusolverDnCheevd(DnHandle handle, EigMode jobz, FillMode uplo, int n, cuComplex* A, int lda, float* W, cuComplex* work, int lwork, int* info)
    Status cusolverDnZheevd(DnHandle handle, EigMode jobz, FillMode uplo, int n, cuDoubleComplex* A, int lda, double* W, cuDoubleComplex* work, int lwork, int* info)
    Status cusolverDnSsyevj_bufferSize(DnHandle handle, EigMode jobz, FillMode uplo, int n, const float* A, int lda, const float* W, int* lwork, syevjInfo_t params)
    Status cusolverDnDsyevj_bufferSize(DnHandle handle, EigMode jobz, FillMode uplo, int n, const double* A, int lda, const double* W, int* lwork, syevjInfo_t params)
    Status cusolverDnCheevj_bufferSize(DnHandle handle, EigMode jobz, FillMode uplo, int n, const cuComplex* A, int lda, const float* W, int* lwork, syevjInfo_t params)
    Status cusolverDnZheevj_bufferSize(DnHandle handle, EigMode jobz, FillMode uplo, int n, const cuDoubleComplex* A, int lda, const double* W, int* lwork, syevjInfo_t params)
    Status cusolverDnSsyevj(DnHandle handle, EigMode jobz, FillMode uplo, int n, float* A, int lda, float* W, float* work, int lwork, int* info, syevjInfo_t params)
    Status cusolverDnDsyevj(DnHandle handle, EigMode jobz, FillMode uplo, int n, double* A, int lda, double* W, double* work, int lwork, int* info, syevjInfo_t params)
    Status cusolverDnCheevj(DnHandle handle, EigMode jobz, FillMode uplo, int n, cuComplex* A, int lda, float* W, cuComplex* work, int lwork, int* info, syevjInfo_t params)
    Status cusolverDnZheevj(DnHandle handle, EigMode jobz, FillMode uplo, int n, cuDoubleComplex* A, int lda, double* W, cuDoubleComplex* work, int lwork, int* info, syevjInfo_t params)
    Status cusolverDnSsyevjBatched_bufferSize(DnHandle handle, EigMode jobz, FillMode uplo, int n, const float* A, int lda, const float* W, int* lwork, syevjInfo_t params, int batchSize)
    Status cusolverDnDsyevjBatched_bufferSize(DnHandle handle, EigMode jobz, FillMode uplo, int n, const double* A, int lda, const double* W, int* lwork, syevjInfo_t params, int batchSize)
    Status cusolverDnCheevjBatched_bufferSize(DnHandle handle, EigMode jobz, FillMode uplo, int n, const cuComplex* A, int lda, const float* W, int* lwork, syevjInfo_t params, int batchSize)
    Status cusolverDnZheevjBatched_bufferSize(DnHandle handle, EigMode jobz, FillMode uplo, int n, const cuDoubleComplex* A, int lda, const double* W, int* lwork, syevjInfo_t params, int batchSize)
    Status cusolverDnSsyevjBatched(DnHandle handle, EigMode jobz, FillMode uplo, int n, float* A, int lda, float* W, float* work, int lwork, int* info, syevjInfo_t params, int batchSize)
    Status cusolverDnDsyevjBatched(DnHandle handle, EigMode jobz, FillMode uplo, int n, double* A, int lda, double* W, double* work, int lwork, int* info, syevjInfo_t params, int batchSize)
    Status cusolverDnCheevjBatched(DnHandle handle, EigMode jobz, FillMode uplo, int n, cuComplex* A, int lda, float* W, cuComplex* work, int lwork, int* info, syevjInfo_t params, int batchSize)
    Status cusolverDnZheevjBatched(DnHandle handle, EigMode jobz, FillMode uplo, int n, cuDoubleComplex* A, int lda, double* W, cuDoubleComplex* work, int lwork, int* info, syevjInfo_t params, int batchSize)

    # cuSOLVER Sparse LAPACK Function - Helper Function
    Status cusolverSpCreate(SpHandle* handle)
    Status cusolverSpDestroy(SpHandle handle)
    Status cusolverSpSetStream(SpHandle handle, driver.Stream streamId)
    Status cusolverSpGetStream(SpHandle handle, driver.Stream* streamId)

    # cuSOLVER Sparse LAPACK Function - High Level Function
    Status cusolverSpScsrlsvchol(SpHandle handle, int m, int nnz, const MatDescr descrA, const float* csrVal, const int* csrRowPtr, const int* csrColInd, const float* b, float tol, int reorder, float* x, int* singularity)
    Status cusolverSpDcsrlsvchol(SpHandle handle, int m, int nnz, const MatDescr descrA, const double* csrVal, const int* csrRowPtr, const int* csrColInd, const double* b, double tol, int reorder, double* x, int* singularity)
    Status cusolverSpCcsrlsvchol(SpHandle handle, int m, int nnz, const MatDescr descrA, const cuComplex* csrVal, const int* csrRowPtr, const int* csrColInd, const cuComplex* b, float tol, int reorder, cuComplex* x, int* singularity)
    Status cusolverSpZcsrlsvchol(SpHandle handle, int m, int nnz, const MatDescr descrA, const cuDoubleComplex* csrVal, const int* csrRowPtr, const int* csrColInd, const cuDoubleComplex* b, double tol, int reorder, cuDoubleComplex* x, int* singularity)
    Status cusolverSpScsrlsvqr(SpHandle handle, int m, int nnz, const MatDescr descrA, const float* csrVal, const int* csrRowPtr, const int* csrColInd, const float* b, float tol, int reorder, float* x, int* singularity)
    Status cusolverSpDcsrlsvqr(SpHandle handle, int m, int nnz, const MatDescr descrA, const double* csrVal, const int* csrRowPtr, const int* csrColInd, const double* b, double tol, int reorder, double* x, int* singularity)
    Status cusolverSpCcsrlsvqr(SpHandle handle, int m, int nnz, const MatDescr descrA, const cuComplex* csrVal, const int* csrRowPtr, const int* csrColInd, const cuComplex* b, float tol, int reorder, cuComplex* x, int* singularity)
    Status cusolverSpZcsrlsvqr(SpHandle handle, int m, int nnz, const MatDescr descrA, const cuDoubleComplex* csrVal, const int* csrRowPtr, const int* csrColInd, const cuDoubleComplex* b, double tol, int reorder, cuDoubleComplex* x, int* singularity)
    Status cusolverSpScsreigvsi(SpHandle handle, int m, int nnz, const MatDescr descrA, const float* csrValA, const int* csrRowPtrA, const int* csrColIndA, float mu0, const float* x0, int maxite, float eps, float* mu, float* x)
    Status cusolverSpDcsreigvsi(SpHandle handle, int m, int nnz, const MatDescr descrA, const double* csrValA, const int* csrRowPtrA, const int* csrColIndA, double mu0, const double* x0, int maxite, double eps, double* mu, double* x)
    Status cusolverSpCcsreigvsi(SpHandle handle, int m, int nnz, const MatDescr descrA, const cuComplex* csrValA, const int* csrRowPtrA, const int* csrColIndA, cuComplex mu0, const cuComplex* x0, int maxite, float eps, cuComplex* mu, cuComplex* x)
    Status cusolverSpZcsreigvsi(SpHandle handle, int m, int nnz, const MatDescr descrA, const cuDoubleComplex* csrValA, const int* csrRowPtrA, const int* csrColIndA, cuDoubleComplex mu0, const cuDoubleComplex* x0, int maxite, double eps, cuDoubleComplex* mu, cuDoubleComplex* x)

    # libraryPropertyType_t
    int MAJOR_VERSION
    int MINOR_VERSION
    int PATCH_LEVEL


########################################
# Error handling

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


########################################
# Version

cpdef tuple _getVersion():
    return (getProperty(MAJOR_VERSION),
            getProperty(MINOR_VERSION),
            getProperty(PATCH_LEVEL))


########################################
# Library Attributes

cpdef int getProperty(int type) except? -1:
    cdef int value
    status = cusolverGetProperty(<LibraryPropertyType>type, &value)
    check_status(status)
    return value


##################################################
# cuSOLVER Dense LAPACK Function - Helper Function

cpdef intptr_t create() except? 0:
    cdef DnHandle handle
    status = cusolverDnCreate(&handle)
    check_status(status)
    return <intptr_t>handle

cpdef destroy(intptr_t handle):
    status = cusolverDnDestroy(<DnHandle>handle)
    check_status(status)

cpdef setStream(intptr_t handle, size_t streamId):
    status = cusolverDnSetStream(<DnHandle>handle, <driver.Stream>streamId)
    check_status(status)

cpdef size_t getStream(intptr_t handle) except? 0:
    cdef driver.Stream streamId
    status = cusolverDnGetStream(<DnHandle>handle, &streamId)
    check_status(status)
    return <size_t>streamId

cpdef size_t createSyevjInfo() except? 0:
    cdef syevjInfo_t info
    status = cusolverDnCreateSyevjInfo(&info)
    check_status(status)
    return <size_t>info

cpdef destroySyevjInfo(size_t info):
    status = cusolverDnDestroySyevjInfo(<syevjInfo_t>info)
    check_status(status)

cpdef xsyevjSetTolerance(size_t info, double tolerance):
    status = cusolverDnXsyevjSetTolerance(<syevjInfo_t>info, tolerance)
    check_status(status)

cpdef xsyevjSetMaxSweeps(size_t info, int max_sweeps):
    status = cusolverDnXsyevjSetMaxSweeps(<syevjInfo_t>info, max_sweeps)
    check_status(status)

cpdef xsyevjSetSortEig(size_t info, int sort_eig):
    status = cusolverDnXsyevjSetSortEig(<syevjInfo_t>info, sort_eig)
    check_status(status)

cpdef double xsyevjGetResidual(intptr_t handle, size_t info) except? 0:
    cdef double residual
    status = cusolverDnXsyevjGetResidual(<DnHandle>handle, <syevjInfo_t>info, &residual)
    check_status(status)
    return residual

cpdef int xsyevjGetSweeps(intptr_t handle, size_t info) except? 0:
    cdef int executed_sweeps
    status = cusolverDnXsyevjGetSweeps(<DnHandle>handle, <syevjInfo_t>info, &executed_sweeps)
    check_status(status)
    return executed_sweeps

cpdef size_t createGesvdjInfo() except? 0:
    cdef gesvdjInfo_t info
    status = cusolverDnCreateGesvdjInfo(&info)
    check_status(status)
    return <size_t>info

cpdef destroyGesvdjInfo(size_t info):
    status = cusolverDnDestroyGesvdjInfo(<gesvdjInfo_t>info)
    check_status(status)

cpdef xgesvdjSetTolerance(size_t info, double tolerance):
    status = cusolverDnXgesvdjSetTolerance(<gesvdjInfo_t>info, tolerance)
    check_status(status)

cpdef xgesvdjSetMaxSweeps(size_t info, int max_sweeps):
    status = cusolverDnXgesvdjSetMaxSweeps(<gesvdjInfo_t>info, max_sweeps)
    check_status(status)

cpdef xgesvdjSetSortEig(size_t info, int sort_svd):
    status = cusolverDnXgesvdjSetSortEig(<gesvdjInfo_t>info, sort_svd)
    check_status(status)

cpdef double xgesvdjGetResidual(intptr_t handle, size_t info) except? 0:
    cdef double residual
    status = cusolverDnXgesvdjGetResidual(<DnHandle>handle, <gesvdjInfo_t>info, &residual)
    check_status(status)
    return residual

cpdef int xgesvdjGetSweeps(intptr_t handle, size_t info) except? 0:
    cdef int executed_sweeps
    status = cusolverDnXgesvdjGetSweeps(<DnHandle>handle, <gesvdjInfo_t>info, &executed_sweeps)
    check_status(status)
    return executed_sweeps


######################################################
# cuSOLVER Dense LAPACK Function - Dense Linear Solver

cpdef int spotrf_bufferSize(intptr_t handle, int uplo, int n, intptr_t A, int lda) except? 0:
    cdef int Lwork
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnSpotrf_bufferSize(<DnHandle>handle, <FillMode>uplo, n, <float*>A, lda, &Lwork)
    check_status(status)
    return Lwork

cpdef int dpotrf_bufferSize(intptr_t handle, int uplo, int n, intptr_t A, int lda) except? 0:
    cdef int Lwork
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnDpotrf_bufferSize(<DnHandle>handle, <FillMode>uplo, n, <double*>A, lda, &Lwork)
    check_status(status)
    return Lwork

cpdef int cpotrf_bufferSize(intptr_t handle, int uplo, int n, intptr_t A, int lda) except? 0:
    cdef int Lwork
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnCpotrf_bufferSize(<DnHandle>handle, <FillMode>uplo, n, <cuComplex*>A, lda, &Lwork)
    check_status(status)
    return Lwork

cpdef int zpotrf_bufferSize(intptr_t handle, int uplo, int n, intptr_t A, int lda) except? 0:
    cdef int Lwork
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnZpotrf_bufferSize(<DnHandle>handle, <FillMode>uplo, n, <cuDoubleComplex*>A, lda, &Lwork)
    check_status(status)
    return Lwork

cpdef spotrf(intptr_t handle, int uplo, int n, intptr_t A, int lda, intptr_t Workspace, int Lwork, intptr_t devInfo):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnSpotrf(<DnHandle>handle, <FillMode>uplo, n, <float*>A, lda, <float*>Workspace, Lwork, <int*>devInfo)
    check_status(status)

cpdef dpotrf(intptr_t handle, int uplo, int n, intptr_t A, int lda, intptr_t Workspace, int Lwork, intptr_t devInfo):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnDpotrf(<DnHandle>handle, <FillMode>uplo, n, <double*>A, lda, <double*>Workspace, Lwork, <int*>devInfo)
    check_status(status)

cpdef cpotrf(intptr_t handle, int uplo, int n, intptr_t A, int lda, intptr_t Workspace, int Lwork, intptr_t devInfo):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnCpotrf(<DnHandle>handle, <FillMode>uplo, n, <cuComplex*>A, lda, <cuComplex*>Workspace, Lwork, <int*>devInfo)
    check_status(status)

cpdef zpotrf(intptr_t handle, int uplo, int n, intptr_t A, int lda, intptr_t Workspace, int Lwork, intptr_t devInfo):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnZpotrf(<DnHandle>handle, <FillMode>uplo, n, <cuDoubleComplex*>A, lda, <cuDoubleComplex*>Workspace, Lwork, <int*>devInfo)
    check_status(status)

cpdef spotrs(intptr_t handle, int uplo, int n, int nrhs, intptr_t A, int lda, intptr_t B, int ldb, intptr_t devInfo):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnSpotrs(<DnHandle>handle, <FillMode>uplo, n, nrhs, <const float*>A, lda, <float*>B, ldb, <int*>devInfo)
    check_status(status)

cpdef dpotrs(intptr_t handle, int uplo, int n, int nrhs, intptr_t A, int lda, intptr_t B, int ldb, intptr_t devInfo):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnDpotrs(<DnHandle>handle, <FillMode>uplo, n, nrhs, <const double*>A, lda, <double*>B, ldb, <int*>devInfo)
    check_status(status)

cpdef cpotrs(intptr_t handle, int uplo, int n, int nrhs, intptr_t A, int lda, intptr_t B, int ldb, intptr_t devInfo):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnCpotrs(<DnHandle>handle, <FillMode>uplo, n, nrhs, <const cuComplex*>A, lda, <cuComplex*>B, ldb, <int*>devInfo)
    check_status(status)

cpdef zpotrs(intptr_t handle, int uplo, int n, int nrhs, intptr_t A, int lda, intptr_t B, int ldb, intptr_t devInfo):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnZpotrs(<DnHandle>handle, <FillMode>uplo, n, nrhs, <const cuDoubleComplex*>A, lda, <cuDoubleComplex*>B, ldb, <int*>devInfo)
    check_status(status)

cpdef spotrfBatched(intptr_t handle, int uplo, int n, intptr_t Aarray, int lda, intptr_t infoArray, int batchSize):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnSpotrfBatched(<DnHandle>handle, <FillMode>uplo, n, <float**>Aarray, lda, <int*>infoArray, batchSize)
    check_status(status)

cpdef dpotrfBatched(intptr_t handle, int uplo, int n, intptr_t Aarray, int lda, intptr_t infoArray, int batchSize):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnDpotrfBatched(<DnHandle>handle, <FillMode>uplo, n, <double**>Aarray, lda, <int*>infoArray, batchSize)
    check_status(status)

cpdef cpotrfBatched(intptr_t handle, int uplo, int n, intptr_t Aarray, int lda, intptr_t infoArray, int batchSize):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnCpotrfBatched(<DnHandle>handle, <FillMode>uplo, n, <cuComplex**>Aarray, lda, <int*>infoArray, batchSize)
    check_status(status)

cpdef zpotrfBatched(intptr_t handle, int uplo, int n, intptr_t Aarray, int lda, intptr_t infoArray, int batchSize):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnZpotrfBatched(<DnHandle>handle, <FillMode>uplo, n, <cuDoubleComplex**>Aarray, lda, <int*>infoArray, batchSize)
    check_status(status)

cpdef spotrsBatched(intptr_t handle, int uplo, int n, int nrhs, intptr_t A, int lda, intptr_t B, int ldb, intptr_t d_info, int batchSize):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnSpotrsBatched(<DnHandle>handle, <FillMode>uplo, n, nrhs, <float**>A, lda, <float**>B, ldb, <int*>d_info, batchSize)
    check_status(status)

cpdef dpotrsBatched(intptr_t handle, int uplo, int n, int nrhs, intptr_t A, int lda, intptr_t B, int ldb, intptr_t d_info, int batchSize):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnDpotrsBatched(<DnHandle>handle, <FillMode>uplo, n, nrhs, <double**>A, lda, <double**>B, ldb, <int*>d_info, batchSize)
    check_status(status)

cpdef cpotrsBatched(intptr_t handle, int uplo, int n, int nrhs, intptr_t A, int lda, intptr_t B, int ldb, intptr_t d_info, int batchSize):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnCpotrsBatched(<DnHandle>handle, <FillMode>uplo, n, nrhs, <cuComplex**>A, lda, <cuComplex**>B, ldb, <int*>d_info, batchSize)
    check_status(status)

cpdef zpotrsBatched(intptr_t handle, int uplo, int n, int nrhs, intptr_t A, int lda, intptr_t B, int ldb, intptr_t d_info, int batchSize):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnZpotrsBatched(<DnHandle>handle, <FillMode>uplo, n, nrhs, <cuDoubleComplex**>A, lda, <cuDoubleComplex**>B, ldb, <int*>d_info, batchSize)
    check_status(status)

cpdef int sgetrf_bufferSize(intptr_t handle, int m, int n, intptr_t A, int lda) except? 0:
    cdef int Lwork
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnSgetrf_bufferSize(<DnHandle>handle, m, n, <float*>A, lda, &Lwork)
    check_status(status)
    return Lwork

cpdef int dgetrf_bufferSize(intptr_t handle, int m, int n, intptr_t A, int lda) except? 0:
    cdef int Lwork
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnDgetrf_bufferSize(<DnHandle>handle, m, n, <double*>A, lda, &Lwork)
    check_status(status)
    return Lwork

cpdef int cgetrf_bufferSize(intptr_t handle, int m, int n, intptr_t A, int lda) except? 0:
    cdef int Lwork
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnCgetrf_bufferSize(<DnHandle>handle, m, n, <cuComplex*>A, lda, &Lwork)
    check_status(status)
    return Lwork

cpdef int zgetrf_bufferSize(intptr_t handle, int m, int n, intptr_t A, int lda) except? 0:
    cdef int Lwork
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnZgetrf_bufferSize(<DnHandle>handle, m, n, <cuDoubleComplex*>A, lda, &Lwork)
    check_status(status)
    return Lwork

cpdef sgetrf(intptr_t handle, int m, int n, intptr_t A, int lda, intptr_t Workspace, intptr_t devIpiv, intptr_t devInfo):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnSgetrf(<DnHandle>handle, m, n, <float*>A, lda, <float*>Workspace, <int*>devIpiv, <int*>devInfo)
    check_status(status)

cpdef dgetrf(intptr_t handle, int m, int n, intptr_t A, int lda, intptr_t Workspace, intptr_t devIpiv, intptr_t devInfo):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnDgetrf(<DnHandle>handle, m, n, <double*>A, lda, <double*>Workspace, <int*>devIpiv, <int*>devInfo)
    check_status(status)

cpdef cgetrf(intptr_t handle, int m, int n, intptr_t A, int lda, intptr_t Workspace, intptr_t devIpiv, intptr_t devInfo):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnCgetrf(<DnHandle>handle, m, n, <cuComplex*>A, lda, <cuComplex*>Workspace, <int*>devIpiv, <int*>devInfo)
    check_status(status)

cpdef zgetrf(intptr_t handle, int m, int n, intptr_t A, int lda, intptr_t Workspace, intptr_t devIpiv, intptr_t devInfo):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnZgetrf(<DnHandle>handle, m, n, <cuDoubleComplex*>A, lda, <cuDoubleComplex*>Workspace, <int*>devIpiv, <int*>devInfo)
    check_status(status)

cpdef sgetrs(intptr_t handle, int trans, int n, int nrhs, intptr_t A, int lda, intptr_t devIpiv, intptr_t B, int ldb, intptr_t devInfo):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnSgetrs(<DnHandle>handle, <Operation>trans, n, nrhs, <const float*>A, lda, <const int*>devIpiv, <float*>B, ldb, <int*>devInfo)
    check_status(status)

cpdef dgetrs(intptr_t handle, int trans, int n, int nrhs, intptr_t A, int lda, intptr_t devIpiv, intptr_t B, int ldb, intptr_t devInfo):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnDgetrs(<DnHandle>handle, <Operation>trans, n, nrhs, <const double*>A, lda, <const int*>devIpiv, <double*>B, ldb, <int*>devInfo)
    check_status(status)

cpdef cgetrs(intptr_t handle, int trans, int n, int nrhs, intptr_t A, int lda, intptr_t devIpiv, intptr_t B, int ldb, intptr_t devInfo):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnCgetrs(<DnHandle>handle, <Operation>trans, n, nrhs, <const cuComplex*>A, lda, <const int*>devIpiv, <cuComplex*>B, ldb, <int*>devInfo)
    check_status(status)

cpdef zgetrs(intptr_t handle, int trans, int n, int nrhs, intptr_t A, int lda, intptr_t devIpiv, intptr_t B, int ldb, intptr_t devInfo):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnZgetrs(<DnHandle>handle, <Operation>trans, n, nrhs, <const cuDoubleComplex*>A, lda, <const int*>devIpiv, <cuDoubleComplex*>B, ldb, <int*>devInfo)
    check_status(status)

cpdef int sgeqrf_bufferSize(intptr_t handle, int m, int n, intptr_t A, int lda) except? 0:
    cdef int lwork
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnSgeqrf_bufferSize(<DnHandle>handle, m, n, <float*>A, lda, &lwork)
    check_status(status)
    return lwork

cpdef int dgeqrf_bufferSize(intptr_t handle, int m, int n, intptr_t A, int lda) except? 0:
    cdef int lwork
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnDgeqrf_bufferSize(<DnHandle>handle, m, n, <double*>A, lda, &lwork)
    check_status(status)
    return lwork

cpdef int cgeqrf_bufferSize(intptr_t handle, int m, int n, intptr_t A, int lda) except? 0:
    cdef int lwork
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnCgeqrf_bufferSize(<DnHandle>handle, m, n, <cuComplex*>A, lda, &lwork)
    check_status(status)
    return lwork

cpdef int zgeqrf_bufferSize(intptr_t handle, int m, int n, intptr_t A, int lda) except? 0:
    cdef int lwork
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnZgeqrf_bufferSize(<DnHandle>handle, m, n, <cuDoubleComplex*>A, lda, &lwork)
    check_status(status)
    return lwork

cpdef sgeqrf(intptr_t handle, int m, int n, intptr_t A, int lda, intptr_t TAU, intptr_t Workspace, int Lwork, intptr_t devInfo):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnSgeqrf(<DnHandle>handle, m, n, <float*>A, lda, <float*>TAU, <float*>Workspace, Lwork, <int*>devInfo)
    check_status(status)

cpdef dgeqrf(intptr_t handle, int m, int n, intptr_t A, int lda, intptr_t TAU, intptr_t Workspace, int Lwork, intptr_t devInfo):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnDgeqrf(<DnHandle>handle, m, n, <double*>A, lda, <double*>TAU, <double*>Workspace, Lwork, <int*>devInfo)
    check_status(status)

cpdef cgeqrf(intptr_t handle, int m, int n, intptr_t A, int lda, intptr_t TAU, intptr_t Workspace, int Lwork, intptr_t devInfo):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnCgeqrf(<DnHandle>handle, m, n, <cuComplex*>A, lda, <cuComplex*>TAU, <cuComplex*>Workspace, Lwork, <int*>devInfo)
    check_status(status)

cpdef zgeqrf(intptr_t handle, int m, int n, intptr_t A, int lda, intptr_t TAU, intptr_t Workspace, int Lwork, intptr_t devInfo):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnZgeqrf(<DnHandle>handle, m, n, <cuDoubleComplex*>A, lda, <cuDoubleComplex*>TAU, <cuDoubleComplex*>Workspace, Lwork, <int*>devInfo)
    check_status(status)

cpdef int sorgqr_bufferSize(intptr_t handle, int m, int n, int k, intptr_t A, int lda, intptr_t tau) except? 0:
    cdef int lwork
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnSorgqr_bufferSize(<DnHandle>handle, m, n, k, <const float*>A, lda, <const float*>tau, &lwork)
    check_status(status)
    return lwork

cpdef int dorgqr_bufferSize(intptr_t handle, int m, int n, int k, intptr_t A, int lda, intptr_t tau) except? 0:
    cdef int lwork
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnDorgqr_bufferSize(<DnHandle>handle, m, n, k, <const double*>A, lda, <const double*>tau, &lwork)
    check_status(status)
    return lwork

cpdef int cungqr_bufferSize(intptr_t handle, int m, int n, int k, intptr_t A, int lda, intptr_t tau) except? 0:
    cdef int lwork
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnCungqr_bufferSize(<DnHandle>handle, m, n, k, <const cuComplex*>A, lda, <const cuComplex*>tau, &lwork)
    check_status(status)
    return lwork

cpdef int zungqr_bufferSize(intptr_t handle, int m, int n, int k, intptr_t A, int lda, intptr_t tau) except? 0:
    cdef int lwork
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnZungqr_bufferSize(<DnHandle>handle, m, n, k, <const cuDoubleComplex*>A, lda, <const cuDoubleComplex*>tau, &lwork)
    check_status(status)
    return lwork

cpdef sorgqr(intptr_t handle, int m, int n, int k, intptr_t A, int lda, intptr_t tau, intptr_t work, int lwork, intptr_t info):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnSorgqr(<DnHandle>handle, m, n, k, <float*>A, lda, <const float*>tau, <float*>work, lwork, <int*>info)
    check_status(status)

cpdef dorgqr(intptr_t handle, int m, int n, int k, intptr_t A, int lda, intptr_t tau, intptr_t work, int lwork, intptr_t info):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnDorgqr(<DnHandle>handle, m, n, k, <double*>A, lda, <const double*>tau, <double*>work, lwork, <int*>info)
    check_status(status)

cpdef cungqr(intptr_t handle, int m, int n, int k, intptr_t A, int lda, intptr_t tau, intptr_t work, int lwork, intptr_t info):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnCungqr(<DnHandle>handle, m, n, k, <cuComplex*>A, lda, <const cuComplex*>tau, <cuComplex*>work, lwork, <int*>info)
    check_status(status)

cpdef zungqr(intptr_t handle, int m, int n, int k, intptr_t A, int lda, intptr_t tau, intptr_t work, int lwork, intptr_t info):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnZungqr(<DnHandle>handle, m, n, k, <cuDoubleComplex*>A, lda, <const cuDoubleComplex*>tau, <cuDoubleComplex*>work, lwork, <int*>info)
    check_status(status)

cpdef int sormqr_bufferSize(intptr_t handle, int side, int trans, int m, int n, int k, intptr_t A, int lda, intptr_t tau, intptr_t C, int ldc) except? 0:
    cdef int lwork
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnSormqr_bufferSize(<DnHandle>handle, <SideMode>side, <Operation>trans, m, n, k, <const float*>A, lda, <const float*>tau, <const float*>C, ldc, &lwork)
    check_status(status)
    return lwork

cpdef int dormqr_bufferSize(intptr_t handle, int side, int trans, int m, int n, int k, intptr_t A, int lda, intptr_t tau, intptr_t C, int ldc) except? 0:
    cdef int lwork
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnDormqr_bufferSize(<DnHandle>handle, <SideMode>side, <Operation>trans, m, n, k, <const double*>A, lda, <const double*>tau, <const double*>C, ldc, &lwork)
    check_status(status)
    return lwork

cpdef int cunmqr_bufferSize(intptr_t handle, int side, int trans, int m, int n, int k, intptr_t A, int lda, intptr_t tau, intptr_t C, int ldc) except? 0:
    cdef int lwork
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnCunmqr_bufferSize(<DnHandle>handle, <SideMode>side, <Operation>trans, m, n, k, <const cuComplex*>A, lda, <const cuComplex*>tau, <const cuComplex*>C, ldc, &lwork)
    check_status(status)
    return lwork

cpdef int zunmqr_bufferSize(intptr_t handle, int side, int trans, int m, int n, int k, intptr_t A, int lda, intptr_t tau, intptr_t C, int ldc) except? 0:
    cdef int lwork
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnZunmqr_bufferSize(<DnHandle>handle, <SideMode>side, <Operation>trans, m, n, k, <const cuDoubleComplex*>A, lda, <const cuDoubleComplex*>tau, <const cuDoubleComplex*>C, ldc, &lwork)
    check_status(status)
    return lwork

cpdef sormqr(intptr_t handle, int side, int trans, int m, int n, int k, intptr_t A, int lda, intptr_t tau, intptr_t C, int ldc, intptr_t work, int lwork, intptr_t devInfo):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnSormqr(<DnHandle>handle, <SideMode>side, <Operation>trans, m, n, k, <const float*>A, lda, <const float*>tau, <float*>C, ldc, <float*>work, lwork, <int*>devInfo)
    check_status(status)

cpdef dormqr(intptr_t handle, int side, int trans, int m, int n, int k, intptr_t A, int lda, intptr_t tau, intptr_t C, int ldc, intptr_t work, int lwork, intptr_t devInfo):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnDormqr(<DnHandle>handle, <SideMode>side, <Operation>trans, m, n, k, <const double*>A, lda, <const double*>tau, <double*>C, ldc, <double*>work, lwork, <int*>devInfo)
    check_status(status)

cpdef cunmqr(intptr_t handle, int side, int trans, int m, int n, int k, intptr_t A, int lda, intptr_t tau, intptr_t C, int ldc, intptr_t work, int lwork, intptr_t devInfo):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnCunmqr(<DnHandle>handle, <SideMode>side, <Operation>trans, m, n, k, <const cuComplex*>A, lda, <const cuComplex*>tau, <cuComplex*>C, ldc, <cuComplex*>work, lwork, <int*>devInfo)
    check_status(status)

cpdef zunmqr(intptr_t handle, int side, int trans, int m, int n, int k, intptr_t A, int lda, intptr_t tau, intptr_t C, int ldc, intptr_t work, int lwork, intptr_t devInfo):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnZunmqr(<DnHandle>handle, <SideMode>side, <Operation>trans, m, n, k, <const cuDoubleComplex*>A, lda, <const cuDoubleComplex*>tau, <cuDoubleComplex*>C, ldc, <cuDoubleComplex*>work, lwork, <int*>devInfo)
    check_status(status)

cpdef int ssytrf_bufferSize(intptr_t handle, int n, intptr_t A, int lda) except? 0:
    cdef int lwork
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnSsytrf_bufferSize(<DnHandle>handle, n, <float*>A, lda, &lwork)
    check_status(status)
    return lwork

cpdef int dsytrf_bufferSize(intptr_t handle, int n, intptr_t A, int lda) except? 0:
    cdef int lwork
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnDsytrf_bufferSize(<DnHandle>handle, n, <double*>A, lda, &lwork)
    check_status(status)
    return lwork

cpdef int csytrf_bufferSize(intptr_t handle, int n, intptr_t A, int lda) except? 0:
    cdef int lwork
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnCsytrf_bufferSize(<DnHandle>handle, n, <cuComplex*>A, lda, &lwork)
    check_status(status)
    return lwork

cpdef int zsytrf_bufferSize(intptr_t handle, int n, intptr_t A, int lda) except? 0:
    cdef int lwork
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnZsytrf_bufferSize(<DnHandle>handle, n, <cuDoubleComplex*>A, lda, &lwork)
    check_status(status)
    return lwork

cpdef ssytrf(intptr_t handle, int uplo, int n, intptr_t A, int lda, intptr_t ipiv, intptr_t work, int lwork, intptr_t info):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnSsytrf(<DnHandle>handle, <FillMode>uplo, n, <float*>A, lda, <int*>ipiv, <float*>work, lwork, <int*>info)
    check_status(status)

cpdef dsytrf(intptr_t handle, int uplo, int n, intptr_t A, int lda, intptr_t ipiv, intptr_t work, int lwork, intptr_t info):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnDsytrf(<DnHandle>handle, <FillMode>uplo, n, <double*>A, lda, <int*>ipiv, <double*>work, lwork, <int*>info)
    check_status(status)

cpdef csytrf(intptr_t handle, int uplo, int n, intptr_t A, int lda, intptr_t ipiv, intptr_t work, int lwork, intptr_t info):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnCsytrf(<DnHandle>handle, <FillMode>uplo, n, <cuComplex*>A, lda, <int*>ipiv, <cuComplex*>work, lwork, <int*>info)
    check_status(status)

cpdef zsytrf(intptr_t handle, int uplo, int n, intptr_t A, int lda, intptr_t ipiv, intptr_t work, int lwork, intptr_t info):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnZsytrf(<DnHandle>handle, <FillMode>uplo, n, <cuDoubleComplex*>A, lda, <int*>ipiv, <cuDoubleComplex*>work, lwork, <int*>info)
    check_status(status)

cpdef size_t zzgesv_bufferSize(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0:
    cdef size_t lwork_bytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnZZgesv_bufferSize(<DnHandle>handle, n, nrhs, <cuDoubleComplex*>dA, ldda, <int*>dipiv, <cuDoubleComplex*>dB, lddb, <cuDoubleComplex*>dX, lddx, <void*>dWorkspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes

cpdef size_t zcgesv_bufferSize(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0:
    cdef size_t lwork_bytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnZCgesv_bufferSize(<DnHandle>handle, n, nrhs, <cuDoubleComplex*>dA, ldda, <int*>dipiv, <cuDoubleComplex*>dB, lddb, <cuDoubleComplex*>dX, lddx, <void*>dWorkspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes

cpdef size_t zkgesv_bufferSize(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0:
    cdef size_t lwork_bytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnZKgesv_bufferSize(<DnHandle>handle, n, nrhs, <cuDoubleComplex*>dA, ldda, <int*>dipiv, <cuDoubleComplex*>dB, lddb, <cuDoubleComplex*>dX, lddx, <void*>dWorkspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes

cpdef size_t zegesv_bufferSize(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0:
    cdef size_t lwork_bytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnZEgesv_bufferSize(<DnHandle>handle, n, nrhs, <cuDoubleComplex*>dA, ldda, <int*>dipiv, <cuDoubleComplex*>dB, lddb, <cuDoubleComplex*>dX, lddx, <void*>dWorkspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes

cpdef size_t zygesv_bufferSize(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0:
    cdef size_t lwork_bytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnZYgesv_bufferSize(<DnHandle>handle, n, nrhs, <cuDoubleComplex*>dA, ldda, <int*>dipiv, <cuDoubleComplex*>dB, lddb, <cuDoubleComplex*>dX, lddx, <void*>dWorkspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes

cpdef size_t ccgesv_bufferSize(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0:
    cdef size_t lwork_bytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnCCgesv_bufferSize(<DnHandle>handle, n, nrhs, <cuComplex*>dA, ldda, <int*>dipiv, <cuComplex*>dB, lddb, <cuComplex*>dX, lddx, <void*>dWorkspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes

cpdef size_t ckgesv_bufferSize(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0:
    cdef size_t lwork_bytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnCKgesv_bufferSize(<DnHandle>handle, n, nrhs, <cuComplex*>dA, ldda, <int*>dipiv, <cuComplex*>dB, lddb, <cuComplex*>dX, lddx, <void*>dWorkspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes

cpdef size_t cegesv_bufferSize(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0:
    cdef size_t lwork_bytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnCEgesv_bufferSize(<DnHandle>handle, n, nrhs, <cuComplex*>dA, ldda, <int*>dipiv, <cuComplex*>dB, lddb, <cuComplex*>dX, lddx, <void*>dWorkspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes

cpdef size_t cygesv_bufferSize(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0:
    cdef size_t lwork_bytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnCYgesv_bufferSize(<DnHandle>handle, n, nrhs, <cuComplex*>dA, ldda, <int*>dipiv, <cuComplex*>dB, lddb, <cuComplex*>dX, lddx, <void*>dWorkspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes

cpdef size_t ddgesv_bufferSize(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0:
    cdef size_t lwork_bytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnDDgesv_bufferSize(<DnHandle>handle, n, nrhs, <double*>dA, ldda, <int*>dipiv, <double*>dB, lddb, <double*>dX, lddx, <void*>dWorkspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes

cpdef size_t dsgesv_bufferSize(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0:
    cdef size_t lwork_bytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnDSgesv_bufferSize(<DnHandle>handle, n, nrhs, <double*>dA, ldda, <int*>dipiv, <double*>dB, lddb, <double*>dX, lddx, <void*>dWorkspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes

cpdef size_t dhgesv_bufferSize(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0:
    cdef size_t lwork_bytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnDHgesv_bufferSize(<DnHandle>handle, n, nrhs, <double*>dA, ldda, <int*>dipiv, <double*>dB, lddb, <double*>dX, lddx, <void*>dWorkspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes

cpdef size_t dbgesv_bufferSize(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0:
    cdef size_t lwork_bytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnDBgesv_bufferSize(<DnHandle>handle, n, nrhs, <double*>dA, ldda, <int*>dipiv, <double*>dB, lddb, <double*>dX, lddx, <void*>dWorkspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes

cpdef size_t dxgesv_bufferSize(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0:
    cdef size_t lwork_bytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnDXgesv_bufferSize(<DnHandle>handle, n, nrhs, <double*>dA, ldda, <int*>dipiv, <double*>dB, lddb, <double*>dX, lddx, <void*>dWorkspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes

cpdef size_t ssgesv_bufferSize(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0:
    cdef size_t lwork_bytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnSSgesv_bufferSize(<DnHandle>handle, n, nrhs, <float*>dA, ldda, <int*>dipiv, <float*>dB, lddb, <float*>dX, lddx, <void*>dWorkspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes

cpdef size_t shgesv_bufferSize(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0:
    cdef size_t lwork_bytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnSHgesv_bufferSize(<DnHandle>handle, n, nrhs, <float*>dA, ldda, <int*>dipiv, <float*>dB, lddb, <float*>dX, lddx, <void*>dWorkspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes

cpdef size_t sbgesv_bufferSize(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0:
    cdef size_t lwork_bytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnSBgesv_bufferSize(<DnHandle>handle, n, nrhs, <float*>dA, ldda, <int*>dipiv, <float*>dB, lddb, <float*>dX, lddx, <void*>dWorkspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes

cpdef size_t sxgesv_bufferSize(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0:
    cdef size_t lwork_bytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnSXgesv_bufferSize(<DnHandle>handle, n, nrhs, <float*>dA, ldda, <int*>dipiv, <float*>dB, lddb, <float*>dX, lddx, <void*>dWorkspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes

cpdef zzgesv(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t iter, intptr_t d_info):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnZZgesv(<DnHandle>handle, n, nrhs, <cuDoubleComplex*>dA, ldda, <int*>dipiv, <cuDoubleComplex*>dB, lddb, <cuDoubleComplex*>dX, lddx, <void*>dWorkspace, lwork_bytes, <int*>iter, <int*>d_info)
    check_status(status)

cpdef zcgesv(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t iter, intptr_t d_info):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnZCgesv(<DnHandle>handle, n, nrhs, <cuDoubleComplex*>dA, ldda, <int*>dipiv, <cuDoubleComplex*>dB, lddb, <cuDoubleComplex*>dX, lddx, <void*>dWorkspace, lwork_bytes, <int*>iter, <int*>d_info)
    check_status(status)

cpdef zkgesv(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t iter, intptr_t d_info):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnZKgesv(<DnHandle>handle, n, nrhs, <cuDoubleComplex*>dA, ldda, <int*>dipiv, <cuDoubleComplex*>dB, lddb, <cuDoubleComplex*>dX, lddx, <void*>dWorkspace, lwork_bytes, <int*>iter, <int*>d_info)
    check_status(status)

cpdef zegesv(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t iter, intptr_t d_info):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnZEgesv(<DnHandle>handle, n, nrhs, <cuDoubleComplex*>dA, ldda, <int*>dipiv, <cuDoubleComplex*>dB, lddb, <cuDoubleComplex*>dX, lddx, <void*>dWorkspace, lwork_bytes, <int*>iter, <int*>d_info)
    check_status(status)

cpdef zygesv(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t iter, intptr_t d_info):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnZYgesv(<DnHandle>handle, n, nrhs, <cuDoubleComplex*>dA, ldda, <int*>dipiv, <cuDoubleComplex*>dB, lddb, <cuDoubleComplex*>dX, lddx, <void*>dWorkspace, lwork_bytes, <int*>iter, <int*>d_info)
    check_status(status)

cpdef ccgesv(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t iter, intptr_t d_info):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnCCgesv(<DnHandle>handle, n, nrhs, <cuComplex*>dA, ldda, <int*>dipiv, <cuComplex*>dB, lddb, <cuComplex*>dX, lddx, <void*>dWorkspace, lwork_bytes, <int*>iter, <int*>d_info)
    check_status(status)

cpdef cegesv(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t iter, intptr_t d_info):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnCEgesv(<DnHandle>handle, n, nrhs, <cuComplex*>dA, ldda, <int*>dipiv, <cuComplex*>dB, lddb, <cuComplex*>dX, lddx, <void*>dWorkspace, lwork_bytes, <int*>iter, <int*>d_info)
    check_status(status)

cpdef ckgesv(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t iter, intptr_t d_info):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnCKgesv(<DnHandle>handle, n, nrhs, <cuComplex*>dA, ldda, <int*>dipiv, <cuComplex*>dB, lddb, <cuComplex*>dX, lddx, <void*>dWorkspace, lwork_bytes, <int*>iter, <int*>d_info)
    check_status(status)

cpdef cygesv(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t iter, intptr_t d_info):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnCYgesv(<DnHandle>handle, n, nrhs, <cuComplex*>dA, ldda, <int*>dipiv, <cuComplex*>dB, lddb, <cuComplex*>dX, lddx, <void*>dWorkspace, lwork_bytes, <int*>iter, <int*>d_info)
    check_status(status)

cpdef ddgesv(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t iter, intptr_t d_info):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnDDgesv(<DnHandle>handle, n, nrhs, <double*>dA, ldda, <int*>dipiv, <double*>dB, lddb, <double*>dX, lddx, <void*>dWorkspace, lwork_bytes, <int*>iter, <int*>d_info)
    check_status(status)

cpdef dsgesv(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t iter, intptr_t d_info):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnDSgesv(<DnHandle>handle, n, nrhs, <double*>dA, ldda, <int*>dipiv, <double*>dB, lddb, <double*>dX, lddx, <void*>dWorkspace, lwork_bytes, <int*>iter, <int*>d_info)
    check_status(status)

cpdef dhgesv(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t iter, intptr_t d_info):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnDHgesv(<DnHandle>handle, n, nrhs, <double*>dA, ldda, <int*>dipiv, <double*>dB, lddb, <double*>dX, lddx, <void*>dWorkspace, lwork_bytes, <int*>iter, <int*>d_info)
    check_status(status)

cpdef dbgesv(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t iter, intptr_t d_info):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnDBgesv(<DnHandle>handle, n, nrhs, <double*>dA, ldda, <int*>dipiv, <double*>dB, lddb, <double*>dX, lddx, <void*>dWorkspace, lwork_bytes, <int*>iter, <int*>d_info)
    check_status(status)

cpdef dxgesv(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t iter, intptr_t d_info):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnDXgesv(<DnHandle>handle, n, nrhs, <double*>dA, ldda, <int*>dipiv, <double*>dB, lddb, <double*>dX, lddx, <void*>dWorkspace, lwork_bytes, <int*>iter, <int*>d_info)
    check_status(status)

cpdef ssgesv(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t iter, intptr_t d_info):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnSSgesv(<DnHandle>handle, n, nrhs, <float*>dA, ldda, <int*>dipiv, <float*>dB, lddb, <float*>dX, lddx, <void*>dWorkspace, lwork_bytes, <int*>iter, <int*>d_info)
    check_status(status)

cpdef shgesv(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t iter, intptr_t d_info):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnSHgesv(<DnHandle>handle, n, nrhs, <float*>dA, ldda, <int*>dipiv, <float*>dB, lddb, <float*>dX, lddx, <void*>dWorkspace, lwork_bytes, <int*>iter, <int*>d_info)
    check_status(status)

cpdef sbgesv(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t iter, intptr_t d_info):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnSBgesv(<DnHandle>handle, n, nrhs, <float*>dA, ldda, <int*>dipiv, <float*>dB, lddb, <float*>dX, lddx, <void*>dWorkspace, lwork_bytes, <int*>iter, <int*>d_info)
    check_status(status)

cpdef sxgesv(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t iter, intptr_t d_info):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnSXgesv(<DnHandle>handle, n, nrhs, <float*>dA, ldda, <int*>dipiv, <float*>dB, lddb, <float*>dX, lddx, <void*>dWorkspace, lwork_bytes, <int*>iter, <int*>d_info)
    check_status(status)

cpdef size_t zzgels_bufferSize(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0:
    cdef size_t lwork_bytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnZZgels_bufferSize(<DnHandle>handle, m, n, nrhs, <cuDoubleComplex*>dA, ldda, <cuDoubleComplex*>dB, lddb, <cuDoubleComplex*>dX, lddx, <void*>dWorkspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes

cpdef size_t zcgels_bufferSize(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0:
    cdef size_t lwork_bytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnZCgels_bufferSize(<DnHandle>handle, m, n, nrhs, <cuDoubleComplex*>dA, ldda, <cuDoubleComplex*>dB, lddb, <cuDoubleComplex*>dX, lddx, <void*>dWorkspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes

cpdef size_t zkgels_bufferSize(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0:
    cdef size_t lwork_bytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnZKgels_bufferSize(<DnHandle>handle, m, n, nrhs, <cuDoubleComplex*>dA, ldda, <cuDoubleComplex*>dB, lddb, <cuDoubleComplex*>dX, lddx, <void*>dWorkspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes

cpdef size_t zegels_bufferSize(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0:
    cdef size_t lwork_bytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnZEgels_bufferSize(<DnHandle>handle, m, n, nrhs, <cuDoubleComplex*>dA, ldda, <cuDoubleComplex*>dB, lddb, <cuDoubleComplex*>dX, lddx, <void*>dWorkspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes

cpdef size_t zygels_bufferSize(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0:
    cdef size_t lwork_bytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnZYgels_bufferSize(<DnHandle>handle, m, n, nrhs, <cuDoubleComplex*>dA, ldda, <cuDoubleComplex*>dB, lddb, <cuDoubleComplex*>dX, lddx, <void*>dWorkspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes

cpdef size_t ccgels_bufferSize(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0:
    cdef size_t lwork_bytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnCCgels_bufferSize(<DnHandle>handle, m, n, nrhs, <cuComplex*>dA, ldda, <cuComplex*>dB, lddb, <cuComplex*>dX, lddx, <void*>dWorkspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes

cpdef size_t ckgels_bufferSize(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0:
    cdef size_t lwork_bytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnCKgels_bufferSize(<DnHandle>handle, m, n, nrhs, <cuComplex*>dA, ldda, <cuComplex*>dB, lddb, <cuComplex*>dX, lddx, <void*>dWorkspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes

cpdef size_t cegels_bufferSize(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0:
    cdef size_t lwork_bytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnCEgels_bufferSize(<DnHandle>handle, m, n, nrhs, <cuComplex*>dA, ldda, <cuComplex*>dB, lddb, <cuComplex*>dX, lddx, <void*>dWorkspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes

cpdef size_t cygels_bufferSize(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0:
    cdef size_t lwork_bytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnCYgels_bufferSize(<DnHandle>handle, m, n, nrhs, <cuComplex*>dA, ldda, <cuComplex*>dB, lddb, <cuComplex*>dX, lddx, <void*>dWorkspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes

cpdef size_t ddgels_bufferSize(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0:
    cdef size_t lwork_bytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnDDgels_bufferSize(<DnHandle>handle, m, n, nrhs, <double*>dA, ldda, <double*>dB, lddb, <double*>dX, lddx, <void*>dWorkspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes

cpdef size_t dsgels_bufferSize(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0:
    cdef size_t lwork_bytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnDSgels_bufferSize(<DnHandle>handle, m, n, nrhs, <double*>dA, ldda, <double*>dB, lddb, <double*>dX, lddx, <void*>dWorkspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes

cpdef size_t dhgels_bufferSize(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0:
    cdef size_t lwork_bytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnDHgels_bufferSize(<DnHandle>handle, m, n, nrhs, <double*>dA, ldda, <double*>dB, lddb, <double*>dX, lddx, <void*>dWorkspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes

cpdef size_t dbgels_bufferSize(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0:
    cdef size_t lwork_bytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnDBgels_bufferSize(<DnHandle>handle, m, n, nrhs, <double*>dA, ldda, <double*>dB, lddb, <double*>dX, lddx, <void*>dWorkspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes

cpdef size_t dxgels_bufferSize(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0:
    cdef size_t lwork_bytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnDXgels_bufferSize(<DnHandle>handle, m, n, nrhs, <double*>dA, ldda, <double*>dB, lddb, <double*>dX, lddx, <void*>dWorkspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes

cpdef size_t ssgels_bufferSize(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0:
    cdef size_t lwork_bytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnSSgels_bufferSize(<DnHandle>handle, m, n, nrhs, <float*>dA, ldda, <float*>dB, lddb, <float*>dX, lddx, <void*>dWorkspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes

cpdef size_t shgels_bufferSize(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0:
    cdef size_t lwork_bytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnSHgels_bufferSize(<DnHandle>handle, m, n, nrhs, <float*>dA, ldda, <float*>dB, lddb, <float*>dX, lddx, <void*>dWorkspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes

cpdef size_t sbgels_bufferSize(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0:
    cdef size_t lwork_bytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnSBgels_bufferSize(<DnHandle>handle, m, n, nrhs, <float*>dA, ldda, <float*>dB, lddb, <float*>dX, lddx, <void*>dWorkspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes

cpdef size_t sxgels_bufferSize(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0:
    cdef size_t lwork_bytes
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnSXgels_bufferSize(<DnHandle>handle, m, n, nrhs, <float*>dA, ldda, <float*>dB, lddb, <float*>dX, lddx, <void*>dWorkspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes

cpdef zzgels(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t iter, intptr_t d_info):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnZZgels(<DnHandle>handle, m, n, nrhs, <cuDoubleComplex*>dA, ldda, <cuDoubleComplex*>dB, lddb, <cuDoubleComplex*>dX, lddx, <void*>dWorkspace, lwork_bytes, <int*>iter, <int*>d_info)
    check_status(status)

cpdef zcgels(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t iter, intptr_t d_info):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnZCgels(<DnHandle>handle, m, n, nrhs, <cuDoubleComplex*>dA, ldda, <cuDoubleComplex*>dB, lddb, <cuDoubleComplex*>dX, lddx, <void*>dWorkspace, lwork_bytes, <int*>iter, <int*>d_info)
    check_status(status)

cpdef zkgels(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t iter, intptr_t d_info):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnZKgels(<DnHandle>handle, m, n, nrhs, <cuDoubleComplex*>dA, ldda, <cuDoubleComplex*>dB, lddb, <cuDoubleComplex*>dX, lddx, <void*>dWorkspace, lwork_bytes, <int*>iter, <int*>d_info)
    check_status(status)

cpdef zegels(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t iter, intptr_t d_info):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnZEgels(<DnHandle>handle, m, n, nrhs, <cuDoubleComplex*>dA, ldda, <cuDoubleComplex*>dB, lddb, <cuDoubleComplex*>dX, lddx, <void*>dWorkspace, lwork_bytes, <int*>iter, <int*>d_info)
    check_status(status)

cpdef zygels(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t iter, intptr_t d_info):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnZYgels(<DnHandle>handle, m, n, nrhs, <cuDoubleComplex*>dA, ldda, <cuDoubleComplex*>dB, lddb, <cuDoubleComplex*>dX, lddx, <void*>dWorkspace, lwork_bytes, <int*>iter, <int*>d_info)
    check_status(status)

cpdef ccgels(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t iter, intptr_t d_info):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnCCgels(<DnHandle>handle, m, n, nrhs, <cuComplex*>dA, ldda, <cuComplex*>dB, lddb, <cuComplex*>dX, lddx, <void*>dWorkspace, lwork_bytes, <int*>iter, <int*>d_info)
    check_status(status)

cpdef ckgels(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t iter, intptr_t d_info):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnCKgels(<DnHandle>handle, m, n, nrhs, <cuComplex*>dA, ldda, <cuComplex*>dB, lddb, <cuComplex*>dX, lddx, <void*>dWorkspace, lwork_bytes, <int*>iter, <int*>d_info)
    check_status(status)

cpdef cegels(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t iter, intptr_t d_info):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnCEgels(<DnHandle>handle, m, n, nrhs, <cuComplex*>dA, ldda, <cuComplex*>dB, lddb, <cuComplex*>dX, lddx, <void*>dWorkspace, lwork_bytes, <int*>iter, <int*>d_info)
    check_status(status)

cpdef cygels(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t iter, intptr_t d_info):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnCYgels(<DnHandle>handle, m, n, nrhs, <cuComplex*>dA, ldda, <cuComplex*>dB, lddb, <cuComplex*>dX, lddx, <void*>dWorkspace, lwork_bytes, <int*>iter, <int*>d_info)
    check_status(status)

cpdef ddgels(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t iter, intptr_t d_info):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnDDgels(<DnHandle>handle, m, n, nrhs, <double*>dA, ldda, <double*>dB, lddb, <double*>dX, lddx, <void*>dWorkspace, lwork_bytes, <int*>iter, <int*>d_info)
    check_status(status)

cpdef dsgels(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t iter, intptr_t d_info):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnDSgels(<DnHandle>handle, m, n, nrhs, <double*>dA, ldda, <double*>dB, lddb, <double*>dX, lddx, <void*>dWorkspace, lwork_bytes, <int*>iter, <int*>d_info)
    check_status(status)

cpdef dhgels(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t iter, intptr_t d_info):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnDHgels(<DnHandle>handle, m, n, nrhs, <double*>dA, ldda, <double*>dB, lddb, <double*>dX, lddx, <void*>dWorkspace, lwork_bytes, <int*>iter, <int*>d_info)
    check_status(status)

cpdef dbgels(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t iter, intptr_t d_info):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnDBgels(<DnHandle>handle, m, n, nrhs, <double*>dA, ldda, <double*>dB, lddb, <double*>dX, lddx, <void*>dWorkspace, lwork_bytes, <int*>iter, <int*>d_info)
    check_status(status)

cpdef dxgels(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t iter, intptr_t d_info):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnDXgels(<DnHandle>handle, m, n, nrhs, <double*>dA, ldda, <double*>dB, lddb, <double*>dX, lddx, <void*>dWorkspace, lwork_bytes, <int*>iter, <int*>d_info)
    check_status(status)

cpdef ssgels(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t iter, intptr_t d_info):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnSSgels(<DnHandle>handle, m, n, nrhs, <float*>dA, ldda, <float*>dB, lddb, <float*>dX, lddx, <void*>dWorkspace, lwork_bytes, <int*>iter, <int*>d_info)
    check_status(status)

cpdef shgels(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t iter, intptr_t d_info):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnSHgels(<DnHandle>handle, m, n, nrhs, <float*>dA, ldda, <float*>dB, lddb, <float*>dX, lddx, <void*>dWorkspace, lwork_bytes, <int*>iter, <int*>d_info)
    check_status(status)

cpdef sbgels(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t iter, intptr_t d_info):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnSBgels(<DnHandle>handle, m, n, nrhs, <float*>dA, ldda, <float*>dB, lddb, <float*>dX, lddx, <void*>dWorkspace, lwork_bytes, <int*>iter, <int*>d_info)
    check_status(status)

cpdef sxgels(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t iter, intptr_t d_info):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnSXgels(<DnHandle>handle, m, n, nrhs, <float*>dA, ldda, <float*>dB, lddb, <float*>dX, lddx, <void*>dWorkspace, lwork_bytes, <int*>iter, <int*>d_info)
    check_status(status)


##########################################################
# cuSOLVER Dense LAPACK Function - Dense Eigenvalue Solver

cpdef int sgebrd_bufferSize(intptr_t handle, int m, int n) except? 0:
    cdef int Lwork
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnSgebrd_bufferSize(<DnHandle>handle, m, n, &Lwork)
    check_status(status)
    return Lwork

cpdef int dgebrd_bufferSize(intptr_t handle, int m, int n) except? 0:
    cdef int Lwork
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnDgebrd_bufferSize(<DnHandle>handle, m, n, &Lwork)
    check_status(status)
    return Lwork

cpdef int cgebrd_bufferSize(intptr_t handle, int m, int n) except? 0:
    cdef int Lwork
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnCgebrd_bufferSize(<DnHandle>handle, m, n, &Lwork)
    check_status(status)
    return Lwork

cpdef int zgebrd_bufferSize(intptr_t handle, int m, int n) except? 0:
    cdef int Lwork
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnZgebrd_bufferSize(<DnHandle>handle, m, n, &Lwork)
    check_status(status)
    return Lwork

cpdef sgebrd(intptr_t handle, int m, int n, intptr_t A, int lda, intptr_t D, intptr_t E, intptr_t TAUQ, intptr_t TAUP, intptr_t Work, int Lwork, intptr_t devInfo):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnSgebrd(<DnHandle>handle, m, n, <float*>A, lda, <float*>D, <float*>E, <float*>TAUQ, <float*>TAUP, <float*>Work, Lwork, <int*>devInfo)
    check_status(status)

cpdef dgebrd(intptr_t handle, int m, int n, intptr_t A, int lda, intptr_t D, intptr_t E, intptr_t TAUQ, intptr_t TAUP, intptr_t Work, int Lwork, intptr_t devInfo):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnDgebrd(<DnHandle>handle, m, n, <double*>A, lda, <double*>D, <double*>E, <double*>TAUQ, <double*>TAUP, <double*>Work, Lwork, <int*>devInfo)
    check_status(status)

cpdef cgebrd(intptr_t handle, int m, int n, intptr_t A, int lda, intptr_t D, intptr_t E, intptr_t TAUQ, intptr_t TAUP, intptr_t Work, int Lwork, intptr_t devInfo):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnCgebrd(<DnHandle>handle, m, n, <cuComplex*>A, lda, <float*>D, <float*>E, <cuComplex*>TAUQ, <cuComplex*>TAUP, <cuComplex*>Work, Lwork, <int*>devInfo)
    check_status(status)

cpdef zgebrd(intptr_t handle, int m, int n, intptr_t A, int lda, intptr_t D, intptr_t E, intptr_t TAUQ, intptr_t TAUP, intptr_t Work, int Lwork, intptr_t devInfo):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnZgebrd(<DnHandle>handle, m, n, <cuDoubleComplex*>A, lda, <double*>D, <double*>E, <cuDoubleComplex*>TAUQ, <cuDoubleComplex*>TAUP, <cuDoubleComplex*>Work, Lwork, <int*>devInfo)
    check_status(status)

cpdef int sgesvd_bufferSize(intptr_t handle, int m, int n) except? 0:
    cdef int lwork
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnSgesvd_bufferSize(<DnHandle>handle, m, n, &lwork)
    check_status(status)
    return lwork

cpdef int dgesvd_bufferSize(intptr_t handle, int m, int n) except? 0:
    cdef int lwork
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnDgesvd_bufferSize(<DnHandle>handle, m, n, &lwork)
    check_status(status)
    return lwork

cpdef int cgesvd_bufferSize(intptr_t handle, int m, int n) except? 0:
    cdef int lwork
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnCgesvd_bufferSize(<DnHandle>handle, m, n, &lwork)
    check_status(status)
    return lwork

cpdef int zgesvd_bufferSize(intptr_t handle, int m, int n) except? 0:
    cdef int lwork
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnZgesvd_bufferSize(<DnHandle>handle, m, n, &lwork)
    check_status(status)
    return lwork

cpdef sgesvd(intptr_t handle, signed char jobu, signed char jobvt, int m, int n, intptr_t A, int lda, intptr_t S, intptr_t U, int ldu, intptr_t VT, int ldvt, intptr_t work, int lwork, intptr_t rwork, intptr_t info):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnSgesvd(<DnHandle>handle, jobu, jobvt, m, n, <float*>A, lda, <float*>S, <float*>U, ldu, <float*>VT, ldvt, <float*>work, lwork, <float*>rwork, <int*>info)
    check_status(status)

cpdef dgesvd(intptr_t handle, signed char jobu, signed char jobvt, int m, int n, intptr_t A, int lda, intptr_t S, intptr_t U, int ldu, intptr_t VT, int ldvt, intptr_t work, int lwork, intptr_t rwork, intptr_t info):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnDgesvd(<DnHandle>handle, jobu, jobvt, m, n, <double*>A, lda, <double*>S, <double*>U, ldu, <double*>VT, ldvt, <double*>work, lwork, <double*>rwork, <int*>info)
    check_status(status)

cpdef cgesvd(intptr_t handle, signed char jobu, signed char jobvt, int m, int n, intptr_t A, int lda, intptr_t S, intptr_t U, int ldu, intptr_t VT, int ldvt, intptr_t work, int lwork, intptr_t rwork, intptr_t info):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnCgesvd(<DnHandle>handle, jobu, jobvt, m, n, <cuComplex*>A, lda, <float*>S, <cuComplex*>U, ldu, <cuComplex*>VT, ldvt, <cuComplex*>work, lwork, <float*>rwork, <int*>info)
    check_status(status)

cpdef zgesvd(intptr_t handle, signed char jobu, signed char jobvt, int m, int n, intptr_t A, int lda, intptr_t S, intptr_t U, int ldu, intptr_t VT, int ldvt, intptr_t work, int lwork, intptr_t rwork, intptr_t info):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnZgesvd(<DnHandle>handle, jobu, jobvt, m, n, <cuDoubleComplex*>A, lda, <double*>S, <cuDoubleComplex*>U, ldu, <cuDoubleComplex*>VT, ldvt, <cuDoubleComplex*>work, lwork, <double*>rwork, <int*>info)
    check_status(status)

cpdef int sgesvdj_bufferSize(intptr_t handle, int jobz, int econ, int m, int n, intptr_t A, int lda, intptr_t S, intptr_t U, int ldu, intptr_t V, int ldv, size_t params) except? 0:
    cdef int lwork
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnSgesvdj_bufferSize(<DnHandle>handle, <EigMode>jobz, econ, m, n, <const float*>A, lda, <const float*>S, <const float*>U, ldu, <const float*>V, ldv, &lwork, <gesvdjInfo_t>params)
    check_status(status)
    return lwork

cpdef int dgesvdj_bufferSize(intptr_t handle, int jobz, int econ, int m, int n, intptr_t A, int lda, intptr_t S, intptr_t U, int ldu, intptr_t V, int ldv, size_t params) except? 0:
    cdef int lwork
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnDgesvdj_bufferSize(<DnHandle>handle, <EigMode>jobz, econ, m, n, <const double*>A, lda, <const double*>S, <const double*>U, ldu, <const double*>V, ldv, &lwork, <gesvdjInfo_t>params)
    check_status(status)
    return lwork

cpdef int cgesvdj_bufferSize(intptr_t handle, int jobz, int econ, int m, int n, intptr_t A, int lda, intptr_t S, intptr_t U, int ldu, intptr_t V, int ldv, size_t params) except? 0:
    cdef int lwork
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnCgesvdj_bufferSize(<DnHandle>handle, <EigMode>jobz, econ, m, n, <const cuComplex*>A, lda, <const float*>S, <const cuComplex*>U, ldu, <const cuComplex*>V, ldv, &lwork, <gesvdjInfo_t>params)
    check_status(status)
    return lwork

cpdef int zgesvdj_bufferSize(intptr_t handle, int jobz, int econ, int m, int n, intptr_t A, int lda, intptr_t S, intptr_t U, int ldu, intptr_t V, int ldv, size_t params) except? 0:
    cdef int lwork
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnZgesvdj_bufferSize(<DnHandle>handle, <EigMode>jobz, econ, m, n, <const cuDoubleComplex*>A, lda, <const double*>S, <const cuDoubleComplex*>U, ldu, <const cuDoubleComplex*>V, ldv, &lwork, <gesvdjInfo_t>params)
    check_status(status)
    return lwork

cpdef sgesvdj(intptr_t handle, int jobz, int econ, int m, int n, intptr_t A, int lda, intptr_t S, intptr_t U, int ldu, intptr_t V, int ldv, intptr_t work, int lwork, intptr_t info, size_t params):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnSgesvdj(<DnHandle>handle, <EigMode>jobz, econ, m, n, <float*>A, lda, <float*>S, <float*>U, ldu, <float*>V, ldv, <float*>work, lwork, <int*>info, <gesvdjInfo_t>params)
    check_status(status)

cpdef dgesvdj(intptr_t handle, int jobz, int econ, int m, int n, intptr_t A, int lda, intptr_t S, intptr_t U, int ldu, intptr_t V, int ldv, intptr_t work, int lwork, intptr_t info, size_t params):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnDgesvdj(<DnHandle>handle, <EigMode>jobz, econ, m, n, <double*>A, lda, <double*>S, <double*>U, ldu, <double*>V, ldv, <double*>work, lwork, <int*>info, <gesvdjInfo_t>params)
    check_status(status)

cpdef cgesvdj(intptr_t handle, int jobz, int econ, int m, int n, intptr_t A, int lda, intptr_t S, intptr_t U, int ldu, intptr_t V, int ldv, intptr_t work, int lwork, intptr_t info, size_t params):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnCgesvdj(<DnHandle>handle, <EigMode>jobz, econ, m, n, <cuComplex*>A, lda, <float*>S, <cuComplex*>U, ldu, <cuComplex*>V, ldv, <cuComplex*>work, lwork, <int*>info, <gesvdjInfo_t>params)
    check_status(status)

cpdef zgesvdj(intptr_t handle, int jobz, int econ, int m, int n, intptr_t A, int lda, intptr_t S, intptr_t U, int ldu, intptr_t V, int ldv, intptr_t work, int lwork, intptr_t info, size_t params):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnZgesvdj(<DnHandle>handle, <EigMode>jobz, econ, m, n, <cuDoubleComplex*>A, lda, <double*>S, <cuDoubleComplex*>U, ldu, <cuDoubleComplex*>V, ldv, <cuDoubleComplex*>work, lwork, <int*>info, <gesvdjInfo_t>params)
    check_status(status)

cpdef int sgesvdjBatched_bufferSize(intptr_t handle, int jobz, int m, int n, intptr_t A, int lda, intptr_t S, intptr_t U, int ldu, intptr_t V, int ldv, size_t params, int batchSize) except? 0:
    cdef int lwork
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnSgesvdjBatched_bufferSize(<DnHandle>handle, <EigMode>jobz, m, n, <const float*>A, lda, <const float*>S, <const float*>U, ldu, <const float*>V, ldv, &lwork, <gesvdjInfo_t>params, batchSize)
    check_status(status)
    return lwork

cpdef int dgesvdjBatched_bufferSize(intptr_t handle, int jobz, int m, int n, intptr_t A, int lda, intptr_t S, intptr_t U, int ldu, intptr_t V, int ldv, size_t params, int batchSize) except? 0:
    cdef int lwork
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnDgesvdjBatched_bufferSize(<DnHandle>handle, <EigMode>jobz, m, n, <const double*>A, lda, <const double*>S, <const double*>U, ldu, <const double*>V, ldv, &lwork, <gesvdjInfo_t>params, batchSize)
    check_status(status)
    return lwork

cpdef int cgesvdjBatched_bufferSize(intptr_t handle, int jobz, int m, int n, intptr_t A, int lda, intptr_t S, intptr_t U, int ldu, intptr_t V, int ldv, size_t params, int batchSize) except? 0:
    cdef int lwork
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnCgesvdjBatched_bufferSize(<DnHandle>handle, <EigMode>jobz, m, n, <const cuComplex*>A, lda, <const float*>S, <const cuComplex*>U, ldu, <const cuComplex*>V, ldv, &lwork, <gesvdjInfo_t>params, batchSize)
    check_status(status)
    return lwork

cpdef int zgesvdjBatched_bufferSize(intptr_t handle, int jobz, int m, int n, intptr_t A, int lda, intptr_t S, intptr_t U, int ldu, intptr_t V, int ldv, size_t params, int batchSize) except? 0:
    cdef int lwork
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnZgesvdjBatched_bufferSize(<DnHandle>handle, <EigMode>jobz, m, n, <const cuDoubleComplex*>A, lda, <const double*>S, <const cuDoubleComplex*>U, ldu, <const cuDoubleComplex*>V, ldv, &lwork, <gesvdjInfo_t>params, batchSize)
    check_status(status)
    return lwork

cpdef sgesvdjBatched(intptr_t handle, int jobz, int m, int n, intptr_t A, int lda, intptr_t S, intptr_t U, int ldu, intptr_t V, int ldv, intptr_t work, int lwork, intptr_t info, size_t params, int batchSize):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnSgesvdjBatched(<DnHandle>handle, <EigMode>jobz, m, n, <float*>A, lda, <float*>S, <float*>U, ldu, <float*>V, ldv, <float*>work, lwork, <int*>info, <gesvdjInfo_t>params, batchSize)
    check_status(status)

cpdef dgesvdjBatched(intptr_t handle, int jobz, int m, int n, intptr_t A, int lda, intptr_t S, intptr_t U, int ldu, intptr_t V, int ldv, intptr_t work, int lwork, intptr_t info, size_t params, int batchSize):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnDgesvdjBatched(<DnHandle>handle, <EigMode>jobz, m, n, <double*>A, lda, <double*>S, <double*>U, ldu, <double*>V, ldv, <double*>work, lwork, <int*>info, <gesvdjInfo_t>params, batchSize)
    check_status(status)

cpdef cgesvdjBatched(intptr_t handle, int jobz, int m, int n, intptr_t A, int lda, intptr_t S, intptr_t U, int ldu, intptr_t V, int ldv, intptr_t work, int lwork, intptr_t info, size_t params, int batchSize):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnCgesvdjBatched(<DnHandle>handle, <EigMode>jobz, m, n, <cuComplex*>A, lda, <float*>S, <cuComplex*>U, ldu, <cuComplex*>V, ldv, <cuComplex*>work, lwork, <int*>info, <gesvdjInfo_t>params, batchSize)
    check_status(status)

cpdef zgesvdjBatched(intptr_t handle, int jobz, int m, int n, intptr_t A, int lda, intptr_t S, intptr_t U, int ldu, intptr_t V, int ldv, intptr_t work, int lwork, intptr_t info, size_t params, int batchSize):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnZgesvdjBatched(<DnHandle>handle, <EigMode>jobz, m, n, <cuDoubleComplex*>A, lda, <double*>S, <cuDoubleComplex*>U, ldu, <cuDoubleComplex*>V, ldv, <cuDoubleComplex*>work, lwork, <int*>info, <gesvdjInfo_t>params, batchSize)
    check_status(status)

cpdef int sgesvdaStridedBatched_bufferSize(intptr_t handle, int jobz, int rank, int m, int n, intptr_t d_A, int lda, long long int strideA, intptr_t d_S, long long int strideS, intptr_t d_U, int ldu, long long int strideU, intptr_t d_V, int ldv, long long int strideV, int batchSize) except? 0:
    cdef int lwork
    status = cusolverDnSgesvdaStridedBatched_bufferSize(<DnHandle>handle, <EigMode>jobz, rank, m, n, <const float*>d_A, lda, strideA, <const float*>d_S, strideS, <const float*>d_U, ldu, strideU, <const float*>d_V, ldv, strideV, &lwork, batchSize)
    check_status(status)
    return lwork

cpdef int dgesvdaStridedBatched_bufferSize(intptr_t handle, int jobz, int rank, int m, int n, intptr_t d_A, int lda, long long int strideA, intptr_t d_S, long long int strideS, intptr_t d_U, int ldu, long long int strideU, intptr_t d_V, int ldv, long long int strideV, int batchSize) except? 0:
    cdef int lwork
    status = cusolverDnDgesvdaStridedBatched_bufferSize(<DnHandle>handle, <EigMode>jobz, rank, m, n, <const double*>d_A, lda, strideA, <const double*>d_S, strideS, <const double*>d_U, ldu, strideU, <const double*>d_V, ldv, strideV, &lwork, batchSize)
    check_status(status)
    return lwork

cpdef int cgesvdaStridedBatched_bufferSize(intptr_t handle, int jobz, int rank, int m, int n, intptr_t d_A, int lda, long long int strideA, intptr_t d_S, long long int strideS, intptr_t d_U, int ldu, long long int strideU, intptr_t d_V, int ldv, long long int strideV, int batchSize) except? 0:
    cdef int lwork
    status = cusolverDnCgesvdaStridedBatched_bufferSize(<DnHandle>handle, <EigMode>jobz, rank, m, n, <const cuComplex*>d_A, lda, strideA, <const float*>d_S, strideS, <const cuComplex*>d_U, ldu, strideU, <const cuComplex*>d_V, ldv, strideV, &lwork, batchSize)
    check_status(status)
    return lwork

cpdef int zgesvdaStridedBatched_bufferSize(intptr_t handle, int jobz, int rank, int m, int n, intptr_t d_A, int lda, long long int strideA, intptr_t d_S, long long int strideS, intptr_t d_U, int ldu, long long int strideU, intptr_t d_V, int ldv, long long int strideV, int batchSize) except? 0:
    cdef int lwork
    status = cusolverDnZgesvdaStridedBatched_bufferSize(<DnHandle>handle, <EigMode>jobz, rank, m, n, <const cuDoubleComplex*>d_A, lda, strideA, <const double*>d_S, strideS, <const cuDoubleComplex*>d_U, ldu, strideU, <const cuDoubleComplex*>d_V, ldv, strideV, &lwork, batchSize)
    check_status(status)
    return lwork

cpdef sgesvdaStridedBatched(intptr_t handle, int jobz, int rank, int m, int n, intptr_t d_A, int lda, long long int strideA, intptr_t d_S, long long int strideS, intptr_t d_U, int ldu, long long int strideU, intptr_t d_V, int ldv, long long int strideV, intptr_t d_work, int lwork, intptr_t d_info, intptr_t h_R_nrmF, int batchSize):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnSgesvdaStridedBatched(<DnHandle>handle, <EigMode>jobz, rank, m, n, <const float*>d_A, lda, strideA, <float*>d_S, strideS, <float*>d_U, ldu, strideU, <float*>d_V, ldv, strideV, <float*>d_work, lwork, <int*>d_info, <double*>h_R_nrmF, batchSize)
    check_status(status)

cpdef dgesvdaStridedBatched(intptr_t handle, int jobz, int rank, int m, int n, intptr_t d_A, int lda, long long int strideA, intptr_t d_S, long long int strideS, intptr_t d_U, int ldu, long long int strideU, intptr_t d_V, int ldv, long long int strideV, intptr_t d_work, int lwork, intptr_t d_info, intptr_t h_R_nrmF, int batchSize):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnDgesvdaStridedBatched(<DnHandle>handle, <EigMode>jobz, rank, m, n, <const double*>d_A, lda, strideA, <double*>d_S, strideS, <double*>d_U, ldu, strideU, <double*>d_V, ldv, strideV, <double*>d_work, lwork, <int*>d_info, <double*>h_R_nrmF, batchSize)
    check_status(status)

cpdef cgesvdaStridedBatched(intptr_t handle, int jobz, int rank, int m, int n, intptr_t d_A, int lda, long long int strideA, intptr_t d_S, long long int strideS, intptr_t d_U, int ldu, long long int strideU, intptr_t d_V, int ldv, long long int strideV, intptr_t d_work, int lwork, intptr_t d_info, intptr_t h_R_nrmF, int batchSize):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnCgesvdaStridedBatched(<DnHandle>handle, <EigMode>jobz, rank, m, n, <const cuComplex*>d_A, lda, strideA, <float*>d_S, strideS, <cuComplex*>d_U, ldu, strideU, <cuComplex*>d_V, ldv, strideV, <cuComplex*>d_work, lwork, <int*>d_info, <double*>h_R_nrmF, batchSize)
    check_status(status)

cpdef zgesvdaStridedBatched(intptr_t handle, int jobz, int rank, int m, int n, intptr_t d_A, int lda, long long int strideA, intptr_t d_S, long long int strideS, intptr_t d_U, int ldu, long long int strideU, intptr_t d_V, int ldv, long long int strideV, intptr_t d_work, int lwork, intptr_t d_info, intptr_t h_R_nrmF, int batchSize):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnZgesvdaStridedBatched(<DnHandle>handle, <EigMode>jobz, rank, m, n, <const cuDoubleComplex*>d_A, lda, strideA, <double*>d_S, strideS, <cuDoubleComplex*>d_U, ldu, strideU, <cuDoubleComplex*>d_V, ldv, strideV, <cuDoubleComplex*>d_work, lwork, <int*>d_info, <double*>h_R_nrmF, batchSize)
    check_status(status)

cpdef int ssyevd_bufferSize(intptr_t handle, int jobz, int uplo, int n, intptr_t A, int lda, intptr_t W) except? 0:
    cdef int lwork
    status = cusolverDnSsyevd_bufferSize(<DnHandle>handle, <EigMode>jobz, <FillMode>uplo, n, <const float*>A, lda, <const float*>W, &lwork)
    check_status(status)
    return lwork

cpdef int dsyevd_bufferSize(intptr_t handle, int jobz, int uplo, int n, intptr_t A, int lda, intptr_t W) except? 0:
    cdef int lwork
    status = cusolverDnDsyevd_bufferSize(<DnHandle>handle, <EigMode>jobz, <FillMode>uplo, n, <const double*>A, lda, <const double*>W, &lwork)
    check_status(status)
    return lwork

cpdef int cheevd_bufferSize(intptr_t handle, int jobz, int uplo, int n, intptr_t A, int lda, intptr_t W) except? 0:
    cdef int lwork
    status = cusolverDnCheevd_bufferSize(<DnHandle>handle, <EigMode>jobz, <FillMode>uplo, n, <const cuComplex*>A, lda, <const float*>W, &lwork)
    check_status(status)
    return lwork

cpdef int zheevd_bufferSize(intptr_t handle, int jobz, int uplo, int n, intptr_t A, int lda, intptr_t W) except? 0:
    cdef int lwork
    status = cusolverDnZheevd_bufferSize(<DnHandle>handle, <EigMode>jobz, <FillMode>uplo, n, <const cuDoubleComplex*>A, lda, <const double*>W, &lwork)
    check_status(status)
    return lwork

cpdef ssyevd(intptr_t handle, int jobz, int uplo, int n, intptr_t A, int lda, intptr_t W, intptr_t work, int lwork, intptr_t info):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnSsyevd(<DnHandle>handle, <EigMode>jobz, <FillMode>uplo, n, <float*>A, lda, <float*>W, <float*>work, lwork, <int*>info)
    check_status(status)

cpdef dsyevd(intptr_t handle, int jobz, int uplo, int n, intptr_t A, int lda, intptr_t W, intptr_t work, int lwork, intptr_t info):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnDsyevd(<DnHandle>handle, <EigMode>jobz, <FillMode>uplo, n, <double*>A, lda, <double*>W, <double*>work, lwork, <int*>info)
    check_status(status)

cpdef cheevd(intptr_t handle, int jobz, int uplo, int n, intptr_t A, int lda, intptr_t W, intptr_t work, int lwork, intptr_t info):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnCheevd(<DnHandle>handle, <EigMode>jobz, <FillMode>uplo, n, <cuComplex*>A, lda, <float*>W, <cuComplex*>work, lwork, <int*>info)
    check_status(status)

cpdef zheevd(intptr_t handle, int jobz, int uplo, int n, intptr_t A, int lda, intptr_t W, intptr_t work, int lwork, intptr_t info):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnZheevd(<DnHandle>handle, <EigMode>jobz, <FillMode>uplo, n, <cuDoubleComplex*>A, lda, <double*>W, <cuDoubleComplex*>work, lwork, <int*>info)
    check_status(status)

cpdef int ssyevj_bufferSize(intptr_t handle, int jobz, int uplo, int n, intptr_t A, int lda, intptr_t W, size_t params) except? 0:
    cdef int lwork
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnSsyevj_bufferSize(<DnHandle>handle, <EigMode>jobz, <FillMode>uplo, n, <const float*>A, lda, <const float*>W, &lwork, <syevjInfo_t>params)
    check_status(status)
    return lwork

cpdef int dsyevj_bufferSize(intptr_t handle, int jobz, int uplo, int n, intptr_t A, int lda, intptr_t W, size_t params) except? 0:
    cdef int lwork
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnDsyevj_bufferSize(<DnHandle>handle, <EigMode>jobz, <FillMode>uplo, n, <const double*>A, lda, <const double*>W, &lwork, <syevjInfo_t>params)
    check_status(status)
    return lwork

cpdef int cheevj_bufferSize(intptr_t handle, int jobz, int uplo, int n, intptr_t A, int lda, intptr_t W, size_t params) except? 0:
    cdef int lwork
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnCheevj_bufferSize(<DnHandle>handle, <EigMode>jobz, <FillMode>uplo, n, <const cuComplex*>A, lda, <const float*>W, &lwork, <syevjInfo_t>params)
    check_status(status)
    return lwork

cpdef int zheevj_bufferSize(intptr_t handle, int jobz, int uplo, int n, intptr_t A, int lda, intptr_t W, size_t params) except? 0:
    cdef int lwork
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnZheevj_bufferSize(<DnHandle>handle, <EigMode>jobz, <FillMode>uplo, n, <const cuDoubleComplex*>A, lda, <const double*>W, &lwork, <syevjInfo_t>params)
    check_status(status)
    return lwork

cpdef ssyevj(intptr_t handle, int jobz, int uplo, int n, intptr_t A, int lda, intptr_t W, intptr_t work, int lwork, intptr_t info, size_t params):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnSsyevj(<DnHandle>handle, <EigMode>jobz, <FillMode>uplo, n, <float*>A, lda, <float*>W, <float*>work, lwork, <int*>info, <syevjInfo_t>params)
    check_status(status)

cpdef dsyevj(intptr_t handle, int jobz, int uplo, int n, intptr_t A, int lda, intptr_t W, intptr_t work, int lwork, intptr_t info, size_t params):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnDsyevj(<DnHandle>handle, <EigMode>jobz, <FillMode>uplo, n, <double*>A, lda, <double*>W, <double*>work, lwork, <int*>info, <syevjInfo_t>params)
    check_status(status)

cpdef cheevj(intptr_t handle, int jobz, int uplo, int n, intptr_t A, int lda, intptr_t W, intptr_t work, int lwork, intptr_t info, size_t params):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnCheevj(<DnHandle>handle, <EigMode>jobz, <FillMode>uplo, n, <cuComplex*>A, lda, <float*>W, <cuComplex*>work, lwork, <int*>info, <syevjInfo_t>params)
    check_status(status)

cpdef zheevj(intptr_t handle, int jobz, int uplo, int n, intptr_t A, int lda, intptr_t W, intptr_t work, int lwork, intptr_t info, size_t params):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnZheevj(<DnHandle>handle, <EigMode>jobz, <FillMode>uplo, n, <cuDoubleComplex*>A, lda, <double*>W, <cuDoubleComplex*>work, lwork, <int*>info, <syevjInfo_t>params)
    check_status(status)

cpdef int ssyevjBatched_bufferSize(intptr_t handle, int jobz, int uplo, int n, intptr_t A, int lda, intptr_t W, size_t params, int batchSize) except? 0:
    cdef int lwork
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnSsyevjBatched_bufferSize(<DnHandle>handle, <EigMode>jobz, <FillMode>uplo, n, <const float*>A, lda, <const float*>W, &lwork, <syevjInfo_t>params, batchSize)
    check_status(status)
    return lwork

cpdef int dsyevjBatched_bufferSize(intptr_t handle, int jobz, int uplo, int n, intptr_t A, int lda, intptr_t W, size_t params, int batchSize) except? 0:
    cdef int lwork
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnDsyevjBatched_bufferSize(<DnHandle>handle, <EigMode>jobz, <FillMode>uplo, n, <const double*>A, lda, <const double*>W, &lwork, <syevjInfo_t>params, batchSize)
    check_status(status)
    return lwork

cpdef int cheevjBatched_bufferSize(intptr_t handle, int jobz, int uplo, int n, intptr_t A, int lda, intptr_t W, size_t params, int batchSize) except? 0:
    cdef int lwork
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnCheevjBatched_bufferSize(<DnHandle>handle, <EigMode>jobz, <FillMode>uplo, n, <const cuComplex*>A, lda, <const float*>W, &lwork, <syevjInfo_t>params, batchSize)
    check_status(status)
    return lwork

cpdef int zheevjBatched_bufferSize(intptr_t handle, int jobz, int uplo, int n, intptr_t A, int lda, intptr_t W, size_t params, int batchSize) except? 0:
    cdef int lwork
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnZheevjBatched_bufferSize(<DnHandle>handle, <EigMode>jobz, <FillMode>uplo, n, <const cuDoubleComplex*>A, lda, <const double*>W, &lwork, <syevjInfo_t>params, batchSize)
    check_status(status)
    return lwork

cpdef ssyevjBatched(intptr_t handle, int jobz, int uplo, int n, intptr_t A, int lda, intptr_t W, intptr_t work, int lwork, intptr_t info, size_t params, int batchSize):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnSsyevjBatched(<DnHandle>handle, <EigMode>jobz, <FillMode>uplo, n, <float*>A, lda, <float*>W, <float*>work, lwork, <int*>info, <syevjInfo_t>params, batchSize)
    check_status(status)

cpdef dsyevjBatched(intptr_t handle, int jobz, int uplo, int n, intptr_t A, int lda, intptr_t W, intptr_t work, int lwork, intptr_t info, size_t params, int batchSize):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnDsyevjBatched(<DnHandle>handle, <EigMode>jobz, <FillMode>uplo, n, <double*>A, lda, <double*>W, <double*>work, lwork, <int*>info, <syevjInfo_t>params, batchSize)
    check_status(status)

cpdef cheevjBatched(intptr_t handle, int jobz, int uplo, int n, intptr_t A, int lda, intptr_t W, intptr_t work, int lwork, intptr_t info, size_t params, int batchSize):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnCheevjBatched(<DnHandle>handle, <EigMode>jobz, <FillMode>uplo, n, <cuComplex*>A, lda, <float*>W, <cuComplex*>work, lwork, <int*>info, <syevjInfo_t>params, batchSize)
    check_status(status)

cpdef zheevjBatched(intptr_t handle, int jobz, int uplo, int n, intptr_t A, int lda, intptr_t W, intptr_t work, int lwork, intptr_t info, size_t params, int batchSize):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverDnZheevjBatched(<DnHandle>handle, <EigMode>jobz, <FillMode>uplo, n, <cuDoubleComplex*>A, lda, <double*>W, <cuDoubleComplex*>work, lwork, <int*>info, <syevjInfo_t>params, batchSize)
    check_status(status)


###################################################
# cuSOLVER Sparse LAPACK Function - Helper Function

cpdef intptr_t spCreate() except? 0:
    cdef SpHandle handle
    status = cusolverSpCreate(&handle)
    check_status(status)
    return <intptr_t>handle

cpdef spDestroy(intptr_t handle):
    status = cusolverSpDestroy(<SpHandle>handle)
    check_status(status)

cpdef spSetStream(intptr_t handle, size_t streamId):
    status = cusolverSpSetStream(<SpHandle>handle, <driver.Stream>streamId)
    check_status(status)

cpdef size_t spGetStream(intptr_t handle) except? 0:
    cdef driver.Stream streamId
    status = cusolverSpGetStream(<SpHandle>handle, &streamId)
    check_status(status)
    return <size_t>streamId


#######################################################
# cuSOLVER Sparse LAPACK Function - High Level Function

cpdef scsrlsvchol(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrVal, intptr_t csrRowPtr, intptr_t csrColInd, intptr_t b, float tol, int reorder, intptr_t x, intptr_t singularity):
    if stream_module.enable_current_stream:
        spSetStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverSpScsrlsvchol(<SpHandle>handle, m, nnz, <const MatDescr>descrA, <const float*>csrVal, <const int*>csrRowPtr, <const int*>csrColInd, <const float*>b, tol, reorder, <float*>x, <int*>singularity)
    check_status(status)

cpdef dcsrlsvchol(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrVal, intptr_t csrRowPtr, intptr_t csrColInd, intptr_t b, double tol, int reorder, intptr_t x, intptr_t singularity):
    if stream_module.enable_current_stream:
        spSetStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverSpDcsrlsvchol(<SpHandle>handle, m, nnz, <const MatDescr>descrA, <const double*>csrVal, <const int*>csrRowPtr, <const int*>csrColInd, <const double*>b, tol, reorder, <double*>x, <int*>singularity)
    check_status(status)

cpdef ccsrlsvchol(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrVal, intptr_t csrRowPtr, intptr_t csrColInd, intptr_t b, float tol, int reorder, intptr_t x, intptr_t singularity):
    if stream_module.enable_current_stream:
        spSetStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverSpCcsrlsvchol(<SpHandle>handle, m, nnz, <const MatDescr>descrA, <const cuComplex*>csrVal, <const int*>csrRowPtr, <const int*>csrColInd, <const cuComplex*>b, tol, reorder, <cuComplex*>x, <int*>singularity)
    check_status(status)

cpdef zcsrlsvchol(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrVal, intptr_t csrRowPtr, intptr_t csrColInd, intptr_t b, double tol, int reorder, intptr_t x, intptr_t singularity):
    if stream_module.enable_current_stream:
        spSetStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverSpZcsrlsvchol(<SpHandle>handle, m, nnz, <const MatDescr>descrA, <const cuDoubleComplex*>csrVal, <const int*>csrRowPtr, <const int*>csrColInd, <const cuDoubleComplex*>b, tol, reorder, <cuDoubleComplex*>x, <int*>singularity)
    check_status(status)

cpdef scsrlsvqr(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrVal, intptr_t csrRowPtr, intptr_t csrColInd, intptr_t b, float tol, int reorder, intptr_t x, intptr_t singularity):
    if stream_module.enable_current_stream:
        spSetStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverSpScsrlsvqr(<SpHandle>handle, m, nnz, <const MatDescr>descrA, <const float*>csrVal, <const int*>csrRowPtr, <const int*>csrColInd, <const float*>b, tol, reorder, <float*>x, <int*>singularity)
    check_status(status)

cpdef dcsrlsvqr(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrVal, intptr_t csrRowPtr, intptr_t csrColInd, intptr_t b, double tol, int reorder, intptr_t x, intptr_t singularity):
    if stream_module.enable_current_stream:
        spSetStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverSpDcsrlsvqr(<SpHandle>handle, m, nnz, <const MatDescr>descrA, <const double*>csrVal, <const int*>csrRowPtr, <const int*>csrColInd, <const double*>b, tol, reorder, <double*>x, <int*>singularity)
    check_status(status)

cpdef ccsrlsvqr(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrVal, intptr_t csrRowPtr, intptr_t csrColInd, intptr_t b, float tol, int reorder, intptr_t x, intptr_t singularity):
    if stream_module.enable_current_stream:
        spSetStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverSpCcsrlsvqr(<SpHandle>handle, m, nnz, <const MatDescr>descrA, <const cuComplex*>csrVal, <const int*>csrRowPtr, <const int*>csrColInd, <const cuComplex*>b, tol, reorder, <cuComplex*>x, <int*>singularity)
    check_status(status)

cpdef zcsrlsvqr(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrVal, intptr_t csrRowPtr, intptr_t csrColInd, intptr_t b, double tol, int reorder, intptr_t x, intptr_t singularity):
    if stream_module.enable_current_stream:
        spSetStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverSpZcsrlsvqr(<SpHandle>handle, m, nnz, <const MatDescr>descrA, <const cuDoubleComplex*>csrVal, <const int*>csrRowPtr, <const int*>csrColInd, <const cuDoubleComplex*>b, tol, reorder, <cuDoubleComplex*>x, <int*>singularity)
    check_status(status)

cpdef scsreigvsi(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrValA, intptr_t csrRowPtrA, intptr_t csrColIndA, float mu0, intptr_t x0, int maxite, float eps, intptr_t mu, intptr_t x):
    if stream_module.enable_current_stream:
        spSetStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverSpScsreigvsi(<SpHandle>handle, m, nnz, <const MatDescr>descrA, <const float*>csrValA, <const int*>csrRowPtrA, <const int*>csrColIndA, mu0, <const float*>x0, maxite, eps, <float*>mu, <float*>x)
    check_status(status)

cpdef dcsreigvsi(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrValA, intptr_t csrRowPtrA, intptr_t csrColIndA, double mu0, intptr_t x0, int maxite, double eps, intptr_t mu, intptr_t x):
    if stream_module.enable_current_stream:
        spSetStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverSpDcsreigvsi(<SpHandle>handle, m, nnz, <const MatDescr>descrA, <const double*>csrValA, <const int*>csrRowPtrA, <const int*>csrColIndA, mu0, <const double*>x0, maxite, eps, <double*>mu, <double*>x)
    check_status(status)

cpdef ccsreigvsi(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrValA, intptr_t csrRowPtrA, intptr_t csrColIndA, size_t mu0, intptr_t x0, int maxite, float eps, intptr_t mu, intptr_t x):
    if stream_module.enable_current_stream:
        spSetStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverSpCcsreigvsi(<SpHandle>handle, m, nnz, <const MatDescr>descrA, <const cuComplex*>csrValA, <const int*>csrRowPtrA, <const int*>csrColIndA, (<cuComplex*>mu0)[0], <const cuComplex*>x0, maxite, eps, <cuComplex*>mu, <cuComplex*>x)
    check_status(status)

cpdef zcsreigvsi(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrValA, intptr_t csrRowPtrA, intptr_t csrColIndA, size_t mu0, intptr_t x0, int maxite, double eps, intptr_t mu, intptr_t x):
    if stream_module.enable_current_stream:
        spSetStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverSpZcsreigvsi(<SpHandle>handle, m, nnz, <const MatDescr>descrA, <const cuDoubleComplex*>csrValA, <const int*>csrRowPtrA, <const int*>csrColIndA, (<cuDoubleComplex*>mu0)[0], <const cuDoubleComplex*>x0, maxite, eps, <cuDoubleComplex*>mu, <cuDoubleComplex*>x)
    check_status(status)
