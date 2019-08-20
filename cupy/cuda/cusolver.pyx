"""Thin wrapper of CUSOLVER."""

cimport cython  # NOQA

from cupy.cuda cimport driver
from cupy.cuda cimport stream as stream_module

###############################################################################
# Extern
###############################################################################

cdef extern from 'cupy_cuComplex.h':
    ctypedef struct cuComplex 'cuComplex':
        float x, y

    ctypedef struct cuDoubleComplex 'cuDoubleComplex':
        double x, y

cdef extern from 'cupy_cusolver.h' nogil:
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

    # TODO(anaruse): potrfBatched and potrsBatched

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

    int cusolverSpScsrlsvqr(
        SpHandle handle, int m, int nnz, const MatDescr descrA,
        const float* csrValA, const int* csrRowPtrA, const int* csrColIndA,
        const float* b, float tol, int reorder, float* x, int* singularity)

    int cusolverSpDcsrlsvqr(
        SpHandle handle, int m, int nnz, const MatDescr descrA,
        const double* csrValA, const int* csrRowPtrA, const int* csrColIndA,
        const double* b, double tol, int reorder, double* x, int* singularity)

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
}


class CUSOLVERError(RuntimeError):

    def __init__(self, status):
        self.status = status
        super(CUSOLVERError, self).__init__(STATUS[status])

    def __reduce__(self):
        return (type(self), (self.status,))


@cython.profile(False)
cpdef inline check_status(int status):
    if status != 0:
        raise CUSOLVERError(status)


###############################################################################
# Context
###############################################################################

cpdef size_t create() except? 0:
    cdef Handle handle
    with nogil:
        status = cusolverDnCreate(&handle)
    check_status(status)
    return <size_t>handle


cpdef size_t spCreate() except? 0:
    cdef SpHandle handle
    with nogil:
        status = cusolverSpCreate(&handle)
    check_status(status)
    return <size_t>handle


cpdef destroy(size_t handle):
    with nogil:
        status = cusolverDnDestroy(<Handle>handle)
    check_status(status)


cpdef spDestroy(size_t handle):
    with nogil:
        status = cusolverSpDestroy(<SpHandle>handle)
    check_status(status)


###############################################################################
# Stream
###############################################################################

cpdef setStream(size_t handle, size_t stream):
    with nogil:
        status = cusolverDnSetStream(<Handle>handle, <driver.Stream>stream)
    check_status(status)


cpdef size_t getStream(size_t handle) except? 0:
    cdef driver.Stream stream
    with nogil:
        status = cusolverDnGetStream(<Handle>handle, &stream)
    check_status(status)
    return <size_t>stream


cpdef spSetStream(size_t handle, size_t stream):
    with nogil:
        status = cusolverSpSetStream(<SpHandle>handle, <driver.Stream>stream)
    check_status(status)


cpdef size_t spGetStream(size_t handle) except *:
    cdef driver.Stream stream
    with nogil:
        status = cusolverSpGetStream(<SpHandle>handle, &stream)
    check_status(status)
    return <size_t>stream


###########################################################################
# Dense LAPACK Functions (Linear Solver)
###########################################################################

# Cholesky factorization
cpdef int spotrf_bufferSize(size_t handle, int uplo,
                            int n, size_t A, int lda) except? -1:
    cdef int lwork
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnSpotrf_bufferSize(
            <Handle>handle, <FillMode>uplo, n,
            <float*>A, <int>lda, &lwork)
    check_status(status)
    return lwork

cpdef int dpotrf_bufferSize(size_t handle, int uplo,
                            int n, size_t A, int lda) except? -1:
    cdef int lwork
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnDpotrf_bufferSize(
            <Handle>handle, <FillMode>uplo, n,
            <double*>A, <int>lda, &lwork)
    check_status(status)
    return lwork

cpdef int cpotrf_bufferSize(size_t handle, int uplo,
                            int n, size_t A, int lda) except? -1:
    cdef int lwork
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnCpotrf_bufferSize(
            <Handle>handle, <FillMode>uplo, n,
            <cuComplex*>A, <int>lda, &lwork)
    check_status(status)
    return lwork

cpdef int zpotrf_bufferSize(size_t handle, int uplo,
                            int n, size_t A, int lda) except? -1:
    cdef int lwork
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnZpotrf_bufferSize(
            <Handle>handle, <FillMode>uplo, n,
            <cuDoubleComplex*>A, <int>lda, &lwork)
    check_status(status)
    return lwork

cpdef spotrf(size_t handle, int uplo, int n, size_t A, int lda,
             size_t work, int lwork, size_t devInfo):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnSpotrf(
            <Handle>handle, <FillMode>uplo, n, <float*>A,
            lda, <float*>work, lwork, <int*>devInfo)
    check_status(status)

cpdef dpotrf(size_t handle, int uplo, int n, size_t A, int lda,
             size_t work, int lwork, size_t devInfo):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnDpotrf(
            <Handle>handle, <FillMode>uplo, n, <double*>A,
            lda, <double*>work, lwork, <int*>devInfo)
    check_status(status)

cpdef cpotrf(size_t handle, int uplo, int n, size_t A, int lda,
             size_t work, int lwork, size_t devInfo):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnCpotrf(
            <Handle>handle, <FillMode>uplo, n, <cuComplex*>A,
            lda, <cuComplex*>work, lwork, <int*>devInfo)
    check_status(status)

cpdef zpotrf(size_t handle, int uplo, int n, size_t A, int lda,
             size_t work, int lwork, size_t devInfo):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnZpotrf(
            <Handle>handle, <FillMode>uplo, n, <cuDoubleComplex*>A,
            lda, <cuDoubleComplex*>work, lwork, <int*>devInfo)
    check_status(status)

cpdef spotrs(size_t handle, int uplo, int n, int nrhs,
             size_t A, int lda, size_t B, int ldb, size_t devInfo):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnSpotrs(
            <Handle>handle, <FillMode>uplo, n, nrhs,
            <const float*>A, lda, <float*>B, ldb,
            <int*>devInfo)
    check_status(status)

cpdef dpotrs(size_t handle, int uplo, int n, int nrhs,
             size_t A, int lda, size_t B, int ldb, size_t devInfo):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnDpotrs(
            <Handle>handle, <FillMode>uplo, n, nrhs,
            <const double*>A, lda, <double*>B, ldb,
            <int*>devInfo)
    check_status(status)

cpdef cpotrs(size_t handle, int uplo, int n, int nrhs,
             size_t A, int lda, size_t B, int ldb, size_t devInfo):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnCpotrs(
            <Handle>handle, <FillMode>uplo, n, nrhs,
            <const cuComplex*>A, lda, <cuComplex*>B, ldb,
            <int*>devInfo)
    check_status(status)

cpdef zpotrs(size_t handle, int uplo, int n, int nrhs,
             size_t A, int lda, size_t B, int ldb, size_t devInfo):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnZpotrs(
            <Handle>handle, <FillMode>uplo, n, nrhs,
            <const cuDoubleComplex*>A, lda, <cuDoubleComplex*>B, ldb,
            <int*>devInfo)
    check_status(status)


# LU factorization
cpdef int sgetrf_bufferSize(size_t handle, int m, int n,
                            size_t A, int lda) except? -1:
    cdef int lwork
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnSgetrf_bufferSize(
            <Handle>handle, m, n, <float*>A, lda, &lwork)
    check_status(status)
    return lwork

cpdef int dgetrf_bufferSize(size_t handle, int m, int n,
                            size_t A, int lda) except? -1:
    cdef int lwork
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnDgetrf_bufferSize(
            <Handle>handle, m, n, <double*>A, lda, &lwork)
    check_status(status)
    return lwork

cpdef int cgetrf_bufferSize(size_t handle, int m, int n,
                            size_t A, int lda) except? -1:
    cdef int lwork
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnCgetrf_bufferSize(
            <Handle>handle, m, n, <cuComplex*>A, lda, &lwork)
    check_status(status)
    return lwork

cpdef int zgetrf_bufferSize(size_t handle, int m, int n,
                            size_t A, int lda) except? -1:
    cdef int lwork
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnZgetrf_bufferSize(
            <Handle>handle, m, n, <cuDoubleComplex*>A, lda, &lwork)
    check_status(status)
    return lwork

cpdef sgetrf(size_t handle, int m, int n, size_t A, int lda,
             size_t work, size_t devIpiv, size_t devInfo):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnSgetrf(
            <Handle>handle, m, n, <float*>A, lda,
            <float*>work, <int*>devIpiv, <int*>devInfo)
    check_status(status)

cpdef dgetrf(size_t handle, int m, int n, size_t A, int lda,
             size_t work, size_t devIpiv, size_t devInfo):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnDgetrf(
            <Handle>handle, m, n, <double*>A, lda,
            <double*>work, <int*>devIpiv, <int*>devInfo)
    check_status(status)

cpdef cgetrf(size_t handle, int m, int n, size_t A, int lda,
             size_t work, size_t devIpiv, size_t devInfo):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnCgetrf(
            <Handle>handle, m, n, <cuComplex*>A, lda,
            <cuComplex*>work, <int*>devIpiv, <int*>devInfo)
    check_status(status)

cpdef zgetrf(size_t handle, int m, int n, size_t A, int lda,
             size_t work, size_t devIpiv, size_t devInfo):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnZgetrf(
            <Handle>handle, m, n, <cuDoubleComplex*>A, lda,
            <cuDoubleComplex*>work, <int*>devIpiv, <int*>devInfo)
    check_status(status)


# LU solve
cpdef sgetrs(size_t handle, int trans, int n, int nrhs,
             size_t A, int lda, size_t devIpiv,
             size_t B, int ldb, size_t devInfo):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnSgetrs(
            <Handle>handle, <Operation>trans, n, nrhs,
            <const float*> A, lda, <const int*>devIpiv,
            <float*>B, ldb, <int*> devInfo)
    check_status(status)

cpdef dgetrs(size_t handle, int trans, int n, int nrhs,
             size_t A, int lda, size_t devIpiv,
             size_t B, int ldb, size_t devInfo):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnDgetrs(
            <Handle>handle, <Operation>trans, n, nrhs,
            <const double*> A, lda, <const int*>devIpiv,
            <double*>B, ldb, <int*> devInfo)
    check_status(status)

cpdef cgetrs(size_t handle, int trans, int n, int nrhs,
             size_t A, int lda, size_t devIpiv,
             size_t B, int ldb, size_t devInfo):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnCgetrs(
            <Handle>handle, <Operation>trans, n, nrhs,
            <const cuComplex*> A, lda, <const int*>devIpiv,
            <cuComplex*>B, ldb, <int*> devInfo)
    check_status(status)

cpdef zgetrs(size_t handle, int trans, int n, int nrhs,
             size_t A, int lda, size_t devIpiv,
             size_t B, int ldb, size_t devInfo):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnZgetrs(
            <Handle>handle, <Operation>trans, n, nrhs,
            <const cuDoubleComplex*> A, lda, <const int*>devIpiv,
            <cuDoubleComplex*>B, ldb, <int*> devInfo)
    check_status(status)


# QR factorization
cpdef int sgeqrf_bufferSize(size_t handle, int m, int n,
                            size_t A, int lda) except? -1:
    cdef int lwork
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnSgeqrf_bufferSize(
            <Handle>handle, m, n, <float*>A, lda, &lwork)
    check_status(status)
    return lwork

cpdef int dgeqrf_bufferSize(size_t handle, int m, int n,
                            size_t A, int lda) except? -1:
    cdef int lwork
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnDgeqrf_bufferSize(
            <Handle>handle, m, n, <double*>A, lda, &lwork)
    check_status(status)
    return lwork

cpdef int cgeqrf_bufferSize(size_t handle, int m, int n,
                            size_t A, int lda) except? -1:
    cdef int lwork
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnCgeqrf_bufferSize(
            <Handle>handle, m, n, <cuComplex*>A, lda, &lwork)
    check_status(status)
    return lwork

cpdef int zgeqrf_bufferSize(size_t handle, int m, int n,
                            size_t A, int lda) except? -1:
    cdef int lwork
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnZgeqrf_bufferSize(
            <Handle>handle, m, n, <cuDoubleComplex*>A, lda, &lwork)
    check_status(status)
    return lwork

cpdef sgeqrf(size_t handle, int m, int n, size_t A, int lda,
             size_t tau, size_t work, int lwork, size_t devInfo):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnSgeqrf(
            <Handle>handle, m, n, <float*>A, lda,
            <float*>tau, <float*>work, lwork,
            <int*>devInfo)
    check_status(status)

cpdef dgeqrf(size_t handle, int m, int n, size_t A, int lda,
             size_t tau, size_t work, int lwork, size_t devInfo):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnDgeqrf(
            <Handle>handle, m, n, <double*>A, lda,
            <double*>tau, <double*>work, lwork,
            <int*>devInfo)
    check_status(status)

cpdef cgeqrf(size_t handle, int m, int n, size_t A, int lda,
             size_t tau, size_t work, int lwork, size_t devInfo):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnCgeqrf(
            <Handle>handle, m, n, <cuComplex*>A, lda,
            <cuComplex*>tau, <cuComplex*>work, lwork,
            <int*>devInfo)
    check_status(status)

cpdef zgeqrf(size_t handle, int m, int n, size_t A, int lda,
             size_t tau, size_t work, int lwork, size_t devInfo):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnZgeqrf(
            <Handle>handle, m, n, <cuDoubleComplex*>A, lda,
            <cuDoubleComplex*>tau, <cuDoubleComplex*>work, lwork,
            <int*>devInfo)
    check_status(status)


# Generate unitary matrix Q from QR factorization
cpdef int sorgqr_bufferSize(size_t handle, int m, int n, int k,
                            size_t A, int lda, size_t tau) except? -1:
    cdef int lwork
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnSorgqr_bufferSize(
            <Handle>handle, m, n, k, <const float*>A, lda,
            <const float*>tau, &lwork)
    check_status(status)
    return lwork

cpdef int dorgqr_bufferSize(size_t handle, int m, int n, int k,
                            size_t A, int lda, size_t tau) except? -1:
    cdef int lwork
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnDorgqr_bufferSize(
            <Handle>handle, m, n, k, <const double*>A, lda,
            <const double*>tau, &lwork)
    check_status(status)
    return lwork

cpdef int cungqr_bufferSize(size_t handle, int m, int n, int k,
                            size_t A, int lda, size_t tau) except? -1:
    cdef int lwork
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnCungqr_bufferSize(
            <Handle>handle, m, n, k, <const cuComplex*>A, lda,
            <const cuComplex*>tau, &lwork)
    check_status(status)
    return lwork

cpdef int zungqr_bufferSize(size_t handle, int m, int n, int k,
                            size_t A, int lda, size_t tau) except? -1:
    cdef int lwork
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnZungqr_bufferSize(
            <Handle>handle, m, n, k, <const cuDoubleComplex*>A, lda,
            <const cuDoubleComplex*>tau, &lwork)
    check_status(status)
    return lwork

cpdef sorgqr(size_t handle, int m, int n, int k, size_t A, int lda,
             size_t tau, size_t work, int lwork, size_t devInfo):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnSorgqr(
            <Handle>handle, m, n, k, <float*>A, lda,
            <const float*>tau, <float*>work, lwork,
            <int*>devInfo)
    check_status(status)

cpdef dorgqr(size_t handle, int m, int n, int k, size_t A, int lda,
             size_t tau, size_t work, int lwork, size_t devInfo):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnDorgqr(
            <Handle>handle, m, n, k, <double*>A, lda,
            <const double*>tau, <double*>work, lwork,
            <int*>devInfo)
    check_status(status)

cpdef cungqr(size_t handle, int m, int n, int k, size_t A, int lda,
             size_t tau, size_t work, int lwork, size_t devInfo):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnCungqr(
            <Handle>handle, m, n, k, <cuComplex*>A, lda,
            <const cuComplex*>tau, <cuComplex*>work, lwork,
            <int*>devInfo)
    check_status(status)

cpdef zungqr(size_t handle, int m, int n, int k, size_t A, int lda,
             size_t tau, size_t work, int lwork, size_t devInfo):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnZungqr(
            <Handle>handle, m, n, k, <cuDoubleComplex*>A, lda,
            <const cuDoubleComplex*>tau, <cuDoubleComplex*>work, lwork,
            <int*>devInfo)
    check_status(status)


# Compute Q**T*b in solve min||A*x = b||
cpdef int sormqr_bufferSize(size_t handle, int side, int trans,
                            int m, int n, int k, size_t A, int lda, size_t tau,
                            size_t C, int ldc) except? -1:
    cdef int lwork
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnSormqr_bufferSize(
            <Handle>handle, <SideMode>side, <Operation>trans, m, n, k,
            <const float*>A, lda, <const float*>tau,
            <float*>C, ldc, &lwork)
    check_status(status)
    return lwork

cpdef int dormqr_bufferSize(size_t handle, int side, int trans,
                            int m, int n, int k, size_t A, int lda, size_t tau,
                            size_t C, int ldc) except? -1:
    cdef int lwork
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnDormqr_bufferSize(
            <Handle>handle, <SideMode>side, <Operation>trans, m, n, k,
            <const double*>A, lda, <const double*>tau,
            <double*>C, ldc, &lwork)
    check_status(status)
    return lwork

cpdef int cunmqr_bufferSize(size_t handle, int side, int trans,
                            int m, int n, int k, size_t A, int lda, size_t tau,
                            size_t C, int ldc) except? -1:
    cdef int lwork
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnCunmqr_bufferSize(
            <Handle>handle, <SideMode>side, <Operation>trans, m, n, k,
            <const cuComplex*>A, lda, <const cuComplex*>tau,
            <cuComplex*>C, ldc, &lwork)
    check_status(status)
    return lwork

cpdef int zunmqr_bufferSize(size_t handle, int side, int trans,
                            int m, int n, int k, size_t A, int lda, size_t tau,
                            size_t C, int ldc) except? -1:
    cdef int lwork
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnZunmqr_bufferSize(
            <Handle>handle, <SideMode>side, <Operation>trans, m, n, k,
            <const cuDoubleComplex*>A, lda, <const cuDoubleComplex*>tau,
            <cuDoubleComplex*>C, ldc, &lwork)
    check_status(status)
    return lwork


cpdef sormqr(size_t handle, int side, int trans,
             int m, int n, int k, size_t A, int lda, size_t tau,
             size_t C, int ldc, size_t work, int lwork, size_t devInfo):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnSormqr(
            <Handle>handle, <SideMode>side, <Operation>trans, m, n, k,
            <const float*>A, lda, <const float*>tau,
            <float*>C, ldc,
            <float*>work, lwork, <int*>devInfo)
    check_status(status)

cpdef dormqr(size_t handle, int side, int trans,
             int m, int n, int k, size_t A, int lda, size_t tau,
             size_t C, int ldc, size_t work, int lwork, size_t devInfo):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnDormqr(
            <Handle>handle, <SideMode>side, <Operation>trans, m, n, k,
            <const double*>A, lda, <const double*>tau,
            <double*>C, ldc,
            <double*>work, lwork, <int*>devInfo)
    check_status(status)

cpdef cunmqr(size_t handle, int side, int trans,
             int m, int n, int k, size_t A, int lda, size_t tau,
             size_t C, int ldc, size_t work, int lwork, size_t devInfo):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnCunmqr(
            <Handle>handle, <SideMode>side, <Operation>trans, m, n, k,
            <const cuComplex*>A, lda, <const cuComplex*>tau,
            <cuComplex*>C, ldc,
            <cuComplex*>work, lwork, <int*>devInfo)
    check_status(status)

cpdef zunmqr(size_t handle, int side, int trans,
             int m, int n, int k, size_t A, int lda, size_t tau,
             size_t C, int ldc, size_t work, int lwork, size_t devInfo):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnZunmqr(
            <Handle>handle, <SideMode>side, <Operation>trans, m, n, k,
            <const cuDoubleComplex*>A, lda, <const cuDoubleComplex*>tau,
            <cuDoubleComplex*>C, ldc,
            <cuDoubleComplex*>work, lwork, <int*>devInfo)
    check_status(status)

# (obsoleted)
cpdef cormqr(size_t handle, int side, int trans,
             int m, int n, int k, size_t A, int lda, size_t tau,
             size_t C, int ldc, size_t work, int lwork, size_t devInfo):
    return cunmqr(handle, side, trans, m, n, k, A, lda, tau,
                  C, ldc, work, lwork, devInfo)

# (obsoleted)
cpdef zormqr(size_t handle, int side, int trans,
             int m, int n, int k, size_t A, int lda, size_t tau,
             size_t C, int ldc, size_t work, int lwork, size_t devInfo):
    return zunmqr(handle, side, trans, m, n, k, A, lda, tau,
                  C, ldc, work, lwork, devInfo)


# L*D*L**T,U*D*U**T factorization
cpdef int ssytrf_bufferSize(size_t handle, int n, size_t A,
                            int lda) except? -1:
    cdef int lwork
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnSsytrf_bufferSize(
            <Handle>handle, n, <float*>A, lda, &lwork)
    check_status(status)
    return lwork

cpdef int dsytrf_bufferSize(size_t handle, int n, size_t A,
                            int lda) except? -1:
    cdef int lwork
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnDsytrf_bufferSize(
            <Handle>handle, n, <double*>A, lda, &lwork)
    check_status(status)
    return lwork

cpdef int csytrf_bufferSize(size_t handle, int n, size_t A,
                            int lda) except? -1:
    cdef int lwork
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnCsytrf_bufferSize(
            <Handle>handle, n, <cuComplex*>A, lda, &lwork)
    check_status(status)
    return lwork

cpdef int zsytrf_bufferSize(size_t handle, int n, size_t A,
                            int lda) except? -1:
    cdef int lwork
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnZsytrf_bufferSize(
            <Handle>handle, n, <cuDoubleComplex*>A, lda, &lwork)
    check_status(status)
    return lwork

cpdef ssytrf(size_t handle, int uplo, int n, size_t A, int lda,
             size_t ipiv, size_t work, int lwork, size_t devInfo):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnSsytrf(
            <Handle>handle, <FillMode>uplo, n, <float*>A, lda,
            <int*>ipiv, <float*>work, lwork, <int*>devInfo)
    check_status(status)

cpdef dsytrf(size_t handle, int uplo, int n, size_t A, int lda,
             size_t ipiv, size_t work, int lwork, size_t devInfo):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnDsytrf(
            <Handle>handle, <FillMode>uplo, n, <double*>A, lda,
            <int*>ipiv, <double*>work, lwork, <int*>devInfo)
    check_status(status)

cpdef csytrf(size_t handle, int uplo, int n, size_t A, int lda,
             size_t ipiv, size_t work, int lwork, size_t devInfo):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnCsytrf(
            <Handle>handle, <FillMode>uplo, n, <cuComplex*>A, lda,
            <int*>ipiv, <cuComplex*>work, lwork, <int*>devInfo)
    check_status(status)

cpdef zsytrf(size_t handle, int uplo, int n, size_t A, int lda,
             size_t ipiv, size_t work, int lwork, size_t devInfo):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnZsytrf(
            <Handle>handle, <FillMode>uplo, n, <cuDoubleComplex*>A, lda,
            <int*>ipiv, <cuDoubleComplex*>work, lwork, <int*>devInfo)
    check_status(status)


###############################################################################
# Dense LAPACK Functions (Eigenvalue Solver)
###############################################################################

# Bidiagonal factorization
cpdef int sgebrd_bufferSize(size_t handle, int m, int n) except? -1:
    cdef int lwork
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnSgebrd_bufferSize(<Handle>handle, m, n, &lwork)
    check_status(status)
    return lwork

cpdef int dgebrd_bufferSize(size_t handle, int m, int n) except? -1:
    cdef int lwork
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnDgebrd_bufferSize(<Handle>handle, m, n, &lwork)
    check_status(status)
    return lwork

cpdef int cgebrd_bufferSize(size_t handle, int m, int n) except? -1:
    cdef int lwork
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnCgebrd_bufferSize(<Handle>handle, m, n, &lwork)
    check_status(status)
    return lwork

cpdef int zgebrd_bufferSize(size_t handle, int m, int n) except? -1:
    cdef int lwork
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnZgebrd_bufferSize(<Handle>handle, m, n, &lwork)
    check_status(status)
    return lwork

cpdef sgebrd(size_t handle, int m, int n, size_t A, int lda,
             size_t D, size_t E, size_t tauQ, size_t tauP,
             size_t Work, int lwork, size_t devInfo):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnSgebrd(
            <Handle>handle, m, n,
            <float*>A, lda,
            <float*>D, <float*>E,
            <float*>tauQ, <float*>tauP,
            <float*>Work, lwork, <int*>devInfo)
    check_status(status)

cpdef dgebrd(size_t handle, int m, int n, size_t A, int lda,
             size_t D, size_t E, size_t tauQ, size_t tauP,
             size_t Work, int lwork, size_t devInfo):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnDgebrd(
            <Handle>handle, m, n,
            <double*>A, lda,
            <double*>D, <double*>E,
            <double*>tauQ, <double*>tauP,
            <double*>Work, lwork, <int*>devInfo)
    check_status(status)

cpdef cgebrd(size_t handle, int m, int n, size_t A, int lda,
             size_t D, size_t E, size_t tauQ, size_t tauP,
             size_t Work, int lwork, size_t devInfo):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnCgebrd(
            <Handle>handle, m, n,
            <cuComplex*>A, lda,
            <float*>D, <float*>E,
            <cuComplex*>tauQ, <cuComplex*>tauP,
            <cuComplex*>Work, lwork, <int*>devInfo)
    check_status(status)

cpdef zgebrd(size_t handle, int m, int n, size_t A, int lda,
             size_t D, size_t E, size_t tauQ, size_t tauP,
             size_t Work, int lwork, size_t devInfo):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnZgebrd(
            <Handle>handle, m, n,
            <cuDoubleComplex*>A, lda,
            <double*>D, <double*>E,
            <cuDoubleComplex*>tauQ, <cuDoubleComplex*>tauP,
            <cuDoubleComplex*>Work, lwork, <int*>devInfo)
    check_status(status)


# Singular value decomposition, A = U * Sigma * V^H
cpdef int sgesvd_bufferSize(size_t handle, int m, int n) except? -1:
    cdef int lwork
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnSgesvd_bufferSize(<Handle>handle, m, n, &lwork)
    check_status(status)
    return lwork

cpdef int dgesvd_bufferSize(size_t handle, int m, int n) except? -1:
    cdef int lwork
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnDgesvd_bufferSize(<Handle>handle, m, n, &lwork)
    check_status(status)
    return lwork

cpdef int cgesvd_bufferSize(size_t handle, int m, int n) except? -1:
    cdef int lwork
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnCgesvd_bufferSize(<Handle>handle, m, n, &lwork)
    check_status(status)
    return lwork

cpdef int zgesvd_bufferSize(size_t handle, int m, int n) except? -1:
    cdef int lwork
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnZgesvd_bufferSize(<Handle>handle, m, n, &lwork)
    check_status(status)
    return lwork

cpdef sgesvd(size_t handle, char jobu, char jobvt, int m, int n, size_t A,
             int lda, size_t S, size_t U, int ldu, size_t VT, int ldvt,
             size_t Work, int lwork, size_t rwork, size_t devInfo):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnSgesvd(
            <Handle>handle, jobu, jobvt, m, n, <float*>A, lda,
            <float*>S, <float*>U, ldu, <float*>VT, ldvt,
            <float*>Work, lwork, <float*>rwork, <int*>devInfo)
    check_status(status)

cpdef dgesvd(size_t handle, char jobu, char jobvt, int m, int n, size_t A,
             int lda, size_t S, size_t U, int ldu, size_t VT, int ldvt,
             size_t Work, int lwork, size_t rwork, size_t devInfo):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnDgesvd(
            <Handle>handle, jobu, jobvt, m, n, <double*>A, lda,
            <double*>S, <double*>U, ldu, <double*>VT, ldvt,
            <double*>Work, lwork, <double*>rwork, <int*>devInfo)
    check_status(status)

cpdef cgesvd(size_t handle, char jobu, char jobvt, int m, int n, size_t A,
             int lda, size_t S, size_t U, int ldu, size_t VT, int ldvt,
             size_t Work, int lwork, size_t rwork, size_t devInfo):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnCgesvd(
            <Handle>handle, jobu, jobvt, m, n, <cuComplex*>A, lda,
            <float*>S, <cuComplex*>U, ldu, <cuComplex*>VT, ldvt,
            <cuComplex*>Work, lwork, <float*>rwork, <int*>devInfo)
    check_status(status)

cpdef zgesvd(size_t handle, char jobu, char jobvt, int m, int n, size_t A,
             int lda, size_t S, size_t U, int ldu, size_t VT, int ldvt,
             size_t Work, int lwork, size_t rwork, size_t devInfo):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnZgesvd(
            <Handle>handle, jobu, jobvt, m, n, <cuDoubleComplex*>A, lda,
            <double*>S, <cuDoubleComplex*>U, ldu, <cuDoubleComplex*>VT, ldvt,
            <cuDoubleComplex*>Work, lwork, <double*>rwork, <int*>devInfo)
    check_status(status)


# Standard symmetric eigenvalue solver
cpdef int ssyevd_bufferSize(size_t handle, int jobz, int uplo, int n,
                            size_t A, int lda, size_t W) except? -1:
    cdef int lwork, status
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnSsyevd_bufferSize(
            <Handle>handle, <EigMode>jobz, <FillMode>uplo, n,
            <const float*>A,
            lda, <const float*>W, &lwork)
    check_status(status)
    return lwork

cpdef int dsyevd_bufferSize(size_t handle, int jobz, int uplo, int n,
                            size_t A, int lda, size_t W) except? -1:
    cdef int lwork, status
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnDsyevd_bufferSize(
            <Handle>handle, <EigMode>jobz, <FillMode>uplo, n,
            <const double*>A,
            lda, <const double*>W, &lwork)
    check_status(status)
    return lwork

cpdef int cheevd_bufferSize(size_t handle, int jobz, int uplo, int n,
                            size_t A, int lda, size_t W) except? -1:
    cdef int lwork, status
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnCheevd_bufferSize(
            <Handle>handle, <EigMode>jobz, <FillMode>uplo, n,
            <const cuComplex*>A,
            lda, <const float*>W, &lwork)
    check_status(status)
    return lwork

cpdef int zheevd_bufferSize(size_t handle, int jobz, int uplo, int n,
                            size_t A, int lda, size_t W) except? -1:
    cdef int lwork, status
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnZheevd_bufferSize(
            <Handle>handle, <EigMode>jobz, <FillMode>uplo, n,
            <const cuDoubleComplex*>A,
            lda, <const double*>W, &lwork)
    check_status(status)
    return lwork

cpdef ssyevd(size_t handle, int jobz, int uplo, int n, size_t A, int lda,
             size_t W, size_t work, int lwork, size_t info):
    cdef int status
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnSsyevd(
            <Handle>handle, <EigMode>jobz, <FillMode>uplo, n,
            <float*>A, lda, <float*>W,
            <float*>work, lwork, <int*>info)
    check_status(status)

cpdef dsyevd(size_t handle, int jobz, int uplo, int n, size_t A, int lda,
             size_t W, size_t work, int lwork, size_t info):
    cdef int status
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnDsyevd(
            <Handle>handle, <EigMode>jobz, <FillMode>uplo, n,
            <double*>A, lda, <double*>W,
            <double*>work, lwork, <int*>info)
    check_status(status)

cpdef cheevd(size_t handle, int jobz, int uplo, int n, size_t A, int lda,
             size_t W, size_t work, int lwork, size_t info):
    cdef int status
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnCheevd(
            <Handle>handle, <EigMode>jobz, <FillMode>uplo, n,
            <cuComplex*>A, lda, <float*>W,
            <cuComplex*>work, lwork, <int*>info)
    check_status(status)

cpdef zheevd(size_t handle, int jobz, int uplo, int n, size_t A, int lda,
             size_t W, size_t work, int lwork, size_t info):
    cdef int status
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnZheevd(
            <Handle>handle, <EigMode>jobz, <FillMode>uplo, n,
            <cuDoubleComplex*>A, lda, <double*>W,
            <cuDoubleComplex*>work, lwork, <int*>info)
    check_status(status)


###############################################################################
# Sparse LAPACK Functions
###############################################################################
cpdef scsrlsvchol(size_t handle, int m, int nnz, size_t descrA, size_t csrValA,
                  size_t csrRowPtrA, size_t csrColIndA, size_t b, float tol,
                  int reorder, size_t x, size_t singularity):
    cdef int status
    with nogil:
        status = cusolverSpScsrlsvchol(
            <SpHandle>handle, m, nnz, <const MatDescr> descrA,
            <const float*> csrValA, <const int*> csrRowPtrA,
            <const int*> csrColIndA, <const float*> b,
            tol, reorder, <float*> x, <int*> singularity)
    check_status(status)

cpdef dcsrlsvchol(size_t handle, int m, int nnz, size_t descrA, size_t csrValA,
                  size_t csrRowPtrA, size_t csrColIndA, size_t b, double tol,
                  int reorder, size_t x, size_t singularity):
    cdef int status
    with nogil:
        status = cusolverSpDcsrlsvchol(
            <SpHandle>handle, m, nnz, <const MatDescr> descrA,
            <const double*> csrValA, <const int*> csrRowPtrA,
            <const int*> csrColIndA, <const double*> b,
            tol, reorder, <double*> x, <int*> singularity)
    check_status(status)

cpdef scsrlsvqr(size_t handle, int m, int nnz, size_t descrA, size_t csrValA,
                size_t csrRowPtrA, size_t csrColIndA, size_t b, float tol,
                int reorder, size_t x, size_t singularity):
    cdef int status
    spSetStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverSpScsrlsvqr(
            <SpHandle>handle, m, nnz, <const MatDescr> descrA,
            <const float*> csrValA, <const int*> csrRowPtrA,
            <const int*> csrColIndA, <const float*> b,
            tol, reorder, <float*> x, <int*> singularity)
    check_status(status)

cpdef dcsrlsvqr(size_t handle, int m, int nnz, size_t descrA, size_t csrValA,
                size_t csrRowPtrA, size_t csrColIndA, size_t b, double tol,
                int reorder, size_t x, size_t singularity):
    cdef int status
    spSetStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverSpDcsrlsvqr(
            <SpHandle>handle, m, nnz, <const MatDescr> descrA,
            <const double*> csrValA, <const int*> csrRowPtrA,
            <const int*> csrColIndA, <const double*> b,
            tol, reorder, <double*> x, <int*> singularity)
    check_status(status)
