"""Thin wrapper of CUSOLVER."""
cimport cython

from cupy.cuda cimport driver
from cupy.cuda cimport runtime


###############################################################################
# Extern
###############################################################################

cdef extern from 'cupy_cusolver.h':
    # Context
    int cusolverDnCreate(Handle* handle) nogil
    int cusolverDnDestroy(Handle handle) nogil

    # Stream
    int cusolverDnGetStream(Handle handle, driver.Stream* streamId) nogil
    int cusolverDnSetStream(Handle handle, driver.Stream streamId) nogil

    # Linear Equations
    int cusolverDnSpotrf_bufferSize(Handle handle, FillMode uplo, int n,
                                    float* A, int lda, int* lwork) nogil
    int cusolverDnDpotrf_bufferSize(Handle handle, FillMode uplo, int n,
                                    double* A, int lda, int* lwork) nogil
    int cusolverDnSpotrf(Handle handle, FillMode uplo, int n, float* A,
                         int lda, float* work, int lwork, int* devInfo) nogil
    int cusolverDnDpotrf(Handle handle, FillMode uplo, int n, double *A,
                         int lda, double* work, int lwork, int* devInfo) nogil

    int cusolverDnSpotrs(Handle handle, FillMode uplo, int n, int nrhs,
                         const float* A, int lda, float* B, int ldb,
                         int* devInfo) nogil
    int cusolverDnDpotrs(Handle handle, FillMode uplo, int n, int nrhs,
                         const double* A, int lda, double* B, int ldb,
                         int *devInfo) nogil

    int cusolverDnSgetrf(Handle handle, int m, int n, float* A, int lda,
                         float* work, int* devIpiv, int* devInfo) nogil
    int cusolverDnDgetrf(Handle handle, int m, int n, double* A, int lda,
                         double* work, int* devIpiv, int* devInfo) nogil

    int cusolverDnSgetrs(Handle handle, Operation trans,
                         int n, int nrhs, const float* A, int lda,
                         const int* devIpiv, float* B, int ldb,
                         int* devInfo) nogil
    int cusolverDnDgetrs(Handle handle, Operation trans,
                         int n, int nrhs, const double* A, int lda,
                         const int* devIpiv, double* B, int ldb,
                         int* devInfo) nogil

    int cusolverDnSgeqrf_bufferSize(Handle handle, int m, int n,
                                    float* A, int lda, int* lwork) nogil
    int cusolverDnDgeqrf_bufferSize(Handle handle, int m, int n,
                                    double* A, int lda, int* lwork) nogil
    int cusolverDnSgeqrf(Handle handle, int m, int n, float* A, int lda,
                         float* tau, float* work, int lwork,
                         int* devInfo) nogil
    int cusolverDnDgeqrf(Handle handle, int m, int n, double* A, int lda,
                         double* tau, double* work, int lwork,
                         int* devInfo) nogil

    # The actual definition of cusolverDn(S|D)orgqr_bufferSize
    # is different from the reference
    int cusolverDnSorgqr_bufferSize(Handle handle, int m, int n, int k,
                                    const float* A, int lda,
                                    const float* tau, int* lwork) nogil
    int cusolverDnDorgqr_bufferSize(Handle handle, int m, int n, int k,
                                    const double* A, int lda,
                                    const double* tau, int* lwork) nogil
    int cusolverDnSorgqr(Handle handle, int m, int n, int k,
                         float* A, int lda, const float* tau,
                         float* work, int lwork, int* devInfo) nogil
    int cusolverDnDorgqr(Handle handle, int m, int n, int k,
                         double* A, int lda, const double* tau,
                         double* work, int lwork, int* devInfo) nogil

    int cusolverDnSormqr(Handle handle, SideMode side, Operation trans,
                         int m, int n, int k, const float* A, int lda,
                         const float* tau, float* C, int ldc, float* work,
                         int lwork, int* devInfo) nogil
    int cusolverDnDormqr(Handle handle, SideMode side, Operation trans,
                         int m, int n, int k, const double* A, int lda,
                         const double* tau, double* C, int ldc, double* work,
                         int lwork, int* devInfo) nogil

    int cusolverDnSsytrf(Handle handle, FillMode uplo, int n, float* A,
                         int lda, int* ipiv, float* work, int lwork,
                         int* devInfo) nogil
    int cusolverDnDsytrf(Handle handle, FillMode uplo, int n, double* A,
                         int lda, int* ipiv, double* work, int lwork,
                         int* devInfo) nogil

    int cusolverDnSgebrd(Handle handle, int m, int n, float* A, int lda,
                         float* D, float* E, float* tauQ, float* tauP,
                         float* Work, int lwork, int* devInfo) nogil
    int cusolverDnDgebrd(Handle handle, int m, int n, double* A, int lda,
                         double* D, double* E, double* tauQ, double* tauP,
                         double* Work, int lwork, int* devInfo) nogil

    int cusolverDnSgesvd_bufferSize(Handle handle, int m, int n,
                                    int* lwork) nogil
    int cusolverDnDgesvd_bufferSize(Handle handle, int m, int n,
                                    int* lwork) nogil
    int cusolverDnSgesvd(Handle handle, char jobu, char jobvt, int m, int n,
                         float* A, int lda, float* S, float* U, int ldu,
                         float* VT, int ldvt, float* Work, int lwork,
                         float* rwork, int* devInfo) nogil
    int cusolverDnDgesvd(Handle handle, char jobu, char jobvt, int m, int n,
                         double* A, int lda, double* S, double* U, int ldu,
                         double* VT, int ldvt, double* Work, int lwork,
                         double* rwork, int* devInfo) nogil

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


@cython.profile(False)
cpdef inline check_status(int status):
    if status != 0:
        raise CUSOLVERError(status)


###############################################################################
# Context
###############################################################################

cpdef size_t create() except *:
    cdef Handle handle
    with nogil:
        status = cusolverDnCreate(&handle)
    check_status(status)
    return <size_t>handle


cpdef void destroy(size_t handle) except *:
    with nogil:
        status = cusolverDnDestroy(<Handle>handle)
    check_status(status)

###############################################################################
# Stream
###############################################################################

cpdef setStream(size_t handle, size_t stream):
    with nogil:
        status = cusolverDnSetStream(<Handle>handle, <Stream>stream)
    check_status(status)


cpdef size_t getStream(size_t handle) except *:
    cdef driver.Stream stream
    with nogil:
        status = cusolverDnGetStream(<Handle>handle, &stream)
    check_status(status)
    return <size_t>stream

###############################################################################
# dense LAPACK Functions
###############################################################################

cpdef int spotrf_bufferSize(size_t handle, int uplo,
                            int n, size_t A, int lda) except *:
    cdef int lwork
    with nogil:
        status = cusolverDnSpotrf_bufferSize(
            <Handle>handle, <FillMode>uplo, n, <float*>A, <int>lda, &lwork)
    check_status(status)
    return lwork

cpdef int dpotrf_bufferSize(size_t handle, int uplo,
                            int n, size_t A, int lda) except *:
    cdef int lwork
    with nogil:
        status = cusolverDnDpotrf_bufferSize(
            <Handle>handle, <FillMode>uplo, n, <double*>A, <int>lda, &lwork)
    check_status(status)
    return lwork

cpdef spotrf(size_t handle, int uplo, int n, size_t A, int lda,
             size_t work, int lwork, size_t devInfo):
    with nogil:
        status = cusolverDnSpotrf(
            <Handle>handle, <FillMode>uplo, n, <float*>A,
            lda, <float*>work, lwork, <int*>devInfo)
    check_status(status)

cpdef dpotrf(size_t handle, int uplo, int n, size_t A, int lda,
             size_t work, int lwork, size_t devInfo):
    with nogil:
        status = cusolverDnDpotrf(
            <Handle>handle, <FillMode>uplo, n, <double*>A,
            lda, <double*>work, lwork, <int*>devInfo)
    check_status(status)

cpdef spotrs(size_t handle, int uplo, int n, int nrhs,
             size_t A, int lda, size_t B, int ldb, size_t devInfo):
    with nogil:
        status = cusolverDnSpotrs(
            <Handle>handle, <FillMode>uplo, n, nrhs,
            <const float*>A, lda, <float*>B, ldb, <int*>devInfo)
    check_status(status)

cpdef dpotrs(size_t handle, int uplo, int n, int nrhs,
             size_t A, int lda, size_t B, int ldb, size_t devInfo):
    with nogil:
        status = cusolverDnDpotrs(
            <Handle>handle, <FillMode>uplo, n, nrhs,
            <const double*>A, lda, <double*>B, ldb, <int*>devInfo)
    check_status(status)

cpdef sgetrf(size_t handle, int m, int n, size_t A, int lda,
             size_t work, size_t devIpiv, size_t devInfo):
    with nogil:
        status = cusolverDnSgetrf(
            <Handle>handle, m, n, <float*>A, lda,
            <float*>work, <int*>devIpiv, <int*>devInfo)
    check_status(status)

cpdef dgetrf(size_t handle, int m, int n, size_t A, int lda,
             size_t work, size_t devIpiv, size_t devInfo):
    with nogil:
        status = cusolverDnDgetrf(
            <Handle>handle, m, n, <double*>A, lda,
            <double*>work, <int*>devIpiv, <int*>devInfo)
    check_status(status)

cpdef sgetrs(size_t handle, int trans, int n, int nrhs,
             size_t A, int lda, size_t devIpiv,
             size_t B, int ldb, size_t devInfo):
    with nogil:
        status = cusolverDnSgetrs(
            <Handle>handle, <Operation>trans, n, nrhs,
            <const float*> A, lda, <const int*>devIpiv,
            <float*>B, ldb, <int*> devInfo)
    check_status(status)

cpdef dgetrs(size_t handle, int trans, int n, int nrhs,
             size_t A, int lda, size_t devIpiv,
             size_t B, int ldb, size_t devInfo):
    with nogil:
        status = cusolverDnDgetrs(
            <Handle>handle, <Operation>trans, n, nrhs,
            <const double*> A, lda, <const int*>devIpiv,
            <double*>B, ldb, <int*> devInfo)
    check_status(status)

cpdef int sgeqrf_bufferSize(size_t handle, int m, int n,
                            size_t A, int lda) except *:
    cdef int lwork
    with nogil:
        status = cusolverDnSgeqrf_bufferSize(
            <Handle>handle, m, n, <float*>A, lda, &lwork)
    check_status(status)
    return lwork

cpdef int dgeqrf_bufferSize(size_t handle, int m, int n,
                            size_t A, int lda) except *:
    cdef int lwork
    with nogil:
        status = cusolverDnDgeqrf_bufferSize(
            <Handle>handle, m, n, <double*>A, lda, &lwork)
    check_status(status)
    return lwork

cpdef sgeqrf(size_t handle, int m, int n, size_t A, int lda,
             size_t tau, size_t work, int lwork, size_t devInfo):
    with nogil:
        status = cusolverDnSgeqrf(
            <Handle>handle, m, n, <float*>A, lda,
            <float*>tau, <float*>work, lwork, <int*>devInfo)
    check_status(status)

cpdef dgeqrf(size_t handle, int m, int n, size_t A, int lda,
             size_t tau, size_t work, int lwork, size_t devInfo):
    with nogil:
        status = cusolverDnDgeqrf(
            <Handle>handle, m, n, <double*>A, lda,
            <double*>tau, <double*>work, lwork, <int*>devInfo)
    check_status(status)

cpdef int sorgqr_bufferSize(size_t handle, int m, int n, int k,
                            size_t A, int lda, size_t tau) except *:
    cdef int lwork
    with nogil:
        status = cusolverDnSorgqr_bufferSize(
            <Handle>handle, m, n, k, <const float*>A, lda,
            <const float*>tau, &lwork)
    check_status(status)
    return lwork

cpdef int dorgqr_bufferSize(size_t handle, int m, int n, int k,
                            size_t A, int lda, size_t tau) except *:
    cdef int lwork
    with nogil:
        status = cusolverDnDorgqr_bufferSize(
            <Handle>handle, m, n, k, <double*>A, lda,
            <const double*> tau, &lwork)
    check_status(status)
    return lwork

cpdef sorgqr(size_t handle, int m, int n, int k, size_t A, int lda,
             size_t tau, size_t work, int lwork, size_t devInfo):
    with nogil:
        status = cusolverDnSorgqr(
            <Handle>handle, m, n, k, <float*>A, lda,
            <const float*>tau, <float*>work, lwork, <int*>devInfo)
    check_status(status)

cpdef dorgqr(size_t handle, int m, int n, int k, size_t A, int lda,
             size_t tau, size_t work, int lwork, size_t devInfo):
    with nogil:
        status = cusolverDnDorgqr(
            <Handle>handle, m, n, k, <double*>A, lda,
            <const double*>tau, <double*>work, lwork, <int*>devInfo)
    check_status(status)

cpdef sormqr(size_t handle, int side, int trans,
             int m, int n, int k, size_t A, int lda, size_t tau,
             size_t C, int ldc, size_t work, int lwork, size_t devInfo):
    with nogil:
        status = cusolverDnSormqr(
            <Handle>handle, <SideMode>side, <Operation>trans, m, n, k,
            <const float*>A, lda, <const float*>tau, <float*>C, ldc,
            <float*>work, lwork, <int*>devInfo)
    check_status(status)

cpdef dormqr(size_t handle, int side, int trans,
             int m, int n, int k, size_t A, int lda, size_t tau,
             size_t C, int ldc, size_t work, int lwork, size_t devInfo):
    with nogil:
        status = cusolverDnDormqr(
            <Handle>handle, <SideMode>side, <Operation>trans, m, n, k,
            <const double*>A, lda, <const double*>tau, <double*>C, ldc,
            <double*>work, lwork, <int*>devInfo)
    check_status(status)

cpdef ssytrf(size_t handle, int uplo, int n, size_t A, int lda,
             size_t ipiv, size_t work, int lwork, size_t devInfo):
    with nogil:
        status = cusolverDnSsytrf(
            <Handle>handle, <FillMode>uplo, n, <float*>A, lda,
            <int*>ipiv, <float*>work, lwork, <int*>devInfo)
    check_status(status)

cpdef dsytrf(size_t handle, int uplo, int n, size_t A, int lda,
             size_t ipiv, size_t work, int lwork, size_t devInfo):
    with nogil:
        status = cusolverDnDsytrf(
            <Handle>handle, <FillMode>uplo, n, <double*>A, lda,
            <int*>ipiv, <double*>work, lwork, <int*>devInfo)
    check_status(status)

cpdef sgebrd(size_t handle, int m, int n, size_t A, int lda,
             size_t D, size_t E, size_t tauQ, size_t tauP,
             size_t Work, int lwork, size_t devInfo):
    with nogil:
        status = cusolverDnSgebrd(
            <Handle>handle, m, n, <float*>A, lda,
            <float*>D, <float*>E, <float*>tauQ, <float*>tauP,
            <float*>Work, lwork, <int*>devInfo)
    check_status(status)

cpdef dgebrd(size_t handle, int m, int n, size_t A, int lda,
             size_t D, size_t E, size_t tauQ, size_t tauP,
             size_t Work, int lwork, size_t devInfo):
    with nogil:
        status = cusolverDnDgebrd(
            <Handle>handle, m, n, <double*>A, lda,
            <double*>D, <double*>E, <double*>tauQ, <double*>tauP,
            <double*>Work, lwork, <int*>devInfo)
    check_status(status)

cpdef int sgesvd_bufferSize(size_t handle, int m, int n) except *:
    cdef int lwork
    with nogil:
        status = cusolverDnSgesvd_bufferSize(<Handle>handle, m, n, &lwork)
    check_status(status)
    return lwork

cpdef int dgesvd_bufferSize(size_t handle, int m, int n) except *:
    cdef int lwork
    with nogil:
        status = cusolverDnDgesvd_bufferSize(<Handle>handle, m, n, &lwork)
    check_status(status)
    return lwork

cpdef sgesvd(size_t handle, char jobu, char jobvt, int m, int n, size_t A,
             int lda, size_t S, size_t U, int ldu, size_t VT, int ldvt,
             size_t Work, int lwork, size_t rwork, size_t devInfo):
    with nogil:
        status = cusolverDnSgesvd(
            <Handle>handle, jobu, jobvt, m, n, <float*>A,
            lda, <float*>S, <float*>U, ldu, <float*>VT, ldvt,
            <float*>Work, lwork, <float*>rwork, <int*>devInfo)
    check_status(status)

cpdef dgesvd(size_t handle, char jobu, char jobvt, int m, int n, size_t A,
             int lda, size_t S, size_t U, int ldu, size_t VT, int ldvt,
             size_t Work, int lwork, size_t rwork, size_t devInfo):
    with nogil:
        status = cusolverDnDgesvd(
            <Handle>handle, jobu, jobvt, m, n, <double*>A,
            lda, <double*>S, <double*>U, ldu, <double*>VT, ldvt,
            <double*>Work, lwork, <double*>rwork, <int*>devInfo)
    check_status(status)
