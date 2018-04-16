"""Thin wrapper of CUSOLVER."""
cimport cython
cimport cusparse

from cupy.cuda cimport driver
from cupy.cuda cimport runtime
from cupy.cuda cimport stream as stream_module

###############################################################################
# Extern
###############################################################################

cdef extern from 'cupy_cusolver.h' nogil:
    # Context
    int cusolverDnCreate(Handle* handle)
    int cusolverSpCreate(SpHandle* handle)
    int cusolverDnDestroy(Handle handle)

    # Stream
    int cusolverDnGetStream(Handle handle, driver.Stream* streamId)
    int cusolverSpGetStream(SpHandle handle, driver.Stream* streamId)
    int cusolverDnSetStream(Handle handle, driver.Stream streamId)
    int cusolverSpSetStream(SpHandle handle, driver.Stream streamId)

    # Linear Equations
    int cusolverDnSpotrf_bufferSize(Handle handle, FillMode uplo, int n,
                                    float* A, int lda, int* lwork)
    int cusolverDnDpotrf_bufferSize(Handle handle, FillMode uplo, int n,
                                    double* A, int lda, int* lwork)
    int cusolverDnSpotrf(Handle handle, FillMode uplo, int n, float* A,
                         int lda, float* work, int lwork, int* devInfo)
    int cusolverDnDpotrf(Handle handle, FillMode uplo, int n, double *A,
                         int lda, double* work, int lwork, int* devInfo)

    int cusolverDnSpotrs(Handle handle, FillMode uplo, int n, int nrhs,
                         const float* A, int lda, float* B, int ldb,
                         int* devInfo)
    int cusolverDnDpotrs(Handle handle, FillMode uplo, int n, int nrhs,
                         const double* A, int lda, double* B, int ldb,
                         int *devInfo)

    int cusolverDnSgetrf(Handle handle, int m, int n, float* A, int lda,
                         float* work, int* devIpiv, int* devInfo)
    int cusolverDnDgetrf(Handle handle, int m, int n, double* A, int lda,
                         double* work, int* devIpiv, int* devInfo)

    int cusolverDnSgetrs(Handle handle, Operation trans,
                         int n, int nrhs, const float* A, int lda,
                         const int* devIpiv, float* B, int ldb,
                         int* devInfo)
    int cusolverDnDgetrs(Handle handle, Operation trans,
                         int n, int nrhs, const double* A, int lda,
                         const int* devIpiv, double* B, int ldb,
                         int* devInfo)

    int cusolverDnSgetrf_bufferSize(Handle handle, int m, int n,
                                    float *A, int lda, int* lwork)
    int cusolverDnDgetrf_bufferSize(Handle handle, int m, int n,
                                    double *A, int lda, int* lwork)
    int cusolverDnSgeqrf_bufferSize(Handle handle, int m, int n,
                                    float* A, int lda, int* lwork)
    int cusolverDnDgeqrf_bufferSize(Handle handle, int m, int n,
                                    double* A, int lda, int* lwork)
    int cusolverDnSgeqrf(Handle handle, int m, int n, float* A, int lda,
                         float* tau, float* work, int lwork,
                         int* devInfo)
    int cusolverDnDgeqrf(Handle handle, int m, int n, double* A, int lda,
                         double* tau, double* work, int lwork,
                         int* devInfo)

    # The actual definition of cusolverDn(S|D)orgqr_bufferSize
    # is different from the reference
    int cusolverDnSorgqr_bufferSize(Handle handle, int m, int n, int k,
                                    const float* A, int lda,
                                    const float* tau, int* lwork)
    int cusolverDnDorgqr_bufferSize(Handle handle, int m, int n, int k,
                                    const double* A, int lda,
                                    const double* tau, int* lwork)
    int cusolverDnSorgqr(Handle handle, int m, int n, int k,
                         float* A, int lda, const float* tau,
                         float* work, int lwork, int* devInfo)
    int cusolverDnDorgqr(Handle handle, int m, int n, int k,
                         double* A, int lda, const double* tau,
                         double* work, int lwork, int* devInfo)

    int cusolverDnSormqr(Handle handle, SideMode side, Operation trans,
                         int m, int n, int k, const float* A, int lda,
                         const float* tau, float* C, int ldc, float* work,
                         int lwork, int* devInfo)
    int cusolverDnDormqr(Handle handle, SideMode side, Operation trans,
                         int m, int n, int k, const double* A, int lda,
                         const double* tau, double* C, int ldc, double* work,
                         int lwork, int* devInfo)

    int cusolverDnSsytrf(Handle handle, FillMode uplo, int n, float* A,
                         int lda, int* ipiv, float* work, int lwork,
                         int* devInfo)
    int cusolverDnDsytrf(Handle handle, FillMode uplo, int n, double* A,
                         int lda, int* ipiv, double* work, int lwork,
                         int* devInfo)

    int cusolverDnSgebrd(Handle handle, int m, int n, float* A, int lda,
                         float* D, float* E, float* tauQ, float* tauP,
                         float* Work, int lwork, int* devInfo)
    int cusolverDnDgebrd(Handle handle, int m, int n, double* A, int lda,
                         double* D, double* E, double* tauQ, double* tauP,
                         double* Work, int lwork, int* devInfo)

    int cusolverDnSgesvd_bufferSize(Handle handle, int m, int n,
                                    int* lwork)
    int cusolverDnDgesvd_bufferSize(Handle handle, int m, int n,
                                    int* lwork)
    int cusolverDnSgesvd(Handle handle, char jobu, char jobvt, int m, int n,
                         float* A, int lda, float* S, float* U, int ldu,
                         float* VT, int ldvt, float* Work, int lwork,
                         float* rwork, int* devInfo)
    int cusolverDnDgesvd(Handle handle, char jobu, char jobvt, int m, int n,
                         double* A, int lda, double* S, double* U, int ldu,
                         double* VT, int ldvt, double* Work, int lwork,
                         double* rwork, int* devInfo)

    int cusolverDnSsyevd_bufferSize(
        Handle handle, EigMode jobz, FillMode uplo, int n, const float* A,
        int lda, const float* W, int* lwork)
    int cusolverDnDsyevd_bufferSize(
        Handle handle, EigMode jobz, FillMode uplo, int n, const double* A,
        int lda, const double* W, int* lwork)
    int cusolverDnSsyevd(
        Handle handle, EigMode jobz, FillMode uplo, int n, float* A, int lda,
        float* W, float* work, int lwork, int* info)
    int cusolverDnDsyevd(
        Handle handle, EigMode jobz, FillMode uplo, int n, double* A, int lda,
        double* W, double* work, int lwork, int* info)

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


cpdef size_t spCreate() except *:
    cdef SpHandle handle
    with nogil:
        status = cusolverSpCreate(&handle)
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
        status = cusolverDnSetStream(<Handle>handle, <driver.Stream>stream)
    check_status(status)


cpdef size_t getStream(size_t handle) except *:
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


###############################################################################
# dense LAPACK Functions
###############################################################################

cpdef int spotrf_bufferSize(size_t handle, int uplo,
                            int n, size_t A, int lda) except *:
    cdef int lwork
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnSpotrf_bufferSize(
            <Handle>handle, <FillMode>uplo, n, <float*>A, <int>lda, &lwork)
    check_status(status)
    return lwork

cpdef int dpotrf_bufferSize(size_t handle, int uplo,
                            int n, size_t A, int lda) except *:
    cdef int lwork
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnDpotrf_bufferSize(
            <Handle>handle, <FillMode>uplo, n, <double*>A, <int>lda, &lwork)
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

cpdef spotrs(size_t handle, int uplo, int n, int nrhs,
             size_t A, int lda, size_t B, int ldb, size_t devInfo):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnSpotrs(
            <Handle>handle, <FillMode>uplo, n, nrhs,
            <const float*>A, lda, <float*>B, ldb, <int*>devInfo)
    check_status(status)

cpdef dpotrs(size_t handle, int uplo, int n, int nrhs,
             size_t A, int lda, size_t B, int ldb, size_t devInfo):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnDpotrs(
            <Handle>handle, <FillMode>uplo, n, nrhs,
            <const double*>A, lda, <double*>B, ldb, <int*>devInfo)
    check_status(status)

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

cpdef int sgetrf_bufferSize(size_t handle, int m, int n,
                            size_t A, int lda) except *:
    cdef int lwork
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnSgetrf_bufferSize(
            <Handle>handle, m, n, <float*>A, lda, &lwork)
    check_status(status)
    return lwork

cpdef int dgetrf_bufferSize(size_t handle, int m, int n,
                            size_t A, int lda) except *:
    cdef int lwork
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnDgetrf_bufferSize(
            <Handle>handle, m, n, <double*>A, lda, &lwork)
    check_status(status)
    return lwork

cpdef int sgeqrf_bufferSize(size_t handle, int m, int n,
                            size_t A, int lda) except *:
    cdef int lwork
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnSgeqrf_bufferSize(
            <Handle>handle, m, n, <float*>A, lda, &lwork)
    check_status(status)
    return lwork

cpdef int dgeqrf_bufferSize(size_t handle, int m, int n,
                            size_t A, int lda) except *:
    cdef int lwork
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnDgeqrf_bufferSize(
            <Handle>handle, m, n, <double*>A, lda, &lwork)
    check_status(status)
    return lwork

cpdef sgeqrf(size_t handle, int m, int n, size_t A, int lda,
             size_t tau, size_t work, int lwork, size_t devInfo):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnSgeqrf(
            <Handle>handle, m, n, <float*>A, lda,
            <float*>tau, <float*>work, lwork, <int*>devInfo)
    check_status(status)

cpdef dgeqrf(size_t handle, int m, int n, size_t A, int lda,
             size_t tau, size_t work, int lwork, size_t devInfo):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnDgeqrf(
            <Handle>handle, m, n, <double*>A, lda,
            <double*>tau, <double*>work, lwork, <int*>devInfo)
    check_status(status)

cpdef int sorgqr_bufferSize(size_t handle, int m, int n, int k,
                            size_t A, int lda, size_t tau) except *:
    cdef int lwork
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnSorgqr_bufferSize(
            <Handle>handle, m, n, k, <const float*>A, lda,
            <const float*>tau, &lwork)
    check_status(status)
    return lwork

cpdef int dorgqr_bufferSize(size_t handle, int m, int n, int k,
                            size_t A, int lda, size_t tau) except *:
    cdef int lwork
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnDorgqr_bufferSize(
            <Handle>handle, m, n, k, <double*>A, lda,
            <const double*> tau, &lwork)
    check_status(status)
    return lwork

cpdef sorgqr(size_t handle, int m, int n, int k, size_t A, int lda,
             size_t tau, size_t work, int lwork, size_t devInfo):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnSorgqr(
            <Handle>handle, m, n, k, <float*>A, lda,
            <const float*>tau, <float*>work, lwork, <int*>devInfo)
    check_status(status)

cpdef dorgqr(size_t handle, int m, int n, int k, size_t A, int lda,
             size_t tau, size_t work, int lwork, size_t devInfo):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnDorgqr(
            <Handle>handle, m, n, k, <double*>A, lda,
            <const double*>tau, <double*>work, lwork, <int*>devInfo)
    check_status(status)

cpdef sormqr(size_t handle, int side, int trans,
             int m, int n, int k, size_t A, int lda, size_t tau,
             size_t C, int ldc, size_t work, int lwork, size_t devInfo):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnSormqr(
            <Handle>handle, <SideMode>side, <Operation>trans, m, n, k,
            <const float*>A, lda, <const float*>tau, <float*>C, ldc,
            <float*>work, lwork, <int*>devInfo)
    check_status(status)

cpdef dormqr(size_t handle, int side, int trans,
             int m, int n, int k, size_t A, int lda, size_t tau,
             size_t C, int ldc, size_t work, int lwork, size_t devInfo):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnDormqr(
            <Handle>handle, <SideMode>side, <Operation>trans, m, n, k,
            <const double*>A, lda, <const double*>tau, <double*>C, ldc,
            <double*>work, lwork, <int*>devInfo)
    check_status(status)

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

cpdef sgebrd(size_t handle, int m, int n, size_t A, int lda,
             size_t D, size_t E, size_t tauQ, size_t tauP,
             size_t Work, int lwork, size_t devInfo):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnSgebrd(
            <Handle>handle, m, n, <float*>A, lda,
            <float*>D, <float*>E, <float*>tauQ, <float*>tauP,
            <float*>Work, lwork, <int*>devInfo)
    check_status(status)

cpdef dgebrd(size_t handle, int m, int n, size_t A, int lda,
             size_t D, size_t E, size_t tauQ, size_t tauP,
             size_t Work, int lwork, size_t devInfo):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnDgebrd(
            <Handle>handle, m, n, <double*>A, lda,
            <double*>D, <double*>E, <double*>tauQ, <double*>tauP,
            <double*>Work, lwork, <int*>devInfo)
    check_status(status)

cpdef int sgesvd_bufferSize(size_t handle, int m, int n) except *:
    cdef int lwork
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnSgesvd_bufferSize(<Handle>handle, m, n, &lwork)
    check_status(status)
    return lwork

cpdef int dgesvd_bufferSize(size_t handle, int m, int n) except *:
    cdef int lwork
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnDgesvd_bufferSize(<Handle>handle, m, n, &lwork)
    check_status(status)
    return lwork

cpdef sgesvd(size_t handle, char jobu, char jobvt, int m, int n, size_t A,
             int lda, size_t S, size_t U, int ldu, size_t VT, int ldvt,
             size_t Work, int lwork, size_t rwork, size_t devInfo):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnSgesvd(
            <Handle>handle, jobu, jobvt, m, n, <float*>A,
            lda, <float*>S, <float*>U, ldu, <float*>VT, ldvt,
            <float*>Work, lwork, <float*>rwork, <int*>devInfo)
    check_status(status)

cpdef dgesvd(size_t handle, char jobu, char jobvt, int m, int n, size_t A,
             int lda, size_t S, size_t U, int ldu, size_t VT, int ldvt,
             size_t Work, int lwork, size_t rwork, size_t devInfo):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnDgesvd(
            <Handle>handle, jobu, jobvt, m, n, <double*>A,
            lda, <double*>S, <double*>U, ldu, <double*>VT, ldvt,
            <double*>Work, lwork, <double*>rwork, <int*>devInfo)
    check_status(status)

cpdef int ssyevd_bufferSize(size_t handle, int jobz, int uplo, int n,
                            size_t A, int lda, size_t W):
    cdef int lwork, status
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnSsyevd_bufferSize(
            <Handle>handle, <EigMode>jobz, <FillMode>uplo, n, <const float*>A,
            lda, <const float*>W, &lwork)
    check_status(status)
    return lwork

cpdef int dsyevd_bufferSize(size_t handle, int jobz, int uplo, int n,
                            size_t A, int lda, size_t W):
    cdef int lwork, status
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnDsyevd_bufferSize(
            <Handle>handle, <EigMode>jobz, <FillMode>uplo, n, <const double*>A,
            lda, <const double*>W, &lwork)
    check_status(status)
    return lwork

cpdef ssyevd(size_t handle, int jobz, int uplo, int n, size_t A, int lda,
             size_t W, size_t work, int lwork, size_t info):
    cdef int status
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnSsyevd(
            <Handle>handle, <EigMode>jobz, <FillMode>uplo, n, <float*>A, lda,
            <float*>W, <float*>work, lwork, <int*>info)
    check_status(status)

cpdef dsyevd(size_t handle, int jobz, int uplo, int n, size_t A, int lda,
             size_t W, size_t work, int lwork, size_t info):
    cdef int status
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cusolverDnDsyevd(
            <Handle>handle, <EigMode>jobz, <FillMode>uplo, n, <double*>A, lda,
            <double*>W, <double*>work, lwork, <int*>info)
    check_status(status)

###############################################################################
# sparse LAPACK Functions
###############################################################################

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
