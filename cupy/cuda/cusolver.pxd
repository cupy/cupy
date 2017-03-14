"""Thin wrapper of CUSOLVER."""


###############################################################################
# Types
###############################################################################

from cupy.cuda.driver cimport Stream

cdef extern from *:
    ctypedef void* Handle 'cusolverDnHandle_t'

    ctypedef int Operation 'cublasOperation_t'
    ctypedef int SideMode 'cublasSideMode_t'
    ctypedef int FillMode 'cublasFillMode_t'

###############################################################################
# Context
###############################################################################

cpdef size_t create() except *
cpdef void destroy(size_t handle) except *

###############################################################################
# Stream
###############################################################################

cpdef setStream(size_t handle, size_t stream)
cpdef size_t getStream(size_t handle) except *

###############################################################################
# dense LAPACK Functions
###############################################################################

cpdef int spotrf_bufferSize(size_t handle, int uplo,
                            int n, size_t A, int lda) except *
cpdef int dpotrf_bufferSize(size_t handle, int uplo,
                            int n, size_t A, int lda) except *
cpdef spotrf(size_t handle, int uplo, int n, size_t A, int lda,
             size_t work, int lwork, size_t devInfo)
cpdef dpotrf(size_t handle, int uplo, int n, size_t A, int lda,
             size_t work, int lwork, size_t devInfo)

cpdef spotrs(size_t handle, int uplo, int n, int nrhs,
             size_t A, int lda, size_t B, int ldb, size_t devInfo)
cpdef dpotrs(size_t handle, int uplo, int n, int nrhs,
             size_t A, int lda, size_t B, int ldb, size_t devInfo)

cpdef sgetrf(size_t handle, int m, int n, size_t A, int lda,
             size_t work, size_t devIpiv, size_t devInfo)
cpdef dgetrf(size_t handle, int m, int n, size_t A, int lda,
             size_t work, size_t devIpiv, size_t devInfo)

cpdef sgetrs(size_t handle, int trans, int n, int nrhs,
             size_t A, int lda, size_t devIpiv,
             size_t B, int ldb, size_t devInfo)
cpdef dgetrs(size_t handle, int trans, int n, int nrhs,
             size_t A, int lda, size_t devIpiv,
             size_t B, int ldb, size_t devInfo)

cpdef int sgeqrf_bufferSize(size_t handle, int m, int n,
                            size_t A, int lda) except *
cpdef int dgeqrf_bufferSize(size_t handle, int m, int n,
                            size_t A, int lda) except *
cpdef sgeqrf(size_t handle, int m, int n, size_t A, int lda,
             size_t tau, size_t work, int lwork, size_t devInfo)
cpdef dgeqrf(size_t handle, int m, int n, size_t A, int lda,
             size_t tau, size_t work, int lwork, size_t devInfo)

cpdef int sorgqr_bufferSize(size_t handle, int m, int n, int k,
                            size_t A, int lda, size_t tau) except *
cpdef int dorgqr_bufferSize(size_t handle, int m, int n, int k,
                            size_t A, int lda, size_t tau) except *
cpdef sorgqr(size_t handle, int m, int n, int k, size_t A, int lda,
             size_t tau, size_t work, int lwork, size_t devInfo)
cpdef dorgqr(size_t handle, int m, int n, int k, size_t A, int lda,
             size_t tau, size_t work, int lwork, size_t devInfo)

cpdef sormqr(size_t handle, int side, int trans,
             int m, int n, int k, size_t A, int lda, size_t tau,
             size_t C, int ldc, size_t work, int lwork, size_t devInfo)
cpdef dormqr(size_t handle, int side, int trans,
             int m, int n, int k, size_t A, int lda, size_t tau,
             size_t C, int ldc, size_t work, int lwork, size_t devInfo)

cpdef ssytrf(size_t handle, int uplo, int n, size_t A, int lda,
             size_t ipiv, size_t work, int lwork, size_t devInfo)
cpdef dsytrf(size_t handle, int uplo, int n, size_t A, int lda,
             size_t ipiv, size_t work, int lwork, size_t devInfo)

cpdef sgebrd(size_t handle, int m, int n, size_t A, int lda,
             size_t D, size_t E, size_t tauQ, size_t tauP,
             size_t Work, int lwork, size_t devInfo)
cpdef dgebrd(size_t handle, int m, int n, size_t A, int lda,
             size_t D, size_t E, size_t tauQ, size_t tauP,
             size_t Work, int lwork, size_t devInfo)

cpdef int sgesvd_bufferSize(size_t handle, int m, int n) except *
cpdef int dgesvd_bufferSize(size_t handle, int m, int n) except *
cpdef sgesvd(size_t handle, char jobu, char jobvt, int m, int n, size_t A,
             int lda, size_t S, size_t U, int ldu, size_t VT, int ldvt,
             size_t Work, int lwork, size_t rwork, size_t devInfo)
cpdef dgesvd(size_t handle, char jobu, char jobvt, int m, int n, size_t A,
             int lda, size_t S, size_t U, int ldu, size_t VT, int ldvt,
             size_t Work, int lwork, size_t rwork, size_t devInfo)
