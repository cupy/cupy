"""Thin wrapper of CUSOLVER."""
from libc.stdint cimport intptr_t


###############################################################################
# Types
###############################################################################

from cupy.cuda.driver cimport Stream

cdef extern from *:
    ctypedef void* LibraryPropertyType 'libraryPropertyType_t'

    ctypedef void* Handle 'cusolverDnHandle_t'
    ctypedef void* SpHandle 'cusolverSpHandle_t'

    ctypedef int Operation 'cublasOperation_t'
    ctypedef int SideMode 'cublasSideMode_t'
    ctypedef int FillMode 'cublasFillMode_t'

    ctypedef int EigType 'cusolverEigType_t'
    ctypedef int EigMode 'cusolverEigMode_t'

    ctypedef void* MatDescr 'cusparseMatDescr_t'

    ctypedef void* cuComplex 'cuComplex'
    ctypedef void* cuDoubleComplex 'cuDoubleComplex'

###############################################################################
# Enum
###############################################################################

cpdef enum:
    CUSOLVER_EIG_TYPE_1 = 1
    CUSOLVER_EIG_TYPE_2 = 2
    CUSOLVER_EIG_TYPE_3 = 3

    CUSOLVER_EIG_MODE_NOVECTOR = 0
    CUSOLVER_EIG_MODE_VECTOR = 1

###############################################################################
# Library Attributes
###############################################################################

cpdef int getProperty(int type)
cpdef tuple _getVersion()

###############################################################################
# Context
###############################################################################

cpdef intptr_t create() except? 0
cpdef intptr_t spCreate() except? 0
cpdef destroy(intptr_t handle)
cpdef spDestroy(intptr_t handle)

###############################################################################
# Stream
###############################################################################

cpdef setStream(intptr_t handle, size_t stream)
cpdef size_t getStream(intptr_t handle) except? 0

###############################################################################
# Dense LAPACK Functions (Linear Solver)
###############################################################################

# Cholesky factorization
cpdef int spotrf_bufferSize(intptr_t handle, int uplo,
                            int n, size_t A, int lda) except? -1
cpdef int dpotrf_bufferSize(intptr_t handle, int uplo,
                            int n, size_t A, int lda) except? -1
cpdef int cpotrf_bufferSize(intptr_t handle, int uplo,
                            int n, size_t A, int lda) except? -1
cpdef int zpotrf_bufferSize(intptr_t handle, int uplo,
                            int n, size_t A, int lda) except? -1

cpdef spotrf(intptr_t handle, int uplo, int n, size_t A, int lda,
             size_t work, int lwork, size_t devInfo)
cpdef dpotrf(intptr_t handle, int uplo, int n, size_t A, int lda,
             size_t work, int lwork, size_t devInfo)
cpdef cpotrf(intptr_t handle, int uplo, int n, size_t A, int lda,
             size_t work, int lwork, size_t devInfo)
cpdef zpotrf(intptr_t handle, int uplo, int n, size_t A, int lda,
             size_t work, int lwork, size_t devInfo)

cpdef spotrs(intptr_t handle, int uplo, int n, int nrhs,
             size_t A, int lda, size_t B, int ldb, size_t devInfo)
cpdef dpotrs(intptr_t handle, int uplo, int n, int nrhs,
             size_t A, int lda, size_t B, int ldb, size_t devInfo)
cpdef cpotrs(intptr_t handle, int uplo, int n, int nrhs,
             size_t A, int lda, size_t B, int ldb, size_t devInfo)
cpdef zpotrs(intptr_t handle, int uplo, int n, int nrhs,
             size_t A, int lda, size_t B, int ldb, size_t devInfo)

# TODO(anaruse): potrfBatched and potrsBatched

# LU factorization
cpdef int sgetrf_bufferSize(intptr_t handle, int m, int n,
                            size_t A, int lda) except? -1
cpdef int dgetrf_bufferSize(intptr_t handle, int m, int n,
                            size_t A, int lda) except? -1
cpdef int cgetrf_bufferSize(intptr_t handle, int m, int n,
                            size_t A, int lda) except? -1
cpdef int zgetrf_bufferSize(intptr_t handle, int m, int n,
                            size_t A, int lda) except? -1

cpdef sgetrf(intptr_t handle, int m, int n, size_t A, int lda,
             size_t work, size_t devIpiv, size_t devInfo)
cpdef dgetrf(intptr_t handle, int m, int n, size_t A, int lda,
             size_t work, size_t devIpiv, size_t devInfo)
cpdef cgetrf(intptr_t handle, int m, int n, size_t A, int lda,
             size_t work, size_t devIpiv, size_t devInfo)
cpdef zgetrf(intptr_t handle, int m, int n, size_t A, int lda,
             size_t work, size_t devIpiv, size_t devInfo)

# TODO(anaruse): laswp

# LU solve
cpdef sgetrs(intptr_t handle, int trans, int n, int nrhs,
             size_t A, int lda, size_t devIpiv,
             size_t B, int ldb, size_t devInfo)
cpdef dgetrs(intptr_t handle, int trans, int n, int nrhs,
             size_t A, int lda, size_t devIpiv,
             size_t B, int ldb, size_t devInfo)
cpdef cgetrs(intptr_t handle, int trans, int n, int nrhs,
             size_t A, int lda, size_t devIpiv,
             size_t B, int ldb, size_t devInfo)
cpdef zgetrs(intptr_t handle, int trans, int n, int nrhs,
             size_t A, int lda, size_t devIpiv,
             size_t B, int ldb, size_t devInfo)

# QR factorization
cpdef int sgeqrf_bufferSize(intptr_t handle, int m, int n,
                            size_t A, int lda) except? -1
cpdef int dgeqrf_bufferSize(intptr_t handle, int m, int n,
                            size_t A, int lda) except? -1
cpdef int cgeqrf_bufferSize(intptr_t handle, int m, int n,
                            size_t A, int lda) except? -1
cpdef int zgeqrf_bufferSize(intptr_t handle, int m, int n,
                            size_t A, int lda) except? -1

cpdef sgeqrf(intptr_t handle, int m, int n, size_t A, int lda,
             size_t tau, size_t work, int lwork, size_t devInfo)
cpdef dgeqrf(intptr_t handle, int m, int n, size_t A, int lda,
             size_t tau, size_t work, int lwork, size_t devInfo)
cpdef cgeqrf(intptr_t handle, int m, int n, size_t A, int lda,
             size_t tau, size_t work, int lwork, size_t devInfo)
cpdef zgeqrf(intptr_t handle, int m, int n, size_t A, int lda,
             size_t tau, size_t work, int lwork, size_t devInfo)

# Generate unitary matrix Q from QR factorization
cpdef int sorgqr_bufferSize(intptr_t handle, int m, int n, int k,
                            size_t A, int lda, size_t tau) except? -1
cpdef int dorgqr_bufferSize(intptr_t handle, int m, int n, int k,
                            size_t A, int lda, size_t tau) except? -1
cpdef int cungqr_bufferSize(intptr_t handle, int m, int n, int k,
                            size_t A, int lda, size_t tau) except? -1
cpdef int zungqr_bufferSize(intptr_t handle, int m, int n, int k,
                            size_t A, int lda, size_t tau) except? -1

cpdef sorgqr(intptr_t handle, int m, int n, int k, size_t A, int lda,
             size_t tau, size_t work, int lwork, size_t devInfo)
cpdef dorgqr(intptr_t handle, int m, int n, int k, size_t A, int lda,
             size_t tau, size_t work, int lwork, size_t devInfo)
cpdef cungqr(intptr_t handle, int m, int n, int k, size_t A, int lda,
             size_t tau, size_t work, int lwork, size_t devInfo)
cpdef zungqr(intptr_t handle, int m, int n, int k, size_t A, int lda,
             size_t tau, size_t work, int lwork, size_t devInfo)

# Compute Q**T*b in solve min||A*x = b||
cpdef int sormqr_bufferSize(intptr_t handle, int side, int trans,
                            int m, int n, int k, size_t A, int lda, size_t tau,
                            size_t C, int ldc) except? -1
cpdef int dormqr_bufferSize(intptr_t handle, int side, int trans,
                            int m, int n, int k, size_t A, int lda, size_t tau,
                            size_t C, int ldc) except? -1
cpdef int cunmqr_bufferSize(intptr_t handle, int side, int trans,
                            int m, int n, int k, size_t A, int lda, size_t tau,
                            size_t C, int ldc) except? -1
cpdef int zunmqr_bufferSize(intptr_t handle, int side, int trans,
                            int m, int n, int k, size_t A, int lda, size_t tau,
                            size_t C, int ldc) except? -1

cpdef sormqr(intptr_t handle, int side, int trans,
             int m, int n, int k, size_t A, int lda, size_t tau, size_t C,
             int ldc, size_t work, int lwork, size_t devInfo)
cpdef dormqr(intptr_t handle, int side, int trans,
             int m, int n, int k, size_t A, int lda, size_t tau, size_t C,
             int ldc, size_t work, int lwork, size_t devInfo)
cpdef cunmqr(intptr_t handle, int side, int trans,
             int m, int n, int k, size_t A, int lda, size_t tau, size_t C,
             int ldc, size_t work, int lwork, size_t devInfo)
cpdef zunmqr(intptr_t handle, int side, int trans,
             int m, int n, int k, size_t A, int lda, size_t tau, size_t C,
             int ldc, size_t work, int lwork, size_t devInfo)
cpdef cormqr(intptr_t handle, int side, int trans,
             int m, int n, int k, size_t A, int lda, size_t tau, size_t C,
             int ldc, size_t work, int lwork, size_t devInfo)  # (obsoleted)
cpdef zormqr(intptr_t handle, int side, int trans,
             int m, int n, int k, size_t A, int lda, size_t tau, size_t C,
             int ldc, size_t work, int lwork, size_t devInfo)  # (obsoleted)

# L*D*L**T,U*D*U**T factorization
cpdef int ssytrf_bufferSize(intptr_t handle, int n, size_t A,
                            int lda) except? -1
cpdef int dsytrf_bufferSize(intptr_t handle, int n, size_t A,
                            int lda) except? -1
cpdef int csytrf_bufferSize(intptr_t handle, int n, size_t A,
                            int lda) except? -1
cpdef int zsytrf_bufferSize(intptr_t handle, int n, size_t A,
                            int lda) except? -1

cpdef ssytrf(intptr_t handle, int uplo, int n, size_t A, int lda,
             size_t ipiv, size_t work, int lwork, size_t devInfo)
cpdef dsytrf(intptr_t handle, int uplo, int n, size_t A, int lda,
             size_t ipiv, size_t work, int lwork, size_t devInfo)
cpdef csytrf(intptr_t handle, int uplo, int n, size_t A, int lda,
             size_t ipiv, size_t work, int lwork, size_t devInfo)
cpdef zsytrf(intptr_t handle, int uplo, int n, size_t A, int lda,
             size_t ipiv, size_t work, int lwork, size_t devInfo)

###############################################################################
# Dense LAPACK Functions (Eigenvalue Solver)
###############################################################################

# Bidiagonal factorization
cpdef int sgebrd_bufferSize(intptr_t handle, int m, int n) except? -1
cpdef int dgebrd_bufferSize(intptr_t handle, int m, int n) except? -1
cpdef int cgebrd_bufferSize(intptr_t handle, int m, int n) except? -1
cpdef int zgebrd_bufferSize(intptr_t handle, int m, int n) except? -1

cpdef sgebrd(intptr_t handle, int m, int n, size_t A, int lda,
             size_t D, size_t E, size_t tauQ, size_t tauP,
             size_t Work, int lwork, size_t devInfo)
cpdef dgebrd(intptr_t handle, int m, int n, size_t A, int lda,
             size_t D, size_t E, size_t tauQ, size_t tauP,
             size_t Work, int lwork, size_t devInfo)
cpdef cgebrd(intptr_t handle, int m, int n, size_t A, int lda,
             size_t D, size_t E, size_t tauQ, size_t tauP,
             size_t Work, int lwork, size_t devInfo)
cpdef zgebrd(intptr_t handle, int m, int n, size_t A, int lda,
             size_t D, size_t E, size_t tauQ, size_t tauP,
             size_t Work, int lwork, size_t devInfo)

# TODO(anaruse): orgbr/ungbr, sytrd/hetrd, orgtr/ungtr, ormtr/unmtr

# Singular value decomposition, A = U * Sigma * V^H
cpdef int sgesvd_bufferSize(intptr_t handle, int m, int n) except? -1
cpdef int dgesvd_bufferSize(intptr_t handle, int m, int n) except? -1
cpdef int cgesvd_bufferSize(intptr_t handle, int m, int n) except? -1
cpdef int zgesvd_bufferSize(intptr_t handle, int m, int n) except? -1

cpdef sgesvd(intptr_t handle, char jobu, char jobvt, int m, int n, size_t A,
             int lda, size_t S, size_t U, int ldu, size_t VT, int ldvt,
             size_t Work, int lwork, size_t rwork, size_t devInfo)
cpdef dgesvd(intptr_t handle, char jobu, char jobvt, int m, int n, size_t A,
             int lda, size_t S, size_t U, int ldu, size_t VT, int ldvt,
             size_t Work, int lwork, size_t rwork, size_t devInfo)
cpdef cgesvd(intptr_t handle, char jobu, char jobvt, int m, int n, size_t A,
             int lda, size_t S, size_t U, int ldu, size_t VT, int ldvt,
             size_t Work, int lwork, size_t rwork, size_t devInfo)
cpdef zgesvd(intptr_t handle, char jobu, char jobvt, int m, int n, size_t A,
             int lda, size_t S, size_t U, int ldu, size_t VT, int ldvt,
             size_t Work, int lwork, size_t rwork, size_t devInfo)

# Standard symmetric eigenvalue solver
cpdef int ssyevd_bufferSize(intptr_t handle, int jobz, int uplo, int n,
                            size_t A, int lda, size_t W) except? -1
cpdef int dsyevd_bufferSize(intptr_t handle, int jobz, int uplo, int n,
                            size_t A, int lda, size_t W) except? -1
cpdef int cheevd_bufferSize(intptr_t handle, int jobz, int uplo, int n,
                            size_t A, int lda, size_t W) except? -1
cpdef int zheevd_bufferSize(intptr_t handle, int jobz, int uplo, int n,
                            size_t A, int lda, size_t W) except? -1

cpdef ssyevd(intptr_t handle, int jobz, int uplo, int n, size_t A, int lda,
             size_t W, size_t work, int lwork, size_t info)
cpdef dsyevd(intptr_t handle, int jobz, int uplo, int n, size_t A, int lda,
             size_t W, size_t work, int lwork, size_t info)
cpdef cheevd(intptr_t handle, int jobz, int uplo, int n, size_t A, int lda,
             size_t W, size_t work, int lwork, size_t info)
cpdef zheevd(intptr_t handle, int jobz, int uplo, int n, size_t A, int lda,
             size_t W, size_t work, int lwork, size_t info)

# TODO(anaruse); sygvd/hegvd, sygvd/hegvd

###############################################################################
# Sparse LAPACK Functions
###############################################################################

cpdef scsrlsvchol(intptr_t handle, int m, int nnz, size_t descrA,
                  size_t csrValA, size_t csrRowPtrA, size_t csrColIndA,
                  size_t b, float tol, int reorder, size_t x,
                  size_t singularity)
cpdef dcsrlsvchol(intptr_t handle, int m, int nnz, size_t descrA,
                  size_t csrValA, size_t csrRowPtrA, size_t csrColIndA,
                  size_t b, double tol, int reorder, size_t x,
                  size_t singularity)
cpdef ccsrlsvchol(intptr_t handle, int m, int nnz, size_t descrA,
                  size_t csrVal, size_t csrRowPtr, size_t csrColInd, size_t b,
                  float tol, int reorder, size_t x, size_t singularity)
cpdef zcsrlsvchol(intptr_t handle, int m, int nnz, size_t descrA,
                  size_t csrVal, size_t csrRowPtr, size_t csrColInd, size_t b,
                  double tol, int reorder, size_t x, size_t singularity)

cpdef scsrlsvqr(intptr_t handle, int m, int nnz, size_t descrA, size_t csrValA,
                size_t csrRowPtrA, size_t csrColIndA, size_t b, float tol,
                int reorder, size_t x, size_t singularity)
cpdef dcsrlsvqr(intptr_t handle, int m, int nnz, size_t descrA, size_t csrValA,
                size_t csrRowPtrA, size_t csrColIndA, size_t b, double tol,
                int reorder, size_t x, size_t singularity)
cpdef ccsrlsvqr(intptr_t handle, int m, int nnz, size_t descrA, size_t csrVal,
                size_t csrRowPtr, size_t csrColInd, size_t b, float tol,
                int reorder, size_t x, size_t singularity)
cpdef zcsrlsvqr(intptr_t handle, int m, int nnz, size_t descrA, size_t csrVal,
                size_t csrRowPtr, size_t csrColInd, size_t b, double tol,
                int reorder, size_t x, size_t singularity)

cpdef scsreigvsi(intptr_t handle, int m, int nnz, size_t descrA,
                 size_t csrValA, size_t csrRowPtrA, size_t csrColIndA,
                 float mu0, size_t x0, int maxite, float eps, size_t mu,
                 size_t x)
cpdef dcsreigvsi(intptr_t handle, int m, int nnz, size_t descrA,
                 size_t csrValA, size_t csrRowPtrA, size_t csrColIndA,
                 double mu0, size_t x0, int maxite, double eps, size_t mu,
                 size_t x)
cpdef ccsreigvsi(intptr_t handle, int m, int nnz, size_t descrA,
                 size_t csrValA, size_t csrRowPtrA, size_t csrColIndA,
                 size_t mu0, size_t x0, int maxite, float eps, size_t mu,
                 size_t x)
cpdef zcsreigvsi(intptr_t handle, int m, int nnz, size_t descrA,
                 size_t csrValA, size_t csrRowPtrA, size_t csrColIndA,
                 size_t mu0, size_t x0, int maxite, double eps, size_t mu,
                 size_t x)
