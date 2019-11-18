"""Thin wrapper of CUSOLVER."""


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

cpdef size_t create() except? 0
cpdef size_t spCreate() except? 0
cpdef destroy(size_t handle)
cpdef spDestroy(size_t handle)

###############################################################################
# Stream
###############################################################################

cpdef setStream(size_t handle, size_t stream)
cpdef size_t getStream(size_t handle) except? 0

###############################################################################
# Dense LAPACK Functions (Linear Solver)
###############################################################################

# Cholesky factorization
cpdef int spotrf_bufferSize(size_t handle, int uplo,
                            int n, size_t A, int lda) except? -1
cpdef int dpotrf_bufferSize(size_t handle, int uplo,
                            int n, size_t A, int lda) except? -1
cpdef int cpotrf_bufferSize(size_t handle, int uplo,
                            int n, size_t A, int lda) except? -1
cpdef int zpotrf_bufferSize(size_t handle, int uplo,
                            int n, size_t A, int lda) except? -1

cpdef spotrf(size_t handle, int uplo, int n, size_t A, int lda,
             size_t work, int lwork, size_t devInfo)
cpdef dpotrf(size_t handle, int uplo, int n, size_t A, int lda,
             size_t work, int lwork, size_t devInfo)
cpdef cpotrf(size_t handle, int uplo, int n, size_t A, int lda,
             size_t work, int lwork, size_t devInfo)
cpdef zpotrf(size_t handle, int uplo, int n, size_t A, int lda,
             size_t work, int lwork, size_t devInfo)

cpdef spotrs(size_t handle, int uplo, int n, int nrhs,
             size_t A, int lda, size_t B, int ldb, size_t devInfo)
cpdef dpotrs(size_t handle, int uplo, int n, int nrhs,
             size_t A, int lda, size_t B, int ldb, size_t devInfo)
cpdef cpotrs(size_t handle, int uplo, int n, int nrhs,
             size_t A, int lda, size_t B, int ldb, size_t devInfo)
cpdef zpotrs(size_t handle, int uplo, int n, int nrhs,
             size_t A, int lda, size_t B, int ldb, size_t devInfo)

# TODO(anaruse): potrfBatched and potrsBatched

# LU factorization
cpdef int sgetrf_bufferSize(size_t handle, int m, int n,
                            size_t A, int lda) except? -1
cpdef int dgetrf_bufferSize(size_t handle, int m, int n,
                            size_t A, int lda) except? -1
cpdef int cgetrf_bufferSize(size_t handle, int m, int n,
                            size_t A, int lda) except? -1
cpdef int zgetrf_bufferSize(size_t handle, int m, int n,
                            size_t A, int lda) except? -1

cpdef sgetrf(size_t handle, int m, int n, size_t A, int lda,
             size_t work, size_t devIpiv, size_t devInfo)
cpdef dgetrf(size_t handle, int m, int n, size_t A, int lda,
             size_t work, size_t devIpiv, size_t devInfo)
cpdef cgetrf(size_t handle, int m, int n, size_t A, int lda,
             size_t work, size_t devIpiv, size_t devInfo)
cpdef zgetrf(size_t handle, int m, int n, size_t A, int lda,
             size_t work, size_t devIpiv, size_t devInfo)

# TODO(anaruse): laswp

# LU solve
cpdef sgetrs(size_t handle, int trans, int n, int nrhs,
             size_t A, int lda, size_t devIpiv,
             size_t B, int ldb, size_t devInfo)
cpdef dgetrs(size_t handle, int trans, int n, int nrhs,
             size_t A, int lda, size_t devIpiv,
             size_t B, int ldb, size_t devInfo)
cpdef cgetrs(size_t handle, int trans, int n, int nrhs,
             size_t A, int lda, size_t devIpiv,
             size_t B, int ldb, size_t devInfo)
cpdef zgetrs(size_t handle, int trans, int n, int nrhs,
             size_t A, int lda, size_t devIpiv,
             size_t B, int ldb, size_t devInfo)

# QR factorization
cpdef int sgeqrf_bufferSize(size_t handle, int m, int n,
                            size_t A, int lda) except? -1
cpdef int dgeqrf_bufferSize(size_t handle, int m, int n,
                            size_t A, int lda) except? -1
cpdef int cgeqrf_bufferSize(size_t handle, int m, int n,
                            size_t A, int lda) except? -1
cpdef int zgeqrf_bufferSize(size_t handle, int m, int n,
                            size_t A, int lda) except? -1

cpdef sgeqrf(size_t handle, int m, int n, size_t A, int lda,
             size_t tau, size_t work, int lwork, size_t devInfo)
cpdef dgeqrf(size_t handle, int m, int n, size_t A, int lda,
             size_t tau, size_t work, int lwork, size_t devInfo)
cpdef cgeqrf(size_t handle, int m, int n, size_t A, int lda,
             size_t tau, size_t work, int lwork, size_t devInfo)
cpdef zgeqrf(size_t handle, int m, int n, size_t A, int lda,
             size_t tau, size_t work, int lwork, size_t devInfo)

# Generate unitary matrix Q from QR factorization
cpdef int sorgqr_bufferSize(size_t handle, int m, int n, int k,
                            size_t A, int lda, size_t tau) except? -1
cpdef int dorgqr_bufferSize(size_t handle, int m, int n, int k,
                            size_t A, int lda, size_t tau) except? -1
cpdef int cungqr_bufferSize(size_t handle, int m, int n, int k,
                            size_t A, int lda, size_t tau) except? -1
cpdef int zungqr_bufferSize(size_t handle, int m, int n, int k,
                            size_t A, int lda, size_t tau) except? -1

cpdef sorgqr(size_t handle, int m, int n, int k, size_t A, int lda,
             size_t tau, size_t work, int lwork, size_t devInfo)
cpdef dorgqr(size_t handle, int m, int n, int k, size_t A, int lda,
             size_t tau, size_t work, int lwork, size_t devInfo)
cpdef cungqr(size_t handle, int m, int n, int k, size_t A, int lda,
             size_t tau, size_t work, int lwork, size_t devInfo)
cpdef zungqr(size_t handle, int m, int n, int k, size_t A, int lda,
             size_t tau, size_t work, int lwork, size_t devInfo)

# Compute Q**T*b in solve min||A*x = b||
cpdef int sormqr_bufferSize(size_t handle, int side, int trans,
                            int m, int n, int k, size_t A, int lda, size_t tau,
                            size_t C, int ldc) except? -1
cpdef int dormqr_bufferSize(size_t handle, int side, int trans,
                            int m, int n, int k, size_t A, int lda, size_t tau,
                            size_t C, int ldc) except? -1
cpdef int cunmqr_bufferSize(size_t handle, int side, int trans,
                            int m, int n, int k, size_t A, int lda, size_t tau,
                            size_t C, int ldc) except? -1
cpdef int zunmqr_bufferSize(size_t handle, int side, int trans,
                            int m, int n, int k, size_t A, int lda, size_t tau,
                            size_t C, int ldc) except? -1

cpdef sormqr(size_t handle, int side, int trans,
             int m, int n, int k, size_t A, int lda, size_t tau, size_t C,
             int ldc, size_t work, int lwork, size_t devInfo)
cpdef dormqr(size_t handle, int side, int trans,
             int m, int n, int k, size_t A, int lda, size_t tau, size_t C,
             int ldc, size_t work, int lwork, size_t devInfo)
cpdef cunmqr(size_t handle, int side, int trans,
             int m, int n, int k, size_t A, int lda, size_t tau, size_t C,
             int ldc, size_t work, int lwork, size_t devInfo)
cpdef zunmqr(size_t handle, int side, int trans,
             int m, int n, int k, size_t A, int lda, size_t tau, size_t C,
             int ldc, size_t work, int lwork, size_t devInfo)
cpdef cormqr(size_t handle, int side, int trans,
             int m, int n, int k, size_t A, int lda, size_t tau, size_t C,
             int ldc, size_t work, int lwork, size_t devInfo)  # (obsoleted)
cpdef zormqr(size_t handle, int side, int trans,
             int m, int n, int k, size_t A, int lda, size_t tau, size_t C,
             int ldc, size_t work, int lwork, size_t devInfo)  # (obsoleted)

# L*D*L**T,U*D*U**T factorization
cpdef int ssytrf_bufferSize(size_t handle, int n, size_t A, int lda) except? -1
cpdef int dsytrf_bufferSize(size_t handle, int n, size_t A, int lda) except? -1
cpdef int csytrf_bufferSize(size_t handle, int n, size_t A, int lda) except? -1
cpdef int zsytrf_bufferSize(size_t handle, int n, size_t A, int lda) except? -1

cpdef ssytrf(size_t handle, int uplo, int n, size_t A, int lda,
             size_t ipiv, size_t work, int lwork, size_t devInfo)
cpdef dsytrf(size_t handle, int uplo, int n, size_t A, int lda,
             size_t ipiv, size_t work, int lwork, size_t devInfo)
cpdef csytrf(size_t handle, int uplo, int n, size_t A, int lda,
             size_t ipiv, size_t work, int lwork, size_t devInfo)
cpdef zsytrf(size_t handle, int uplo, int n, size_t A, int lda,
             size_t ipiv, size_t work, int lwork, size_t devInfo)

###############################################################################
# Dense LAPACK Functions (Eigenvalue Solver)
###############################################################################

# Bidiagonal factorization
cpdef int sgebrd_bufferSize(size_t handle, int m, int n) except? -1
cpdef int dgebrd_bufferSize(size_t handle, int m, int n) except? -1
cpdef int cgebrd_bufferSize(size_t handle, int m, int n) except? -1
cpdef int zgebrd_bufferSize(size_t handle, int m, int n) except? -1

cpdef sgebrd(size_t handle, int m, int n, size_t A, int lda,
             size_t D, size_t E, size_t tauQ, size_t tauP,
             size_t Work, int lwork, size_t devInfo)
cpdef dgebrd(size_t handle, int m, int n, size_t A, int lda,
             size_t D, size_t E, size_t tauQ, size_t tauP,
             size_t Work, int lwork, size_t devInfo)
cpdef cgebrd(size_t handle, int m, int n, size_t A, int lda,
             size_t D, size_t E, size_t tauQ, size_t tauP,
             size_t Work, int lwork, size_t devInfo)
cpdef zgebrd(size_t handle, int m, int n, size_t A, int lda,
             size_t D, size_t E, size_t tauQ, size_t tauP,
             size_t Work, int lwork, size_t devInfo)

# TODO(anaruse): orgbr/ungbr, sytrd/hetrd, orgtr/ungtr, ormtr/unmtr

# Singular value decomposition, A = U * Sigma * V^H
cpdef int sgesvd_bufferSize(size_t handle, int m, int n) except? -1
cpdef int dgesvd_bufferSize(size_t handle, int m, int n) except? -1
cpdef int cgesvd_bufferSize(size_t handle, int m, int n) except? -1
cpdef int zgesvd_bufferSize(size_t handle, int m, int n) except? -1

cpdef sgesvd(size_t handle, char jobu, char jobvt, int m, int n, size_t A,
             int lda, size_t S, size_t U, int ldu, size_t VT, int ldvt,
             size_t Work, int lwork, size_t rwork, size_t devInfo)
cpdef dgesvd(size_t handle, char jobu, char jobvt, int m, int n, size_t A,
             int lda, size_t S, size_t U, int ldu, size_t VT, int ldvt,
             size_t Work, int lwork, size_t rwork, size_t devInfo)
cpdef cgesvd(size_t handle, char jobu, char jobvt, int m, int n, size_t A,
             int lda, size_t S, size_t U, int ldu, size_t VT, int ldvt,
             size_t Work, int lwork, size_t rwork, size_t devInfo)
cpdef zgesvd(size_t handle, char jobu, char jobvt, int m, int n, size_t A,
             int lda, size_t S, size_t U, int ldu, size_t VT, int ldvt,
             size_t Work, int lwork, size_t rwork, size_t devInfo)

# Standard symmetric eigenvalue solver
cpdef int ssyevd_bufferSize(size_t handle, int jobz, int uplo, int n,
                            size_t A, int lda, size_t W) except? -1
cpdef int dsyevd_bufferSize(size_t handle, int jobz, int uplo, int n,
                            size_t A, int lda, size_t W) except? -1
cpdef int cheevd_bufferSize(size_t handle, int jobz, int uplo, int n,
                            size_t A, int lda, size_t W) except? -1
cpdef int zheevd_bufferSize(size_t handle, int jobz, int uplo, int n,
                            size_t A, int lda, size_t W) except? -1

cpdef ssyevd(size_t handle, int jobz, int uplo, int n, size_t A, int lda,
             size_t W, size_t work, int lwork, size_t info)
cpdef dsyevd(size_t handle, int jobz, int uplo, int n, size_t A, int lda,
             size_t W, size_t work, int lwork, size_t info)
cpdef cheevd(size_t handle, int jobz, int uplo, int n, size_t A, int lda,
             size_t W, size_t work, int lwork, size_t info)
cpdef zheevd(size_t handle, int jobz, int uplo, int n, size_t A, int lda,
             size_t W, size_t work, int lwork, size_t info)

# TODO(anaruse); sygvd/hegvd, sygvd/hegvd

###############################################################################
# Sparse LAPACK Functions
###############################################################################

cpdef scsrlsvchol(size_t handle, int m, int nnz, size_t descrA, size_t csrValA,
                  size_t csrRowPtrA, size_t csrColIndA, size_t b, float tol,
                  int reorder, size_t x, size_t singularity)
cpdef dcsrlsvchol(size_t handle, int m, int nnz, size_t descrA, size_t csrValA,
                  size_t csrRowPtrA, size_t csrColIndA, size_t b, double tol,
                  int reorder, size_t x, size_t singularity)
cpdef ccsrlsvchol(size_t handle, int m, int nnz, size_t descrA, size_t csrVal,
                  size_t csrRowPtr, size_t csrColInd, size_t b, float tol,
                  int reorder, size_t x, size_t singularity)
cpdef zcsrlsvchol(size_t handle, int m, int nnz, size_t descrA, size_t csrVal,
                  size_t csrRowPtr, size_t csrColInd, size_t b, double tol,
                  int reorder, size_t x, size_t singularity)

cpdef scsrlsvqr(size_t handle, int m, int nnz, size_t descrA, size_t csrValA,
                size_t csrRowPtrA, size_t csrColIndA, size_t b, float tol,
                int reorder, size_t x, size_t singularity)
cpdef dcsrlsvqr(size_t handle, int m, int nnz, size_t descrA, size_t csrValA,
                size_t csrRowPtrA, size_t csrColIndA, size_t b, double tol,
                int reorder, size_t x, size_t singularity)
cpdef ccsrlsvqr(size_t handle, int m, int nnz, size_t descrA, size_t csrVal,
                size_t csrRowPtr, size_t csrColInd, size_t b, float tol,
                int reorder, size_t x, size_t singularity)
cpdef zcsrlsvqr(size_t handle, int m, int nnz, size_t descrA, size_t csrVal,
                size_t csrRowPtr, size_t csrColInd, size_t b, double tol,
                int reorder, size_t x, size_t singularity)

cpdef scsreigvsi(size_t handle, int m, int nnz, size_t descrA, size_t csrValA,
                 size_t csrRowPtrA, size_t csrColIndA, float mu0,
                 size_t x0, int maxite, float eps, size_t mu, size_t x)
cpdef dcsreigvsi(size_t handle, int m, int nnz, size_t descrA, size_t csrValA,
                 size_t csrRowPtrA, size_t csrColIndA, double mu0,
                 size_t x0, int maxite, double eps, size_t mu, size_t x)
cpdef ccsreigvsi(size_t handle, int m, int nnz, size_t descrA, size_t csrValA,
                 size_t csrRowPtrA, size_t csrColIndA, size_t mu0,
                 size_t x0, int maxite, float eps, size_t mu, size_t x)
cpdef zcsreigvsi(size_t handle, int m, int nnz, size_t descrA, size_t csrValA,
                 size_t csrRowPtrA, size_t csrColIndA, size_t mu0,
                 size_t x0, int maxite, double eps, size_t mu, size_t x)
