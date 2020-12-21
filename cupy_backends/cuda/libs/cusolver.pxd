"""Thin wrapper of CUSOLVER."""
from libc.stdint cimport intptr_t


###############################################################################
# Types
###############################################################################
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

    ctypedef void* GesvdjInfo 'gesvdjInfo_t'
    ctypedef void* SyevjInfo 'syevjInfo_t'

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

cpdef int getProperty(int type) except? -1
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

cpdef spotrfBatched(intptr_t handle, int uplo, int n, size_t Aarray, int lda,
                    size_t infoArray, int batchSize)
cpdef dpotrfBatched(intptr_t handle, int uplo, int n, size_t Aarray, int lda,
                    size_t infoArray, int batchSize)
cpdef cpotrfBatched(intptr_t handle, int uplo, int n, size_t Aarray, int lda,
                    size_t infoArray, int batchSize)
cpdef zpotrfBatched(intptr_t handle, int uplo, int n, size_t Aarray, int lda,
                    size_t infoArray, int batchSize)

cpdef spotrsBatched(intptr_t handle, int uplo, int n, int nrhs, size_t Aarray,
                    int lda, size_t Barray, int ldb, size_t devInfo,
                    int batchSize)
cpdef dpotrsBatched(intptr_t handle, int uplo, int n, int nrhs, size_t Aarray,
                    int lda, size_t Barray, int ldb, size_t devInfo,
                    int batchSize)
cpdef cpotrsBatched(intptr_t handle, int uplo, int n, int nrhs, size_t Aarray,
                    int lda, size_t Barray, int ldb, size_t devInfo,
                    int batchSize)
cpdef zpotrsBatched(intptr_t handle, int uplo, int n, int nrhs, size_t Aarray,
                    int lda, size_t Barray, int ldb, size_t devInfo,
                    int batchSize)

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

# Solve A * X = B using iterative refinement
cpdef size_t zzgesv_bufferSize(intptr_t handle, int n, int nrhs, size_t dA,
                               int ldda, size_t dipiv, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1
cpdef size_t zcgesv_bufferSize(intptr_t handle, int n, int nrhs, size_t dA,
                               int ldda, size_t dipiv, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1
cpdef size_t zygesv_bufferSize(intptr_t handle, int n, int nrhs, size_t dA,
                               int ldda, size_t dipiv, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1
cpdef size_t zkgesv_bufferSize(intptr_t handle, int n, int nrhs, size_t dA,
                               int ldda, size_t dipiv, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1
cpdef size_t ccgesv_bufferSize(intptr_t handle, int n, int nrhs, size_t dA,
                               int ldda, size_t dipiv, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1
cpdef size_t cygesv_bufferSize(intptr_t handle, int n, int nrhs, size_t dA,
                               int ldda, size_t dipiv, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1
cpdef size_t ckgesv_bufferSize(intptr_t handle, int n, int nrhs, size_t dA,
                               int ldda, size_t dipiv, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1
cpdef size_t ddgesv_bufferSize(intptr_t handle, int n, int nrhs, size_t dA,
                               int ldda, size_t dipiv, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1
cpdef size_t dsgesv_bufferSize(intptr_t handle, int n, int nrhs, size_t dA,
                               int ldda, size_t dipiv, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1
cpdef size_t dxgesv_bufferSize(intptr_t handle, int n, int nrhs, size_t dA,
                               int ldda, size_t dipiv, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1
cpdef size_t dhgesv_bufferSize(intptr_t handle, int n, int nrhs, size_t dA,
                               int ldda, size_t dipiv, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1
cpdef size_t ssgesv_bufferSize(intptr_t handle, int n, int nrhs, size_t dA,
                               int ldda, size_t dipiv, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1
cpdef size_t sxgesv_bufferSize(intptr_t handle, int n, int nrhs, size_t dA,
                               int ldda, size_t dipiv, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1
cpdef size_t shgesv_bufferSize(intptr_t handle, int n, int nrhs, size_t dA,
                               int ldda, size_t dipiv, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1

cpdef int zzgesv(intptr_t handle, int n, int nrhs, size_t dA, int ldda,
                 size_t dipiv, size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork_bytes, size_t dInfo)
cpdef int zcgesv(intptr_t handle, int n, int nrhs, size_t dA, int ldda,
                 size_t dipiv, size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork_bytes, size_t dInfo)
cpdef int zygesv(intptr_t handle, int n, int nrhs, size_t dA, int ldda,
                 size_t dipiv, size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork_bytes, size_t dInfo)
cpdef int zkgesv(intptr_t handle, int n, int nrhs, size_t dA, int ldda,
                 size_t dipiv, size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork_bytes, size_t dInfo)
cpdef int ccgesv(intptr_t handle, int n, int nrhs, size_t dA, int ldda,
                 size_t dipiv, size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork_bytes, size_t dInfo)
cpdef int ckgesv(intptr_t handle, int n, int nrhs, size_t dA, int ldda,
                 size_t dipiv, size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork_bytes, size_t dInfo)
cpdef int cygesv(intptr_t handle, int n, int nrhs, size_t dA, int ldda,
                 size_t dipiv, size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork_bytes, size_t dInfo)
cpdef int ddgesv(intptr_t handle, int n, int nrhs, size_t dA, int ldda,
                 size_t dipiv, size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork_bytes, size_t dInfo)
cpdef int dsgesv(intptr_t handle, int n, int nrhs, size_t dA, int ldda,
                 size_t dipiv, size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork_bytes, size_t dInfo)
cpdef int dxgesv(intptr_t handle, int n, int nrhs, size_t dA, int ldda,
                 size_t dipiv, size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork_bytes, size_t dInfo)
cpdef int dhgesv(intptr_t handle, int n, int nrhs, size_t dA, int ldda,
                 size_t dipiv, size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork_bytes, size_t dInfo)
cpdef int ssgesv(intptr_t handle, int n, int nrhs, size_t dA, int ldda,
                 size_t dipiv, size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork_bytes, size_t dInfo)
cpdef int sxgesv(intptr_t handle, int n, int nrhs, size_t dA, int ldda,
                 size_t dipiv, size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork_bytes, size_t dInfo)
cpdef int shgesv(intptr_t handle, int n, int nrhs, size_t dA, int ldda,
                 size_t dipiv, size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork_bytes, size_t dInfo)

# Compute least-saure solution of A * X = B using iterative refinement
cpdef size_t zzgels_bufferSize(intptr_t handle, int m, int n, int nrhs,
                               size_t dA, int ldda, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1
cpdef size_t zcgels_bufferSize(intptr_t handle, int m, int n, int nrhs,
                               size_t dA, int ldda, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1
cpdef size_t zygels_bufferSize(intptr_t handle, int m, int n, int nrhs,
                               size_t dA, int ldda, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1
cpdef size_t zkgels_bufferSize(intptr_t handle, int m, int n, int nrhs,
                               size_t dA, int ldda, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1
cpdef size_t ccgels_bufferSize(intptr_t handle, int m, int n, int nrhs,
                               size_t dA, int ldda, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1
cpdef size_t cygels_bufferSize(intptr_t handle, int m, int n, int nrhs,
                               size_t dA, int ldda, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1
cpdef size_t ckgels_bufferSize(intptr_t handle, int m, int n, int nrhs,
                               size_t dA, int ldda, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1
cpdef size_t ddgels_bufferSize(intptr_t handle, int m, int n, int nrhs,
                               size_t dA, int ldda, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1
cpdef size_t dsgels_bufferSize(intptr_t handle, int m, int n, int nrhs,
                               size_t dA, int ldda, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1
cpdef size_t dxgels_bufferSize(intptr_t handle, int m, int n, int nrhs,
                               size_t dA, int ldda, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1
cpdef size_t dhgels_bufferSize(intptr_t handle, int m, int n, int nrhs,
                               size_t dA, int ldda, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1
cpdef size_t ssgels_bufferSize(intptr_t handle, int m, int n, int nrhs,
                               size_t dA, int ldda, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1
cpdef size_t sxgels_bufferSize(intptr_t handle, int m, int n, int nrhs,
                               size_t dA, int ldda, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1
cpdef size_t shgels_bufferSize(intptr_t handle, int m, int n, int nrhs,
                               size_t dA, int ldda, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1

cpdef int zzgels(intptr_t handle, int m, int n, int nrhs, size_t dA, int ldda,
                 size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork_bytes, size_t dInfo)
cpdef int zcgels(intptr_t handle, int m, int n, int nrhs, size_t dA, int ldda,
                 size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork_bytes, size_t dInfo)
cpdef int zygels(intptr_t handle, int m, int n, int nrhs, size_t dA, int ldda,
                 size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork_bytes, size_t dInfo)
cpdef int zkgels(intptr_t handle, int m, int n, int nrhs, size_t dA, int ldda,
                 size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork_bytes, size_t dInfo)
cpdef int ccgels(intptr_t handle, int m, int n, int nrhs, size_t dA, int ldda,
                 size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork_bytes, size_t dInfo)
cpdef int ckgels(intptr_t handle, int m, int n, int nrhs, size_t dA, int ldda,
                 size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork_bytes, size_t dInfo)
cpdef int cygels(intptr_t handle, int m, int n, int nrhs, size_t dA, int ldda,
                 size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork_bytes, size_t dInfo)
cpdef int ddgels(intptr_t handle, int m, int n, int nrhs, size_t dA, int ldda,
                 size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork_bytes, size_t dInfo)
cpdef int dsgels(intptr_t handle, int m, int n, int nrhs, size_t dA, int ldda,
                 size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork_bytes, size_t dInfo)
cpdef int dxgels(intptr_t handle, int m, int n, int nrhs, size_t dA, int ldda,
                 size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork_bytes, size_t dInfo)
cpdef int dhgels(intptr_t handle, int m, int n, int nrhs, size_t dA, int ldda,
                 size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork_bytes, size_t dInfo)
cpdef int ssgels(intptr_t handle, int m, int n, int nrhs, size_t dA, int ldda,
                 size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork_bytes, size_t dInfo)
cpdef int sxgels(intptr_t handle, int m, int n, int nrhs, size_t dA, int ldda,
                 size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork_bytes, size_t dInfo)
cpdef int shgels(intptr_t handle, int m, int n, int nrhs, size_t dA, int ldda,
                 size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork_bytes, size_t dInfo)

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

# gesvdj ... Singular value decomposition using Jacobi mathod
cpdef intptr_t createGesvdjInfo() except? 0
cpdef destroyGesvdjInfo(intptr_t info)

cpdef xgesvdjSetTolerance(intptr_t info, double tolerance)
cpdef xgesvdjSetMaxSweeps(intptr_t info, int max_sweeps)
cpdef xgesvdjSetSortEig(intptr_t info, int sort_svd)
cpdef double xgesvdjGetResidual(intptr_t handle, intptr_t info)
cpdef int xgesvdjGetSweeps(intptr_t handle, intptr_t info)

cpdef int sgesvdj_bufferSize(intptr_t handle, int jobz, int econ, int m, int n,
                             intptr_t A, int lda, intptr_t S, intptr_t U,
                             int ldu, intptr_t V, int ldv, intptr_t params)
cpdef int dgesvdj_bufferSize(intptr_t handle, int jobz, int econ, int m, int n,
                             intptr_t A, int lda, intptr_t S, intptr_t U,
                             int ldu, intptr_t V, int ldv, intptr_t params)
cpdef int cgesvdj_bufferSize(intptr_t handle, int jobz, int econ, int m, int n,
                             intptr_t A, int lda, intptr_t S, intptr_t U,
                             int ldu, intptr_t V, int ldv, intptr_t params)
cpdef int zgesvdj_bufferSize(intptr_t handle, int jobz, int econ, int m, int n,
                             intptr_t A, int lda, intptr_t S, intptr_t U,
                             int ldu, intptr_t V, int ldv, intptr_t params)

cpdef sgesvdj(intptr_t handle, int jobz, int econ, int m, int n, intptr_t A,
              int lda, intptr_t S, intptr_t U, int ldu, intptr_t V, int ldv,
              intptr_t work, int lwork, intptr_t info, intptr_t params)
cpdef dgesvdj(intptr_t handle, int jobz, int econ, int m, int n, intptr_t A,
              int lda, intptr_t S, intptr_t U, int ldu, intptr_t V, int ldv,
              intptr_t work, int lwork, intptr_t info, intptr_t params)
cpdef cgesvdj(intptr_t handle, int jobz, int econ, int m, int n, intptr_t A,
              int lda, intptr_t S, intptr_t U, int ldu, intptr_t V, int ldv,
              intptr_t work, int lwork, intptr_t info, intptr_t params)
cpdef zgesvdj(intptr_t handle, int jobz, int econ, int m, int n, intptr_t A,
              int lda, intptr_t S, intptr_t U, int ldu, intptr_t V, int ldv,
              intptr_t work, int lwork, intptr_t info, intptr_t params)

cpdef int sgesvdjBatched_bufferSize(
    intptr_t handle, int jobz, int m, int n,
    intptr_t A, int lda, intptr_t S, intptr_t U,
    int ldu, intptr_t V, int ldv, intptr_t params,
    int batchSize) except? -1
cpdef int dgesvdjBatched_bufferSize(
    intptr_t handle, int jobz, int m, int n,
    intptr_t A, int lda, intptr_t S, intptr_t U,
    int ldu, intptr_t V, int ldv, intptr_t params,
    int batchSize) except? -1
cpdef int cgesvdjBatched_bufferSize(
    intptr_t handle, int jobz, int m, int n,
    intptr_t A, int lda, intptr_t S, intptr_t U,
    int ldu, intptr_t V, int ldv, intptr_t params,
    int batchSize) except? -1
cpdef int zgesvdjBatched_bufferSize(
    intptr_t handle, int jobz, int m, int n,
    intptr_t A, int lda, intptr_t S, intptr_t U,
    int ldu, intptr_t V, int ldv, intptr_t params,
    int batchSize) except? -1

cpdef sgesvdjBatched(
    intptr_t handle, int jobz, int m, int n, intptr_t A,
    int lda, intptr_t S, intptr_t U, int ldu, intptr_t V, int ldv,
    intptr_t work, int lwork, intptr_t info, intptr_t params, int batchSize)
cpdef dgesvdjBatched(
    intptr_t handle, int jobz, int m, int n, intptr_t A,
    int lda, intptr_t S, intptr_t U, int ldu, intptr_t V, int ldv,
    intptr_t work, int lwork, intptr_t info, intptr_t params, int batchSize)
cpdef cgesvdjBatched(
    intptr_t handle, int jobz, int m, int n, intptr_t A,
    int lda, intptr_t S, intptr_t U, int ldu, intptr_t V, int ldv,
    intptr_t work, int lwork, intptr_t info, intptr_t params, int batchSize)
cpdef zgesvdjBatched(
    intptr_t handle, int jobz, int m, int n, intptr_t A,
    int lda, intptr_t S, intptr_t U, int ldu, intptr_t V, int ldv,
    intptr_t work, int lwork, intptr_t info, intptr_t params, int batchSize)

# gesvda ... Approximate singular value decomposition
cpdef int sgesvdaStridedBatched_bufferSize(
    intptr_t handle, int jobz, int rank, int m, int n, intptr_t d_A,
    int lda, long long int strideA, intptr_t d_S, long long int strideS,
    intptr_t d_U, int ldu, long long int strideU, intptr_t d_V, int ldv,
    long long int strideV, int batchSize)
cpdef int dgesvdaStridedBatched_bufferSize(
    intptr_t handle, int jobz, int rank, int m, int n, intptr_t d_A,
    int lda, long long int strideA, intptr_t d_S, long long int strideS,
    intptr_t d_U, int ldu, long long int strideU, intptr_t d_V, int ldv,
    long long int strideV, int batchSize)
cpdef int cgesvdaStridedBatched_bufferSize(
    intptr_t handle, int jobz, int rank, int m, int n, intptr_t d_A,
    int lda, long long int strideA, intptr_t d_S, long long int strideS,
    intptr_t d_U, int ldu, long long int strideU, intptr_t d_V, int ldv,
    long long int strideV, int batchSize)
cpdef int zgesvdaStridedBatched_bufferSize(
    intptr_t handle, int jobz, int rank, int m, int n, intptr_t d_A,
    int lda, long long int strideA, intptr_t d_S, long long int strideS,
    intptr_t d_U, int ldu, long long int strideU, intptr_t d_V, int ldv,
    long long int strideV, int batchSize)

cpdef sgesvdaStridedBatched(
    intptr_t handle, int jobz, int rank, int m, int n, intptr_t d_A,
    int lda, long long int strideA, intptr_t d_S, long long int strideS,
    intptr_t d_U, int ldu, long long int strideU, intptr_t d_V, int ldv,
    long long int strideV, intptr_t d_work, int lwork, intptr_t d_info,
    intptr_t h_R_nrmF, int batchSize)
cpdef dgesvdaStridedBatched(
    intptr_t handle, int jobz, int rank, int m, int n, intptr_t d_A,
    int lda, long long int strideA, intptr_t d_S, long long int strideS,
    intptr_t d_U, int ldu, long long int strideU, intptr_t d_V, int ldv,
    long long int strideV, intptr_t d_work, int lwork, intptr_t d_info,
    intptr_t h_R_nrmF, int batchSize)
cpdef cgesvdaStridedBatched(
    intptr_t handle, int jobz, int rank, int m, int n, intptr_t d_A,
    int lda, long long int strideA, intptr_t d_S, long long int strideS,
    intptr_t d_U, int ldu, long long int strideU, intptr_t d_V, int ldv,
    long long int strideV, intptr_t d_work, int lwork, intptr_t d_info,
    intptr_t h_R_nrmF, int batchSize)
cpdef zgesvdaStridedBatched(
    intptr_t handle, int jobz, int rank, int m, int n, intptr_t d_A,
    int lda, long long int strideA, intptr_t d_S, long long int strideS,
    intptr_t d_U, int ldu, long long int strideU, intptr_t d_V, int ldv,
    long long int strideV, intptr_t d_work, int lwork, intptr_t d_info,
    intptr_t h_R_nrmF, int batchSize)

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

# syevj ... Symmetric eigenvalue solver via Jacobi method
cpdef intptr_t createSyevjInfo() except? 0
cpdef destroySyevjInfo(intptr_t info)

cpdef xsyevjSetTolerance(intptr_t info, double tolerance)
cpdef xsyevjSetMaxSweeps(intptr_t info, int max_sweeps)
cpdef xsyevjSetSortEig(intptr_t info, int sort_eig)
cpdef double xsyevjGetResidual(intptr_t handle, intptr_t info)
cpdef int xsyevjGetSweeps(intptr_t handle, intptr_t info)

cpdef int ssyevj_bufferSize(
    intptr_t handle, int jobz, int uplo, int n,
    size_t A, int lda, size_t W, intptr_t params) except? -1
cpdef int dsyevj_bufferSize(
    intptr_t handle, int jobz, int uplo, int n,
    size_t A, int lda, size_t W, intptr_t params) except? -1
cpdef int cheevj_bufferSize(
    intptr_t handle, int jobz, int uplo, int n,
    size_t A, int lda, size_t W, intptr_t params) except? -1
cpdef int zheevj_bufferSize(
    intptr_t handle, int jobz, int uplo, int n,
    size_t A, int lda, size_t W, intptr_t params) except? -1
cpdef ssyevj(intptr_t handle, int jobz, int uplo, int n, size_t A, int lda,
             size_t W, size_t work, int lwork, size_t info, intptr_t params)
cpdef dsyevj(intptr_t handle, int jobz, int uplo, int n, size_t A, int lda,
             size_t W, size_t work, int lwork, size_t info, intptr_t params)
cpdef cheevj(intptr_t handle, int jobz, int uplo, int n, size_t A, int lda,
             size_t W, size_t work, int lwork, size_t info, intptr_t params)
cpdef zheevj(intptr_t handle, int jobz, int uplo, int n, size_t A, int lda,
             size_t W, size_t work, int lwork, size_t info, intptr_t params)

cpdef int ssyevjBatched_bufferSize(
    intptr_t handle, int jobz, int uplo, int n,
    size_t A, int lda, size_t W, intptr_t params, int batchSize) except? -1
cpdef int dsyevjBatched_bufferSize(
    intptr_t handle, int jobz, int uplo, int n,
    size_t A, int lda, size_t W, intptr_t params, int batchSize) except? -1
cpdef int cheevjBatched_bufferSize(
    intptr_t handle, int jobz, int uplo, int n,
    size_t A, int lda, size_t W, intptr_t params, int batchSize) except? -1
cpdef int zheevjBatched_bufferSize(
    intptr_t handle, int jobz, int uplo, int n,
    size_t A, int lda, size_t W, intptr_t params, int batchSize) except? -1
cpdef ssyevjBatched(
    intptr_t handle, int jobz, int uplo, int n, size_t A, int lda,
    size_t W, size_t work, int lwork, size_t info, intptr_t params,
    int batchSize)
cpdef dsyevjBatched(
    intptr_t handle, int jobz, int uplo, int n, size_t A, int lda,
    size_t W, size_t work, int lwork, size_t info, intptr_t params,
    int batchSize)
cpdef cheevjBatched(
    intptr_t handle, int jobz, int uplo, int n, size_t A, int lda,
    size_t W, size_t work, int lwork, size_t info, intptr_t params,
    int batchSize)
cpdef zheevjBatched(
    intptr_t handle, int jobz, int uplo, int n, size_t A, int lda,
    size_t W, size_t work, int lwork, size_t info, intptr_t params,
    int batchSize)

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
