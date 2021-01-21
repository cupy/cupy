# This code was automatically generated. Do not modify it directly.

from libc.stdint cimport intptr_t


########################################
# Types

cdef extern from *:
    ctypedef int DataType 'cudaDataType'

    ctypedef int Operation 'cublasOperation_t'
    ctypedef int SideMode 'cublasSideMode_t'
    ctypedef int FillMode 'cublasFillMode_t'

    ctypedef void* MatDescr 'cusparseMatDescr_t'

cdef extern from *:
    ctypedef void* DnHandle 'cusolverDnHandle_t'
    ctypedef void* syevjInfo_t 'syevjInfo_t'
    ctypedef void* gesvdjInfo_t 'gesvdjInfo_t'
    ctypedef void* DnIRSParams 'cusolverDnIRSParams_t'
    ctypedef void* DnIRSInfos 'cusolverDnIRSInfos_t'
    ctypedef void* DnParams 'cusolverDnParams_t'
    ctypedef void* SpHandle 'cusolverSpHandle_t'
    ctypedef void* csrqrInfo_t 'csrqrInfo_t'


########################################
# Enumerators

cdef extern from *:
    ctypedef int DnFunction 'cusolverDnFunction_t'
    ctypedef int Status 'cusolverStatus_t'
    ctypedef int EigType 'cusolverEigType_t'
    ctypedef int EigMode 'cusolverEigMode_t'
    ctypedef int EigRange 'cusolverEigRange_t'
    ctypedef int Norm 'cusolverNorm_t'
    ctypedef int IRSRefinement 'cusolverIRSRefinement_t'
    ctypedef int PrecType 'cusolverPrecType_t'
    ctypedef int AlgMode 'cusolverAlgMode_t'
    ctypedef int StorevMode 'cusolverStorevMode_t'
    ctypedef int DirectMode 'cusolverDirectMode_t'

cpdef enum:
    CUSOLVERDN_GETRF = 0

cpdef enum:
    CUSOLVER_STATUS_SUCCESS = 0
    CUSOLVER_STATUS_NOT_INITIALIZED = 1
    CUSOLVER_STATUS_ALLOC_FAILED = 2
    CUSOLVER_STATUS_INVALID_VALUE = 3
    CUSOLVER_STATUS_ARCH_MISMATCH = 4
    CUSOLVER_STATUS_MAPPING_ERROR = 5
    CUSOLVER_STATUS_EXECUTION_FAILED = 6
    CUSOLVER_STATUS_INTERNAL_ERROR = 7
    CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED = 8
    CUSOLVER_STATUS_NOT_SUPPORTED = 9
    CUSOLVER_STATUS_ZERO_PIVOT = 10
    CUSOLVER_STATUS_INVALID_LICENSE = 11
    CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED = 12
    CUSOLVER_STATUS_IRS_PARAMS_INVALID = 13
    CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC = 14
    CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE = 15
    CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER = 16
    CUSOLVER_STATUS_IRS_INTERNAL_ERROR = 20
    CUSOLVER_STATUS_IRS_NOT_SUPPORTED = 21
    CUSOLVER_STATUS_IRS_OUT_OF_RANGE = 22
    CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES = 23
    CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED = 25
    CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED = 26
    CUSOLVER_STATUS_IRS_MATRIX_SINGULAR = 30
    CUSOLVER_STATUS_INVALID_WORKSPACE = 31

cpdef enum:
    CUSOLVER_EIG_TYPE_1 = 1
    CUSOLVER_EIG_TYPE_2 = 2
    CUSOLVER_EIG_TYPE_3 = 3

cpdef enum:
    CUSOLVER_EIG_MODE_NOVECTOR = 0
    CUSOLVER_EIG_MODE_VECTOR = 1

cpdef enum:
    CUSOLVER_EIG_RANGE_ALL = 1001
    CUSOLVER_EIG_RANGE_I = 1002
    CUSOLVER_EIG_RANGE_V = 1003

cpdef enum:
    CUSOLVER_INF_NORM = 104
    CUSOLVER_MAX_NORM = 105
    CUSOLVER_ONE_NORM = 106
    CUSOLVER_FRO_NORM = 107

cpdef enum:
    CUSOLVER_IRS_REFINE_NOT_SET = 1100
    CUSOLVER_IRS_REFINE_NONE = 1101
    CUSOLVER_IRS_REFINE_CLASSICAL = 1102
    CUSOLVER_IRS_REFINE_CLASSICAL_GMRES = 1103
    CUSOLVER_IRS_REFINE_GMRES = 1104
    CUSOLVER_IRS_REFINE_GMRES_GMRES = 1105
    CUSOLVER_IRS_REFINE_GMRES_NOPCOND = 1106
    CUSOLVER_PREC_DD = 1150
    CUSOLVER_PREC_SS = 1151
    CUSOLVER_PREC_SHT = 1152

cpdef enum:
    CUSOLVER_R_8I = 1201
    CUSOLVER_R_8U = 1202
    CUSOLVER_R_64F = 1203
    CUSOLVER_R_32F = 1204
    CUSOLVER_R_16F = 1205
    CUSOLVER_R_16BF = 1206
    CUSOLVER_R_TF32 = 1207
    CUSOLVER_R_AP = 1208
    CUSOLVER_C_8I = 1211
    CUSOLVER_C_8U = 1212
    CUSOLVER_C_64F = 1213
    CUSOLVER_C_32F = 1214
    CUSOLVER_C_16F = 1215
    CUSOLVER_C_16BF = 1216
    CUSOLVER_C_TF32 = 1217
    CUSOLVER_C_AP = 1218

cpdef enum:
    CUSOLVER_ALG_0 = 0
    CUSOLVER_ALG_1 = 1

cpdef enum:
    CUBLAS_STOREV_COLUMNWISE = 0
    CUBLAS_STOREV_ROWWISE = 1

cpdef enum:
    CUBLAS_DIRECT_FORWARD = 0
    CUBLAS_DIRECT_BACKWARD = 1


########################################
# Library Attributes

cpdef int getProperty(int type) except? -1


##################################################
# cuSOLVER Dense LAPACK Function - Helper Function

cpdef intptr_t create() except? 0

cpdef destroy(intptr_t handle)

cpdef setStream(intptr_t handle, size_t streamId)

cpdef size_t getStream(intptr_t handle) except? 0

cpdef size_t createSyevjInfo() except? 0

cpdef destroySyevjInfo(size_t info)

cpdef xsyevjSetTolerance(size_t info, double tolerance)

cpdef xsyevjSetMaxSweeps(size_t info, int max_sweeps)

cpdef xsyevjSetSortEig(size_t info, int sort_eig)

cpdef double xsyevjGetResidual(intptr_t handle, size_t info) except? 0

cpdef int xsyevjGetSweeps(intptr_t handle, size_t info) except? 0

cpdef size_t createGesvdjInfo() except? 0

cpdef destroyGesvdjInfo(size_t info)

cpdef xgesvdjSetTolerance(size_t info, double tolerance)

cpdef xgesvdjSetMaxSweeps(size_t info, int max_sweeps)

cpdef xgesvdjSetSortEig(size_t info, int sort_svd)

cpdef double xgesvdjGetResidual(intptr_t handle, size_t info) except? 0

cpdef int xgesvdjGetSweeps(intptr_t handle, size_t info) except? 0


######################################################
# cuSOLVER Dense LAPACK Function - Dense Linear Solver

cpdef int spotrf_bufferSize(intptr_t handle, int uplo, int n, intptr_t A, int lda) except? 0
cpdef int dpotrf_bufferSize(intptr_t handle, int uplo, int n, intptr_t A, int lda) except? 0
cpdef int cpotrf_bufferSize(intptr_t handle, int uplo, int n, intptr_t A, int lda) except? 0
cpdef int zpotrf_bufferSize(intptr_t handle, int uplo, int n, intptr_t A, int lda) except? 0

cpdef spotrf(intptr_t handle, int uplo, int n, intptr_t A, int lda, intptr_t Workspace, int Lwork, intptr_t devInfo)
cpdef dpotrf(intptr_t handle, int uplo, int n, intptr_t A, int lda, intptr_t Workspace, int Lwork, intptr_t devInfo)
cpdef cpotrf(intptr_t handle, int uplo, int n, intptr_t A, int lda, intptr_t Workspace, int Lwork, intptr_t devInfo)
cpdef zpotrf(intptr_t handle, int uplo, int n, intptr_t A, int lda, intptr_t Workspace, int Lwork, intptr_t devInfo)

cpdef spotrs(intptr_t handle, int uplo, int n, int nrhs, intptr_t A, int lda, intptr_t B, int ldb, intptr_t devInfo)
cpdef dpotrs(intptr_t handle, int uplo, int n, int nrhs, intptr_t A, int lda, intptr_t B, int ldb, intptr_t devInfo)
cpdef cpotrs(intptr_t handle, int uplo, int n, int nrhs, intptr_t A, int lda, intptr_t B, int ldb, intptr_t devInfo)
cpdef zpotrs(intptr_t handle, int uplo, int n, int nrhs, intptr_t A, int lda, intptr_t B, int ldb, intptr_t devInfo)

cpdef spotrfBatched(intptr_t handle, int uplo, int n, intptr_t Aarray, int lda, intptr_t infoArray, int batchSize)
cpdef dpotrfBatched(intptr_t handle, int uplo, int n, intptr_t Aarray, int lda, intptr_t infoArray, int batchSize)
cpdef cpotrfBatched(intptr_t handle, int uplo, int n, intptr_t Aarray, int lda, intptr_t infoArray, int batchSize)
cpdef zpotrfBatched(intptr_t handle, int uplo, int n, intptr_t Aarray, int lda, intptr_t infoArray, int batchSize)

cpdef spotrsBatched(intptr_t handle, int uplo, int n, int nrhs, intptr_t A, int lda, intptr_t B, int ldb, intptr_t d_info, int batchSize)
cpdef dpotrsBatched(intptr_t handle, int uplo, int n, int nrhs, intptr_t A, int lda, intptr_t B, int ldb, intptr_t d_info, int batchSize)
cpdef cpotrsBatched(intptr_t handle, int uplo, int n, int nrhs, intptr_t A, int lda, intptr_t B, int ldb, intptr_t d_info, int batchSize)
cpdef zpotrsBatched(intptr_t handle, int uplo, int n, int nrhs, intptr_t A, int lda, intptr_t B, int ldb, intptr_t d_info, int batchSize)

cpdef int sgetrf_bufferSize(intptr_t handle, int m, int n, intptr_t A, int lda) except? 0
cpdef int dgetrf_bufferSize(intptr_t handle, int m, int n, intptr_t A, int lda) except? 0
cpdef int cgetrf_bufferSize(intptr_t handle, int m, int n, intptr_t A, int lda) except? 0
cpdef int zgetrf_bufferSize(intptr_t handle, int m, int n, intptr_t A, int lda) except? 0

cpdef sgetrf(intptr_t handle, int m, int n, intptr_t A, int lda, intptr_t Workspace, intptr_t devIpiv, intptr_t devInfo)
cpdef dgetrf(intptr_t handle, int m, int n, intptr_t A, int lda, intptr_t Workspace, intptr_t devIpiv, intptr_t devInfo)
cpdef cgetrf(intptr_t handle, int m, int n, intptr_t A, int lda, intptr_t Workspace, intptr_t devIpiv, intptr_t devInfo)
cpdef zgetrf(intptr_t handle, int m, int n, intptr_t A, int lda, intptr_t Workspace, intptr_t devIpiv, intptr_t devInfo)

cpdef sgetrs(intptr_t handle, int trans, int n, int nrhs, intptr_t A, int lda, intptr_t devIpiv, intptr_t B, int ldb, intptr_t devInfo)
cpdef dgetrs(intptr_t handle, int trans, int n, int nrhs, intptr_t A, int lda, intptr_t devIpiv, intptr_t B, int ldb, intptr_t devInfo)
cpdef cgetrs(intptr_t handle, int trans, int n, int nrhs, intptr_t A, int lda, intptr_t devIpiv, intptr_t B, int ldb, intptr_t devInfo)
cpdef zgetrs(intptr_t handle, int trans, int n, int nrhs, intptr_t A, int lda, intptr_t devIpiv, intptr_t B, int ldb, intptr_t devInfo)

cpdef int sgeqrf_bufferSize(intptr_t handle, int m, int n, intptr_t A, int lda) except? 0
cpdef int dgeqrf_bufferSize(intptr_t handle, int m, int n, intptr_t A, int lda) except? 0
cpdef int cgeqrf_bufferSize(intptr_t handle, int m, int n, intptr_t A, int lda) except? 0
cpdef int zgeqrf_bufferSize(intptr_t handle, int m, int n, intptr_t A, int lda) except? 0

cpdef sgeqrf(intptr_t handle, int m, int n, intptr_t A, int lda, intptr_t TAU, intptr_t Workspace, int Lwork, intptr_t devInfo)
cpdef dgeqrf(intptr_t handle, int m, int n, intptr_t A, int lda, intptr_t TAU, intptr_t Workspace, int Lwork, intptr_t devInfo)
cpdef cgeqrf(intptr_t handle, int m, int n, intptr_t A, int lda, intptr_t TAU, intptr_t Workspace, int Lwork, intptr_t devInfo)
cpdef zgeqrf(intptr_t handle, int m, int n, intptr_t A, int lda, intptr_t TAU, intptr_t Workspace, int Lwork, intptr_t devInfo)

cpdef int sorgqr_bufferSize(intptr_t handle, int m, int n, int k, intptr_t A, int lda, intptr_t tau) except? 0
cpdef int dorgqr_bufferSize(intptr_t handle, int m, int n, int k, intptr_t A, int lda, intptr_t tau) except? 0
cpdef int cungqr_bufferSize(intptr_t handle, int m, int n, int k, intptr_t A, int lda, intptr_t tau) except? 0
cpdef int zungqr_bufferSize(intptr_t handle, int m, int n, int k, intptr_t A, int lda, intptr_t tau) except? 0

cpdef sorgqr(intptr_t handle, int m, int n, int k, intptr_t A, int lda, intptr_t tau, intptr_t work, int lwork, intptr_t info)
cpdef dorgqr(intptr_t handle, int m, int n, int k, intptr_t A, int lda, intptr_t tau, intptr_t work, int lwork, intptr_t info)
cpdef cungqr(intptr_t handle, int m, int n, int k, intptr_t A, int lda, intptr_t tau, intptr_t work, int lwork, intptr_t info)
cpdef zungqr(intptr_t handle, int m, int n, int k, intptr_t A, int lda, intptr_t tau, intptr_t work, int lwork, intptr_t info)

cpdef int sormqr_bufferSize(intptr_t handle, int side, int trans, int m, int n, int k, intptr_t A, int lda, intptr_t tau, intptr_t C, int ldc) except? 0
cpdef int dormqr_bufferSize(intptr_t handle, int side, int trans, int m, int n, int k, intptr_t A, int lda, intptr_t tau, intptr_t C, int ldc) except? 0
cpdef int cunmqr_bufferSize(intptr_t handle, int side, int trans, int m, int n, int k, intptr_t A, int lda, intptr_t tau, intptr_t C, int ldc) except? 0
cpdef int zunmqr_bufferSize(intptr_t handle, int side, int trans, int m, int n, int k, intptr_t A, int lda, intptr_t tau, intptr_t C, int ldc) except? 0

cpdef sormqr(intptr_t handle, int side, int trans, int m, int n, int k, intptr_t A, int lda, intptr_t tau, intptr_t C, int ldc, intptr_t work, int lwork, intptr_t devInfo)
cpdef dormqr(intptr_t handle, int side, int trans, int m, int n, int k, intptr_t A, int lda, intptr_t tau, intptr_t C, int ldc, intptr_t work, int lwork, intptr_t devInfo)
cpdef cunmqr(intptr_t handle, int side, int trans, int m, int n, int k, intptr_t A, int lda, intptr_t tau, intptr_t C, int ldc, intptr_t work, int lwork, intptr_t devInfo)
cpdef zunmqr(intptr_t handle, int side, int trans, int m, int n, int k, intptr_t A, int lda, intptr_t tau, intptr_t C, int ldc, intptr_t work, int lwork, intptr_t devInfo)

cpdef int ssytrf_bufferSize(intptr_t handle, int n, intptr_t A, int lda) except? 0
cpdef int dsytrf_bufferSize(intptr_t handle, int n, intptr_t A, int lda) except? 0
cpdef int csytrf_bufferSize(intptr_t handle, int n, intptr_t A, int lda) except? 0
cpdef int zsytrf_bufferSize(intptr_t handle, int n, intptr_t A, int lda) except? 0

cpdef ssytrf(intptr_t handle, int uplo, int n, intptr_t A, int lda, intptr_t ipiv, intptr_t work, int lwork, intptr_t info)
cpdef dsytrf(intptr_t handle, int uplo, int n, intptr_t A, int lda, intptr_t ipiv, intptr_t work, int lwork, intptr_t info)
cpdef csytrf(intptr_t handle, int uplo, int n, intptr_t A, int lda, intptr_t ipiv, intptr_t work, int lwork, intptr_t info)
cpdef zsytrf(intptr_t handle, int uplo, int n, intptr_t A, int lda, intptr_t ipiv, intptr_t work, int lwork, intptr_t info)

cpdef size_t zzgesv_bufferSize(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0
cpdef size_t zcgesv_bufferSize(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0
cpdef size_t zkgesv_bufferSize(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0
cpdef size_t zegesv_bufferSize(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0
cpdef size_t zygesv_bufferSize(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0
cpdef size_t ccgesv_bufferSize(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0
cpdef size_t ckgesv_bufferSize(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0
cpdef size_t cegesv_bufferSize(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0
cpdef size_t cygesv_bufferSize(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0
cpdef size_t ddgesv_bufferSize(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0
cpdef size_t dsgesv_bufferSize(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0
cpdef size_t dhgesv_bufferSize(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0
cpdef size_t dbgesv_bufferSize(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0
cpdef size_t dxgesv_bufferSize(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0
cpdef size_t ssgesv_bufferSize(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0
cpdef size_t shgesv_bufferSize(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0
cpdef size_t sbgesv_bufferSize(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0
cpdef size_t sxgesv_bufferSize(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0

cpdef int zzgesv(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t d_info) except? 0
cpdef int zcgesv(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t d_info) except? 0
cpdef int zkgesv(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t d_info) except? 0
cpdef int zegesv(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t d_info) except? 0
cpdef int zygesv(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t d_info) except? 0
cpdef int ccgesv(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t d_info) except? 0
cpdef int cegesv(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t d_info) except? 0
cpdef int ckgesv(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t d_info) except? 0
cpdef int cygesv(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t d_info) except? 0
cpdef int ddgesv(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t d_info) except? 0
cpdef int dsgesv(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t d_info) except? 0
cpdef int dhgesv(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t d_info) except? 0
cpdef int dbgesv(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t d_info) except? 0
cpdef int dxgesv(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t d_info) except? 0
cpdef int ssgesv(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t d_info) except? 0
cpdef int shgesv(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t d_info) except? 0
cpdef int sbgesv(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t d_info) except? 0
cpdef int sxgesv(intptr_t handle, int n, int nrhs, intptr_t dA, int ldda, intptr_t dipiv, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t d_info) except? 0

cpdef size_t zzgels_bufferSize(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0
cpdef size_t zcgels_bufferSize(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0
cpdef size_t zkgels_bufferSize(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0
cpdef size_t zegels_bufferSize(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0
cpdef size_t zygels_bufferSize(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0
cpdef size_t ccgels_bufferSize(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0
cpdef size_t ckgels_bufferSize(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0
cpdef size_t cegels_bufferSize(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0
cpdef size_t cygels_bufferSize(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0
cpdef size_t ddgels_bufferSize(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0
cpdef size_t dsgels_bufferSize(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0
cpdef size_t dhgels_bufferSize(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0
cpdef size_t dbgels_bufferSize(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0
cpdef size_t dxgels_bufferSize(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0
cpdef size_t ssgels_bufferSize(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0
cpdef size_t shgels_bufferSize(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0
cpdef size_t sbgels_bufferSize(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0
cpdef size_t sxgels_bufferSize(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace) except? 0

cpdef int zzgels(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t d_info) except? 0
cpdef int zcgels(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t d_info) except? 0
cpdef int zkgels(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t d_info) except? 0
cpdef int zegels(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t d_info) except? 0
cpdef int zygels(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t d_info) except? 0
cpdef int ccgels(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t d_info) except? 0
cpdef int ckgels(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t d_info) except? 0
cpdef int cegels(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t d_info) except? 0
cpdef int cygels(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t d_info) except? 0
cpdef int ddgels(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t d_info) except? 0
cpdef int dsgels(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t d_info) except? 0
cpdef int dhgels(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t d_info) except? 0
cpdef int dbgels(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t d_info) except? 0
cpdef int dxgels(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t d_info) except? 0
cpdef int ssgels(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t d_info) except? 0
cpdef int shgels(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t d_info) except? 0
cpdef int sbgels(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t d_info) except? 0
cpdef int sxgels(intptr_t handle, int m, int n, int nrhs, intptr_t dA, int ldda, intptr_t dB, int lddb, intptr_t dX, int lddx, intptr_t dWorkspace, size_t lwork_bytes, intptr_t d_info) except? 0


##########################################################
# cuSOLVER Dense LAPACK Function - Dense Eigenvalue Solver

cpdef int sgebrd_bufferSize(intptr_t handle, int m, int n) except? 0
cpdef int dgebrd_bufferSize(intptr_t handle, int m, int n) except? 0
cpdef int cgebrd_bufferSize(intptr_t handle, int m, int n) except? 0
cpdef int zgebrd_bufferSize(intptr_t handle, int m, int n) except? 0

cpdef sgebrd(intptr_t handle, int m, int n, intptr_t A, int lda, intptr_t D, intptr_t E, intptr_t TAUQ, intptr_t TAUP, intptr_t Work, int Lwork, intptr_t devInfo)
cpdef dgebrd(intptr_t handle, int m, int n, intptr_t A, int lda, intptr_t D, intptr_t E, intptr_t TAUQ, intptr_t TAUP, intptr_t Work, int Lwork, intptr_t devInfo)
cpdef cgebrd(intptr_t handle, int m, int n, intptr_t A, int lda, intptr_t D, intptr_t E, intptr_t TAUQ, intptr_t TAUP, intptr_t Work, int Lwork, intptr_t devInfo)
cpdef zgebrd(intptr_t handle, int m, int n, intptr_t A, int lda, intptr_t D, intptr_t E, intptr_t TAUQ, intptr_t TAUP, intptr_t Work, int Lwork, intptr_t devInfo)

cpdef int sgesvd_bufferSize(intptr_t handle, int m, int n) except? 0
cpdef int dgesvd_bufferSize(intptr_t handle, int m, int n) except? 0
cpdef int cgesvd_bufferSize(intptr_t handle, int m, int n) except? 0
cpdef int zgesvd_bufferSize(intptr_t handle, int m, int n) except? 0

cpdef sgesvd(intptr_t handle, signed char jobu, signed char jobvt, int m, int n, intptr_t A, int lda, intptr_t S, intptr_t U, int ldu, intptr_t VT, int ldvt, intptr_t work, int lwork, intptr_t rwork, intptr_t info)
cpdef dgesvd(intptr_t handle, signed char jobu, signed char jobvt, int m, int n, intptr_t A, int lda, intptr_t S, intptr_t U, int ldu, intptr_t VT, int ldvt, intptr_t work, int lwork, intptr_t rwork, intptr_t info)
cpdef cgesvd(intptr_t handle, signed char jobu, signed char jobvt, int m, int n, intptr_t A, int lda, intptr_t S, intptr_t U, int ldu, intptr_t VT, int ldvt, intptr_t work, int lwork, intptr_t rwork, intptr_t info)
cpdef zgesvd(intptr_t handle, signed char jobu, signed char jobvt, int m, int n, intptr_t A, int lda, intptr_t S, intptr_t U, int ldu, intptr_t VT, int ldvt, intptr_t work, int lwork, intptr_t rwork, intptr_t info)

cpdef int sgesvdj_bufferSize(intptr_t handle, int jobz, int econ, int m, int n, intptr_t A, int lda, intptr_t S, intptr_t U, int ldu, intptr_t V, int ldv, size_t params) except? 0
cpdef int dgesvdj_bufferSize(intptr_t handle, int jobz, int econ, int m, int n, intptr_t A, int lda, intptr_t S, intptr_t U, int ldu, intptr_t V, int ldv, size_t params) except? 0
cpdef int cgesvdj_bufferSize(intptr_t handle, int jobz, int econ, int m, int n, intptr_t A, int lda, intptr_t S, intptr_t U, int ldu, intptr_t V, int ldv, size_t params) except? 0
cpdef int zgesvdj_bufferSize(intptr_t handle, int jobz, int econ, int m, int n, intptr_t A, int lda, intptr_t S, intptr_t U, int ldu, intptr_t V, int ldv, size_t params) except? 0

cpdef sgesvdj(intptr_t handle, int jobz, int econ, int m, int n, intptr_t A, int lda, intptr_t S, intptr_t U, int ldu, intptr_t V, int ldv, intptr_t work, int lwork, intptr_t info, size_t params)
cpdef dgesvdj(intptr_t handle, int jobz, int econ, int m, int n, intptr_t A, int lda, intptr_t S, intptr_t U, int ldu, intptr_t V, int ldv, intptr_t work, int lwork, intptr_t info, size_t params)
cpdef cgesvdj(intptr_t handle, int jobz, int econ, int m, int n, intptr_t A, int lda, intptr_t S, intptr_t U, int ldu, intptr_t V, int ldv, intptr_t work, int lwork, intptr_t info, size_t params)
cpdef zgesvdj(intptr_t handle, int jobz, int econ, int m, int n, intptr_t A, int lda, intptr_t S, intptr_t U, int ldu, intptr_t V, int ldv, intptr_t work, int lwork, intptr_t info, size_t params)

cpdef int sgesvdjBatched_bufferSize(intptr_t handle, int jobz, int m, int n, intptr_t A, int lda, intptr_t S, intptr_t U, int ldu, intptr_t V, int ldv, size_t params, int batchSize) except? 0
cpdef int dgesvdjBatched_bufferSize(intptr_t handle, int jobz, int m, int n, intptr_t A, int lda, intptr_t S, intptr_t U, int ldu, intptr_t V, int ldv, size_t params, int batchSize) except? 0
cpdef int cgesvdjBatched_bufferSize(intptr_t handle, int jobz, int m, int n, intptr_t A, int lda, intptr_t S, intptr_t U, int ldu, intptr_t V, int ldv, size_t params, int batchSize) except? 0
cpdef int zgesvdjBatched_bufferSize(intptr_t handle, int jobz, int m, int n, intptr_t A, int lda, intptr_t S, intptr_t U, int ldu, intptr_t V, int ldv, size_t params, int batchSize) except? 0

cpdef sgesvdjBatched(intptr_t handle, int jobz, int m, int n, intptr_t A, int lda, intptr_t S, intptr_t U, int ldu, intptr_t V, int ldv, intptr_t work, int lwork, intptr_t info, size_t params, int batchSize)
cpdef dgesvdjBatched(intptr_t handle, int jobz, int m, int n, intptr_t A, int lda, intptr_t S, intptr_t U, int ldu, intptr_t V, int ldv, intptr_t work, int lwork, intptr_t info, size_t params, int batchSize)
cpdef cgesvdjBatched(intptr_t handle, int jobz, int m, int n, intptr_t A, int lda, intptr_t S, intptr_t U, int ldu, intptr_t V, int ldv, intptr_t work, int lwork, intptr_t info, size_t params, int batchSize)
cpdef zgesvdjBatched(intptr_t handle, int jobz, int m, int n, intptr_t A, int lda, intptr_t S, intptr_t U, int ldu, intptr_t V, int ldv, intptr_t work, int lwork, intptr_t info, size_t params, int batchSize)

cpdef int sgesvdaStridedBatched_bufferSize(intptr_t handle, int jobz, int rank, int m, int n, intptr_t d_A, int lda, long long int strideA, intptr_t d_S, long long int strideS, intptr_t d_U, int ldu, long long int strideU, intptr_t d_V, int ldv, long long int strideV, int batchSize) except? 0
cpdef int dgesvdaStridedBatched_bufferSize(intptr_t handle, int jobz, int rank, int m, int n, intptr_t d_A, int lda, long long int strideA, intptr_t d_S, long long int strideS, intptr_t d_U, int ldu, long long int strideU, intptr_t d_V, int ldv, long long int strideV, int batchSize) except? 0
cpdef int cgesvdaStridedBatched_bufferSize(intptr_t handle, int jobz, int rank, int m, int n, intptr_t d_A, int lda, long long int strideA, intptr_t d_S, long long int strideS, intptr_t d_U, int ldu, long long int strideU, intptr_t d_V, int ldv, long long int strideV, int batchSize) except? 0
cpdef int zgesvdaStridedBatched_bufferSize(intptr_t handle, int jobz, int rank, int m, int n, intptr_t d_A, int lda, long long int strideA, intptr_t d_S, long long int strideS, intptr_t d_U, int ldu, long long int strideU, intptr_t d_V, int ldv, long long int strideV, int batchSize) except? 0

cpdef sgesvdaStridedBatched(intptr_t handle, int jobz, int rank, int m, int n, intptr_t d_A, int lda, long long int strideA, intptr_t d_S, long long int strideS, intptr_t d_U, int ldu, long long int strideU, intptr_t d_V, int ldv, long long int strideV, intptr_t d_work, int lwork, intptr_t d_info, intptr_t h_R_nrmF, int batchSize)
cpdef dgesvdaStridedBatched(intptr_t handle, int jobz, int rank, int m, int n, intptr_t d_A, int lda, long long int strideA, intptr_t d_S, long long int strideS, intptr_t d_U, int ldu, long long int strideU, intptr_t d_V, int ldv, long long int strideV, intptr_t d_work, int lwork, intptr_t d_info, intptr_t h_R_nrmF, int batchSize)
cpdef cgesvdaStridedBatched(intptr_t handle, int jobz, int rank, int m, int n, intptr_t d_A, int lda, long long int strideA, intptr_t d_S, long long int strideS, intptr_t d_U, int ldu, long long int strideU, intptr_t d_V, int ldv, long long int strideV, intptr_t d_work, int lwork, intptr_t d_info, intptr_t h_R_nrmF, int batchSize)
cpdef zgesvdaStridedBatched(intptr_t handle, int jobz, int rank, int m, int n, intptr_t d_A, int lda, long long int strideA, intptr_t d_S, long long int strideS, intptr_t d_U, int ldu, long long int strideU, intptr_t d_V, int ldv, long long int strideV, intptr_t d_work, int lwork, intptr_t d_info, intptr_t h_R_nrmF, int batchSize)

cpdef int ssyevd_bufferSize(intptr_t handle, int jobz, int uplo, int n, intptr_t A, int lda, intptr_t W) except? 0
cpdef int dsyevd_bufferSize(intptr_t handle, int jobz, int uplo, int n, intptr_t A, int lda, intptr_t W) except? 0
cpdef int cheevd_bufferSize(intptr_t handle, int jobz, int uplo, int n, intptr_t A, int lda, intptr_t W) except? 0
cpdef int zheevd_bufferSize(intptr_t handle, int jobz, int uplo, int n, intptr_t A, int lda, intptr_t W) except? 0

cpdef ssyevd(intptr_t handle, int jobz, int uplo, int n, intptr_t A, int lda, intptr_t W, intptr_t work, int lwork, intptr_t info)
cpdef dsyevd(intptr_t handle, int jobz, int uplo, int n, intptr_t A, int lda, intptr_t W, intptr_t work, int lwork, intptr_t info)
cpdef cheevd(intptr_t handle, int jobz, int uplo, int n, intptr_t A, int lda, intptr_t W, intptr_t work, int lwork, intptr_t info)
cpdef zheevd(intptr_t handle, int jobz, int uplo, int n, intptr_t A, int lda, intptr_t W, intptr_t work, int lwork, intptr_t info)

cpdef int ssyevj_bufferSize(intptr_t handle, int jobz, int uplo, int n, intptr_t A, int lda, intptr_t W, size_t params) except? 0
cpdef int dsyevj_bufferSize(intptr_t handle, int jobz, int uplo, int n, intptr_t A, int lda, intptr_t W, size_t params) except? 0
cpdef int cheevj_bufferSize(intptr_t handle, int jobz, int uplo, int n, intptr_t A, int lda, intptr_t W, size_t params) except? 0
cpdef int zheevj_bufferSize(intptr_t handle, int jobz, int uplo, int n, intptr_t A, int lda, intptr_t W, size_t params) except? 0

cpdef ssyevj(intptr_t handle, int jobz, int uplo, int n, intptr_t A, int lda, intptr_t W, intptr_t work, int lwork, intptr_t info, size_t params)
cpdef dsyevj(intptr_t handle, int jobz, int uplo, int n, intptr_t A, int lda, intptr_t W, intptr_t work, int lwork, intptr_t info, size_t params)
cpdef cheevj(intptr_t handle, int jobz, int uplo, int n, intptr_t A, int lda, intptr_t W, intptr_t work, int lwork, intptr_t info, size_t params)
cpdef zheevj(intptr_t handle, int jobz, int uplo, int n, intptr_t A, int lda, intptr_t W, intptr_t work, int lwork, intptr_t info, size_t params)

cpdef int ssyevjBatched_bufferSize(intptr_t handle, int jobz, int uplo, int n, intptr_t A, int lda, intptr_t W, size_t params, int batchSize) except? 0
cpdef int dsyevjBatched_bufferSize(intptr_t handle, int jobz, int uplo, int n, intptr_t A, int lda, intptr_t W, size_t params, int batchSize) except? 0
cpdef int cheevjBatched_bufferSize(intptr_t handle, int jobz, int uplo, int n, intptr_t A, int lda, intptr_t W, size_t params, int batchSize) except? 0
cpdef int zheevjBatched_bufferSize(intptr_t handle, int jobz, int uplo, int n, intptr_t A, int lda, intptr_t W, size_t params, int batchSize) except? 0

cpdef ssyevjBatched(intptr_t handle, int jobz, int uplo, int n, intptr_t A, int lda, intptr_t W, intptr_t work, int lwork, intptr_t info, size_t params, int batchSize)
cpdef dsyevjBatched(intptr_t handle, int jobz, int uplo, int n, intptr_t A, int lda, intptr_t W, intptr_t work, int lwork, intptr_t info, size_t params, int batchSize)
cpdef cheevjBatched(intptr_t handle, int jobz, int uplo, int n, intptr_t A, int lda, intptr_t W, intptr_t work, int lwork, intptr_t info, size_t params, int batchSize)
cpdef zheevjBatched(intptr_t handle, int jobz, int uplo, int n, intptr_t A, int lda, intptr_t W, intptr_t work, int lwork, intptr_t info, size_t params, int batchSize)


###################################################
# cuSOLVER Sparse LAPACK Function - Helper Function

cpdef intptr_t spCreate() except? 0

cpdef spDestroy(intptr_t handle)

cpdef spSetStream(intptr_t handle, size_t streamId)

cpdef size_t spGetStream(intptr_t handle) except? 0


#######################################################
# cuSOLVER Sparse LAPACK Function - High Level Function

cpdef scsrlsvchol(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrVal, intptr_t csrRowPtr, intptr_t csrColInd, intptr_t b, float tol, int reorder, intptr_t x, intptr_t singularity)
cpdef dcsrlsvchol(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrVal, intptr_t csrRowPtr, intptr_t csrColInd, intptr_t b, double tol, int reorder, intptr_t x, intptr_t singularity)
cpdef ccsrlsvchol(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrVal, intptr_t csrRowPtr, intptr_t csrColInd, intptr_t b, float tol, int reorder, intptr_t x, intptr_t singularity)
cpdef zcsrlsvchol(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrVal, intptr_t csrRowPtr, intptr_t csrColInd, intptr_t b, double tol, int reorder, intptr_t x, intptr_t singularity)

cpdef scsrlsvqr(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrVal, intptr_t csrRowPtr, intptr_t csrColInd, intptr_t b, float tol, int reorder, intptr_t x, intptr_t singularity)
cpdef dcsrlsvqr(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrVal, intptr_t csrRowPtr, intptr_t csrColInd, intptr_t b, double tol, int reorder, intptr_t x, intptr_t singularity)
cpdef ccsrlsvqr(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrVal, intptr_t csrRowPtr, intptr_t csrColInd, intptr_t b, float tol, int reorder, intptr_t x, intptr_t singularity)
cpdef zcsrlsvqr(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrVal, intptr_t csrRowPtr, intptr_t csrColInd, intptr_t b, double tol, int reorder, intptr_t x, intptr_t singularity)

cpdef scsreigvsi(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrValA, intptr_t csrRowPtrA, intptr_t csrColIndA, float mu0, intptr_t x0, int maxite, float eps, intptr_t mu, intptr_t x)
cpdef dcsreigvsi(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrValA, intptr_t csrRowPtrA, intptr_t csrColIndA, double mu0, intptr_t x0, int maxite, double eps, intptr_t mu, intptr_t x)
cpdef ccsreigvsi(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrValA, intptr_t csrRowPtrA, intptr_t csrColIndA, size_t mu0, intptr_t x0, int maxite, float eps, intptr_t mu, intptr_t x)
cpdef zcsreigvsi(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrValA, intptr_t csrRowPtrA, intptr_t csrColIndA, size_t mu0, intptr_t x0, int maxite, double eps, intptr_t mu, intptr_t x)


########################################
# Version

cpdef tuple _getVersion()
