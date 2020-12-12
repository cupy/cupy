# This code was automatically generated. Do not modify it directly.

from libc.stdint cimport intptr_t, int64_t


########################################
# Opaque data structures

cdef extern from *:
    ctypedef int DataType 'cudaDataType'


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
# Auxiliary structures




# TODO: should also expose functions?

