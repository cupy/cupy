# This code was automatically generated. Do not modify it directly.

from libc.stdint cimport intptr_t, int64_t


########################################
# Opaque pointers

cdef extern from *:
    ctypedef int DataType 'cudaDataType'


########################################
# Enumerators

cdef extern from *:
    ctypedef int Operator 'cutensorOperator_t'
    ctypedef int Status 'cutensorStatus_t'
    ctypedef int Algo 'cutensorAlgo_t'
    ctypedef int WorksizePreference 'cutensorWorksizePreference_t'
    ctypedef int ComputeType 'cutensorComputeType_t'
    ctypedef int ContractionDescriptorAttributes 'cutensorContractionDescriptorAttributes_t'
    ctypedef int ContractionFindAttributes 'cutensorContractionFindAttributes_t'
    ctypedef int AutotuneMode 'cutensorAutotuneMode_t'
    ctypedef int CacheMode 'cutensorCacheMode_t'

cpdef enum:
    CUTENSOR_OP_IDENTITY = 1
    CUTENSOR_OP_SQRT = 2
    CUTENSOR_OP_RELU = 8
    CUTENSOR_OP_CONJ = 9
    CUTENSOR_OP_RCP = 10
    CUTENSOR_OP_SIGMOID = 11
    CUTENSOR_OP_TANH = 12
    CUTENSOR_OP_EXP = 22
    CUTENSOR_OP_LOG = 23
    CUTENSOR_OP_ABS = 24
    CUTENSOR_OP_NEG = 25
    CUTENSOR_OP_SIN = 26
    CUTENSOR_OP_COS = 27
    CUTENSOR_OP_TAN = 28
    CUTENSOR_OP_SINH = 29
    CUTENSOR_OP_COSH = 30
    CUTENSOR_OP_ASIN = 31
    CUTENSOR_OP_ACOS = 32
    CUTENSOR_OP_ATAN = 33
    CUTENSOR_OP_ASINH = 34
    CUTENSOR_OP_ACOSH = 35
    CUTENSOR_OP_ATANH = 36
    CUTENSOR_OP_CEIL = 37
    CUTENSOR_OP_FLOOR = 38
    CUTENSOR_OP_ADD = 3
    CUTENSOR_OP_MUL = 5
    CUTENSOR_OP_MAX = 6
    CUTENSOR_OP_MIN = 7
    CUTENSOR_OP_UNKNOWN = 126

cpdef enum:
    CUTENSOR_STATUS_SUCCESS = 0
    CUTENSOR_STATUS_NOT_INITIALIZED = 1
    CUTENSOR_STATUS_ALLOC_FAILED = 3
    CUTENSOR_STATUS_INVALID_VALUE = 7
    CUTENSOR_STATUS_ARCH_MISMATCH = 8
    CUTENSOR_STATUS_MAPPING_ERROR = 11
    CUTENSOR_STATUS_EXECUTION_FAILED = 13
    CUTENSOR_STATUS_INTERNAL_ERROR = 14
    CUTENSOR_STATUS_NOT_SUPPORTED = 15
    CUTENSOR_STATUS_LICENSE_ERROR = 16
    CUTENSOR_STATUS_CUBLAS_ERROR = 17
    CUTENSOR_STATUS_CUDA_ERROR = 18
    CUTENSOR_STATUS_INSUFFICIENT_WORKSPACE = 19
    CUTENSOR_STATUS_INSUFFICIENT_DRIVER = 20
    CUTENSOR_STATUS_IO_ERROR = 21

cpdef enum:
    CUTENSOR_ALGO_GETT = -4
    CUTENSOR_ALGO_TGETT = -3
    CUTENSOR_ALGO_TTGT = -2
    CUTENSOR_ALGO_DEFAULT = -1

cpdef enum:
    CUTENSOR_WORKSPACE_MIN = 1
    CUTENSOR_WORKSPACE_RECOMMENDED = 2
    CUTENSOR_WORKSPACE_MAX = 3

cpdef enum:
    CUTENSOR_COMPUTE_16F = (1U << 0U)
    CUTENSOR_COMPUTE_16BF = (1U << 10U)
    CUTENSOR_COMPUTE_TF32 = (1U << 12U)
    CUTENSOR_COMPUTE_32F = (1U << 2U)
    CUTENSOR_COMPUTE_64F = (1U << 4U)
    CUTENSOR_COMPUTE_8U = (1U << 6U)
    CUTENSOR_COMPUTE_8I = (1U << 8U)
    CUTENSOR_COMPUTE_32U = (1U << 7U)
    CUTENSOR_COMPUTE_32I = (1U << 9U)
    CUTENSOR_R_MIN_16F = (1U << 0U)
    CUTENSOR_C_MIN_16F = (1U << 1U)
    CUTENSOR_R_MIN_32F = (1U << 2U)
    CUTENSOR_C_MIN_32F = (1U << 3U)
    CUTENSOR_R_MIN_64F = (1U << 4U)
    CUTENSOR_C_MIN_64F = (1U << 5U)
    CUTENSOR_R_MIN_8U = (1U << 6U)
    CUTENSOR_R_MIN_32U = (1U << 7U)
    CUTENSOR_R_MIN_8I = (1U << 8U)
    CUTENSOR_R_MIN_32I = (1U << 9U)
    CUTENSOR_R_MIN_16BF = (1U << 10U)
    CUTENSOR_R_MIN_TF32 = (1U << 11U)
    CUTENSOR_C_MIN_TF32 = (1U << 12U)

cpdef enum:
    CUTENSOR_CONTRACTION_DESCRIPTOR_TAG

cpdef enum:
    CUTENSOR_CONTRACTION_FIND_AUTOTUNE_MODE
    CUTENSOR_CONTRACTION_FIND_CACHE_MODE
    CUTENSOR_CONTRACTION_FIND_INCREMENTAL_COUNT

cpdef enum:
    CUTENSOR_AUTOTUNE_NONE
    CUTENSOR_AUTOTUNE_INCREMENTAL

cpdef enum:
    CUTENSOR_CACHE_MODE_NONE
    CUTENSOR_CACHE_MODE_PEDANTIC


########################################
# cuTENSOR Helper Functions

cpdef init(Handle handle)

cpdef initTensorDescriptor(Handle handle, TensorDescriptor desc, const uint32_t numModes, intptr_t extent, intptr_t stride, size_t dataType, int unaryOp)

cpdef uint32_t getAlignmentRequirement(Handle handle, intptr_t ptr, TensorDescriptor desc) except? 0


########################################
# cuTENSOR Element-wise Operations

cpdef elementwiseTrinary(Handle handle, intptr_t alpha, intptr_t A, TensorDescriptor descA, intptr_t modeA, intptr_t beta, intptr_t B, TensorDescriptor descB, intptr_t modeB, intptr_t gamma, intptr_t C, TensorDescriptor descC, intptr_t modeC, intptr_t D, TensorDescriptor descD, intptr_t modeD, int opAB, int opABC, size_t typeScalar)

cpdef elementwiseBinary(Handle handle, intptr_t alpha, intptr_t A, TensorDescriptor descA, intptr_t modeA, intptr_t gamma, intptr_t C, TensorDescriptor descC, intptr_t modeC, intptr_t D, TensorDescriptor descD, intptr_t modeD, int opAC, size_t typeScalar)


########################################
# cuTENSOR Contraction Operations

cpdef initContractionDescriptor(Handle handle, ContractionDescriptor desc, TensorDescriptor descA, intptr_t modeA, const uint32_t alignmentRequirementA, TensorDescriptor descB, intptr_t modeB, const uint32_t alignmentRequirementB, TensorDescriptor descC, intptr_t modeC, const uint32_t alignmentRequirementC, TensorDescriptor descD, intptr_t modeD, const uint32_t alignmentRequirementD, int typeCompute)

cpdef initContractionFind(Handle handle, ContractionFind find, int algo)

cpdef initContractionPlan(Handle handle, ContractionPlan plan, ContractionDescriptor desc, ContractionFind find, const uint64_t workspaceSize)

cpdef contraction(Handle handle, ContractionPlan plan, intptr_t alpha, intptr_t A, intptr_t B, intptr_t beta, intptr_t C, intptr_t D, intptr_t workspace, uint64_t workspaceSize)

cpdef uint64_t contractionGetWorkspace(Handle handle, ContractionDescriptor desc, ContractionFind find, int pref) except? 0

cpdef int32_t contractionMaxAlgos() except? 0


########################################
# cuTENSOR Reduction Operations

cpdef reduction(Handle handle, intptr_t alpha, intptr_t A, TensorDescriptor descA, intptr_t modeA, intptr_t beta, intptr_t C, TensorDescriptor descC, intptr_t modeC, intptr_t D, TensorDescriptor descD, intptr_t modeD, int opReduce, int typeCompute, intptr_t workspace, uint64_t workspaceSize)

cpdef uint64_t reductionGetWorkspace(Handle handle, intptr_t A, TensorDescriptor descA, intptr_t modeA, intptr_t C, TensorDescriptor descC, intptr_t modeC, intptr_t D, TensorDescriptor descD, intptr_t modeD, int opReduce, int typeCompute) except? 0
