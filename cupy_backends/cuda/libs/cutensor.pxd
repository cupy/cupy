from libc.stdint cimport int32_t, uint32_t, int64_t, uint64_t, intptr_t


###############################################################################
# Enum
###############################################################################

cpdef enum:
    # cutensorAlgo_t (values > 0 correspond to certain algorithms of GETT)
    ALGO_GETT = -4     # NOQA, Choose the GETT algorithm
    ALGO_TGETT = -3    # NOQA, Transpose (A or B) + GETT
    ALGO_TTGT = -2     # NOQA, Transpose-Transpose-GEMM-Transpose (requires additional memory)
    ALGO_DEFAULT = -1  # NOQA, Lets the internal heuristic choose

    # cutensorWorksizePreference_t
    WORKSPACE_MIN = 1          # NOQA, At least one algorithm will be available
    WORKSPACE_RECOMMENDED = 2  # NOQA, The most suitable algorithm will be available
    WORKSPACE_MAX = 3          # NOQA, All algorithms will be available

    # cutensorOperator_t (Unary)
    OP_IDENTITY = 1  # NOQA, Identity operator (i.e., elements are not changed)
    OP_SQRT = 2      # NOQA, Square root
    OP_RELU = 8      # NOQA, Rectified linear unit
    OP_CONJ = 9      # NOQA, Complex conjugate
    OP_RCP = 10      # NOQA, Reciprocal
    OP_SIGMOID = 11  # NOQA, y=1/(1+exp(-x))
    OP_TANH = 12     # NOQA, y=tanh(x)
    OP_EXP = 22      # NOQA, Exponentiation.
    OP_LOG = 23      # NOQA, Log (base e).
    OP_ABS = 24      # NOQA, Absolute value.
    OP_NEG = 25      # NOQA, Negation.
    OP_SIN = 26      # NOQA, Sine.
    OP_COS = 27      # NOQA, Cosine.
    OP_TAN = 28      # NOQA, Tangent.
    OP_SINH = 29     # NOQA, Hyperbolic sine.
    OP_COSH = 30     # NOQA, Hyperbolic cosine.
    OP_ASIN = 31     # NOQA, Inverse sine.
    OP_ACOS = 32     # NOQA, Inverse cosine.
    OP_ATAN = 33     # NOQA, Inverse tangent.
    OP_ASINH = 34    # NOQA, Inverse hyperbolic sine.
    OP_ACOSH = 35    # NOQA, Inverse hyperbolic cosine.
    OP_ATANH = 36    # NOQA, Inverse hyperbolic tangent.
    OP_CEIL = 37     # NOQA, Ceiling.
    OP_FLOOR = 38    # NOQA, Floor.

    # cutensorOperator_t (Binary)
    OP_ADD = 3  # NOQA, Addition of two elements
    OP_MUL = 5  # NOQA, Multiplication of two elements
    OP_MAX = 6  # NOQA, Maximum of two elements
    OP_MIN = 7  # NOQA, Minimum of two elements

    # cutensorStatus_t
    STATUS_SUCCESS = 0
    STATUS_NOT_INITIALIZED = 1
    STATUS_ALLOC_FAILED = 3
    STATUS_INVALID_VALUE = 7
    STATUS_ARCH_MISMATCH = 8  # NOQA, Indicates that the device is either not ready, or the target architecture is not supported.
    STATUS_MAPPING_ERROR = 11
    STATUS_EXECUTION_FAILED = 13
    STATUS_INTERNAL_ERROR = 14
    STATUS_NOT_SUPPORTED = 15
    STATUS_LICENSE_ERROR = 16
    STATUS_CUBLAS_ERROR = 17
    STATUS_CUDA_ERROR = 18
    STATUS_INSUFFICIENT_WORKSPACE = 19
    STATUS_INSUFFICIENT_DRIVER = 20  # NOQA, Indicates that the driver version is insufficient.
    STATUS_IO_ERROR = 21

    # cutensorComputeType_t
    # (*) compute types added in versoin 1.2
    COMPUTE_16F  = 1     # NOQA, half
    COMPUTE_16BF = 1024  # NOQA, bfloat
    COMPUTE_TF32 = 4096  # NOQA, tensor-float-32
    COMPUTE_32F  = 4     # NOQA, float
    COMPUTE_64F  = 16    # NOQA, double
    COMPUTE_8U   = 64    # NOQA, uint8
    COMPUTE_8I   = 256   # NOQA, int8
    COMPUTE_32U  = 128   # NOQA, uint32
    COMPUTE_32I  = 512   # NOQA, int32
    # (*) compute types below will be deprecated in the furture release.
    R_MIN_16F = 1    # NOQA, real as a half
    C_MIN_16F = 2    # NOQA, complex as a half
    R_MIN_32F = 4    # NOQA, real as a float
    C_MIN_32F = 8    # NOQA, complex as a float
    R_MIN_64F = 16   # NOQA, real as a double
    C_MIN_64F = 32   # NOQA, complex as a double
    R_MIN_8U  = 64   # NOQA, real as a uint8
    R_MIN_32U = 128  # NOQA, real as a uint32
    R_MIN_8I  = 256  # NOQA, real as a int8
    R_MIN_32I = 512  # NOQA, real as a int32
    R_MIN_16BF = 1024  # NOQA, real as a bfloat16
    R_MIN_TF32 = 2048  # NOQA, real as a tensorfloat32
    C_MIN_TF32 = 4096  # NOQA, complex as a tensorfloat32


cpdef size_t get_version()


cdef class Handle:

    cdef void* _ptr


cdef class TensorDescriptor:

    cdef void* _ptr


cdef class ContractionDescriptor:

    cdef void* _ptr


cdef class ContractionFind:

    cdef void* _ptr


cdef class ContractionPlan:

    cdef void* _ptr


cpdef init(Handle handle)

cpdef initTensorDescriptor(
    Handle handle,
    TensorDescriptor desc,
    uint32_t numModes,
    intptr_t extent,
    intptr_t stride,
    int dataType,
    int unaryOp)

cpdef elementwiseTrinary(
    Handle handle,
    intptr_t alpha,
    intptr_t A,
    TensorDescriptor descA,
    intptr_t modeA,
    intptr_t beta,
    intptr_t B,
    TensorDescriptor descB,
    intptr_t modeB,
    intptr_t gamma,
    intptr_t C,
    TensorDescriptor descC,
    intptr_t modeC,
    intptr_t D,
    TensorDescriptor descD,
    intptr_t modeD,
    int opAB,
    int opABC,
    int typeScalar)

cpdef elementwiseBinary(
    Handle handle,
    intptr_t alpha,
    intptr_t A,
    TensorDescriptor descA,
    intptr_t modeA,
    intptr_t gamma,
    intptr_t C,
    TensorDescriptor descC,
    intptr_t modeC,
    intptr_t D,
    TensorDescriptor descD,
    intptr_t modeD,
    int opAC,
    int typeScalar)

cpdef initContractionDescriptor(
    Handle handle,
    ContractionDescriptor desc,
    TensorDescriptor descA,
    intptr_t modeA,
    uint32_t alignmentRequirementA,
    TensorDescriptor descB,
    intptr_t modeB,
    uint32_t alignmentRequirementB,
    TensorDescriptor descC,
    intptr_t modeC,
    uint32_t alignmentRequirementC,
    TensorDescriptor descD,
    intptr_t modeD,
    uint32_t alignmentRequirementD,
    int computeType)

cpdef initContractionFind(
    Handle handle,
    ContractionFind find,
    int algo)

cpdef initContractionPlan(
    Handle handle,
    ContractionPlan plan,
    ContractionDescriptor desc,
    ContractionFind find,
    uint64_t workspaceSize)

cpdef contraction(
    Handle handle,
    ContractionPlan plan,
    intptr_t alpha,
    intptr_t A,
    intptr_t B,
    intptr_t beta,
    intptr_t C,
    intptr_t D,
    intptr_t workspace,
    uint64_t workspaceSize)

cpdef uint64_t contractionGetWorkspace(
    Handle handle,
    ContractionDescriptor desc,
    ContractionFind find,
    int pref)

cpdef int32_t contractionMaxAlgos()

cpdef reduction(
    Handle handle,
    intptr_t alpha,
    intptr_t A,
    TensorDescriptor descA,
    intptr_t modeA,
    intptr_t beta,
    intptr_t C,
    TensorDescriptor descC,
    intptr_t modeC,
    intptr_t D,
    TensorDescriptor descD,
    intptr_t modeD,
    int opReduce,
    int minTypeCompute,
    intptr_t workspace,
    uint64_t workspaceSize)

cpdef uint64_t reductionGetWorkspace(
    Handle handle,
    intptr_t A,
    TensorDescriptor descA,
    intptr_t modeA,
    intptr_t C,
    TensorDescriptor descC,
    intptr_t modeC,
    intptr_t D,
    TensorDescriptor descD,
    intptr_t modeD,
    int opReduce,
    int typeCompute)

cpdef uint32_t getAlignmentRequirement(
    Handle handle,
    intptr_t ptr,
    TensorDescriptor desc)
