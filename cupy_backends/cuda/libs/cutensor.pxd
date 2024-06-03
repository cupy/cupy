from libc.stdint cimport int32_t, uint32_t, int64_t, uint64_t, intptr_t

###############################################################################
# Enum
###############################################################################

cpdef enum:
    # cutensorDataType_t
    R_16F = 2  # NOQA, real as a half
    C_16F = 6  # NOQA, complex as a pair of half numbers
    R_16BF = 14  # NOQA, real as a nv_bfloat16
    C_16BF = 15  # NOQA, complex as a pair of nv_bfloat16 numbers
    R_32F = 0  # NOQA, real as a float
    C_32F = 4  # NOQA, complex as a pair of float numbers
    R_64F = 1  # NOQA, real as a double
    C_64F = 5  # NOQA, complex as a pair of double numbers
    R_4I = 16  # NOQA, real as a signed 4-bit int
    C_4I = 17  # NOQA, complex as a pair of signed 4-bit int numbers
    R_4U = 18  # NOQA, real as a unsigned 4-bit int
    C_4U = 19  # NOQA, complex as a pair of unsigned 4-bit int numbers
    R_8I = 3  # NOQA, real as a signed 8-bit int
    C_8I = 7  # NOQA, complex as a pair of signed 8-bit int numbers
    R_8U = 8  # NOQA, real as a unsigned 8-bit int
    C_8U = 9  # NOQA, complex as a pair of unsigned 8-bit int numbers
    R_16I = 20  # NOQA, real as a signed 16-bit int
    C_16I = 21  # NOQA, complex as a pair of signed 16-bit int numbers
    R_16U = 22  # NOQA, real as a unsigned 16-bit int
    C_16U = 23  # NOQA, complex as a pair of unsigned 16-bit int numbers
    R_32I = 10  # NOQA, real as a signed 32-bit int
    C_32I = 11  # NOQA, complex as a pair of signed 32-bit int numbers
    R_32U = 12  # NOQA, real as a unsigned 32-bit int
    C_32U = 13  # NOQA, complex as a pair of unsigned 32-bit int numbers
    R_64I = 24  # NOQA, real as a signed 64-bit int
    C_64I = 25  # NOQA, complex as a pair of signed 64-bit int numbers
    R_64U = 26  # NOQA, real as a unsigned 64-bit int
    C_64U = 27  # NOQA, complex as a pair of unsigned 64-bit int numbers

    # cutensorAlgo_t (values > 0 correspond to certain algorithms of GETT)
    ALGO_DEFAULT_PATIENT = -6  # NOQA, Uses the more accurate but also more time-consuming performance model
    ALGO_GETT = -4             # NOQA, Choose the GETT algorithm
    ALGO_TGETT = -3            # NOQA, Transpose (A or B) + GETT
    ALGO_TTGT = -2             # NOQA, Transpose-Transpose-GEMM-Transpose (requires additional memory)
    ALGO_DEFAULT = -1          # NOQA, Lets the internal heuristic choose

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
    # (*) compute types added in version 1.2
    COMPUTE_16F = 1     # NOQA, half
    COMPUTE_16BF = 1024  # NOQA, bfloat
    COMPUTE_TF32 = 4096  # NOQA, tensor-float-32
    COMPUTE_3XTF32 = 8192  # NOQA, more precise than TF32, but less precise than float
    COMPUTE_32F = 4     # NOQA, float
    COMPUTE_64F = 16    # NOQA, double
    COMPUTE_8U = 64    # NOQA, uint8
    COMPUTE_8I = 256   # NOQA, int8
    COMPUTE_32U = 128   # NOQA, uint32
    COMPUTE_32I = 512   # NOQA, int32
    # (*) compute types below will be deprecated in the future release.
    R_MIN_16F = 1    # NOQA, real as a half
    C_MIN_16F = 2    # NOQA, complex as a half
    R_MIN_32F = 4    # NOQA, real as a float
    C_MIN_32F = 8    # NOQA, complex as a float
    R_MIN_64F = 16   # NOQA, real as a double
    C_MIN_64F = 32   # NOQA, complex as a double
    R_MIN_8U = 64   # NOQA, real as a uint8
    R_MIN_32U = 128  # NOQA, real as a uint32
    R_MIN_8I = 256  # NOQA, real as a int8
    R_MIN_32I = 512  # NOQA, real as a int32
    R_MIN_16BF = 1024  # NOQA, real as a bfloat16
    R_MIN_TF32 = 2048  # NOQA, real as a tensorfloat32
    C_MIN_TF32 = 4096  # NOQA, complex as a tensorfloat32

    # cutensorComputeDescriptor_t alternatives
    COMPUTE_DESC_16F = 1
    COMPUTE_DESC_16BF = 1024
    COMPUTE_DESC_TF32 = 4096
    COMPUTE_DESC_3xTF32 = 8192
    COMPUTE_DESC_32F = 4
    COMPUTE_DESC_64F = 16

    # cutensorJitMode_t
    JIT_MODE_NONE = 0   # NOQA, no kernel will be just-in-time compiled.
    JIT_MODE_DEFAULT = 1,  # NOQA, the corresponding plan will try to compile a dedicated kernel for the given operation. Only supported for GPUs with compute capability >= 8.0 (Ampere or newer).
    JIT_MODE_ALL = 2  # NOQA, the corresponding plan will compile all the kernel candidates for the given contraction.

# Version information
cpdef size_t get_version()
cpdef size_t get_cudart_version()

# Handle creation and destruction
cpdef intptr_t create() except? 0
cpdef destroy(intptr_t handle)

# TensorDescriptor creation and destruction
cpdef intptr_t createTensorDescriptor(
    intptr_t handle,
    uint32_t numModes,
    intptr_t extent,
    intptr_t stride,
    int dataType,
    uint32_t alignmentRequirement) except? 0
cpdef destroyTensorDescriptor(intptr_t desc)

# PlanPreference creation and destruction
cpdef intptr_t createPlanPreference(
    intptr_t handle,
    int algo,
    int jitMode) except? 0
cpdef destroyPlanPreference(intptr_t pref)

cpdef uint64_t estimateWorkspaceSize(
    intptr_t handle,
    intptr_t desc,
    intptr_t planPref,
    int workspacePref)

# Plan creation and destruction
cpdef intptr_t createPlan(
    intptr_t handle,
    intptr_t desc,
    intptr_t pref,
    uint64_t workspaceSizeLimit) except? 0
cpdef destroyPlan(intptr_t plan)

# cutensorElementwiseTrinary
cpdef intptr_t createElementwiseTrinary(
    intptr_t handle,
    intptr_t descA, intptr_t modeA, int opA,
    intptr_t descB, intptr_t modeB, int opB,
    intptr_t descC, intptr_t modeC, int opC,
    intptr_t descD, intptr_t modeD,
    int opAB, int opABC, int descCompute) except? 0
cpdef elementwiseTrinaryExecute(
    intptr_t handle, intptr_t plan,
    intptr_t alpha, intptr_t A,
    intptr_t beta, intptr_t B,
    intptr_t gamma, intptr_t C, intptr_t D)

# cutensorElementwiseBinary
cpdef intptr_t createElementwiseBinary(
    intptr_t handle,
    intptr_t descA, intptr_t modeA, int opA,
    intptr_t descC, intptr_t modeC, int opC,
    intptr_t descD, intptr_t modeD,
    int opAC, int descCompute) except? 0
cpdef elementwiseBinaryExecute(
    intptr_t handle, intptr_t plan,
    intptr_t alpha, intptr_t A,
    intptr_t gamma, intptr_t C, intptr_t D)

# cutensorPermutation
cpdef intptr_t createPermutation(
    intptr_t handle,
    intptr_t descA, intptr_t modeA, int opA,
    intptr_t descB, intptr_t modeB,
    int descCompute) except? 0
cpdef permute(
    intptr_t handle, intptr_t plan,
    intptr_t alpha, intptr_t A, intptr_t B)

# cutensorContraction
cpdef intptr_t createContraction(
    intptr_t handle,
    intptr_t descA, intptr_t modeA, int opA,
    intptr_t descB, intptr_t modeB, int opB,
    intptr_t descC, intptr_t modeC, int opC,
    intptr_t descD, intptr_t modeD,
    int descCompute) except? 0
cpdef contract(
    intptr_t handle, intptr_t plan,
    intptr_t alpha, intptr_t A, intptr_t B,
    intptr_t beta, intptr_t C, intptr_t D,
    intptr_t workspace, uint64_t workspaceSize)

# cutensorReduction
cpdef intptr_t createReduction(
    intptr_t handle,
    intptr_t descA, intptr_t modeA, int opA,
    intptr_t descC, intptr_t modeC, int opC,
    intptr_t descD, intptr_t modeD,
    int opReduce, int descCompute) except? 0
cpdef reduce(
    intptr_t handle, intptr_t plan,
    intptr_t alpha, intptr_t A,
    intptr_t beta, intptr_t C, intptr_t D,
    intptr_t workspace, uint64_t workspaceSize)

#
cpdef destroyOperationDescriptor(intptr_t desc)
