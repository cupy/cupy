# distutils: language = c++

"""Thin wrapper of cuTENSOR."""

cimport cython  # NOQA
from libc.stdint cimport int32_t, uint32_t, int64_t, uint64_t  # NOQA

from cupy.cuda cimport driver
from cupy.cuda cimport stream as stream_module

cdef extern from 'cupy_cutensor.h' nogil:
    ctypedef int Status 'cutensorStatus_t'
    ctypedef int Algo 'cutensorAlgo_t'
    ctypedef int Operator 'cutensorOperator_t'
    ctypedef int WorksizePreference 'cutensorWorksizePreference_t'
    ctypedef int DataType 'cudaDataType_t'

    ctypedef void* Handle 'cutensorHandle_t'
    ctypedef void* TensorDescriptor 'cutensorTensorDescriptor_t'

    const char* cutensorGetErrorString(Status status)

    int cutensorCreate(Handle* handle)

    int cutensorDestroy(Handle handle)

    int cutensorCreateTensorDescriptor(
        TensorDescriptor* desc,
        uint32_t numModes,
        int64_t* extent,
        int64_t* stride,
        DataType dataType,
        Operator unaryOp,
        uint32_t vectorWidth,
        uint32_t vectorModeIndex)

    int cutensorDestroyTensorDescriptor(TensorDescriptor desc)

    int cutensorElementwiseTrinary(
        Handle handle,
        void* alpha,
        void* A, TensorDescriptor descA, int32_t* modeA,
        void* beta,
        void* B, TensorDescriptor descB, int32_t* modeB,
        void* gamma,
        void* C, TensorDescriptor descC, int32_t* modeC,
        void* D, TensorDescriptor descD, int32_t* modeD,
        Operator otAB, Operator otABC,
        DataType typeCompute, driver.Stream stream)

    int cutensorElementwiseBinary(
        Handle handle,
        void* alpha,
        void* A, TensorDescriptor descA, int32_t* modeA,
        void* gamma,
        void* C, TensorDescriptor descC, int32_t* modeC,
        void* D, TensorDescriptor descD, int32_t* modeD,
        Operator otAC,
        DataType typeCompute, driver.Stream stream)

    int cutensorContraction(
        Handle handle,
        void* alpha,
        void* A, TensorDescriptor descA, int32_t* modeA,
        void* B, TensorDescriptor descB, int32_t* modeB,
        void* beta,
        void* C, TensorDescriptor descC, int32_t* modeC,
        void* D, TensorDescriptor descD, int32_t* modeD,
        Operator opOut, DataType typeCompute, Algo algo,
        void* workspace, uint64_t workspaceSize, driver.Stream stream)

    int cutensorContractionGetWorkspace(
        Handle handle,
        void* A, TensorDescriptor descA, int32_t* modeA,
        void* B, TensorDescriptor descB, int32_t* modeB,
        void* C, TensorDescriptor descC, int32_t* modeC,
        void* D, TensorDescriptor descD, int32_t* modeD,
        Operator opOut, DataType typeCompute, Algo algo,
        WorksizePreference pref, uint64_t* workspaceSize)

    int cutensorContractionMaxAlgos(int32_t* maxNumAlgos)

    int cutensorReduction(
        Handle handle,
        void* alpha,
        void* A, TensorDescriptor descA, int32_t* modeA,
        void* beta,
        void* C, TensorDescriptor descC, int32_t* modeC,
        void* D, TensorDescriptor descD, int32_t* modeD,
        Operator opReduce, DataType typeCompute,
        void* workspace, uint64_t workspaceSize,
        driver.Stream stream)

    int cutensorReductionGetWorkspace(
        Handle handle,
        void* A, TensorDescriptor descA, int32_t* modeA,
        void* C, TensorDescriptor descC, int32_t* modeC,
        void* D, TensorDescriptor descD, int32_t* modeD,
        Operator opReduce, DataType typeCompute,
        uint64_t* workspaceSize)


###############################################################################
# Enum
###############################################################################

cpdef enum:
    # cutensorAlgo_t (values > 0 correspond to certain algorithms of GETT)
    ALGO_TGETT = -7           # NOQA, Transpose (A or B) + GETT
    ALGO_GETT = -6            # NOQA, Choose the GETT algorithm
    ALGO_LOG_TENSOR_OP = -5   # NOQA, Loop-over-GEMM approach using tensor cores
    ALGO_LOG = -4             # NOQA, Loop-over-GEMM approach
    ALGO_TTGT_TENSOR_OP = -3  # NOQA, Transpose-Transpose-GEMM-Transpose using tensor cores (requires additional memory)
    ALGO_TTGT = -2            # NOQA, Transpose-Transpose-GEMM-Transpose (requires additional memory)
    ALGO_DEFAULT = -1         # NOQA, Lets the internal heuristic choose

    # cutensorOperator_t (Unary)
    OP_IDENTITY = 1
    OP_SQRT = 2
    OP_RELU = 8
    OP_CONJ = 9
    OP_RCP = 10

    # cutensorOperator_t (Binary)
    OP_ADD = 3
    OP_MUL = 5
    OP_MAX = 6
    OP_MIN = 7

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

    # cutensorWorksizePreference_t
    WORKSPACE_MIN = 1
    WORKSPACE_RECOMMENDED = 2
    WORKSPACE_MAX = 3


###############################################################################
# Error handling
###############################################################################

class CuTensorError(RuntimeError):

    def __init__(self, int status):
        self.status = status
        msg = cutensorGetErrorString(<Status>status)
        super(CuTensorError, self).__init__(msg.decode())

    def __reduce__(self):
        return (type(self), (self.status,))


@cython.profile(False)
cpdef inline check_status(int status):
    if status != STATUS_SUCCESS:
        raise CuTensorError(status)


###############################################################################
# Handler creation/destruction
###############################################################################

cpdef size_t create() except? 0:
    """Initializes the cuTensor library"""
    cdef Handle handle
    with nogil:
        status = cutensorCreate(&handle)
    check_status(status)
    return <size_t>handle


cpdef destroy(size_t handle):
    """Release hardware resources used by cuTensor library"""
    with nogil:
        status = cutensorDestroy(<Handle>handle)
    check_status(status)


###############################################################################
# Tensor descriptor creation/destruction
###############################################################################

cpdef size_t createTensorDescriptor(uint32_t numModes,
                                    size_t extent,
                                    size_t stride,
                                    int dataType,
                                    int unaryOp,
                                    uint32_t vectorWidth,
                                    uint32_t vectorModeIndex):
    """Creates a tesnor descriptor

    Args:
        numModes (uint32_t): number of modes
        extent (int64_t*): extent of each mode (must be larger than zero)
        stride (int64_t*): stride[i] denotes the displacement (stride)
            between two consecutive elements in the ith-mode. This value may be
            NULL, in which case a generalized column-major memory layout is
            assumed (i.e., the strides increase monotonically from left to
            right). Each stride must be larger than zero (a zero stride is
            identical to removing this mode entirely).
        dataType (cudaDataType_t): data type of stored entries.
        unaryOp (cutensorOperator_t): Unary operator that will be applied to
            each element of the corresponding tensor in a lazy fashion (i.e.,
            the algorithm uses this tensor as its operand only once). The
            original data of this tensor remains unchanged.
        vectorWidth (uint32_t): The vectorization-width of the vectorized mode
            (i.e., the number of consecutive elements in that mode). Set this
            value to 1 if no vectorization is desired. Allowed values are
            limited to 1 (this should likely be your default choice), 2, 4, 8,
            16, and 32.
        vectorModeIndex (uint32_t): The position of the mode that is vectorized
            (from left to right, 0-indexed). For instance, vectorModeIndex == i
            means that the mode corresponding to extent[i] and stride[i] is
            vectorized. This value is ignored if the vectorWidth is set to 1.

    Returns:
        desc (cutensorTensorDescriptor): Pointer to the address where the
            allocated tensor descriptor object should be stored
    """
    cdef TensorDescriptor desc
    status = cutensorCreateTensorDescriptor(
        &desc, numModes,
        <int64_t*> extent, <int64_t*> stride,
        <DataType> dataType, <Operator> unaryOp,
        vectorWidth, vectorModeIndex)
    check_status(status)
    return <size_t>desc


cpdef destroyTensorDescriptor(size_t desc):
    """Frees the memory associated to the provided descriptor"""
    status = cutensorDestroyTensorDescriptor(<TensorDescriptor> desc)
    check_status(status)


###############################################################################
# Tensor elementwise operations
###############################################################################

cpdef elementwiseTrinary(size_t handle,
                         size_t alpha,
                         size_t A, size_t descA, size_t modeA,
                         size_t beta,
                         size_t B, size_t descB, size_t modeB,
                         size_t gamma,
                         size_t C, size_t descC, size_t modeC,
                         size_t D, size_t descD, size_t modeD,
                         int opAB, int opABC,
                         int typeCompute):
    """Element-wise tensor operation for three input tensors

    This function performs a element-wise tensor operation of the form:

        D_{Pi^C(i_0,i_1,...,i_nc)} =
            Phi_ABC(Phi_AB(alpha * Psi_A(A_{Pi^A(i_0,i_1,...,i_na)}),
                           beta  * Psi_B(B_{Pi^B(i_0,i_1,...,i_nb)})),
                           gamma * Psi_C(C_{Pi^C(i_0,i_1,...,i_nc)}))

    Where
     - A,B,C,D are multi-mode tensors (of arbitrary data types).
     - Pi^A, Pi^B, Pi^C are permutation operators that permute the modes of A,
       B, and C respectively.
     - Psi_A, Psi_B, Psi_C are unary element-wise operators (e.g., IDENTITY,
       SQR, CONJUGATE).
     - Phi_ABC, Phi_AB are binary element-wise operators (e.g., ADD, MUL, MAX,
       MIN).

    Notice that the broadcasting (of a mode) can be achieved by simply omitting
    that mode from the respective tensor.

    Moreover, modes may appear in any order giving the user a greater
    flexibility. The only restrictions are:
     - modes that appear in A or B must also appear in the output tensor as
       such a case would correspond to a tensor contraction.
     - each mode may appear in each tensor at most once.

    It is guaranteed that an input tensor will not be read if the corresponding
    scalar is zero.

    Finally, the output tensor is padded with zeros, if the extent of the
    vectorized mode ---of the output tensor--- is not a multiple of the
    vector-width (the user has to ensure that the output tensor has sufficient
    memory available to facilitate such a padding). Let the i'th mode be
    vectorized with a vector-width w, then the additional padding is limited to
    (w - (extent[i] % w)) many elements in total.

    Examples:
     - B_{a,b,c,d} = A_{b,d,a,c}
     - C_{a,b,c,d} = 2.2 * A_{b,d,a,c} + 1.3 * B_{c,b,d,a}
     - D_{a,b,c,d} = 2.2 * A_{b,d,a,c} + 1.3 * B_{c,b,d,a} + C_{a,b,c,d}
     - D_{a,b,c,d} = min((2.2 * A_{b,d,a,c} + 1.3 * B_{c,b,d,a}), C_{a,b,c,d})

    Args:
        handle (cutensorHandle_t): Opaque handle holding CUTENSOR's library
            context.
        alpha (void*): Scaling factor for A (see equation above) of the type
            typeCompute. Pointer to the host memory. Note that A is not read if
            alpha is equal to zero, and the corresponding unary operator will
            not be performed.
        A (void*): Multi-mode tensor of type typeA with nmodeA modes. Pointer
            to the GPU-accessable memory (while a host memory pointer is
            acceptable, support for it remains an experimental feature).
        descA (cutensorDescriptor_t): A descriptor that holds the information
            about the data type, modes, and strides of A.
        modeA (int32_t*): Array (in host memory) of size descA->numModes that
            holds the labels of the modes of A (e.g., if A_{a,b,c} => modeA =
            {'a','b','c'}). The modeA[i] corresponds to extent[i] and stride[i]
            w.r.t. the arguments provided to cutensorCreateTensorDescriptor.
        beta (void*): Scaling factor for B (see equation above) of the type
            typeCompute. Pointer to the host memory. Note that B is not read if
            beta is equal to zero, and the corresponding unary operator will
            not be performed.
        B (void*): Multi-mode tensor of type typeB with nmodeB many modes.
            Pointer to the GPU-accessable memory (while a host memory pointer
            is acceptable, support for it remains an experimental feature).
        descB (cutensorDescriptor_t): The B descriptor that holds information
            about the data type, modes, and strides of B.
        modeB (int32_t*): Array (in host memory) of size descB->numModes that
            holds the names of the modes of B. modeB[i] corresponds to
            extent[i] and stride[i] of the cutensorCreateTensorDescriptor
        gamma (void*): Scaling factor for C (see equation above) of type
            typeCompute. Pointer to the host memory. Note that C is not read if
            gamma is equal to zero, and the corresponding unary operator will
            not be performed.
        C (void*): Multi-mode tensor of type typeC with nmodeC many modes.
            Pointer to the GPU-accessable memory (while a host memory pointer
            is acceptable, support for it remains an experimental feature).
        descC (cutensorDescriptor_t): The C descriptor that holds information
            about the data type, modes, and strides of C.
        modeC (int32_t*): Array (in host memory) of size descC->numModes that
            holds the names of the modes of C. The modeC[i] corresponds to
            extent[i] and stride[i] of the cutensorCreateTensorDescriptor.
        D (void*): Multi-mode output tensor of type typeC with nmodeC modes
            that are ordered according to modeD. Pointer to the GPU-accessable
            memory (while a host memory pointer is acceptable, support for it
            remains an experimental feature). Notice that D may alias any input
            tensor if they share the same memory layout (i.e., same tensor
            descriptor).
        descD (cutensorDescriptor_t): The D descriptor that holds information
            about the data type, modes, and strides of D. Notice that we
            currently request descD and descC to be identical.
        modeD (int32_t*): Array (in host memory) of size descD->numModes that
            holds the names of the modes of D. The modeD[i] corresponds to
            extent[i] and stride[i] of the cutensorCreateTensorDescriptor.
        opAB (cutensorOperator_t): Element-wise binary operator
            (see Phi_AB above).
        opABC (cutensorOperator_t): Element-wise binary operator
            (see Phi_ABC above).
        typeCompute (cudaDataType_t): Compute type for the intermediate
            computation.
    """
    cdef size_t stream = stream_module.get_current_stream_ptr()
    status = cutensorElementwiseTrinary(
        <Handle> handle,
        <void*> alpha,
        <void*> A, <TensorDescriptor> descA, <int32_t*> modeA,
        <void*> beta,
        <void*> B, <TensorDescriptor> descB, <int32_t*> modeB,
        <void*> gamma,
        <void*> C, <TensorDescriptor> descC, <int32_t*> modeC,
        <void*> D, <TensorDescriptor> descD, <int32_t*> modeD,
        <Operator> opAB, <Operator> opABC,
        <DataType> typeCompute, <driver.Stream> stream)
    check_status(status)


cpdef elementwiseBinary(size_t handle,
                        size_t alpha,
                        size_t A, size_t descA, size_t modeA,
                        size_t gamma,
                        size_t C, size_t descC, size_t modeC,
                        size_t D, size_t descD, size_t modeD,
                        int opAC, int typeCompute):
    """Element-wise tensor operation for two input tensors

    This function performs a element-wise tensor operation of the form:

        D_{Pi^C(i_0,i_1,...,i_n)} =
            Phi_AC(alpha * Psi_A(A_{Pi^A(i_0,i_1,...,i_n)}),
                   gamma * Psi_C(C_{Pi^C(i_0,i_1,...,i_n)}))

    See elementwiseTrinary() for details.
    """
    cdef size_t stream = stream_module.get_current_stream_ptr()
    status = cutensorElementwiseBinary(
        <Handle> handle,
        <void*> alpha,
        <void*> A, <TensorDescriptor> descA, <int32_t*> modeA,
        <void*> gamma,
        <void*> C, <TensorDescriptor> descC, <int32_t*> modeC,
        <void*> D, <TensorDescriptor> descD, <int32_t*> modeD,
        <Operator> opAC,
        <DataType> typeCompute, <driver.Stream> stream)
    check_status(status)


###############################################################################
# Tensor contraction
###############################################################################

cpdef contraction(size_t handle,
                  size_t alpha,
                  size_t A, size_t descA, size_t modeA,
                  size_t B, size_t descB, size_t modeB,
                  size_t beta,
                  size_t C, size_t descC, size_t modeC,
                  size_t D, size_t descD, size_t modeD,
                  int opOut, int typeCompute, int algo,
                  size_t workspace, uint64_t workspaceSize):
    """General tensor contraction

    This routine computes the tensor contraction
    D = Psi_out(alpha * Psi_A(A) * Psi_B(B) + beta * Psi_C(C)).

    Example:
     - C_{a,b,c,d} = 1.3 * A_{b,e,d,f} * B_{f,e,a,c}

    Args:
        handle (cutensorHandle_t): Opaque handle holding CUTENSOR's library
            context.
        alpha (void*): Scaling for A*B. The data_type_t is determined by
            'typeCompute'. Pointer to the host memory.
        A (void*): Pointer to the data corresponding to A. Pointer to the
            GPU-accessable memory (while a host memory pointer is acceptable,
            support for it remains an experimental feature).
        descA (cutensorDescriptor_t): A descriptor that holds the information
            about the data type, modes and strides of A.
        modeA (int32_t*): Array with 'nmodeA' entries that represent the modes
            of A. The modeA[i] corresponds to extent[i] and stride[i] w.r.t.
            the arguments provided to cutensorCreateTensorDescriptor.
        B (void*): Pointer to the data corresponding to B. Pointer to the
            GPU-accessable memory (while a host memory pointer is acceptable,
            support for it remains an experimental feature).
        descC (cutensorDescriptor_t): The C descriptor that holds information
            about the data type, modes, and strides of C.
        modeB (int32_t*): Array with 'nmodeB' entries that represent the modes
            of B. The modeB[i] corresponds to extent[i] and stride[i] w.r.t.
            the arguments provided to cutensorCreateTensorDescriptor.
        beta (void*): Scaling for C. The data_type_t is determined by
            'typeCompute'. Pointer to the host memory.
        C (void*): Pointer to the data corresponding to C. Pointer to the
            GPU-accessable memory (while a host memory pointer is acceptable,
            support for it remains an experimental feature).
        modeC (int32_t*): Array with 'nmodeC' entries that represent the modes
            of C. The modeC[i] corresponds to extent[i] and stride[i] w.r.t.
            the arguments provided to cutensorCreateTensorDescriptor.
        descC (cutensorDescriptor_t): The C descriptor that holds information
            about the data type, modes, and strides of C.
        D (void*): Pointer to the data corresponding to D (must be identical
            to C for now). Pointer to the GPU-accessable memory (while a host
            memory pointer is acceptable, support for it remains an
            experimental feature).
        modeD (int32_t*): Array with 'nmodeD' entries that represent the modes
            of D (must be identical to modeC for now). The modeD[i] corresponds
            to extent[i] and stride[i] w.r.t. the arguments provided to
            cutensorCreateTensorDescriptor.
        descD (cutensorDescriptor_t): The D descriptor that holds information
            about the data type, modes, and strides of D (must be identical to
            descC for now).
        opOut (cutensorOperator_t): The element-wise unary operator
            (see Psi_out above).
        typeCompute (cudaDataType_t): Datatype of for the intermediate
            computation of typeCompute T = A * B.
        algo (cutenorAlgo_t): Allows users to select a specific algorithm.
            ALGO_DEFAULT lets the heuristic choose the algorithm.
            Any value >= 0 selects a specific GEMM-like algorithm and
            deactivates the heuristic. If a specified algorithm is not
            supported, STATUS_NOT_SUPPORTED is returned.
        workspace (void*): Optional parameter that may be NULL. This pointer
            provides additional workspace, in device memory, to the library for
            additional optimizations.
        workspaceSize (uint64_t): Size of the workspace array in bytes.
    """
    cdef size_t stream = stream_module.get_current_stream_ptr()
    status = cutensorContraction(
        <Handle> handle,
        <void*> alpha,
        <void*> A, <TensorDescriptor> descA, <int32_t*> modeA,
        <void*> B, <TensorDescriptor> descB, <int32_t*> modeB,
        <void*> beta,
        <void*> C, <TensorDescriptor> descC, <int32_t*> modeC,
        <void*> D, <TensorDescriptor> descD, <int32_t*> modeD,
        <Operator> opOut, <DataType> typeCompute, <Algo> algo,
        <void*> workspace, workspaceSize, <driver.Stream> stream)
    check_status(status)


cpdef uint64_t contractionGetWorkspace(size_t handle,
                                       size_t A, size_t descA, size_t modeA,
                                       size_t B, size_t descB, size_t modeB,
                                       size_t C, size_t descC, size_t modeC,
                                       size_t D, size_t descD, size_t modeD,
                                       int opOut, int typeCompute, int algo,
                                       int pref):
    """Determines the required workspaceSize for a given tensor contraction

    Args:
        perf (cutensorWorksizePreference_t): User preference for the workspace.

        See contraction() about other args.

    Returns:
        workspaceSize (uint64_t): The workspace size (in bytes) that is
            required for the given tensor contraction.
    """
    cdef uint64_t workspaceSize
    status = cutensorContractionGetWorkspace(
        <Handle> handle,
        <void*> A, <TensorDescriptor> descA, <int32_t*> modeA,
        <void*> B, <TensorDescriptor> descB, <int32_t*> modeB,
        <void*> C, <TensorDescriptor> descC, <int32_t*> modeC,
        <void*> D, <TensorDescriptor> descD, <int32_t*> modeD,
        <Operator> opOut, <DataType> typeCompute, <Algo> algo,
        <WorksizePreference> pref, &workspaceSize)
    check_status(status)
    return workspaceSize


cpdef int32_t contractionMaxAlgos():
    """Returns the maximum number of algorithms for cutensorContraction()

    You can use the returned integer for auto-tuning purposes (i.e., iterate
    overe all algorithms up to the returned value). Not all algorithms might be
    applicable to your specific problem. cutensorContraction() will return
    CUTENSOR_STATUS_NOT_SUPPORTED if an algorithm is not applicable.

    Returns:
        maxNumAlgos (int32_t): The maximum number of algorithms available for
            cutensorContraction().
    """
    cdef int32_t maxNumAlgos
    status = cutensorContractionMaxAlgos(&maxNumAlgos)
    check_status(status)
    return maxNumAlgos


###############################################################################
# Tensor reduction
###############################################################################

cpdef reduction(size_t handle,
                size_t alpha,
                size_t A, size_t descA, size_t modeA,
                size_t beta,
                size_t C, size_t descC, size_t modeC,
                size_t D, size_t descD, size_t modeD,
                int opReduce, int typeCompute,
                size_t workspace, uint64_t workspaceSize):
    """Tensor reduction

    This routine computes the tensor reduction of the form
    D = alpha * opReduce(opA(A)) + beta * opC(C).

    Example:
     - D_{a,d} = 0.9 * A_{a,b,c,d} + 0.1 * C_{a,d}

    Args:
        handle (cutensorHandle_t): Opaque handle holding CUTENSOR's library
            context.
        alpha (void*): Scaling for A. The data_type_t is determined by
            'typeCompute'. Pointer to the host memory.
        A (void*): Pointer to the data corresponding to A. Pointer to the
            GPU-accessable memory.
        descA (cutensorDescriptor_t): A descriptor that holds the information
            about the data type, modes, strides and unary operator (opA) of A.
        modeA (int32_t*): Array with 'nmodeA' entries that represent the modes
            of A. The modeA[i] corresponds to extent[i] and stride[i] w.r.t.
            the arguments provided to cutensorCreateTensorDescriptor.
        beta (void*): Scaling for C. The data_type_t is determined by
            'typeCompute'. Pointer to the host memory.
        C (void*): Pointer to the data corresponding to C. Pointer to the
            GPU-accessable memory.
        modeC (int32_t*): Array with 'nmodeC' entries that represent the modes
            of C. The modeC[i] corresponds to extent[i] and stride[i] w.r.t.
            the arguments provided to cutensorCreateTensorDescriptor.
        descC (cutensorDescriptor_t): The C descriptor that holds information
            about the data type, modes, strides and unary operator (opC) of C.
        D (void*): Pointer to the data corresponding to D (must be identical
            to C for now). Pointer to the GPU-accessable memory.
        modeD (int32_t*): Array with 'nmodeD' entries that represent the modes
            of D (must be identical to modeC for now). The modeD[i] corresponds
            to extent[i] and stride[i] w.r.t. the arguments provided to
            cutensorCreateTensorDescriptor.
        descD (cutensorDescriptor_t): The D descriptor that holds information
            about the data type, modes, and strides of D (must be identical to
            descC for now).
        opReduce (cutensorOperator_t): Binary operator used to reduce elements
            of A.
        typeCompute (cudaDataType_t): All arithmetic is performed using this
            data type (i.e., it affects the accuracy and performance).
        workspace (void*): Scratchpad (device) memory.
        workspaceSize (uint64_t): Please use reductionGetWorkspace() to query
            the required workspace. That being said, a workspaceSize of zero is
            valid but it can lead to (grossly) suboptimal performance. Hence,
            if you don't want to call reductionGetWorkspace() prior to each
            reduction call (which is not really necessary), then you could
            provide some small and fixed workspace (e.g., 8192 bytes).
    """
    cdef size_t stream = stream_module.get_current_stream_ptr()
    status = cutensorReduction(
        <Handle> handle,
        <void*> alpha,
        <void*> A, <TensorDescriptor> descA, <int32_t*> modeA,
        <void*> beta,
        <void*> C, <TensorDescriptor> descC, <int32_t*> modeC,
        <void*> D, <TensorDescriptor> descD, <int32_t*> modeD,
        <Operator> opReduce, <DataType> typeCompute,
        <void*> workspace, workspaceSize, <driver.Stream> stream)
    check_status(status)


cpdef uint64_t reductionGetWorkspace(size_t handle,
                                     size_t A, size_t descA, size_t modeA,
                                     size_t C, size_t descC, size_t modeC,
                                     size_t D, size_t descD, size_t modeD,
                                     int opReduce, int typeCompute):
    """Determines the required workspaceSize for a given tensor reduction

    Args:
        See reduction() about args.

    Returns:
        workspaceSize (uint64_t): The workspace size (in bytes) that is
            required for the given tensor reduction.
    """
    cdef uint64_t workspaceSize
    status = cutensorReductionGetWorkspace(
        <Handle> handle,
        <void*> A, <TensorDescriptor> descA, <int32_t*> modeA,
        <void*> C, <TensorDescriptor> descC, <int32_t*> modeC,
        <void*> D, <TensorDescriptor> descD, <int32_t*> modeD,
        <Operator> opReduce, <DataType> typeCompute, &workspaceSize)
    check_status(status)
    return workspaceSize
