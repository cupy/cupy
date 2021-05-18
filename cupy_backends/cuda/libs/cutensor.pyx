# distutils: language = c++

"""Thin wrapper of cuTENSOR."""

cimport cython  # NOQA
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.stdint cimport int32_t, uint32_t, int64_t, uint64_t, intptr_t

from cupy_backends.cuda cimport stream as stream_module
from cupy_backends.cuda.api cimport driver

cdef extern from '../../cupy_cutensor.h' nogil:
    ctypedef int Status 'cutensorStatus_t'
    ctypedef int Algo 'cutensorAlgo_t'
    ctypedef int Operator 'cutensorOperator_t'
    ctypedef int WorksizePreference 'cutensorWorksizePreference_t'
    ctypedef int DataType 'cudaDataType_t'
    ctypedef int ComputeType 'cutensorComputeType_t'
    ctypedef struct cutensorHandle_t 'cutensorHandle_t':
        int64_t fields[512]
    ctypedef struct cutensorTensorDescriptor_t 'cutensorTensorDescriptor_t':
        int64_t fields[64]
    ctypedef struct cutensorContractionDescriptor_t \
        'cutensorContractionDescriptor_t':  # NOQA: E125
        int64_t fields[256]
    ctypedef struct cutensorContractionPlan_t 'cutensorContractionPlan_t':
        int64_t fields[640]
    ctypedef struct cutensorContractionFind_t 'cutensorContractionFind_t':
        int64_t fields[64]

    const char* cutensorGetErrorString(Status status)

    int cutensorInit(cutensorHandle_t* handle)

    # TODO(niboshi): Add const to input pointer parameters.

    int cutensorInitTensorDescriptor(
        cutensorHandle_t* handle,
        cutensorTensorDescriptor_t* desc,
        uint32_t numModes,
        int64_t* extent,
        int64_t* stride,
        DataType dataType,
        Operator unaryOp)

    int cutensorElementwiseTrinary(
        cutensorHandle_t* handle,
        void* alpha,
        void* A, cutensorTensorDescriptor_t* descA, int32_t* modeA,
        void* beta,
        void* B, cutensorTensorDescriptor_t* descB, int32_t* modeB,
        void* gamma,
        void* C, cutensorTensorDescriptor_t* descC, int32_t* modeC,
        void* D, cutensorTensorDescriptor_t* descD, int32_t* modeD,
        Operator otAB, Operator otABC,
        DataType typeScalar, driver.Stream stream)

    int cutensorElementwiseBinary(
        cutensorHandle_t* handle,
        void* alpha,
        void* A, cutensorTensorDescriptor_t* descA, int32_t* modeA,
        void* gamma,
        void* C, cutensorTensorDescriptor_t* descC, int32_t* modeC,
        void* D, cutensorTensorDescriptor_t* descD, int32_t* modeD,
        Operator otAC,
        DataType typeScalar, driver.Stream stream)

    int cutensorInitContractionDescriptor(
        cutensorHandle_t* handle,
        cutensorContractionDescriptor_t* desc,
        cutensorTensorDescriptor_t* descA,
        int32_t* modeA,
        uint32_t alignmentReqA,
        cutensorTensorDescriptor_t* descB,
        int32_t* modeB,
        uint32_t alignmentReqB,
        cutensorTensorDescriptor_t* descC,
        int32_t* modeC,
        uint32_t alignmentReqC,
        cutensorTensorDescriptor_t* descD,
        int32_t* modeD,
        uint32_t alignmentReqD,
        ComputeType typeCompute)

    int cutensorInitContractionFind(
        cutensorHandle_t* handle,
        cutensorContractionFind_t* find,
        Algo algo)

    int cutensorContractionGetWorkspace(
        cutensorHandle_t* handle,
        cutensorContractionDescriptor_t* desc,
        cutensorContractionFind_t* find,
        WorksizePreference pref,
        uint64_t *workspaceSize)

    int cutensorInitContractionPlan(
        cutensorHandle_t* handle,
        cutensorContractionPlan_t* plan,
        cutensorContractionDescriptor_t* desc,
        cutensorContractionFind_t* find,
        uint64_t workspaceSize)

    int cutensorContraction(
        cutensorHandle_t* handle,
        cutensorContractionPlan_t* plan,
        void* alpha, void* A, void* B, void* beta, void* C, void* D,
        void *workspace, uint64_t workspaceSize, driver.Stream stream)

    int cutensorContractionMaxAlgos(int32_t* maxNumAlgos)

    int cutensorReduction(
        cutensorHandle_t* handle,
        void* alpha,
        void* A, cutensorTensorDescriptor_t* descA, int32_t* modeA,
        void* beta,
        void* C, cutensorTensorDescriptor_t* descC, int32_t* modeC,
        void* D, cutensorTensorDescriptor_t* descD, int32_t* modeD,
        Operator opReduce, ComputeType typeCompute,
        void* workspace, uint64_t workspaceSize,
        driver.Stream stream)

    int cutensorReductionGetWorkspace(
        cutensorHandle_t* handle,
        void* A, cutensorTensorDescriptor_t* descA, int32_t* modeA,
        void* C, cutensorTensorDescriptor_t* descC, int32_t* modeC,
        void* D, cutensorTensorDescriptor_t* descD, int32_t* modeD,
        Operator opReduce, ComputeType typeCompute,
        uint64_t* workspaceSize)

    int cutensorGetAlignmentRequirement(
        cutensorHandle_t* handle,
        void* ptr,
        cutensorTensorDescriptor_t* desc,
        uint32_t* alignmentReq)

    size_t cutensorGetVersion()


available = True


###############################################################################
# Version information
###############################################################################
cpdef size_t get_version():
    return cutensorGetVersion()


###############################################################################
# Classes
###############################################################################

cdef class Handle:

    def __init__(self):
        self._ptr = PyMem_Malloc(sizeof(cutensorHandle_t))

    def __dealloc__(self):
        PyMem_Free(self._ptr)
        self._ptr = NULL

    @property
    def ptr(self):
        return <intptr_t>self._ptr


cdef class TensorDescriptor:

    def __init__(self):
        self._ptr = PyMem_Malloc(sizeof(cutensorTensorDescriptor_t))

    def __dealloc__(self):
        PyMem_Free(self._ptr)
        self._ptr = NULL

    @property
    def ptr(self):
        return <intptr_t>self._ptr


cdef class ContractionDescriptor:

    def __init__(self):
        self._ptr = PyMem_Malloc(sizeof(cutensorContractionDescriptor_t))

    def __dealloc__(self):
        PyMem_Free(self._ptr)
        self._ptr = NULL

    @property
    def ptr(self):
        return <intptr_t>self._ptr


cdef class ContractionFind:

    def __init__(self):
        self._ptr = PyMem_Malloc(sizeof(cutensorContractionFind_t))

    def __dealloc__(self):
        PyMem_Free(self._ptr)
        self._ptr = NULL

    @property
    def ptr(self):
        return <intptr_t>self._ptr


cdef class ContractionPlan:

    def __init__(self):
        self._ptr = PyMem_Malloc(sizeof(cutensorContractionPlan_t))

    def __dealloc__(self):
        PyMem_Free(self._ptr)
        self._ptr = NULL

    @property
    def ptr(self):
        return <intptr_t>self._ptr


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
cdef inline check_status(int status):
    if status != STATUS_SUCCESS:
        raise CuTensorError(status)


###############################################################################
# Handle initialization
###############################################################################

cpdef init(Handle handle):
    """Initializes the cuTENSOR library"""
    status = cutensorInit(<cutensorHandle_t*> handle._ptr)
    check_status(status)


###############################################################################
# Tensor descriptor initialization
###############################################################################

cpdef initTensorDescriptor(
        Handle handle,
        TensorDescriptor desc,
        uint32_t numModes,
        intptr_t extent,
        intptr_t stride,
        int dataType,
        int unaryOp):
    """Initializes a tesnor descriptor

    Args:
        handle (Handle):
            Opaque handle holding cuTENSOR's library context.
        desc (TensorDescriptor):
            Tensor descriptor.
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
    """
    status = cutensorInitTensorDescriptor(
        <cutensorHandle_t*> handle._ptr,
        <cutensorTensorDescriptor_t*> desc._ptr,
        numModes, <int64_t*> extent, <int64_t*> stride,
        <DataType> dataType, <Operator> unaryOp)
    check_status(status)


###############################################################################
# Tensor elementwise operations
###############################################################################

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
        int typeScalar):
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
        handle (Handle): Opaque handle holding CUTENSOR's library
            context.
        alpha (void*): Scaling factor for A (see equation above) of the type
            typeCompute. Pointer to the host memory. Note that A is not read if
            alpha is equal to zero, and the corresponding unary operator will
            not be performed.
        A (void*): Multi-mode tensor of type typeA with nmodeA modes. Pointer
            to the GPU-accessable memory (while a host memory pointer is
            acceptable, support for it remains an experimental feature).
        descA (TensorDescriptor): A descriptor that holds the information
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
        descB (TensorDescriptor): The B descriptor that holds information
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
        descC (TensorDescriptor): The C descriptor that holds information
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
        descD (TensorDescriptor): The D descriptor that holds information
            about the data type, modes, and strides of D. Notice that we
            currently request descD and descC to be identical.
        modeD (int32_t*): Array (in host memory) of size descD->numModes that
            holds the names of the modes of D. The modeD[i] corresponds to
            extent[i] and stride[i] of the cutensorCreateTensorDescriptor.
        opAB (cutensorOperator_t): Element-wise binary operator
            (see Phi_AB above).
        opABC (cutensorOperator_t): Element-wise binary operator
            (see Phi_ABC above).
        typeScalar (cudaDataType_t): Compute type for the intermediate
            computation.
    """
    cdef intptr_t stream = stream_module.get_current_stream_ptr()
    status = cutensorElementwiseTrinary(
        <cutensorHandle_t*> handle._ptr,
        <void*>alpha,
        <void*>A,
        <cutensorTensorDescriptor_t*> descA._ptr,
        <int32_t*>modeA,
        <void*>beta,
        <void*>B,
        <cutensorTensorDescriptor_t*> descB._ptr,
        <int32_t*>modeB,
        <void*>gamma,
        <void*>C,
        <cutensorTensorDescriptor_t*> descC._ptr,
        <int32_t*>modeC,
        <void*>D,
        <cutensorTensorDescriptor_t*> descD._ptr,
        <int32_t*>modeD,
        <Operator>opAB,
        <Operator>opABC,
        <DataType>typeScalar,
        <driver.Stream>stream)
    check_status(status)


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
        int typeScalar):
    """Element-wise tensor operation for two input tensors

    This function performs a element-wise tensor operation of the form:

        D_{Pi^C(i_0,i_1,...,i_n)} =
            Phi_AC(alpha * Psi_A(A_{Pi^A(i_0,i_1,...,i_n)}),
                   gamma * Psi_C(C_{Pi^C(i_0,i_1,...,i_n)}))

    See elementwiseTrinary() for details.
    """
    cdef intptr_t stream = stream_module.get_current_stream_ptr()
    status = cutensorElementwiseBinary(
        <cutensorHandle_t*> handle._ptr,
        <void*> alpha,
        <void*> A,
        <cutensorTensorDescriptor_t*> descA._ptr,
        <int32_t*> modeA,
        <void*> gamma,
        <void*> C,
        <cutensorTensorDescriptor_t*> descC._ptr,
        <int32_t*> modeC,
        <void*> D,
        <cutensorTensorDescriptor_t*> descD._ptr,
        <int32_t*> modeD,
        <Operator> opAC,
        <DataType> typeScalar,
        <driver.Stream> stream)
    check_status(status)


###############################################################################
# Tensor contraction
###############################################################################

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
        int computeType):
    """Initializes tensor contraction descriptor.

    Args:
        handle (Handle): Opaque handle holding cuTENSOR's library
            context.
        desc (ContractionDescriptor): This opaque struct gets filled
            with the information that encodes the tensor contraction problem.
        descA (TensorDescriptor): A descriptor that holds the
            information about the data type, modes and strides of A.
        modeA (int32_t*): Array with 'nmodeA' entries that represent the modes
            of A. The modeA[i] corresponds to extent[i] and stride[i] w.r.t.
            the arguments provided to cutensorInitTensorDescriptor.
        alighmentReqA (uint32_t): Alignment that cuTENSOR may require for
            A's pointer (in bytes); you can use the helper function
            cutensorGetAlignmentRequirement() to determine the best value for a
            given pointer.
        descB (TensorDescriptor): A descriptor that holds the
            information about the data type, modes and strides of B.
        modeB (int32_t*): Array with 'nmodeB' entries that represent the modes
            of B. The modeB[i] corresponds to extent[i] and stride[i] w.r.t.
            the arguments provided to cutensorInitTensorDescriptor.
        alighmentReqB (uint32_t): Alignment that cuTENSOR may require for
            B's pointer (in bytes); you can use the helper function
            cutensorGetAlignmentRequirement() to determine the best value for a
            given pointer.
        descC (TensorDescriptor): A descriptor that holds the
            information about the data type, modes and strides of C.
        modeC (int32_t*): Array with 'nmodeC' entries that represent the modes
            of C. The modeC[i] corresponds to extent[i] and stride[i] w.r.t.
            the arguments provided to cutensorInitTensorDescriptor.
        alighmentReqC (uint32_t): Alignment that cuTENSOR may require for
            C's pointer (in bytes); you can use the helper function
            cutensorGetAlignmentRequirement() to determine the best value for a
            given pointer.
        descD (TensorDescriptor): A descriptor that holds the
            information about the data type, modes and strides of D.
            (*) must be identical to descC for now.
        modeD (int32_t*): Array with 'nmodeD' entries that represent the modes
            of D. The modeD[i] corresponds to extent[i] and stride[i] w.r.t.
            the arguments provided to cutensorInitTensorDescriptor.
            (*) must be identical to modeD for now.
        alighmentReqD (uint32_t): Alignment that cuTENSOR may require for
            D's pointer (in bytes); you can use the helper function
            cutensorGetAlignmentRequirement() to determine the best value for a
            given pointer.
        computeType (cutensorComputeType_t): Datatype of for the intermediate
            computation of typeCompute T = A * B
    """
    status = cutensorInitContractionDescriptor(
        <cutensorHandle_t*> handle._ptr,
        <cutensorContractionDescriptor_t*> desc._ptr,
        <cutensorTensorDescriptor_t*> descA._ptr,
        <int32_t*> modeA,
        alignmentRequirementA,
        <cutensorTensorDescriptor_t*> descB._ptr,
        <int32_t*> modeB,
        alignmentRequirementB,
        <cutensorTensorDescriptor_t*> descC._ptr,
        <int32_t*> modeC,
        alignmentRequirementC,
        <cutensorTensorDescriptor_t*> descD._ptr,
        <int32_t*> modeD,
        alignmentRequirementD,
        <ComputeType> computeType)
    check_status(status)


cpdef initContractionFind(
        Handle handle,
        ContractionFind find,
        int algo):
    """Limits the search space of viable candidates

    This function gives the user finer control over the candidates that the
    subsequent call to cutensorInitContractionPlan() is allowed to evaluate.

    Args:
        handle (Handle): Opaque handle holding cuTENSOR's library
            context.
        algo (cutensorAlgo_t): Allows users to select a specific algorithm.
            CUTENSOR_ALGO_DEFAULT lets the heuristic choose the algorithm.
            Any value >= 0 selects a specific GEMM-like algorithm and
            deactivates the heuristic. If a specified algorithm is not
            supported CUTENSOR_STATUS_NOT_SUPPORTED is returned.
            See cutensorAlgo_t for additional choices.

    Return:
        find (cutensorContractionFind_t*):
    """
    status = cutensorInitContractionFind(
        <cutensorHandle_t*> handle._ptr,
        <cutensorContractionFind_t*> find._ptr,
        <Algo> algo)
    check_status(status)


cpdef initContractionPlan(
        Handle handle,
        ContractionPlan plan,
        ContractionDescriptor desc,
        ContractionFind find,
        uint64_t workspaceSize):
    """Initializes the contraction plan

    Args:
        handle (Handle): Opaque handle holding cuTENSOR's library
            context.
        plan (ContractionPlan) Opaque handle holding the contraction
            execution plan (i.e., the candidate that will be executed as well
            as all it's runtime parameters for the given tensor contraction
            problem).
        desc (ContractionDescriptor) This opaque struct encodes the
            given tensor contraction problem.
        find (ContractionFind) This opaque struct is used to
            restrict the search space of viable candidates.
        workspaceSize (uint64_t) Available workspace size (in bytes).
    """
    status = cutensorInitContractionPlan(
        <cutensorHandle_t*> handle._ptr,
        <cutensorContractionPlan_t*> plan._ptr,
        <cutensorContractionDescriptor_t*> desc._ptr,
        <cutensorContractionFind_t*> find._ptr,
        workspaceSize)
    check_status(status)


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
        uint64_t workspaceSize):
    """General tensor contraction

    This routine computes the tensor contraction
    D = alpha * Psi_A(A) * Psi_B(B) + beta * Psi_C(C).

    Example:
     - D_{a,b,c,d} = 1.3 * A_{b,e,d,f} * B_{f,e,a,c}

    Args:
        handle (Handle): Opaque handle holding CUTENSOR's library
            context.
        plan (ContractionPlan): Opaque handle holding the
            contraction execution plan.
        alpha (void*): Scaling for A*B. The data_type_t is determined by
            'typeCompute'. Pointer to the host memory.
        A (void*): Pointer to the data corresponding to A. Pointer to the
            GPU-accessable memory.
        B (void*): Pointer to the data corresponding to B. Pointer to the
            GPU-accessable memory.
        beta (void*): Scaling for C. The data_type_t is determined by
            'typeCompute'. Pointer to the host memory.
        C (void*): Pointer to the data corresponding to C. Pointer to the
            GPU-accessable memory.
        D (void*): Pointer to the data corresponding to D (must be identical
            to C for now). Pointer to the GPU-accessable memory.
        workspace (void*): Optional parameter that may be NULL. This pointer
            provides additional workspace, in device memory, to the library for
            additional optimizations.
        workspaceSize (uint64_t): Size of the workspace array in bytes.
    """
    cdef intptr_t stream = stream_module.get_current_stream_ptr()
    status = cutensorContraction(
        <cutensorHandle_t*> handle._ptr,
        <cutensorContractionPlan_t*> plan._ptr,
        <void*> alpha, <void*> A, <void*> B,
        <void*> beta, <void*> C, <void*> D,
        <void*> workspace, workspaceSize, <driver.Stream> stream)
    check_status(status)


cpdef uint64_t contractionGetWorkspace(
        Handle handle,
        ContractionDescriptor desc,
        ContractionFind find,
        int pref):
    """Determines the required workspaceSize for a given tensor contraction

    Args:
        handle (Handle): Opaque handle holding CUTENSOR's library
            context.
        desc (ContractionDescriptor): This opaque struct encodes the
            given tensor contraction problem.
        find (ContractionFind): This opaque struct restricts the
            search space of viable candidates.
        pref (cutensorWorksizePreference_t): User preference for the workspace.

    Return:
        workspaceSize (uint64_t): The workspace size (in bytes) that is
            required for the given tensor contraction.
    """
    cdef uint64_t workspaceSize = 0
    status = cutensorContractionGetWorkspace(
        <cutensorHandle_t*> handle._ptr,
        <cutensorContractionDescriptor_t*> desc._ptr,
        <cutensorContractionFind_t*> find._ptr,
        <WorksizePreference> pref,
        &workspaceSize)
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
    cdef int32_t maxNumAlgos = 0
    status = cutensorContractionMaxAlgos(&maxNumAlgos)
    check_status(status)
    return maxNumAlgos


###############################################################################
# Tensor reduction
###############################################################################

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
        uint64_t workspaceSize):
    """Tensor reduction

    This routine computes the tensor reduction of the form
    D = alpha * opReduce(opA(A)) + beta * opC(C).

    Example:
     - D_{a,d} = 0.9 * A_{a,b,c,d} + 0.1 * C_{a,d}

    Args:
        handle (Handle handle): Opaque handle holding CUTENSOR's library
            context.
        alpha (void*): Scaling for A. The data_type_t is determined by
            'typeCompute'. Pointer to the host memory.
        A (void*): Pointer to the data corresponding to A. Pointer to the
            GPU-accessable memory.
        descA (TensorDescriptor): A descriptor that holds the information
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
        descC (TensorDescriptor): The C descriptor that holds information
            about the data type, modes, strides and unary operator (opC) of C.
        D (void*): Pointer to the data corresponding to D (must be identical
            to C for now). Pointer to the GPU-accessable memory.
        modeD (int32_t*): Array with 'nmodeD' entries that represent the modes
            of D (must be identical to modeC for now). The modeD[i] corresponds
            to extent[i] and stride[i] w.r.t. the arguments provided to
            cutensorCreateTensorDescriptor.
        descD (TensorDescriptor): The D descriptor that holds information
            about the data type, modes, and strides of D (must be identical to
            descC for now).
        opReduce (cutensorOperator_t): Binary operator used to reduce elements
            of A.
        minTypeCompute (cutensorComputeType_t): All arithmetic is performed
            usingthis data type (i.e., it affects the accuracy and
            performance).
        workspace (void*): Scratchpad (device) memory.
        workspaceSize (uint64_t): Please use cutensorReductionGetWorkspace() to
            query the required workspace. While lower values, including zero,
            are valid, they may lead to grossly suboptimal performance.
    """
    cdef intptr_t stream = stream_module.get_current_stream_ptr()
    status = cutensorReduction(
        <cutensorHandle_t*> handle._ptr,
        <void*> alpha,
        <void*> A, <cutensorTensorDescriptor_t*> descA._ptr, <int32_t*> modeA,
        <void*> beta,
        <void*> C, <cutensorTensorDescriptor_t*> descC._ptr, <int32_t*> modeC,
        <void*> D, <cutensorTensorDescriptor_t*> descD._ptr, <int32_t*> modeD,
        <Operator> opReduce, <ComputeType> minTypeCompute,
        <void*> workspace, workspaceSize, <driver.Stream> stream)
    check_status(status)


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
        int typeCompute):
    """Determines the required workspaceSize for a given tensor reduction

    Args:
        See reduction() about args.

    Returns:
        workspaceSize (uint64_t): The workspace size (in bytes) that is
            required for the given tensor reduction.
    """
    cdef uint64_t workspaceSize = 0
    status = cutensorReductionGetWorkspace(
        <cutensorHandle_t*> handle._ptr,
        <void*> A, <cutensorTensorDescriptor_t*> descA._ptr, <int32_t*> modeA,
        <void*> C, <cutensorTensorDescriptor_t*> descC._ptr, <int32_t*> modeC,
        <void*> D, <cutensorTensorDescriptor_t*> descD._ptr, <int32_t*> modeD,
        <Operator> opReduce, <ComputeType> typeCompute, &workspaceSize)
    check_status(status)
    return workspaceSize


cpdef uint32_t getAlignmentRequirement(
        Handle handle,
        intptr_t ptr,
        TensorDescriptor desc):
    """Computes the minimal alignment requirement for a given pointer and
       descriptor

    Args:
        handle (Handle): Opaque handle holding CUTENSOR's library
            context.
        ptr (const void*): Raw pointer to the data of the respective tensor.
        desc (TensorDescriptor): Tensor descriptor for ptr.

    Return:
        alignmentRequirement (uint32_t): Largest alignment requirement that ptr
            can fulfill (in bytes).
    """
    cdef uint32_t alignmentRequirement = 0
    status = cutensorGetAlignmentRequirement(
        <cutensorHandle_t*> handle._ptr,
        <void*> ptr,
        <cutensorTensorDescriptor_t*> desc._ptr,
        &alignmentRequirement)
    check_status(status)
    return alignmentRequirement
