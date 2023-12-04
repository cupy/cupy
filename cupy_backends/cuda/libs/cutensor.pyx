# distutils: language = c++

"""Thin wrapper of cuTENSOR."""

cimport cython  # NOQA
from libc.stdint cimport int32_t, uint32_t, int64_t, uint64_t, intptr_t

from cupy_backends.cuda cimport stream as stream_module
from cupy_backends.cuda.api cimport runtime

cdef extern from '../../cupy_cutensor.h' nogil:
    ctypedef void* Handle_t 'cutensorHandle_t'
    ctypedef void* TensorDescriptor_t 'cutensorTensorDescriptor_t'
    ctypedef void* OperationDescriptor_t 'cutensorOperationDescriptor_t'
    ctypedef void* PlanPreference_t 'cutensorPlanPreference_t'
    ctypedef void* Plan_t 'cutensorPlan_t'

    ctypedef int Status_t 'cutensorStatus_t'
    ctypedef int Algo_t 'cutensorAlgo_t'
    ctypedef int JitMode_t 'cutensorJitMode_t'
    ctypedef int Operator_t 'cutensorOperator_t'
    ctypedef int WorksizePreference_t 'cutensorWorksizePreference_t'
    ctypedef int DataType_t 'cutensorDataType_t'
    ctypedef int ComputeType_t 'cutensorComputeType_t'

    ctypedef struct cutensorComputeDescriptor:
        pass
    # ctypedef cutensorComputeDescriptor* cutensorComputeDescriptor_t
    ctypedef cutensorComputeDescriptor* ComputeDescriptor_t 'cutensorComputeDescriptor_t'  # NOQA
    const ComputeDescriptor_t CUTENSOR_COMPUTE_DESC_16F
    const ComputeDescriptor_t CUTENSOR_COMPUTE_DESC_16BF
    const ComputeDescriptor_t CUTENSOR_COMPUTE_DESC_TF32
    const ComputeDescriptor_t CUTENSOR_COMPUTE_DESC_3XTF32
    const ComputeDescriptor_t CUTENSOR_COMPUTE_DESC_32F
    const ComputeDescriptor_t CUTENSOR_COMPUTE_DESC_64F

    #
    const char* cutensorGetErrorString(Status_t status)
    size_t cutensorGetVersion()
    size_t cutensorGetCudartVersion()

    # Handle creation and destruction
    Status_t cutensorCreate(Handle_t* handle)
    Status_t cutensorDestroy(Handle_t handle)

    # TensorDescriptor creation and destruction
    Status_t cutensorCreateTensorDescriptor(
        Handle_t handle,
        TensorDescriptor_t* desc,
        uint32_t numModes,
        int64_t* extent,
        int64_t* stride,
        DataType_t dataType,
        uint32_t alignmentRequirement)
    Status_t cutensorDestroyTensorDescriptor(TensorDescriptor_t desc)

    # PlanPreference creation and destruction
    Status_t cutensorCreatePlanPreference(
        Handle_t handle,
        PlanPreference_t* pref,
        int algo,
        int jitMode)
    Status_t cutensorDestroyPlanPreference(PlanPreference_t pref)

    Status_t cutensorEstimateWorkspaceSize(
        Handle_t handle,
        OperationDescriptor_t desc,
        PlanPreference_t planPref,
        WorksizePreference_t workspacePref,
        uint64_t* workspaceSizeEstimate)

    # Plan creation and destruction
    Status_t cutensorCreatePlan(
        Handle_t handle,
        Plan_t* plan,
        OperationDescriptor_t desc,
        PlanPreference_t pref,
        uint64_t workspaceSizeLimit)
    Status_t cutensorDestroyPlan(Plan_t plan)

    # cutensorElementwiseTrinary
    Status_t cutensorCreateElementwiseTrinary(
        Handle_t handle, OperationDescriptor_t* desc,
        TensorDescriptor_t descA, int32_t* modeA, Operator_t opA,
        TensorDescriptor_t descB, int32_t* modeB, Operator_t opB,
        TensorDescriptor_t descC, int32_t* modeC, Operator_t opC,
        TensorDescriptor_t descD, int32_t* modeD,
        Operator_t opAB, Operator_t opABC, ComputeDescriptor_t descCompute)

    Status_t cutensorElementwiseTrinaryExecute(
        Handle_t handle, Plan_t plan,
        void* alpha, void* A,
        void* beta, void* B,
        void* gamma, void* C, void* D, runtime.Stream stream)

    # cutensorElementwiseBinary
    Status_t cutensorCreateElementwiseBinary(
        Handle_t handle, OperationDescriptor_t* desc,
        TensorDescriptor_t descA, int32_t* modeA, Operator_t opA,
        TensorDescriptor_t descC, int32_t* modeC, Operator_t opC,
        TensorDescriptor_t descD, int32_t* modeD,
        Operator_t opAC, ComputeDescriptor_t descCompute)

    Status_t cutensorElementwiseBinaryExecute(
        Handle_t handle, Plan_t plan,
        void* alpha, void* A,
        void* gamma, void* C, void* D, runtime.Stream stream)

    # cutensorPermutation
    Status_t cutensorCreatePermutation(
        Handle_t handle, OperationDescriptor_t* desc,
        TensorDescriptor_t descA, int32_t* modeA, Operator_t opA,
        TensorDescriptor_t descB, int32_t* modeB,
        ComputeDescriptor_t descCompute)

    Status_t cutensorPermute(
        Handle_t handle, Plan_t plan,
        void* alpha, void* A,
        void* B, runtime.Stream stream)

    # cutensorContraction
    Status_t cutensorCreateContraction(
        Handle_t handle, OperationDescriptor_t* desc,
        TensorDescriptor_t descA, int32_t* modeA, Operator_t opA,
        TensorDescriptor_t descB, int32_t* modeB, Operator_t opB,
        TensorDescriptor_t descC, int32_t* modeC, Operator_t opC,
        TensorDescriptor_t descD, int32_t* modeD,
        ComputeDescriptor_t descCompute)

    Status_t cutensorContract(
        Handle_t handle, Plan_t plan,
        void* alpha, void* A, void* B,
        void* beta, void* C, void* D,
        void* workspace, uint64_t workspaceSize, runtime.Stream stream)

    # cutensorReduction
    Status_t cutensorCreateReduction(
        Handle_t handle, OperationDescriptor_t* desc,
        TensorDescriptor_t descA, int32_t* modeA, Operator_t opA,
        TensorDescriptor_t descC, int32_t* modeC, Operator_t opC,
        TensorDescriptor_t descD, int32_t* modeD,
        Operator_t opReduce, ComputeDescriptor_t descCompute)

    Status_t cutensorReduce(
        Handle_t handle, Plan_t plan,
        void* alpha, void* A,
        void* beta, void* C, void* D,
        void* workspace, uint64_t workspaceSize, runtime.Stream stream)

    #
    Status_t cutensorDestroyOperationDescriptor(OperationDescriptor_t desc)

    # build-time version
    int CUTENSOR_VERSION


available = True


###############################################################################
# Version information
###############################################################################

cpdef size_t get_version():
    return cutensorGetVersion()

cpdef size_t get_cudart_version():
    return cutensorGetCudartVersion()

###############################################################################
# Error handling
###############################################################################


class CuTensorError(RuntimeError):

    def __init__(self, int status):
        self.status = status
        msg = cutensorGetErrorString(<Status_t>status)
        super(CuTensorError, self).__init__(msg.decode())

    def __reduce__(self):
        return (type(self), (self.status,))


@cython.profile(False)
cdef inline check_status(int status):
    if status != STATUS_SUCCESS:
        raise CuTensorError(status)


###############################################################################
# cutensorHandle creation and destruction
###############################################################################

cpdef intptr_t create() except? 0:
    cdef Handle_t handle
    with nogil:
        status = cutensorCreate(&handle)
    check_status(status)
    return <intptr_t>handle

cpdef destroy(intptr_t handle):
    with nogil:
        status = cutensorDestroy(<Handle_t>handle)
    check_status(status)


###############################################################################
# cutensorTensorDescriptor: creation and destruction
###############################################################################

cpdef intptr_t createTensorDescriptor(
        intptr_t handle,
        uint32_t numModes,
        intptr_t extent,
        intptr_t stride,
        int dataType,
        uint32_t alignmentRequirement) except? 0:
    cdef TensorDescriptor_t desc
    with nogil:
        status = cutensorCreateTensorDescriptor(
            <Handle_t>handle, &desc,
            numModes, <int64_t*>extent, <int64_t*>stride,
            <DataType_t>dataType, alignmentRequirement)
    check_status(status)
    return <intptr_t>desc

cpdef destroyTensorDescriptor(intptr_t desc):
    with nogil:
        status = cutensorDestroyTensorDescriptor(<TensorDescriptor_t>desc)
    check_status(status)

###############################################################################
# cutensorPlanPreference: creation and destruction
###############################################################################

cpdef intptr_t createPlanPreference(
        intptr_t handle,
        int algo,
        int jitMode) except? 0:
    cdef PlanPreference_t pref
    with nogil:
        status = cutensorCreatePlanPreference(
            <Handle_t>handle, &pref, <Algo_t>algo, <JitMode_t>jitMode)
    check_status(status)
    return <intptr_t>pref

cpdef destroyPlanPreference(intptr_t pref):
    with nogil:
        status = cutensorDestroyPlanPreference(<PlanPreference_t>pref)
    check_status(status)

cpdef uint64_t estimateWorkspaceSize(
        intptr_t handle,
        intptr_t desc,
        intptr_t planPref,
        int workspacePref):
    cdef uint64_t workspaceSizeEstimate
    with nogil:
        status = cutensorEstimateWorkspaceSize(
            <Handle_t>handle, <OperationDescriptor_t>desc,
            <PlanPreference_t>planPref, <WorksizePreference_t>workspacePref,
            &workspaceSizeEstimate)
    check_status(status)
    return workspaceSizeEstimate

###############################################################################
# cutensorPlan: creation and destruction
###############################################################################

cpdef intptr_t createPlan(
        intptr_t handle,
        intptr_t desc,
        intptr_t pref,
        uint64_t workspaceSizeLimit) except? 0:
    cdef Plan_t plan
    with nogil:
        status = cutensorCreatePlan(
            <Handle_t>handle, &plan, <OperationDescriptor_t>desc,
            <PlanPreference_t>pref, workspaceSizeLimit)
    check_status(status)
    return <intptr_t>plan

cpdef destroyPlan(intptr_t plan):
    with nogil:
        status = cutensorDestroyPlan(<Plan_t>plan)
    check_status(status)

###############################################################################

cdef ComputeDescriptor_t getComputeDesc(int descCompute) nogil:
    if (descCompute == COMPUTE_16F):
        return CUTENSOR_COMPUTE_DESC_16F
    elif (descCompute == COMPUTE_16BF):
        return CUTENSOR_COMPUTE_DESC_16BF
    elif (descCompute == COMPUTE_TF32):
        return CUTENSOR_COMPUTE_DESC_TF32
    elif (descCompute == COMPUTE_3XTF32):
        return CUTENSOR_COMPUTE_DESC_3XTF32
    elif (descCompute == COMPUTE_32F):
        return CUTENSOR_COMPUTE_DESC_32F
    elif (descCompute == COMPUTE_64F):
        return CUTENSOR_COMPUTE_DESC_64F
    return CUTENSOR_COMPUTE_DESC_32F

###############################################################################
# cutensorElementwiseTrinary
###############################################################################

cpdef intptr_t createElementwiseTrinary(
        intptr_t handle,
        intptr_t descA, intptr_t modeA, int opA,
        intptr_t descB, intptr_t modeB, int opB,
        intptr_t descC, intptr_t modeC, int opC,
        intptr_t descD, intptr_t modeD,
        int opAB, int opABC, int descCompute) except? 0:
    cdef OperationDescriptor_t desc
    with nogil:
        status = cutensorCreateElementwiseTrinary(
            <Handle_t>handle, &desc,
            <TensorDescriptor_t>descA, <int32_t*>modeA, <Operator_t>opA,
            <TensorDescriptor_t>descB, <int32_t*>modeB, <Operator_t>opB,
            <TensorDescriptor_t>descC, <int32_t*>modeC, <Operator_t>opC,
            <TensorDescriptor_t>descD, <int32_t*>modeD,
            <Operator_t>opAB, <Operator_t>opABC,
            getComputeDesc(descCompute))
    check_status(status)
    return <intptr_t>desc

cpdef elementwiseTrinaryExecute(
        intptr_t handle, intptr_t plan,
        intptr_t alpha, intptr_t A,
        intptr_t beta, intptr_t B,
        intptr_t gamma, intptr_t C,
        intptr_t D):
    cdef intptr_t stream = stream_module.get_current_stream_ptr()
    with nogil:
        status = cutensorElementwiseTrinaryExecute(
            <Handle_t>handle, <Plan_t>plan,
            <void*>alpha, <void*>A,
            <void*>beta, <void*>B,
            <void*>gamma, <void*>C, <void*>D,
            <runtime.Stream>stream)
    check_status(status)

###############################################################################
# cutensorElementwiseBinary
###############################################################################

cpdef intptr_t createElementwiseBinary(
        intptr_t handle,
        intptr_t descA, intptr_t modeA, int opA,
        intptr_t descC, intptr_t modeC, int opC,
        intptr_t descD, intptr_t modeD,
        int opAB, int descCompute) except? 0:
    cdef OperationDescriptor_t desc
    with nogil:
        status = cutensorCreateElementwiseBinary(
            <Handle_t>handle, &desc,
            <TensorDescriptor_t>descA, <int32_t*>modeA, <Operator_t>opA,
            <TensorDescriptor_t>descC, <int32_t*>modeC, <Operator_t>opC,
            <TensorDescriptor_t>descD, <int32_t*>modeD,
            <Operator_t>opAB, getComputeDesc(descCompute))
    check_status(status)
    return <intptr_t>desc

cpdef elementwiseBinaryExecute(
        intptr_t handle, intptr_t plan,
        intptr_t alpha, intptr_t A,
        intptr_t gamma, intptr_t C, intptr_t D):
    cdef intptr_t stream = stream_module.get_current_stream_ptr()
    with nogil:
        status = cutensorElementwiseBinaryExecute(
            <Handle_t>handle, <Plan_t>plan,
            <void*>alpha, <void*>A,
            <void*>gamma, <void*>C, <void*>D,
            <runtime.Stream>stream)
    check_status(status)

###############################################################################
# cutensorPermute
###############################################################################

cpdef intptr_t createPermutation(
        intptr_t handle,
        intptr_t descA, intptr_t modeA, int opA,
        intptr_t descB, intptr_t modeB,
        int descCompute) except? 0:
    cdef OperationDescriptor_t desc
    with nogil:
        status = cutensorCreatePermutation(
            <Handle_t>handle, &desc,
            <TensorDescriptor_t>descA, <int32_t*>modeA, <Operator_t>opA,
            <TensorDescriptor_t>descB, <int32_t*>modeB,
            getComputeDesc(descCompute))
    check_status(status)
    return <intptr_t>desc

cpdef permute(
        intptr_t handle, intptr_t plan,
        intptr_t alpha, intptr_t A, intptr_t B):
    cdef intptr_t stream = stream_module.get_current_stream_ptr()
    with nogil:
        status =cutensorPermute(
            <Handle_t>handle, <Plan_t>plan,
            <void*>alpha, <void*>A,
            <void*>B, <runtime.Stream>stream)
    check_status(status)

###############################################################################
# cutensorContraction
###############################################################################

cpdef intptr_t createContraction(
        intptr_t handle,
        intptr_t descA, intptr_t modeA, int opA,
        intptr_t descB, intptr_t modeB, int opB,
        intptr_t descC, intptr_t modeC, int opC,
        intptr_t descD, intptr_t modeD,
        int descCompute) except? 0:
    cdef OperationDescriptor_t desc
    with nogil:
        status = cutensorCreateContraction(
            <Handle_t>handle, &desc,
            <TensorDescriptor_t>descA, <int32_t*>modeA, <Operator_t>opA,
            <TensorDescriptor_t>descB, <int32_t*>modeB, <Operator_t>opB,
            <TensorDescriptor_t>descC, <int32_t*>modeC, <Operator_t>opC,
            <TensorDescriptor_t>descD, <int32_t*>modeD,
            getComputeDesc(descCompute))
    check_status(status)
    return <intptr_t>desc

cpdef contract(
        intptr_t handle, intptr_t plan,
        intptr_t alpha, intptr_t A, intptr_t B,
        intptr_t beta, intptr_t C, intptr_t D,
        intptr_t workspace, uint64_t workspaceSize):
    cdef intptr_t stream = stream_module.get_current_stream_ptr()
    with nogil:
        status =cutensorContract(
            <Handle_t>handle, <Plan_t>plan,
            <void*>alpha, <void*>A, <void*>B,
            <void*>beta, <void*>C, <void*>D,
            <void*>workspace, workspaceSize, <runtime.Stream>stream)
    check_status(status)

###############################################################################
# cutensorReduction
###############################################################################

cpdef intptr_t createReduction(
        intptr_t handle,
        intptr_t descA, intptr_t modeA, int opA,
        intptr_t descC, intptr_t modeC, int opC,
        intptr_t descD, intptr_t modeD,
        int opReduce, int descCompute) except? 0:
    cdef OperationDescriptor_t desc
    with nogil:
        status = cutensorCreateReduction(
            <Handle_t>handle, &desc,
            <TensorDescriptor_t>descA, <int32_t*>modeA, <Operator_t>opA,
            <TensorDescriptor_t>descC, <int32_t*>modeC, <Operator_t>opC,
            <TensorDescriptor_t>descD, <int32_t*>modeD,
            <Operator_t>opReduce, getComputeDesc(descCompute))
    check_status(status)
    return <intptr_t>desc

cpdef reduce(
        intptr_t handle, intptr_t plan,
        intptr_t alpha, intptr_t A,
        intptr_t beta, intptr_t C, intptr_t D,
        intptr_t workspace, uint64_t workspaceSize):
    cdef intptr_t stream = stream_module.get_current_stream_ptr()
    with nogil:
        status = cutensorReduce(
            <Handle_t>handle, <Plan_t>plan,
            <void*>alpha, <void*>A,
            <void*>beta, <void*>C, <void*>D,
            <void*>workspace, workspaceSize, <runtime.Stream>stream)
    check_status(status)

#
cpdef destroyOperationDescriptor(intptr_t desc):
    with nogil:
        status = cutensorDestroyOperationDescriptor(
            <OperationDescriptor_t>desc)
    check_status(status)
