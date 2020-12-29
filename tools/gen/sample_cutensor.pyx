# This code was automatically generated. Do not modify it directly.

cimport cython  # NOQA
from cython.mem cimport PyMem_Malloc, PyMem_Free
from libc.stdint cimport int32_t, uint32_t, int64_t, uint64_t, intptr_t

from cupy_backends.cuda cimport stream as stream_module
from cupy_backends.cuda.api cimport driver

cdef extern from '../../cupy_cutensor.h' nogil:
    ctypedef struct cutensorHandle_t 'cutensorHandle_t':
        int64_t fields[512]
    ctypedef struct cutensorTensorDescriptor_t 'cutensorTensorDescriptor_t':
        int64_t fields[64]
    ctypedef struct cutensorContractionDescriptor_t 'cutensorContractionDescriptor_t':
        int64_t fields[256]
    ctypedef struct cutensorContractionPlan_t 'cutensorContractionPlan_t':
        int64_t fields[640]
    ctypedef struct cutensorContractionFind_t 'cutensorContractionFind_t':
        int64_t fields[64]

    const char* cutensorGetErrorString(Status status)

    size_t cutensorGetVersion()


cdef extern from '../../cupy_cutensor.h' nogil:

    # cuTENSOR Helper Functions
    Status cutensorInit(cutensorHandle_t* handle)
    Status cutensorInitTensorDescriptor(const cutensorHandle_t* const handle, cutensorTensorDescriptor_t* desc, const uint32_t numModes, const int64_t* extent, const int64_t* stride, DataType dataType, Operator unaryOp)
    Status cutensorGetAlignmentRequirement(const cutensorHandle_t* const handle, const void* ptr, const cutensorTensorDescriptor_t* const desc, uint32_t* alignmentRequirement)

    # cuTENSOR Element-wise Operations
    Status cutensorElementwiseTrinary(const cutensorHandle_t* const handle, const void* alpha, const void* A, const cutensorTensorDescriptor_t* const descA, const int32_t* modeA, const void* beta, const void* B, const cutensorTensorDescriptor_t* const descB, const int32_t* modeB, const void* gamma, const void* C, const cutensorTensorDescriptor_t* const descC, const int32_t* modeC, void* D, const cutensorTensorDescriptor_t* const descD, const int32_t* modeD, Operator opAB, Operator opABC, DataType typeScalar, const driver.Stream stream)
    Status cutensorElementwiseBinary(const cutensorHandle_t* const handle, const void* alpha, const void* A, const cutensorTensorDescriptor_t* const descA, const int32_t* modeA, const void* gamma, const void* C, const cutensorTensorDescriptor_t* const descC, const int32_t* modeC, void* D, const cutensorTensorDescriptor_t* const descD, const int32_t* modeD, Operator opAC, DataType typeScalar, driver.Stream stream)

    # cuTENSOR Contraction Operations
    Status cutensorInitContractionDescriptor(const cutensorHandle_t* const handle, cutensorContractionDescriptor_t* desc, const cutensorTensorDescriptor_t* const descA, const int32_t* modeA, const uint32_t alignmentRequirementA, const cutensorTensorDescriptor_t* const descB, const int32_t* modeB, const uint32_t alignmentRequirementB, const cutensorTensorDescriptor_t* const descC, const int32_t* modeC, const uint32_t alignmentRequirementC, const cutensorTensorDescriptor_t* const descD, const int32_t* modeD, const uint32_t alignmentRequirementD, ComputeType typeCompute)
    Status cutensorInitContractionFind(const cutensorHandle_t* const handle, cutensorContractionFind_t* find, const Algo algo)
    Status cutensorInitContractionPlan(const cutensorHandle_t* const handle, cutensorContractionPlan_t* plan, const cutensorContractionDescriptor_t* const desc, const cutensorContractionFind_t* const find, const uint64_t workspaceSize)
    Status cutensorContraction(const cutensorHandle_t* const handle, const cutensorContractionPlan_t* const plan, const void* alpha, const void* A, const void* B, const void* beta, const void* C, void* D, void* workspace, uint64_t workspaceSize, driver.Stream stream)
    Status cutensorContractionGetWorkspace(const cutensorHandle_t* const handle, const cutensorContractionDescriptor_t* const desc, const cutensorContractionFind_t* const find, const WorksizePreference pref, uint64_t* workspaceSize)
    Status cutensorContractionMaxAlgos(int32_t* maxNumAlgos)

    # cuTENSOR Reduction Operations
    Status cutensorReduction(const cutensorHandle_t* const handle, const void* alpha, const void* A, const cutensorTensorDescriptor_t* const descA, const int32_t* modeA, const void* beta, const void* C, const cutensorTensorDescriptor_t* const descC, const int32_t* modeC, void* D, const cutensorTensorDescriptor_t* const descD, const int32_t* modeD, Operator opReduce, ComputeType typeCompute, void* workspace, uint64_t workspaceSize, driver.Stream stream)
    Status cutensorReductionGetWorkspace(const cutensorHandle_t* const handle, const void* A, const cutensorTensorDescriptor_t* const descA, const int32_t* modeA, const void* C, const cutensorTensorDescriptor_t* const descC, const int32_t* modeC, const void* D, const cutensorTensorDescriptor_t* const descD, const int32_t* modeD, Operator opReduce, ComputeType typeCompute, uint64_t* workspaceSize)


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


########################################
# cuTENSOR Helper Functions

cpdef init(Handle handle):
    status = cutensorInit(<cutensorHandle_t*>handle._ptr)
    check_status(status)

cpdef initTensorDescriptor(Handle handle, TensorDescriptor desc, const uint32_t numModes, intptr_t extent, intptr_t stride, size_t dataType, int unaryOp):
    status = cutensorInitTensorDescriptor(<const cutensorHandle_t*>handle._ptr, <cutensorTensorDescriptor_t*>desc._ptr, numModes, <const int64_t*>extent, <const int64_t*>stride, <DataType>dataType, <Operator>unaryOp)
    check_status(status)

cpdef uint32_t getAlignmentRequirement(Handle handle, intptr_t ptr, TensorDescriptor desc) except? 0:
    cdef uint32_t alignmentRequirement
    status = cutensorGetAlignmentRequirement(<const cutensorHandle_t*>handle._ptr, <const void*>ptr, <const cutensorTensorDescriptor_t*>desc._ptr, &alignmentRequirement)
    check_status(status)
    return alignmentRequirement


########################################
# cuTENSOR Element-wise Operations

cpdef elementwiseTrinary(Handle handle, intptr_t alpha, intptr_t A, TensorDescriptor descA, intptr_t modeA, intptr_t beta, intptr_t B, TensorDescriptor descB, intptr_t modeB, intptr_t gamma, intptr_t C, TensorDescriptor descC, intptr_t modeC, intptr_t D, TensorDescriptor descD, intptr_t modeD, int opAB, int opABC, size_t typeScalar):
    cdef intptr_t stream = stream_module.get_current_stream_ptr()
    status = cutensorElementwiseTrinary(<const cutensorHandle_t*>handle._ptr, <const void*>alpha, <const void*>A, <const cutensorTensorDescriptor_t*>descA._ptr, <const int32_t*>modeA, <const void*>beta, <const void*>B, <const cutensorTensorDescriptor_t*>descB._ptr, <const int32_t*>modeB, <const void*>gamma, <const void*>C, <const cutensorTensorDescriptor_t*>descC._ptr, <const int32_t*>modeC, <void*>D, <const cutensorTensorDescriptor_t*>descD._ptr, <const int32_t*>modeD, <Operator>opAB, <Operator>opABC, <DataType>typeScalar, <const driver.Stream>stream)
    check_status(status)

cpdef elementwiseBinary(Handle handle, intptr_t alpha, intptr_t A, TensorDescriptor descA, intptr_t modeA, intptr_t gamma, intptr_t C, TensorDescriptor descC, intptr_t modeC, intptr_t D, TensorDescriptor descD, intptr_t modeD, int opAC, size_t typeScalar):
    cdef intptr_t stream = stream_module.get_current_stream_ptr()
    status = cutensorElementwiseBinary(<const cutensorHandle_t*>handle._ptr, <const void*>alpha, <const void*>A, <const cutensorTensorDescriptor_t*>descA._ptr, <const int32_t*>modeA, <const void*>gamma, <const void*>C, <const cutensorTensorDescriptor_t*>descC._ptr, <const int32_t*>modeC, <void*>D, <const cutensorTensorDescriptor_t*>descD._ptr, <const int32_t*>modeD, <Operator>opAC, <DataType>typeScalar, <driver.Stream>stream)
    check_status(status)


########################################
# cuTENSOR Contraction Operations

cpdef initContractionDescriptor(Handle handle, ContractionDescriptor desc, TensorDescriptor descA, intptr_t modeA, const uint32_t alignmentRequirementA, TensorDescriptor descB, intptr_t modeB, const uint32_t alignmentRequirementB, TensorDescriptor descC, intptr_t modeC, const uint32_t alignmentRequirementC, TensorDescriptor descD, intptr_t modeD, const uint32_t alignmentRequirementD, int typeCompute):
    status = cutensorInitContractionDescriptor(<const cutensorHandle_t*>handle._ptr, <cutensorContractionDescriptor_t*>desc._ptr, <const cutensorTensorDescriptor_t*>descA._ptr, <const int32_t*>modeA, alignmentRequirementA, <const cutensorTensorDescriptor_t*>descB._ptr, <const int32_t*>modeB, alignmentRequirementB, <const cutensorTensorDescriptor_t*>descC._ptr, <const int32_t*>modeC, alignmentRequirementC, <const cutensorTensorDescriptor_t*>descD._ptr, <const int32_t*>modeD, alignmentRequirementD, <ComputeType>typeCompute)
    check_status(status)

cpdef initContractionFind(Handle handle, ContractionFind find, int algo):
    status = cutensorInitContractionFind(<const cutensorHandle_t*>handle._ptr, <cutensorContractionFind_t*>find._ptr, <const Algo>algo)
    check_status(status)

cpdef initContractionPlan(Handle handle, ContractionPlan plan, ContractionDescriptor desc, ContractionFind find, const uint64_t workspaceSize):
    status = cutensorInitContractionPlan(<const cutensorHandle_t*>handle._ptr, <cutensorContractionPlan_t*>plan._ptr, <const cutensorContractionDescriptor_t*>desc._ptr, <const cutensorContractionFind_t*>find._ptr, workspaceSize)
    check_status(status)

cpdef contraction(Handle handle, ContractionPlan plan, intptr_t alpha, intptr_t A, intptr_t B, intptr_t beta, intptr_t C, intptr_t D, intptr_t workspace, uint64_t workspaceSize):
    cdef intptr_t stream = stream_module.get_current_stream_ptr()
    status = cutensorContraction(<const cutensorHandle_t*>handle._ptr, <const cutensorContractionPlan_t*>plan._ptr, <const void*>alpha, <const void*>A, <const void*>B, <const void*>beta, <const void*>C, <void*>D, <void*>workspace, workspaceSize, <driver.Stream>stream)
    check_status(status)

cpdef uint64_t contractionGetWorkspace(Handle handle, ContractionDescriptor desc, ContractionFind find, int pref) except? 0:
    cdef uint64_t workspaceSize
    status = cutensorContractionGetWorkspace(<const cutensorHandle_t*>handle._ptr, <const cutensorContractionDescriptor_t*>desc._ptr, <const cutensorContractionFind_t*>find._ptr, <const WorksizePreference>pref, &workspaceSize)
    check_status(status)
    return workspaceSize

cpdef int32_t contractionMaxAlgos() except? 0:
    cdef int32_t maxNumAlgos
    status = cutensorContractionMaxAlgos(&maxNumAlgos)
    check_status(status)
    return maxNumAlgos


########################################
# cuTENSOR Reduction Operations

cpdef reduction(Handle handle, intptr_t alpha, intptr_t A, TensorDescriptor descA, intptr_t modeA, intptr_t beta, intptr_t C, TensorDescriptor descC, intptr_t modeC, intptr_t D, TensorDescriptor descD, intptr_t modeD, int opReduce, int typeCompute, intptr_t workspace, uint64_t workspaceSize):
    cdef intptr_t stream = stream_module.get_current_stream_ptr()
    status = cutensorReduction(<const cutensorHandle_t*>handle._ptr, <const void*>alpha, <const void*>A, <const cutensorTensorDescriptor_t*>descA._ptr, <const int32_t*>modeA, <const void*>beta, <const void*>C, <const cutensorTensorDescriptor_t*>descC._ptr, <const int32_t*>modeC, <void*>D, <const cutensorTensorDescriptor_t*>descD._ptr, <const int32_t*>modeD, <Operator>opReduce, <ComputeType>typeCompute, <void*>workspace, workspaceSize, <driver.Stream>stream)
    check_status(status)

cpdef uint64_t reductionGetWorkspace(Handle handle, intptr_t A, TensorDescriptor descA, intptr_t modeA, intptr_t C, TensorDescriptor descC, intptr_t modeC, intptr_t D, TensorDescriptor descD, intptr_t modeD, int opReduce, int typeCompute) except? 0:
    cdef uint64_t workspaceSize
    status = cutensorReductionGetWorkspace(<const cutensorHandle_t*>handle._ptr, <const void*>A, <const cutensorTensorDescriptor_t*>descA._ptr, <const int32_t*>modeA, <const void*>C, <const cutensorTensorDescriptor_t*>descC._ptr, <const int32_t*>modeC, <const void*>D, <const cutensorTensorDescriptor_t*>descD._ptr, <const int32_t*>modeD, <Operator>opReduce, <ComputeType>typeCompute, &workspaceSize)
    check_status(status)
    return workspaceSize

