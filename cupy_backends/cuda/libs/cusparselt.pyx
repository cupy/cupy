"""Thin wrapper for cuSPARSELt"""

cimport cython  # NOQA

from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.stdint cimport uint8_t, int32_t, uint32_t, int64_t, intptr_t

from cupy_backends.cuda cimport stream as stream_module
from cupy_backends.cuda.api cimport driver

from cupy_backends.cuda.libs import cusparse

cdef extern from '../../cupy_cusparselt.h' nogil:
    ctypedef int cusparseStatus_t 'cusparseStatus_t'
    ctypedef int cusparseOrder_t 'cusparseOrder_t'
    ctypedef int cudaDataType 'cudaDataType'
    ctypedef int cusparseComputeType 'cusparseComputeType'

    # Opaque Data Structures
    ctypedef struct cusparseLtHandle_t 'cusparseLtHandle_t':
        uint8_t data[1024]
    ctypedef struct cusparseLtMatDescriptor_t 'cusparseLtMatDescriptor_t':
        uint8_t data[1024]
    ctypedef struct cusparseLtMatmulDescriptor_t 'cusparseLtMatmulDescriptor_t':  # NOQA
        uint8_t data[1024]
    ctypedef struct cusparseLtMatmulAlgSelection_t 'cusparseLtMatmulAlgSelection_t':  # NOQA
        uint8_t data[1024]
    ctypedef struct cusparseLtMatmulPlan_t 'cusparseLtMatmulPlan_t':
        uint8_t data[1024]

    # Enumerators
    ctypedef int cusparseLtSparsity_t 'cusparseLtSparsity_t'
    ctypedef int cusparseOperation_t 'cusparseOperation_t'
    ctypedef int cusparseLtMatmulAlg_t 'cusparseLtMatmulAlg_t'
    ctypedef int cusparseLtMatmulAlgAttribute_t 'cusparseLtMatmulAlgAttribute_t'  # NOQA
    ctypedef int cusparseLtPruneAlg_t 'cusparseLtPruneAlg_t'

    # Management Functions
    cusparseStatus_t cusparseLtInit(cusparseLtHandle_t* handle)
    cusparseStatus_t cusparseLtDestroy(const cusparseLtHandle_t* handle)

    # Matmul Functions
    cusparseStatus_t cusparseLtDenseDescriptorInit(
        const cusparseLtHandle_t* handle,
        cusparseLtMatDescriptor_t* matDescr,
        int64_t rows, int64_t cols, int64_t ld, uint32_t alignment,
        cudaDataType valueType, cusparseOrder_t order)
    cusparseStatus_t cusparseLtStructuredDescriptorInit(
        const cusparseLtHandle_t* handle,
        cusparseLtMatDescriptor_t* matDescr,
        int64_t rows, int64_t cols, int64_t ld, uint32_t alignment,
        cudaDataType valueType, cusparseOrder_t order,
        cusparseLtSparsity_t sparsity)
    cusparseStatus_t cusparseLtMatmulDescriptorInit(
        const cusparseLtHandle_t* handle,
        cusparseLtMatmulDescriptor_t* matMulDescr,
        cusparseOperation_t opA,
        cusparseOperation_t opB,
        const cusparseLtMatDescriptor_t* matA,
        const cusparseLtMatDescriptor_t* matB,
        const cusparseLtMatDescriptor_t* matC,
        const cusparseLtMatDescriptor_t* matD,
        cusparseComputeType computeType)
    cusparseStatus_t cusparseLtMatmulAlgSelectionInit(
        const cusparseLtHandle_t* handle,
        cusparseLtMatmulAlgSelection_t* algSelection,
        const cusparseLtMatmulDescriptor_t* matmulDescr,
        cusparseLtMatmulAlg_t alg)
    cusparseStatus_t cusparseLtMatmulAlgSetAttribute(
        const cusparseLtHandle_t* handle,
        cusparseLtMatmulAlgSelection_t* algSelection,
        cusparseLtMatmulAlgAttribute_t attribute,
        const void* data, size_t ataSize)
    cusparseStatus_t cusparseLtMatmulGetWorkspace(
        const cusparseLtHandle_t* handle,
        const cusparseLtMatmulAlgSelection_t* algSelection,
        size_t* workspaceSize)
    cusparseStatus_t cusparseLtMatmulPlanInit(
        const cusparseLtHandle_t* handle,
        cusparseLtMatmulPlan_t* plan,
        const cusparseLtMatmulDescriptor_t* matmulDescr,
        const cusparseLtMatmulAlgSelection_t* algSelection,
        size_t workspaceSize)
    cusparseStatus_t cusparseLtMatmulPlanDestroy(
        const cusparseLtMatmulPlan_t* plan)
    cusparseStatus_t cusparseLtMatmul(
        const cusparseLtHandle_t* handle, const cusparseLtMatmulPlan_t* plan,
        const void* alpha, const void* d_A, const void* d_B,
        const void* beta, const void* d_C, void* d_D,
        void* workspace, driver.Stream* streams, int32_t numStreams)

    # Helper Functions
    cusparseStatus_t cusparseLtSpMMAPrune(
        const cusparseLtHandle_t* handle,
        const cusparseLtMatmulDescriptor_t* matmulDescr,
        const void* d_in, void* d_out,
        cusparseLtPruneAlg_t pruneAlg, driver.Stream stream)
    cusparseStatus_t cusparseLtSpMMAPruneCheck(
        const cusparseLtHandle_t* handle,
        const cusparseLtMatmulDescriptor_t* matmulDescr,
        const void* d_in, int* valid, driver.Stream stream)
    cusparseStatus_t cusparseLtSpMMACompressedSize(
        const cusparseLtHandle_t* handle, const cusparseLtMatmulPlan_t* plan,
        size_t* compressedSize)
    cusparseStatus_t cusparseLtSpMMACompress(
        const cusparseLtHandle_t* handle, const cusparseLtMatmulPlan_t* plan,
        const void* d_dense, void* d_compressed, driver.Stream stream)

    # Build-time version
    int CUSPARSELT_VERSION


###############################################################################
# Classes
###############################################################################

cdef class Handle:
    cdef void * _ptr

    def __init__(self):
        self._ptr = PyMem_Malloc(sizeof(cusparseLtHandle_t))

    def __dealloc__(self):
        PyMem_Free(self._ptr)
        self._ptr = NULL

    @property
    def ptr(self):
        return <intptr_t>self._ptr


cdef class MatDescriptor:
    cdef void * _ptr

    def __init__(self):
        self._ptr = PyMem_Malloc(sizeof(cusparseLtMatDescriptor_t))

    def __dealloc__(self):
        PyMem_Free(self._ptr)
        self._ptr = NULL

    @property
    def ptr(self):
        return <intptr_t>self._ptr


cdef class MatmulDescriptor:
    cdef void * _ptr

    def __init__(self):
        self._ptr = PyMem_Malloc(sizeof(cusparseLtMatmulDescriptor_t))

    def __dealloc__(self):
        PyMem_Free(self._ptr)
        self._ptr = NULL

    @property
    def ptr(self):
        return <intptr_t>self._ptr


cdef class MatmulAlgSelection:
    cdef void * _ptr

    def __init__(self):
        self._ptr = PyMem_Malloc(sizeof(cusparseLtMatmulAlgSelection_t))

    def __dealloc__(self):
        PyMem_Free(self._ptr)
        self._ptr = NULL

    @property
    def ptr(self):
        return <intptr_t>self._ptr


cdef class MatmulPlan:
    cdef void * _ptr

    def __init__(self):
        self._ptr = PyMem_Malloc(sizeof(cusparseLtMatmulPlan_t))

    def __dealloc__(self):
        PyMem_Free(self._ptr)
        self._ptr = NULL

    @property
    def ptr(self):
        return <intptr_t>self._ptr


###############################################################################
# Error handling
###############################################################################

@cython.profile(False)
cpdef inline check_status(int status):
    if status != 0:
        raise cusparse.CuSparseError(status)

###############################################################################
# cuSPARSELt: Library Management Functions
###############################################################################

cpdef init(Handle handle):
    """Initializes the cuSPARSELt library handle"""
    status = cusparseLtInit(<cusparseLtHandle_t*> handle._ptr)
    check_status(status)


cpdef destroy(Handle handle):
    """Releases hardware resources used by the cuSPARSELt library"""
    status = cusparseLtDestroy(<cusparseLtHandle_t*> handle._ptr)
    check_status(status)

###############################################################################
# cuSPARSELt: Matmul Functions
###############################################################################

cpdef denseDescriptorInit(Handle handle, MatDescriptor matDescr,
                          rows, cols, ld, alignment, valueType, order):
    """Initializes the descriptor of a dense matrix"""
    status = cusparseLtDenseDescriptorInit(
        <const cusparseLtHandle_t*> handle._ptr,
        <cusparseLtMatDescriptor_t*> matDescr._ptr,
        <int64_t> rows, <int64_t> cols, <int64_t> ld, <uint32_t> alignment,
        <cudaDataType> valueType, <cusparseOrder_t> order)
    check_status(status)

cpdef structuredDescriptorInit(Handle handle, MatDescriptor matDescr,
                               rows, cols, ld, alignment, valueType, order,
                               sparsity):
    """Initializes the descriptor of a structured matrix."""
    status = cusparseLtStructuredDescriptorInit(
        <const cusparseLtHandle_t*> handle._ptr,
        <cusparseLtMatDescriptor_t*> matDescr._ptr,
        <int64_t> rows, <int64_t> cols, <int64_t> ld, <uint32_t> alignment,
        <cudaDataType> valueType, <cusparseOrder_t> order,
        <cusparseLtSparsity_t> sparsity)
    check_status(status)

cpdef matmulDescriptorInit(Handle handle,
                           MatmulDescriptor matMulDescr,
                           opA, opB,
                           MatDescriptor matA,
                           MatDescriptor matB,
                           MatDescriptor matC,
                           MatDescriptor matD,
                           computeType):
    """Initializes the matrix multiplication descriptor."""
    status = cusparseLtMatmulDescriptorInit(
        <const cusparseLtHandle_t*> handle._ptr,
        <cusparseLtMatmulDescriptor_t*> matMulDescr._ptr,
        <cusparseOperation_t> opA,
        <cusparseOperation_t> opB,
        <const cusparseLtMatDescriptor_t*> matA._ptr,
        <const cusparseLtMatDescriptor_t*> matB._ptr,
        <const cusparseLtMatDescriptor_t*> matC._ptr,
        <const cusparseLtMatDescriptor_t*> matD._ptr,
        <cusparseComputeType> computeType)
    check_status(status)

cpdef matmulAlgSelectionInit(Handle handle, MatmulAlgSelection algSelection,
                             MatmulDescriptor matmulDescr, alg):
    """Initializes the algorithm selection descriptor."""
    status = cusparseLtMatmulAlgSelectionInit(
        <const cusparseLtHandle_t*> handle._ptr,
        <cusparseLtMatmulAlgSelection_t*> algSelection._ptr,
        <const cusparseLtMatmulDescriptor_t*> matmulDescr._ptr,
        <cusparseLtMatmulAlg_t> alg)
    check_status(status)

cpdef matmulAlgSetAttribute(Handle handle, MatmulAlgSelection algSelection,
                            attribute, size_t data, size_t dataSize):
    """Sets the attribute related to algorithm selection descriptor."""
    status = cusparseLtMatmulAlgSetAttribute(
        <const cusparseLtHandle_t*> handle._ptr,
        <cusparseLtMatmulAlgSelection_t*> algSelection._ptr,
        <cusparseLtMatmulAlgAttribute_t> attribute,
        <const void*> data, <size_t> dataSize)
    check_status(status)

cpdef size_t matmulGetWorkspace(Handle handle,
                                MatmulAlgSelection algSelection):
    """Determines the required workspace size"""
    cdef size_t workspaceSize
    status = cusparseLtMatmulGetWorkspace(
        <const cusparseLtHandle_t*> handle._ptr,
        <const cusparseLtMatmulAlgSelection_t*> algSelection._ptr,
        &workspaceSize)
    check_status(status)
    return workspaceSize

cpdef matmulPlanInit(Handle handle, MatmulPlan plan,
                     MatmulDescriptor matmulDescr,
                     MatmulAlgSelection algSelection,
                     size_t workspaceSize):
    """Initializes the plan."""
    status = cusparseLtMatmulPlanInit(
        <const cusparseLtHandle_t*> handle._ptr,
        <cusparseLtMatmulPlan_t*> plan._ptr,
        <const cusparseLtMatmulDescriptor_t*> matmulDescr._ptr,
        <const cusparseLtMatmulAlgSelection_t*> algSelection._ptr,
        <size_t> workspaceSize)
    check_status(status)

cpdef matmulPlanDestroy(MatmulPlan plan):
    """Destroys plan"""
    status = cusparseLtMatmulPlanDestroy(
        <const cusparseLtMatmulPlan_t*> plan._ptr)
    check_status(status)

cpdef matmul(Handle handle, MatmulPlan plan,
             size_t alpha, size_t d_A, size_t d_B,
             size_t beta, size_t d_C, size_t d_D, size_t workspace):
    """Computes the matrix multiplication"""
    status = cusparseLtMatmul(
        <const cusparseLtHandle_t*> handle._ptr,
        <const cusparseLtMatmulPlan_t*> plan._ptr,
        <const void*> alpha, <const void*> d_A, <const void*> d_B,
        <const void*> beta, <const void*> d_C, <void*> d_D,
        <void*> workspace, <driver.Stream*> NULL, <int32_t> 0)
    check_status(status)

###############################################################################
# cuSPARSELt: Helper Functions
###############################################################################

cpdef spMMAPrune(Handle handle, MatmulDescriptor matmulDescr,
                 size_t d_in, size_t d_out, pruneAlg):
    """Prunes a dense matrix d_in"""
    cdef intptr_t stream = stream_module.get_current_stream_ptr()
    status = cusparseLtSpMMAPrune(
        <const cusparseLtHandle_t*> handle._ptr,
        <const cusparseLtMatmulDescriptor_t*> matmulDescr._ptr,
        <const void*> d_in, <void*> d_out,
        <cusparseLtPruneAlg_t> pruneAlg, <driver.Stream> stream)
    check_status(status)

cpdef spMMAPruneCheck(Handle handle, MatmulDescriptor matmulDescr,
                      size_t d_in, size_t valid):
    """checks the correctness of the pruning structure"""
    cdef intptr_t stream = stream_module.get_current_stream_ptr()
    status = cusparseLtSpMMAPruneCheck(
        <const cusparseLtHandle_t*> handle._ptr,
        <const cusparseLtMatmulDescriptor_t*> matmulDescr._ptr,
        <const void*> d_in, <int*> valid, <driver.Stream> stream)
    check_status(status)

cpdef size_t spMMACompressedSize(Handle handle, MatmulPlan plan):
    """Provides the size of the compressed matrix"""
    cdef size_t compressedSize
    status = cusparseLtSpMMACompressedSize(
        <const cusparseLtHandle_t*> handle._ptr,
        <const cusparseLtMatmulPlan_t*> plan._ptr,
        &compressedSize)
    check_status(status)
    return compressedSize

cpdef spMMACompress(Handle handle, MatmulPlan plan,
                    size_t d_dense, size_t d_compressed):
    """Compresses a dense matrix d_dense."""
    cdef intptr_t stream = stream_module.get_current_stream_ptr()
    status = cusparseLtSpMMACompress(
        <const cusparseLtHandle_t*> handle._ptr,
        <const cusparseLtMatmulPlan_t*> plan._ptr,
        <const void*> d_dense, <void*> d_compressed, <driver.Stream> stream)
    check_status(status)


def get_build_version():
    return CUSPARSELT_VERSION
