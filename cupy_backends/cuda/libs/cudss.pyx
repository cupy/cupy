# distutils: language = c++

"""Thin wrapper for cuDSS 0.8.0.

Handles and host/device pointers are passed as integers. Callers own the
referenced buffers and must keep them alive until asynchronous work completes.
"""

cimport cython
from libc.stdint cimport intptr_t, int64_t
from libc.stdio cimport FILE

cdef extern from "../../cupy_cudss.h" nogil:
    ctypedef void* Stream_t "cudaStream_t"
    ctypedef void* cudssHandle_t
    ctypedef void* cudssConfig_t
    ctypedef void* cudssData_t
    ctypedef void* cudssMatrix_t
    int CUDSS_VERSION
    ctypedef int libraryPropertyType
    ctypedef struct cudssDeviceMemHandler_t:
        pass
    ctypedef void (*cudssLoggerCallback_t)(
        int, const char*, const char*) noexcept

    cudssStatus_t cudssCreate(
        cudssHandle_t* handle)
    cudssStatus_t cudssCreateMg(
        cudssHandle_t* handle,
        int device_count,
        const int* device_indices)
    cudssStatus_t cudssDestroy(
        cudssHandle_t handle)
    cudssStatus_t cudssGetProperty(
        libraryPropertyType propertyType,
        int* value)
    cudssStatus_t cudssSetStream(
        cudssHandle_t handle,
        Stream_t stream)
    cudssStatus_t cudssSetMgStreams(
        cudssHandle_t handle,
        const Stream_t* streams,
        int stream_count)
    cudssStatus_t cudssSetDeviceMemHandler(
        cudssHandle_t handle,
        const cudssDeviceMemHandler_t* handler)
    cudssStatus_t cudssGetDeviceMemHandler(
        const cudssHandle_t handle,
        cudssDeviceMemHandler_t* handler)
    cudssStatus_t cudssSetCommLayer(
        cudssHandle_t handle,
        const char* commLibFileName)
    cudssStatus_t cudssSetThreadingLayer(
        cudssHandle_t handle,
        const char* thrLibFileName)
    cudssStatus_t cudssLoggerSetCallback(
        cudssLoggerCallback_t callback)
    cudssStatus_t cudssLoggerSetFile(
        FILE* file)
    cudssStatus_t cudssLoggerOpenFile(
        const char* logFile)
    cudssStatus_t cudssLoggerSetLevel(
        int level)
    cudssStatus_t cudssLoggerSetMask(
        int mask)
    cudssStatus_t cudssLoggerForceDisable()
    cudssStatus_t cudssConfigCreate(
        cudssConfig_t* config)
    cudssStatus_t cudssConfigDestroy(
        cudssConfig_t config)
    cudssStatus_t cudssConfigSet(
        cudssConfig_t config,
        cudssConfigParam_t param,
        const void* value,
        size_t sizeInBytes)
    cudssStatus_t cudssConfigGet(
        const cudssConfig_t config,
        cudssConfigParam_t param,
        void* value,
        size_t sizeInBytes,
        size_t* sizeWritten)
    cudssStatus_t cudssDataCreate(
        cudssHandle_t handle,
        cudssData_t* data)
    cudssStatus_t cudssDataDestroy(
        cudssHandle_t handle,
        cudssData_t data)
    cudssStatus_t cudssDataSet(
        const cudssHandle_t handle,
        cudssData_t data,
        cudssDataParam_t param,
        const void* value,
        size_t sizeInBytes)
    cudssStatus_t cudssDataGet(
        const cudssHandle_t handle,
        const cudssData_t data,
        cudssDataParam_t param,
        void* value,
        size_t sizeInBytes,
        size_t* sizeWritten)
    cudssStatus_t cudssExecute(
        cudssHandle_t handle,
        int phase,
        const cudssConfig_t config,
        cudssData_t data,
        const cudssMatrix_t matrix,
        cudssMatrix_t solution,
        const cudssMatrix_t rhs)
    cudssStatus_t cudssMatrixCreateDn(
        cudssMatrix_t* matrix,
        int64_t nrows,
        int64_t ncols,
        int64_t ld,
        const void* values,
        cudssDataType_t valueType,
        cudssLayout_t layout)
    cudssStatus_t cudssMatrixCreateBatchDn(
        cudssMatrix_t* matrix,
        int64_t batchCount,
        const void* nrows,
        const void* ncols,
        const void* ld,
        const void*const* values,
        cudssDataType_t integerType,
        cudssDataType_t valueType,
        cudssLayout_t layout)
    cudssStatus_t cudssMatrixCreateCsr(
        cudssMatrix_t* matrix,
        int64_t nrows,
        int64_t ncols,
        int64_t nnz,
        const void* rowStart,
        const void* rowEnd,
        const void* colIndices,
        const void* values,
        cudssDataType_t offsetType,
        cudssDataType_t indexType,
        cudssDataType_t valueType,
        cudssMatrixType_t mtype,
        cudssMatrixViewType_t mview,
        cudssIndexBase_t indexBase)
    cudssStatus_t cudssMatrixCreateBatchCsr(
        cudssMatrix_t* matrix,
        int64_t batchCount,
        const void* nrows,
        const void* ncols,
        const void* nnz,
        const void*const* rowStart,
        const void*const* rowEnd,
        const void*const* colIndices,
        const void*const* values,
        cudssDataType_t offsetType,
        cudssDataType_t indexType,
        cudssDataType_t valueType,
        cudssMatrixType_t mtype,
        cudssMatrixViewType_t mview,
        cudssIndexBase_t indexBase)
    cudssStatus_t cudssMatrixDestroy(
        cudssMatrix_t matrix)
    cudssStatus_t cudssMatrixSetValues(
        cudssMatrix_t matrix,
        const void* values)
    cudssStatus_t cudssMatrixSetBatchValues(
        cudssMatrix_t matrix,
        const void*const* values)
    cudssStatus_t cudssMatrixSetCsrPointers(
        cudssMatrix_t matrix,
        const void* rowStart,
        const void* rowEnd,
        const void* colIndices,
        const void* values)
    cudssStatus_t cudssMatrixSetBatchCsrPointers(
        cudssMatrix_t matrix,
        const void*const* rowStart,
        const void*const* rowEnd,
        const void*const* colIndices,
        const void*const* values)
    cudssStatus_t cudssMatrixSetDistributionRow1d(
        cudssMatrix_t matrix,
        int64_t first_row,
        int64_t last_row)
    cudssStatus_t cudssMatrixGetDn(
        const cudssMatrix_t matrix,
        int64_t* nrows,
        int64_t* ncols,
        int64_t* ld,
        void** values,
        cudssDataType_t* valueType,
        cudssLayout_t* layout)
    cudssStatus_t cudssMatrixGetBatchDn(
        const cudssMatrix_t matrix,
        int64_t* batchCount,
        void** nrows,
        void** ncols,
        void** ld,
        void*** values,
        cudssDataType_t* indexType,
        cudssDataType_t* valueType,
        cudssLayout_t* layout)
    cudssStatus_t cudssMatrixGetCsr(
        const cudssMatrix_t matrix,
        int64_t* nrows,
        int64_t* ncols,
        int64_t* nnz,
        void** rowStart,
        void** rowEnd,
        void** colIndices,
        void** values,
        cudssDataType_t* offsetType,
        cudssDataType_t* indexType,
        cudssDataType_t* valueType,
        cudssMatrixType_t* mtype,
        cudssMatrixViewType_t* mview,
        cudssIndexBase_t* indexBase)
    cudssStatus_t cudssMatrixGetBatchCsr(
        const cudssMatrix_t matrix,
        int64_t* batchCount,
        void** nrows,
        void** ncols,
        void** nnz,
        void*** rowStart,
        void*** rowEnd,
        void*** colIndices,
        void*** values,
        cudssDataType_t* offsetType,
        cudssDataType_t* indexType,
        cudssDataType_t* valueType,
        cudssMatrixType_t* mtype,
        cudssMatrixViewType_t* mview,
        cudssIndexBase_t* indexBase)
    cudssStatus_t cudssMatrixGetDistributionRow1d(
        const cudssMatrix_t matrix,
        int64_t* first_row,
        int64_t* last_row)
    cudssStatus_t cudssMatrixGetFormat(
        const cudssMatrix_t matrix,
        int* format)


# Keep constants accessible at module scope, as in the other CUDA wrappers.
for _enum in (cudssDataType_t, cudssStatus_t, cudssConfigParam_t,
              cudssDataParam_t, cudssPhase_t, cudssMatrixFormat_t,
              cudssMatrixType_t, cudssMatrixViewType_t, cudssIndexBase_t,
              cudssLayout_t, cudssReorderingAlg_t, cudssFactorizationAlg_t,
              cudssSolveAlg_t, cudssPivotEpsilonAlg_t, cudssMatchingAlg_t,
              cudssPivotType_t, cudssOpType_t):
    globals().update(_enum.__members__)
del _enum

available = CUDSS_VERSION >= 800


class CuDSSError(RuntimeError):
    def __init__(self, int status):
        self.status = status
        try:
            name = cudssStatus_t(status).name
        except ValueError:
            name = 'Unknown cuDSS error'
        super().__init__(f'{name} (status {status})')

    def __reduce__(self):
        return type(self), (self.status,)


@cython.profile(False)
cpdef inline check_status(int status):
    if status != CUDSS_STATUS_SUCCESS:
        raise CuDSSError(status)


cpdef create():
    """Wrap ``cudssCreate``. Return handle."""
    cdef cudssHandle_t handle = NULL
    with nogil:
        status = cudssCreate(
            &handle)
    check_status(status)
    return <intptr_t>handle


cpdef createMg(int device_count, intptr_t device_indices):
    """Wrap ``cudssCreateMg``. Return handle."""
    cdef cudssHandle_t handle = NULL
    with nogil:
        status = cudssCreateMg(
            &handle,
            device_count,
            <const int*>device_indices)
    check_status(status)
    return <intptr_t>handle


cpdef destroy(intptr_t handle):
    """Wrap ``cudssDestroy``."""
    with nogil:
        status = cudssDestroy(
            <cudssHandle_t>handle)
    check_status(status)


cpdef getProperty(int propertyType):
    """Wrap ``cudssGetProperty``. Return value."""
    cdef int value = <int>0
    with nogil:
        status = cudssGetProperty(
            <libraryPropertyType>propertyType,
            &value)
    check_status(status)
    return value


cpdef setStream(intptr_t handle, intptr_t stream):
    """Wrap ``cudssSetStream``."""
    with nogil:
        status = cudssSetStream(
            <cudssHandle_t>handle,
            <Stream_t>stream)
    check_status(status)


cpdef setMgStreams(intptr_t handle, intptr_t streams, int stream_count):
    """Wrap ``cudssSetMgStreams``."""
    with nogil:
        status = cudssSetMgStreams(
            <cudssHandle_t>handle,
            <const Stream_t*>streams,
            stream_count)
    check_status(status)


cpdef setDeviceMemHandler(intptr_t handle, intptr_t handler):
    """Wrap ``cudssSetDeviceMemHandler``."""
    with nogil:
        status = cudssSetDeviceMemHandler(
            <cudssHandle_t>handle,
            <const cudssDeviceMemHandler_t*>handler)
    check_status(status)


cpdef getDeviceMemHandler(intptr_t handle, intptr_t handler):
    """Wrap ``cudssGetDeviceMemHandler``."""
    with nogil:
        status = cudssGetDeviceMemHandler(
            <const cudssHandle_t>handle,
            <cudssDeviceMemHandler_t*>handler)
    check_status(status)


cpdef setCommLayer(intptr_t handle, filename):
    """Wrap ``cudssSetCommLayer``."""
    cdef bytes encoded = None if filename is None else filename.encode()
    cdef const char* path = NULL
    if encoded is not None:
        path = encoded
    with nogil:
        status = cudssSetCommLayer(
            <cudssHandle_t>handle,
            path)
    check_status(status)


cpdef setThreadingLayer(intptr_t handle, filename):
    """Wrap ``cudssSetThreadingLayer``."""
    cdef bytes encoded = None if filename is None else filename.encode()
    cdef const char* path = NULL
    if encoded is not None:
        path = encoded
    with nogil:
        status = cudssSetThreadingLayer(
            <cudssHandle_t>handle,
            path)
    check_status(status)


cpdef loggerSetCallback(intptr_t callback):
    """Wrap ``cudssLoggerSetCallback``."""
    with nogil:
        status = cudssLoggerSetCallback(
            <cudssLoggerCallback_t>callback)
    check_status(status)


cpdef loggerSetFile(intptr_t file):
    """Wrap ``cudssLoggerSetFile``."""
    with nogil:
        status = cudssLoggerSetFile(
            <FILE*>file)
    check_status(status)


cpdef loggerOpenFile(filename):
    """Wrap ``cudssLoggerOpenFile``."""
    cdef bytes encoded = None if filename is None else filename.encode()
    cdef const char* path = NULL
    if encoded is not None:
        path = encoded
    with nogil:
        status = cudssLoggerOpenFile(
            path)
    check_status(status)


cpdef loggerSetLevel(int level):
    """Wrap ``cudssLoggerSetLevel``."""
    with nogil:
        status = cudssLoggerSetLevel(
            level)
    check_status(status)


cpdef loggerSetMask(int mask):
    """Wrap ``cudssLoggerSetMask``."""
    with nogil:
        status = cudssLoggerSetMask(
            mask)
    check_status(status)


cpdef loggerForceDisable():
    """Wrap ``cudssLoggerForceDisable``."""
    with nogil:
        status = cudssLoggerForceDisable()
    check_status(status)


cpdef configCreate():
    """Wrap ``cudssConfigCreate``. Return config."""
    cdef cudssConfig_t config = NULL
    with nogil:
        status = cudssConfigCreate(
            &config)
    check_status(status)
    return <intptr_t>config


cpdef configDestroy(intptr_t config):
    """Wrap ``cudssConfigDestroy``."""
    with nogil:
        status = cudssConfigDestroy(
            <cudssConfig_t>config)
    check_status(status)


cpdef configSet(
        intptr_t config,
        cudssConfigParam_t param,
        intptr_t value,
        size_t sizeInBytes):
    """Wrap ``cudssConfigSet``."""
    with nogil:
        status = cudssConfigSet(
            <cudssConfig_t>config,
            param,
            <const void*>value,
            sizeInBytes)
    check_status(status)


cpdef configGet(
        intptr_t config,
        cudssConfigParam_t param,
        intptr_t value,
        size_t sizeInBytes):
    """Wrap ``cudssConfigGet``. Return sizeWritten."""
    cdef size_t sizeWritten = <size_t>0
    with nogil:
        status = cudssConfigGet(
            <const cudssConfig_t>config,
            param,
            <void*>value,
            sizeInBytes,
            &sizeWritten)
    check_status(status)
    return sizeWritten


cpdef dataCreate(intptr_t handle):
    """Wrap ``cudssDataCreate``. Return data."""
    cdef cudssData_t data = NULL
    with nogil:
        status = cudssDataCreate(
            <cudssHandle_t>handle,
            &data)
    check_status(status)
    return <intptr_t>data


cpdef dataDestroy(intptr_t handle, intptr_t data):
    """Wrap ``cudssDataDestroy``."""
    with nogil:
        status = cudssDataDestroy(
            <cudssHandle_t>handle,
            <cudssData_t>data)
    check_status(status)


cpdef dataSet(
        intptr_t handle,
        intptr_t data,
        cudssDataParam_t param,
        intptr_t value,
        size_t sizeInBytes):
    """Wrap ``cudssDataSet``."""
    with nogil:
        status = cudssDataSet(
            <const cudssHandle_t>handle,
            <cudssData_t>data,
            param,
            <const void*>value,
            sizeInBytes)
    check_status(status)


cpdef dataGet(
        intptr_t handle,
        intptr_t data,
        cudssDataParam_t param,
        intptr_t value,
        size_t sizeInBytes):
    """Wrap ``cudssDataGet``. Return sizeWritten."""
    cdef size_t sizeWritten = <size_t>0
    with nogil:
        status = cudssDataGet(
            <const cudssHandle_t>handle,
            <const cudssData_t>data,
            param,
            <void*>value,
            sizeInBytes,
            &sizeWritten)
    check_status(status)
    return sizeWritten


cpdef execute(
        intptr_t handle,
        int phase,
        intptr_t config,
        intptr_t data,
        intptr_t matrix,
        intptr_t solution,
        intptr_t rhs):
    """Wrap ``cudssExecute``."""
    with nogil:
        status = cudssExecute(
            <cudssHandle_t>handle,
            phase,
            <const cudssConfig_t>config,
            <cudssData_t>data,
            <const cudssMatrix_t>matrix,
            <cudssMatrix_t>solution,
            <const cudssMatrix_t>rhs)
    check_status(status)


cpdef matrixCreateDn(
        int64_t nrows,
        int64_t ncols,
        int64_t ld,
        intptr_t values,
        cudssDataType_t valueType,
        cudssLayout_t layout):
    """Wrap ``cudssMatrixCreateDn``. Return matrix."""
    cdef cudssMatrix_t matrix = NULL
    with nogil:
        status = cudssMatrixCreateDn(
            &matrix,
            nrows,
            ncols,
            ld,
            <const void*>values,
            valueType,
            layout)
    check_status(status)
    return <intptr_t>matrix


cpdef matrixCreateBatchDn(
        int64_t batchCount,
        intptr_t nrows,
        intptr_t ncols,
        intptr_t ld,
        intptr_t values,
        cudssDataType_t integerType,
        cudssDataType_t valueType,
        cudssLayout_t layout):
    """Wrap ``cudssMatrixCreateBatchDn``. Return matrix."""
    cdef cudssMatrix_t matrix = NULL
    with nogil:
        status = cudssMatrixCreateBatchDn(
            &matrix,
            batchCount,
            <const void*>nrows,
            <const void*>ncols,
            <const void*>ld,
            <const void*const*>values,
            integerType,
            valueType,
            layout)
    check_status(status)
    return <intptr_t>matrix


cpdef matrixCreateCsr(
        int64_t nrows,
        int64_t ncols,
        int64_t nnz,
        intptr_t rowStart,
        intptr_t rowEnd,
        intptr_t colIndices,
        intptr_t values,
        cudssDataType_t offsetType,
        cudssDataType_t indexType,
        cudssDataType_t valueType,
        cudssMatrixType_t mtype,
        cudssMatrixViewType_t mview,
        cudssIndexBase_t indexBase):
    """Wrap ``cudssMatrixCreateCsr``. Return matrix."""
    cdef cudssMatrix_t matrix = NULL
    with nogil:
        status = cudssMatrixCreateCsr(
            &matrix,
            nrows,
            ncols,
            nnz,
            <const void*>rowStart,
            <const void*>rowEnd,
            <const void*>colIndices,
            <const void*>values,
            offsetType,
            indexType,
            valueType,
            mtype,
            mview,
            indexBase)
    check_status(status)
    return <intptr_t>matrix


cpdef matrixCreateBatchCsr(
        int64_t batchCount,
        intptr_t nrows,
        intptr_t ncols,
        intptr_t nnz,
        intptr_t rowStart,
        intptr_t rowEnd,
        intptr_t colIndices,
        intptr_t values,
        cudssDataType_t offsetType,
        cudssDataType_t indexType,
        cudssDataType_t valueType,
        cudssMatrixType_t mtype,
        cudssMatrixViewType_t mview,
        cudssIndexBase_t indexBase):
    """Wrap ``cudssMatrixCreateBatchCsr``. Return matrix."""
    cdef cudssMatrix_t matrix = NULL
    with nogil:
        status = cudssMatrixCreateBatchCsr(
            &matrix,
            batchCount,
            <const void*>nrows,
            <const void*>ncols,
            <const void*>nnz,
            <const void*const*>rowStart,
            <const void*const*>rowEnd,
            <const void*const*>colIndices,
            <const void*const*>values,
            offsetType,
            indexType,
            valueType,
            mtype,
            mview,
            indexBase)
    check_status(status)
    return <intptr_t>matrix


cpdef matrixDestroy(intptr_t matrix):
    """Wrap ``cudssMatrixDestroy``."""
    with nogil:
        status = cudssMatrixDestroy(
            <cudssMatrix_t>matrix)
    check_status(status)


cpdef matrixSetValues(intptr_t matrix, intptr_t values):
    """Wrap ``cudssMatrixSetValues``."""
    with nogil:
        status = cudssMatrixSetValues(
            <cudssMatrix_t>matrix,
            <const void*>values)
    check_status(status)


cpdef matrixSetBatchValues(intptr_t matrix, intptr_t values):
    """Wrap ``cudssMatrixSetBatchValues``."""
    with nogil:
        status = cudssMatrixSetBatchValues(
            <cudssMatrix_t>matrix,
            <const void*const*>values)
    check_status(status)


cpdef matrixSetCsrPointers(
        intptr_t matrix,
        intptr_t rowStart,
        intptr_t rowEnd,
        intptr_t colIndices,
        intptr_t values):
    """Wrap ``cudssMatrixSetCsrPointers``."""
    with nogil:
        status = cudssMatrixSetCsrPointers(
            <cudssMatrix_t>matrix,
            <const void*>rowStart,
            <const void*>rowEnd,
            <const void*>colIndices,
            <const void*>values)
    check_status(status)


cpdef matrixSetBatchCsrPointers(
        intptr_t matrix,
        intptr_t rowStart,
        intptr_t rowEnd,
        intptr_t colIndices,
        intptr_t values):
    """Wrap ``cudssMatrixSetBatchCsrPointers``."""
    with nogil:
        status = cudssMatrixSetBatchCsrPointers(
            <cudssMatrix_t>matrix,
            <const void*const*>rowStart,
            <const void*const*>rowEnd,
            <const void*const*>colIndices,
            <const void*const*>values)
    check_status(status)


cpdef matrixSetDistributionRow1d(
        intptr_t matrix,
        int64_t first_row,
        int64_t last_row):
    """Wrap ``cudssMatrixSetDistributionRow1d``."""
    with nogil:
        status = cudssMatrixSetDistributionRow1d(
            <cudssMatrix_t>matrix,
            first_row,
            last_row)
    check_status(status)


cpdef matrixGetDn(intptr_t matrix):
    """Wrap ``cudssMatrixGetDn``.
    Return nrows, ncols, ld, values, valueType, layout.
    """
    cdef int64_t nrows = <int64_t>0
    cdef int64_t ncols = <int64_t>0
    cdef int64_t ld = <int64_t>0
    cdef void* values = NULL
    cdef cudssDataType_t valueType = <cudssDataType_t>0
    cdef cudssLayout_t layout = <cudssLayout_t>0
    with nogil:
        status = cudssMatrixGetDn(
            <const cudssMatrix_t>matrix,
            &nrows,
            &ncols,
            &ld,
            &values,
            &valueType,
            &layout)
    check_status(status)
    return nrows, ncols, ld, <intptr_t>values, valueType, layout


cpdef matrixGetBatchDn(intptr_t matrix):
    """Wrap ``cudssMatrixGetBatchDn``.
    Return batchCount, nrows, ncols, ld, values, indexType, valueType,
    layout.
    """
    cdef int64_t batchCount = <int64_t>0
    cdef void* nrows = NULL
    cdef void* ncols = NULL
    cdef void* ld = NULL
    cdef void** values = NULL
    cdef cudssDataType_t indexType = <cudssDataType_t>0
    cdef cudssDataType_t valueType = <cudssDataType_t>0
    cdef cudssLayout_t layout = <cudssLayout_t>0
    with nogil:
        status = cudssMatrixGetBatchDn(
            <const cudssMatrix_t>matrix,
            &batchCount,
            &nrows,
            &ncols,
            &ld,
            &values,
            &indexType,
            &valueType,
            &layout)
    check_status(status)
    return (
        batchCount,
        <intptr_t>nrows,
        <intptr_t>ncols,
        <intptr_t>ld,
        <intptr_t>values,
        indexType,
        valueType,
        layout,
    )


cpdef matrixGetCsr(intptr_t matrix):
    """Wrap ``cudssMatrixGetCsr``.
    Return nrows, ncols, nnz, rowStart, rowEnd, colIndices, values,
    offsetType, indexType, valueType, mtype, mview, indexBase.
    """
    cdef int64_t nrows = <int64_t>0
    cdef int64_t ncols = <int64_t>0
    cdef int64_t nnz = <int64_t>0
    cdef void* rowStart = NULL
    cdef void* rowEnd = NULL
    cdef void* colIndices = NULL
    cdef void* values = NULL
    cdef cudssDataType_t offsetType = <cudssDataType_t>0
    cdef cudssDataType_t indexType = <cudssDataType_t>0
    cdef cudssDataType_t valueType = <cudssDataType_t>0
    cdef cudssMatrixType_t mtype = <cudssMatrixType_t>0
    cdef cudssMatrixViewType_t mview = <cudssMatrixViewType_t>0
    cdef cudssIndexBase_t indexBase = <cudssIndexBase_t>0
    with nogil:
        status = cudssMatrixGetCsr(
            <const cudssMatrix_t>matrix,
            &nrows,
            &ncols,
            &nnz,
            &rowStart,
            &rowEnd,
            &colIndices,
            &values,
            &offsetType,
            &indexType,
            &valueType,
            &mtype,
            &mview,
            &indexBase)
    check_status(status)
    return (
        nrows,
        ncols,
        nnz,
        <intptr_t>rowStart,
        <intptr_t>rowEnd,
        <intptr_t>colIndices,
        <intptr_t>values,
        offsetType,
        indexType,
        valueType,
        mtype,
        mview,
        indexBase,
    )


cpdef matrixGetBatchCsr(intptr_t matrix):
    """Wrap ``cudssMatrixGetBatchCsr``.
    Return batchCount, nrows, ncols, nnz, rowStart, rowEnd, colIndices,
    values, offsetType, indexType, valueType, mtype, mview, indexBase.
    """
    cdef int64_t batchCount = <int64_t>0
    cdef void* nrows = NULL
    cdef void* ncols = NULL
    cdef void* nnz = NULL
    cdef void** rowStart = NULL
    cdef void** rowEnd = NULL
    cdef void** colIndices = NULL
    cdef void** values = NULL
    cdef cudssDataType_t offsetType = <cudssDataType_t>0
    cdef cudssDataType_t indexType = <cudssDataType_t>0
    cdef cudssDataType_t valueType = <cudssDataType_t>0
    cdef cudssMatrixType_t mtype = <cudssMatrixType_t>0
    cdef cudssMatrixViewType_t mview = <cudssMatrixViewType_t>0
    cdef cudssIndexBase_t indexBase = <cudssIndexBase_t>0
    with nogil:
        status = cudssMatrixGetBatchCsr(
            <const cudssMatrix_t>matrix,
            &batchCount,
            &nrows,
            &ncols,
            &nnz,
            &rowStart,
            &rowEnd,
            &colIndices,
            &values,
            &offsetType,
            &indexType,
            &valueType,
            &mtype,
            &mview,
            &indexBase)
    check_status(status)
    return (
        batchCount,
        <intptr_t>nrows,
        <intptr_t>ncols,
        <intptr_t>nnz,
        <intptr_t>rowStart,
        <intptr_t>rowEnd,
        <intptr_t>colIndices,
        <intptr_t>values,
        offsetType,
        indexType,
        valueType,
        mtype,
        mview,
        indexBase,
    )


cpdef matrixGetDistributionRow1d(intptr_t matrix):
    """Wrap ``cudssMatrixGetDistributionRow1d``. Return first_row, last_row."""
    cdef int64_t first_row = <int64_t>0
    cdef int64_t last_row = <int64_t>0
    with nogil:
        status = cudssMatrixGetDistributionRow1d(
            <const cudssMatrix_t>matrix,
            &first_row,
            &last_row)
    check_status(status)
    return first_row, last_row


cpdef matrixGetFormat(intptr_t matrix):
    """Wrap ``cudssMatrixGetFormat``. Return format."""
    cdef int format = <int>0
    with nogil:
        status = cudssMatrixGetFormat(
            <const cudssMatrix_t>matrix,
            &format)
    check_status(status)
    return format
