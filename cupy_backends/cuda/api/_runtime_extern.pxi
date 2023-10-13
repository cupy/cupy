import sys as _sys

from cupy_backends.cuda.api cimport driver
from cupy_backends.cuda._softlink cimport SoftLink


# Error handling
ctypedef const char* (*F_cudaGetErrorName)(Error error) nogil
cdef F_cudaGetErrorName cudaGetErrorName

ctypedef const char* (*F_cudaGetErrorString)(Error error) nogil
cdef F_cudaGetErrorString cudaGetErrorString

ctypedef int (*F_cudaGetLastError)() nogil
cdef F_cudaGetLastError cudaGetLastError

# Initialization
ctypedef int (*F_cudaDriverGetVersion)(int* driverVersion) nogil
cdef F_cudaDriverGetVersion cudaDriverGetVersion

ctypedef int (*F_cudaRuntimeGetVersion)(int* runtimeVersion) nogil
cdef F_cudaRuntimeGetVersion cudaRuntimeGetVersion

# Device operations
ctypedef int (*F_cudaGetDevice)(int* device) nogil
cdef F_cudaGetDevice cudaGetDevice

ctypedef int (*F_cudaDeviceGetAttribute)(
    int* value, DeviceAttr attr, int device) nogil
cdef F_cudaDeviceGetAttribute cudaDeviceGetAttribute

ctypedef int (*F_cudaDeviceGetByPCIBusId)(
    int* device, const char* pciBusId) nogil
cdef F_cudaDeviceGetByPCIBusId cudaDeviceGetByPCIBusId

ctypedef int (*F_cudaDeviceGetPCIBusId)(
    char* pciBusId, int len, int device) nogil
cdef F_cudaDeviceGetPCIBusId cudaDeviceGetPCIBusId

ctypedef int (*F_cudaGetDeviceProperties)(DeviceProp* prop, int device) nogil
cdef F_cudaGetDeviceProperties cudaGetDeviceProperties

ctypedef int (*F_cudaGetDeviceCount)(int* count) nogil
cdef F_cudaGetDeviceCount cudaGetDeviceCount

ctypedef int (*F_cudaSetDevice)(int device) nogil
cdef F_cudaSetDevice cudaSetDevice

ctypedef int (*F_cudaDeviceSynchronize)() nogil
cdef F_cudaDeviceSynchronize cudaDeviceSynchronize

# Peer Access
ctypedef int (*F_cudaDeviceCanAccessPeer)(
    int* canAccessPeer, int device, int peerDevice) nogil
cdef F_cudaDeviceCanAccessPeer cudaDeviceCanAccessPeer

ctypedef int (*F_cudaDeviceEnablePeerAccess)(
    int peerDevice, unsigned int flags) nogil
cdef F_cudaDeviceEnablePeerAccess cudaDeviceEnablePeerAccess

ctypedef int (*F_cudaDeviceDisablePeerAccess)(int peerDevice) nogil
cdef F_cudaDeviceDisablePeerAccess cudaDeviceDisablePeerAccess

# Limits
ctypedef int (*F_cudaDeviceGetLimit)(size_t* value, Limit limit) nogil
cdef F_cudaDeviceGetLimit cudaDeviceGetLimit

ctypedef int (*F_cudaDeviceSetLimit)(Limit limit, size_t value) nogil
cdef F_cudaDeviceSetLimit cudaDeviceSetLimit


# IPC
ctypedef int (*F_cudaIpcCloseMemHandle)(void* devPtr) nogil
cdef F_cudaIpcCloseMemHandle cudaIpcCloseMemHandle

ctypedef int (*F_cudaIpcGetEventHandle)(
    IpcEventHandle* handle, driver.Event event) nogil
cdef F_cudaIpcGetEventHandle cudaIpcGetEventHandle

ctypedef int (*F_cudaIpcGetMemHandle)(IpcMemHandle*, void* devPtr) nogil
cdef F_cudaIpcGetMemHandle cudaIpcGetMemHandle

ctypedef int (*F_cudaIpcOpenEventHandle)(
    driver.Event* event, IpcEventHandle handle) nogil
cdef F_cudaIpcOpenEventHandle cudaIpcOpenEventHandle

ctypedef int (*F_cudaIpcOpenMemHandle)(
    void** devPtr, IpcMemHandle handle, unsigned int  flags) nogil
cdef F_cudaIpcOpenMemHandle cudaIpcOpenMemHandle


# Memory management
ctypedef int (*F_cudaMalloc)(void** devPtr, size_t size) nogil
cdef F_cudaMalloc cudaMalloc

ctypedef int (*F_cudaMallocManaged)(
    void** devPtr, size_t size, unsigned int flags) nogil
cdef F_cudaMallocManaged cudaMallocManaged

ctypedef int (*F_cudaMalloc3DArray)(
    Array* array, const ChannelFormatDesc* desc, Extent extent,
    unsigned int flags) nogil
cdef F_cudaMalloc3DArray cudaMalloc3DArray

ctypedef int (*F_cudaMallocArray)(
    Array* array, const ChannelFormatDesc* desc, size_t width, size_t height,
    unsigned int flags) nogil
cdef F_cudaMallocArray cudaMallocArray

ctypedef int (*F_cudaMallocAsync)(void**, size_t, driver.Stream) nogil
cdef F_cudaMallocAsync cudaMallocAsync

ctypedef int (*F_cudaMallocFromPoolAsync)(
    void**, size_t, MemPool, driver.Stream) nogil
cdef F_cudaMallocFromPoolAsync cudaMallocFromPoolAsync

ctypedef int (*F_cudaHostAlloc)(
    void** ptr, size_t size, unsigned int flags) nogil
cdef F_cudaHostAlloc cudaHostAlloc

ctypedef int (*F_cudaHostRegister)(
    void *ptr, size_t size, unsigned int flags) nogil
cdef F_cudaHostRegister cudaHostRegister

ctypedef int (*F_cudaHostUnregister)(void *ptr) nogil
cdef F_cudaHostUnregister cudaHostUnregister

ctypedef int (*F_cudaFree)(void* devPtr) nogil
cdef F_cudaFree cudaFree

ctypedef int (*F_cudaFreeHost)(void* ptr) nogil
cdef F_cudaFreeHost cudaFreeHost

ctypedef int (*F_cudaFreeArray)(Array array) nogil
cdef F_cudaFreeArray cudaFreeArray

ctypedef int (*F_cudaFreeAsync)(void*, driver.Stream) nogil
cdef F_cudaFreeAsync cudaFreeAsync

ctypedef int (*F_cudaMemGetInfo)(size_t* free, size_t* total) nogil
cdef F_cudaMemGetInfo cudaMemGetInfo

ctypedef int (*F_cudaMemcpy)(
    void* dst, const void* src, size_t count, MemoryKind kind) nogil
cdef F_cudaMemcpy cudaMemcpy

ctypedef int (*F_cudaMemcpyAsync)(
    void* dst, const void* src, size_t count, MemoryKind kind,
    driver.Stream stream) nogil
cdef F_cudaMemcpyAsync cudaMemcpyAsync

ctypedef int (*F_cudaMemcpyPeer)(
    void* dst, int dstDevice, const void* src, int srcDevice,
    size_t count) nogil
cdef F_cudaMemcpyPeer cudaMemcpyPeer

ctypedef int (*F_cudaMemcpyPeerAsync)(
    void* dst, int dstDevice, const void* src, int srcDevice, size_t count,
    driver.Stream stream) nogil
cdef F_cudaMemcpyPeerAsync cudaMemcpyPeerAsync

ctypedef int (*F_cudaMemcpy2DFromArray)(
    void* dst, size_t dpitch, Array src, size_t wOffset, size_t hOffset,
    size_t width, size_t height, MemoryKind kind) nogil
cdef F_cudaMemcpy2DFromArray cudaMemcpy2DFromArray

ctypedef int (*F_cudaMemcpy2DFromArrayAsync)(
    void* dst, size_t dpitch, Array src, size_t wOffset, size_t hOffset,
    size_t width, size_t height, MemoryKind kind, driver.Stream stream) nogil
cdef F_cudaMemcpy2DFromArrayAsync cudaMemcpy2DFromArrayAsync

ctypedef int (*F_cudaMemcpy2DToArray)(
    Array dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch,
    size_t width, size_t height, MemoryKind kind) nogil
cdef F_cudaMemcpy2DToArray cudaMemcpy2DToArray

ctypedef int (*F_cudaMemcpy2DToArrayAsync)(
    Array dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch,
    size_t width, size_t height, MemoryKind kind, driver.Stream stream) nogil
cdef F_cudaMemcpy2DToArrayAsync cudaMemcpy2DToArrayAsync

ctypedef int (*F_cudaMemcpy2D)(
    void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
    size_t height, MemoryKind kind) nogil
cdef F_cudaMemcpy2D cudaMemcpy2D

ctypedef int (*F_cudaMemcpy2DAsync)(
    void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
    size_t height, MemoryKind kind, driver.Stream stream) nogil
cdef F_cudaMemcpy2DAsync cudaMemcpy2DAsync

ctypedef int (*F_cudaMemcpy3D)(Memcpy3DParms* Memcpy3DParmsPtr) nogil
cdef F_cudaMemcpy3D cudaMemcpy3D

ctypedef int (*F_cudaMemcpy3DAsync)(
    Memcpy3DParms* Memcpy3DParmsPtr, driver.Stream stream) nogil
cdef F_cudaMemcpy3DAsync cudaMemcpy3DAsync

ctypedef int (*F_cudaMemset)(void* devPtr, int value, size_t count) nogil
cdef F_cudaMemset cudaMemset

ctypedef int (*F_cudaMemsetAsync)(
    void* devPtr, int value, size_t count, driver.Stream stream) nogil
cdef F_cudaMemsetAsync cudaMemsetAsync

ctypedef int (*F_cudaMemPrefetchAsync)(
    const void *devPtr, size_t count, int dstDevice,
    driver.Stream stream) nogil
cdef F_cudaMemPrefetchAsync cudaMemPrefetchAsync

ctypedef int (*F_cudaMemAdvise)(
    const void *devPtr, size_t count, MemoryAdvise advice, int device) nogil
cdef F_cudaMemAdvise cudaMemAdvise

ctypedef int (*F_cudaDeviceGetDefaultMemPool)(MemPool*, int) nogil
cdef F_cudaDeviceGetDefaultMemPool cudaDeviceGetDefaultMemPool

ctypedef int (*F_cudaDeviceGetMemPool)(MemPool*, int) nogil
cdef F_cudaDeviceGetMemPool cudaDeviceGetMemPool

ctypedef int (*F_cudaDeviceSetMemPool)(int, MemPool) nogil
cdef F_cudaDeviceSetMemPool cudaDeviceSetMemPool

ctypedef int (*F_cudaMemPoolCreate)(MemPool*, _MemPoolProps*) nogil
cdef F_cudaMemPoolCreate cudaMemPoolCreate

ctypedef int (*F_cudaMemPoolDestroy)(MemPool) nogil
cdef F_cudaMemPoolDestroy cudaMemPoolDestroy

ctypedef int (*F_cudaMemPoolTrimTo)(MemPool, size_t) nogil
cdef F_cudaMemPoolTrimTo cudaMemPoolTrimTo

ctypedef int (*F_cudaMemPoolGetAttribute)(MemPool, MemPoolAttr, void*) nogil
cdef F_cudaMemPoolGetAttribute cudaMemPoolGetAttribute

ctypedef int (*F_cudaMemPoolSetAttribute)(MemPool, MemPoolAttr, void*) nogil
cdef F_cudaMemPoolSetAttribute cudaMemPoolSetAttribute

ctypedef int (*F_cudaPointerGetAttributes)(
    _PointerAttributes* attributes, const void* ptr) nogil
cdef F_cudaPointerGetAttributes cudaPointerGetAttributes


# Stream and Event
ctypedef int (*F_cudaStreamCreate)(driver.Stream* pStream) nogil
cdef F_cudaStreamCreate cudaStreamCreate

ctypedef int (*F_cudaStreamCreateWithFlags)(
    driver.Stream* pStream, unsigned int flags) nogil
cdef F_cudaStreamCreateWithFlags cudaStreamCreateWithFlags

ctypedef int (*F_cudaStreamDestroy)(driver.Stream stream) nogil
cdef F_cudaStreamDestroy cudaStreamDestroy

ctypedef int (*F_cudaStreamSynchronize)(driver.Stream stream) nogil
cdef F_cudaStreamSynchronize cudaStreamSynchronize

ctypedef int (*F_cudaStreamAddCallback)(
    driver.Stream stream, StreamCallback callback, void* userData,
    unsigned int flags) nogil
cdef F_cudaStreamAddCallback cudaStreamAddCallback

ctypedef int (*F_cudaLaunchHostFunc)(
    driver.Stream stream, HostFn fn, void* userData) nogil
cdef F_cudaLaunchHostFunc cudaLaunchHostFunc

ctypedef int (*F_cudaStreamQuery)(driver.Stream stream) nogil
cdef F_cudaStreamQuery cudaStreamQuery

ctypedef int (*F_cudaStreamWaitEvent)(
    driver.Stream stream, driver.Event event, unsigned int flags) nogil
cdef F_cudaStreamWaitEvent cudaStreamWaitEvent

ctypedef int (*F_cudaStreamBeginCapture)(
    driver.Stream stream, StreamCaptureMode mode) nogil
cdef F_cudaStreamBeginCapture cudaStreamBeginCapture

ctypedef int (*F_cudaStreamEndCapture)(driver.Stream stream, Graph*) nogil
cdef F_cudaStreamEndCapture cudaStreamEndCapture

ctypedef int (*F_cudaStreamIsCapturing)(
    driver.Stream stream, StreamCaptureStatus*) nogil
cdef F_cudaStreamIsCapturing cudaStreamIsCapturing

ctypedef int (*F_cudaEventCreate)(driver.Event* event) nogil
cdef F_cudaEventCreate cudaEventCreate

ctypedef int (*F_cudaEventCreateWithFlags)(
    driver.Event* event, unsigned int flags) nogil
cdef F_cudaEventCreateWithFlags cudaEventCreateWithFlags

ctypedef int (*F_cudaEventDestroy)(driver.Event event) nogil
cdef F_cudaEventDestroy cudaEventDestroy

ctypedef int (*F_cudaEventElapsedTime)(
    float* ms, driver.Event start, driver.Event end) nogil
cdef F_cudaEventElapsedTime cudaEventElapsedTime

ctypedef int (*F_cudaEventQuery)(driver.Event event) nogil
cdef F_cudaEventQuery cudaEventQuery

ctypedef int (*F_cudaEventRecord)(
    driver.Event event, driver.Stream stream) nogil
cdef F_cudaEventRecord cudaEventRecord

ctypedef int (*F_cudaEventSynchronize)(driver.Event event) nogil
cdef F_cudaEventSynchronize cudaEventSynchronize


# Texture
ctypedef int (*F_cudaCreateTextureObject)(
    TextureObject* pTexObject, const ResourceDesc* pResDesc,
    const TextureDesc* pTexDesc, const ResourceViewDesc* pResViewDesc) nogil
cdef F_cudaCreateTextureObject cudaCreateTextureObject

ctypedef int (*F_cudaDestroyTextureObject)(TextureObject texObject) nogil
cdef F_cudaDestroyTextureObject cudaDestroyTextureObject

ctypedef int (*F_cudaGetChannelDesc)(
    ChannelFormatDesc* desc, Array array) nogil
cdef F_cudaGetChannelDesc cudaGetChannelDesc

ctypedef int (*F_cudaGetTextureObjectResourceDesc)(
    ResourceDesc* desc, TextureObject obj) nogil
cdef F_cudaGetTextureObjectResourceDesc cudaGetTextureObjectResourceDesc

ctypedef int (*F_cudaGetTextureObjectTextureDesc)(
    TextureDesc* desc, TextureObject obj) nogil
cdef F_cudaGetTextureObjectTextureDesc cudaGetTextureObjectTextureDesc


# Surface
ctypedef int (*F_cudaCreateSurfaceObject)(
    SurfaceObject* pSurObject, const ResourceDesc* pResDesc) nogil
cdef F_cudaCreateSurfaceObject cudaCreateSurfaceObject

ctypedef int (*F_cudaDestroySurfaceObject)(SurfaceObject surObject) nogil
cdef F_cudaDestroySurfaceObject cudaDestroySurfaceObject


# Graph
ctypedef int (*F_cudaGraphDestroy)(Graph graph) nogil
cdef F_cudaGraphDestroy cudaGraphDestroy

ctypedef int (*F_cudaGraphExecDestroy)(GraphExec graph) nogil
cdef F_cudaGraphExecDestroy cudaGraphExecDestroy

ctypedef int (*F_cudaGraphInstantiate)(
    GraphExec*, Graph, GraphNode*, char*, size_t) nogil
cdef F_cudaGraphInstantiate cudaGraphInstantiate

ctypedef int (*F_cudaGraphLaunch)(GraphExec, driver.Stream) nogil
cdef F_cudaGraphLaunch cudaGraphLaunch

ctypedef int (*F_cudaGraphUpload)(GraphExec, driver.Stream) nogil
cdef F_cudaGraphUpload cudaGraphUpload


cdef extern from '../../cupy_backend_runtime.h' nogil:
    # Inline functions
    Extent make_cudaExtent(size_t w, size_t h, size_t d)
    Pos make_cudaPos(size_t x, size_t y, size_t z)
    PitchedPtr make_cudaPitchedPtr(void* d, size_t p, size_t xsz, size_t ysz)

    # Constants
    int cudaDevAttrComputeCapabilityMajor
    int cudaDevAttrComputeCapabilityMinor

    # Error code
    int cudaErrorMemoryAllocation
    int cudaErrorInvalidValue
    int cudaErrorPeerAccessAlreadyEnabled
    int cudaErrorContextIsDestroyed
    int cudaErrorInvalidResourceHandle


cdef SoftLink _L = None
cdef inline void initialize() except *:
    global _L
    if _L is not None:
        return
    _initialize()


cdef void _initialize() except *:
    global _L
    _L = _get_softlink()

    global cudaGetErrorName
    cudaGetErrorName = <F_cudaGetErrorName>_L.get('GetErrorName')
    global cudaGetErrorString
    cudaGetErrorString = <F_cudaGetErrorString>_L.get('GetErrorString')
    global cudaGetLastError
    cudaGetLastError = <F_cudaGetLastError>_L.get('GetLastError')
    global cudaDriverGetVersion
    cudaDriverGetVersion = <F_cudaDriverGetVersion>_L.get('DriverGetVersion')
    global cudaRuntimeGetVersion
    cudaRuntimeGetVersion = <F_cudaRuntimeGetVersion>_L.get('RuntimeGetVersion')  # noqa
    global cudaGetDevice
    cudaGetDevice = <F_cudaGetDevice>_L.get('GetDevice')
    global cudaDeviceGetAttribute
    cudaDeviceGetAttribute = <F_cudaDeviceGetAttribute>_L.get('DeviceGetAttribute')  # noqa
    global cudaDeviceGetByPCIBusId
    cudaDeviceGetByPCIBusId = <F_cudaDeviceGetByPCIBusId>_L.get('DeviceGetByPCIBusId')  # noqa
    global cudaDeviceGetPCIBusId
    cudaDeviceGetPCIBusId = <F_cudaDeviceGetPCIBusId>_L.get('DeviceGetPCIBusId')  # noqa
    global cudaGetDeviceProperties
    cudaGetDeviceProperties = <F_cudaGetDeviceProperties>_L.get('GetDeviceProperties')  # noqa
    global cudaGetDeviceCount
    cudaGetDeviceCount = <F_cudaGetDeviceCount>_L.get('GetDeviceCount')
    global cudaSetDevice
    cudaSetDevice = <F_cudaSetDevice>_L.get('SetDevice')
    global cudaDeviceSynchronize
    cudaDeviceSynchronize = <F_cudaDeviceSynchronize>_L.get('DeviceSynchronize')  # noqa
    global cudaDeviceCanAccessPeer
    cudaDeviceCanAccessPeer = <F_cudaDeviceCanAccessPeer>_L.get('DeviceCanAccessPeer')  # noqa
    global cudaDeviceEnablePeerAccess
    cudaDeviceEnablePeerAccess = <F_cudaDeviceEnablePeerAccess>_L.get('DeviceEnablePeerAccess')  # noqa
    global cudaDeviceDisablePeerAccess
    cudaDeviceDisablePeerAccess = <F_cudaDeviceDisablePeerAccess>_L.get('DeviceDisablePeerAccess')  # noqa
    global cudaDeviceGetLimit
    cudaDeviceGetLimit = <F_cudaDeviceGetLimit>_L.get('DeviceGetLimit')
    global cudaDeviceSetLimit
    cudaDeviceSetLimit = <F_cudaDeviceSetLimit>_L.get('DeviceSetLimit')
    global cudaIpcCloseMemHandle
    cudaIpcCloseMemHandle = <F_cudaIpcCloseMemHandle>_L.get('IpcCloseMemHandle')  # noqa
    global cudaIpcGetEventHandle
    cudaIpcGetEventHandle = <F_cudaIpcGetEventHandle>_L.get('IpcGetEventHandle')  # noqa
    global cudaIpcGetMemHandle
    cudaIpcGetMemHandle = <F_cudaIpcGetMemHandle>_L.get('IpcGetMemHandle')
    global cudaIpcOpenEventHandle
    cudaIpcOpenEventHandle = <F_cudaIpcOpenEventHandle>_L.get('IpcOpenEventHandle')  # noqa
    global cudaIpcOpenMemHandle
    cudaIpcOpenMemHandle = <F_cudaIpcOpenMemHandle>_L.get('IpcOpenMemHandle')
    global cudaMalloc
    cudaMalloc = <F_cudaMalloc>_L.get('Malloc')
    global cudaMallocManaged
    cudaMallocManaged = <F_cudaMallocManaged>_L.get('MallocManaged')
    global cudaMalloc3DArray
    cudaMalloc3DArray = <F_cudaMalloc3DArray>_L.get('Malloc3DArray')
    global cudaMallocArray
    cudaMallocArray = <F_cudaMallocArray>_L.get('MallocArray')
    global cudaMallocAsync
    cudaMallocAsync = <F_cudaMallocAsync>_L.get('MallocAsync')
    global cudaMallocFromPoolAsync
    cudaMallocFromPoolAsync = <F_cudaMallocFromPoolAsync>_L.get('MallocFromPoolAsync')  # noqa
    global cudaHostAlloc
    cudaHostAlloc = <F_cudaHostAlloc>_L.get('HostAlloc')
    global cudaHostRegister
    cudaHostRegister = <F_cudaHostRegister>_L.get('HostRegister')
    global cudaHostUnregister
    cudaHostUnregister = <F_cudaHostUnregister>_L.get('HostUnregister')
    global cudaFree
    cudaFree = <F_cudaFree>_L.get('Free')
    global cudaFreeHost
    cudaFreeHost = <F_cudaFreeHost>_L.get('FreeHost')
    global cudaFreeArray
    cudaFreeArray = <F_cudaFreeArray>_L.get('FreeArray')
    global cudaFreeAsync
    cudaFreeAsync = <F_cudaFreeAsync>_L.get('FreeAsync')
    global cudaMemGetInfo
    cudaMemGetInfo = <F_cudaMemGetInfo>_L.get('MemGetInfo')
    global cudaMemcpy
    cudaMemcpy = <F_cudaMemcpy>_L.get('Memcpy')
    global cudaMemcpyAsync
    cudaMemcpyAsync = <F_cudaMemcpyAsync>_L.get('MemcpyAsync')
    global cudaMemcpyPeer
    cudaMemcpyPeer = <F_cudaMemcpyPeer>_L.get('MemcpyPeer')
    global cudaMemcpyPeerAsync
    cudaMemcpyPeerAsync = <F_cudaMemcpyPeerAsync>_L.get('MemcpyPeerAsync')
    global cudaMemcpy2DFromArray
    cudaMemcpy2DFromArray = <F_cudaMemcpy2DFromArray>_L.get('Memcpy2DFromArray')  # noqa
    global cudaMemcpy2DFromArrayAsync
    cudaMemcpy2DFromArrayAsync = <F_cudaMemcpy2DFromArrayAsync>_L.get('Memcpy2DFromArrayAsync')  # noqa
    global cudaMemcpy2DToArray
    cudaMemcpy2DToArray = <F_cudaMemcpy2DToArray>_L.get('Memcpy2DToArray')
    global cudaMemcpy2DToArrayAsync
    cudaMemcpy2DToArrayAsync = <F_cudaMemcpy2DToArrayAsync>_L.get('Memcpy2DToArrayAsync')  # noqa
    global cudaMemcpy2D
    cudaMemcpy2D = <F_cudaMemcpy2D>_L.get('Memcpy2D')
    global cudaMemcpy2DAsync
    cudaMemcpy2DAsync = <F_cudaMemcpy2DAsync>_L.get('Memcpy2DAsync')
    global cudaMemcpy3D
    cudaMemcpy3D = <F_cudaMemcpy3D>_L.get('Memcpy3D')
    global cudaMemcpy3DAsync
    cudaMemcpy3DAsync = <F_cudaMemcpy3DAsync>_L.get('Memcpy3DAsync')
    global cudaMemset
    cudaMemset = <F_cudaMemset>_L.get('Memset')
    global cudaMemsetAsync
    cudaMemsetAsync = <F_cudaMemsetAsync>_L.get('MemsetAsync')
    global cudaMemPrefetchAsync
    cudaMemPrefetchAsync = <F_cudaMemPrefetchAsync>_L.get('MemPrefetchAsync')
    global cudaMemAdvise
    cudaMemAdvise = <F_cudaMemAdvise>_L.get('MemAdvise')
    global cudaDeviceGetDefaultMemPool
    cudaDeviceGetDefaultMemPool = <F_cudaDeviceGetDefaultMemPool>_L.get('DeviceGetDefaultMemPool')  # noqa
    global cudaDeviceGetMemPool
    cudaDeviceGetMemPool = <F_cudaDeviceGetMemPool>_L.get('DeviceGetMemPool')
    global cudaDeviceSetMemPool
    cudaDeviceSetMemPool = <F_cudaDeviceSetMemPool>_L.get('DeviceSetMemPool')
    global cudaMemPoolCreate
    cudaMemPoolCreate = <F_cudaMemPoolCreate>_L.get('MemPoolCreate')
    global cudaMemPoolDestroy
    cudaMemPoolDestroy = <F_cudaMemPoolDestroy>_L.get('MemPoolDestroy')
    global cudaMemPoolTrimTo
    cudaMemPoolTrimTo = <F_cudaMemPoolTrimTo>_L.get('MemPoolTrimTo')
    global cudaMemPoolGetAttribute
    cudaMemPoolGetAttribute = <F_cudaMemPoolGetAttribute>_L.get('MemPoolGetAttribute')  # noqa
    global cudaMemPoolSetAttribute
    cudaMemPoolSetAttribute = <F_cudaMemPoolSetAttribute>_L.get('MemPoolSetAttribute')  # noqa
    global cudaPointerGetAttributes
    cudaPointerGetAttributes = <F_cudaPointerGetAttributes>_L.get('PointerGetAttributes')  # noqa
    global cudaStreamCreate
    cudaStreamCreate = <F_cudaStreamCreate>_L.get('StreamCreate')
    global cudaStreamCreateWithFlags
    cudaStreamCreateWithFlags = <F_cudaStreamCreateWithFlags>_L.get('StreamCreateWithFlags')  # noqa
    global cudaStreamDestroy
    cudaStreamDestroy = <F_cudaStreamDestroy>_L.get('StreamDestroy')
    global cudaStreamSynchronize
    cudaStreamSynchronize = <F_cudaStreamSynchronize>_L.get('StreamSynchronize')  # noqa
    global cudaStreamAddCallback
    cudaStreamAddCallback = <F_cudaStreamAddCallback>_L.get('StreamAddCallback')  # noqa
    global cudaLaunchHostFunc
    cudaLaunchHostFunc = <F_cudaLaunchHostFunc>_L.get('LaunchHostFunc')
    global cudaStreamQuery
    cudaStreamQuery = <F_cudaStreamQuery>_L.get('StreamQuery')
    global cudaStreamWaitEvent
    cudaStreamWaitEvent = <F_cudaStreamWaitEvent>_L.get('StreamWaitEvent')
    global cudaStreamBeginCapture
    cudaStreamBeginCapture = <F_cudaStreamBeginCapture>_L.get('StreamBeginCapture')  # noqa
    global cudaStreamEndCapture
    cudaStreamEndCapture = <F_cudaStreamEndCapture>_L.get('StreamEndCapture')
    global cudaStreamIsCapturing
    cudaStreamIsCapturing = <F_cudaStreamIsCapturing>_L.get('StreamIsCapturing')  # noqa
    global cudaEventCreate
    cudaEventCreate = <F_cudaEventCreate>_L.get('EventCreate')
    global cudaEventCreateWithFlags
    cudaEventCreateWithFlags = <F_cudaEventCreateWithFlags>_L.get('EventCreateWithFlags')  # noqa
    global cudaEventDestroy
    cudaEventDestroy = <F_cudaEventDestroy>_L.get('EventDestroy')
    global cudaEventElapsedTime
    cudaEventElapsedTime = <F_cudaEventElapsedTime>_L.get('EventElapsedTime')
    global cudaEventQuery
    cudaEventQuery = <F_cudaEventQuery>_L.get('EventQuery')
    global cudaEventRecord
    cudaEventRecord = <F_cudaEventRecord>_L.get('EventRecord')
    global cudaEventSynchronize
    cudaEventSynchronize = <F_cudaEventSynchronize>_L.get('EventSynchronize')
    global cudaCreateTextureObject
    cudaCreateTextureObject = <F_cudaCreateTextureObject>_L.get('CreateTextureObject')  # noqa
    global cudaDestroyTextureObject
    cudaDestroyTextureObject = <F_cudaDestroyTextureObject>_L.get('DestroyTextureObject')  # noqa
    global cudaGetChannelDesc
    cudaGetChannelDesc = <F_cudaGetChannelDesc>_L.get('GetChannelDesc')
    global cudaGetTextureObjectResourceDesc
    cudaGetTextureObjectResourceDesc = <F_cudaGetTextureObjectResourceDesc>_L.get('GetTextureObjectResourceDesc')  # noqa
    global cudaGetTextureObjectTextureDesc
    cudaGetTextureObjectTextureDesc = <F_cudaGetTextureObjectTextureDesc>_L.get('GetTextureObjectTextureDesc')  # noqa
    global cudaCreateSurfaceObject
    cudaCreateSurfaceObject = <F_cudaCreateSurfaceObject>_L.get('CreateSurfaceObject')  # noqa
    global cudaDestroySurfaceObject
    cudaDestroySurfaceObject = <F_cudaDestroySurfaceObject>_L.get('DestroySurfaceObject')  # noqa
    global cudaGraphDestroy
    cudaGraphDestroy = <F_cudaGraphDestroy>_L.get('GraphDestroy')
    global cudaGraphExecDestroy
    cudaGraphExecDestroy = <F_cudaGraphExecDestroy>_L.get('GraphExecDestroy')
    global cudaGraphInstantiate
    cudaGraphInstantiate = <F_cudaGraphInstantiate>_L.get('GraphInstantiate')
    global cudaGraphLaunch
    cudaGraphLaunch = <F_cudaGraphLaunch>_L.get('GraphLaunch')
    global cudaGraphUpload
    cudaGraphUpload = <F_cudaGraphUpload>_L.get('GraphUpload')


cdef SoftLink _get_softlink():
    cdef int runtime_version
    cdef str prefix = 'cuda'
    cdef object libname = None

    if CUPY_CUDA_VERSION != 0:
        if 11020 <= CUPY_CUDA_VERSION < 12000:
            # CUDA 11.x (11.2+)
            if _sys.platform == 'linux':
                libname = 'libcudart.so.11.0'
            else:
                libname = 'cudart64_110.dll'
        elif 12000 <= CUPY_CUDA_VERSION < 13000:
            # CUDA 12.x
            if _sys.platform == 'linux':
                libname = 'libcudart.so.12'
            else:
                libname = 'cudart64_12.dll'
    elif CUPY_HIP_VERSION != 0:
        prefix = 'hip'
        if CUPY_HIP_VERSION < 5_00_00000:
            # ROCm 4.x
            libname = 'libamdhip64.so.4'
        elif CUPY_HIP_VERSION < 6_00_00000:
            # ROCm 5.x
            libname = 'libamdhip64.so.5'

    return SoftLink(libname, prefix, mandatory=True)
