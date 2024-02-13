from libc.stdint cimport intptr_t, uintmax_t


###############################################################################
# Classes
###############################################################################

cdef class PointerAttributes:
    cdef:
        readonly int device
        readonly intptr_t devicePointer
        readonly intptr_t hostPointer
        readonly int type

cdef class MemPoolProps:
    # flatten the struct & list meaningful members
    cdef:
        int allocType
        int handleType
        int locationType
        int devId


###############################################################################
# Types and Enums
###############################################################################

IF CUPY_USE_CUDA_PYTHON:
    from cuda.ccudart cimport *
    # Aliases for compatibillity with existing CuPy codebase.
    # Keep in sync with names defined in `_runtime_typedef.pxi`.
    # TODO(kmaehashi): Remove these aliases.
    ctypedef cudaError_t Error
    ctypedef cudaDataType_t DataType

    ctypedef cudaDeviceAttr DeviceAttr
    ctypedef cudaMemoryAdvise MemoryAdvise

    ctypedef cudaStream_t Stream
    ctypedef cudaStreamCallback_t StreamCallback
    ctypedef cudaHostFn_t HostFn

    ctypedef cudaChannelFormatKind ChannelFormatKind
    ctypedef cudaChannelFormatDesc ChannelFormatDesc
    ctypedef cudaTextureObject_t TextureObject
    ctypedef cudaSurfaceObject_t SurfaceObject
    ctypedef cudaResourceType ResourceType
    ctypedef cudaTextureAddressMode TextureAddressMode
    ctypedef cudaTextureFilterMode TextureFilterMode
    ctypedef cudaTextureReadMode TextureReadMode
    ctypedef cudaResourceViewDesc ResourceViewDesc
    ctypedef cudaArray_t Array
    ctypedef cudaExtent Extent
    ctypedef cudaPos Pos
    ctypedef cudaPitchedPtr PitchedPtr
    ctypedef cudaMemcpyKind MemoryKind
    ctypedef cudaMipmappedArray_t MipmappedArray

    ctypedef cudaLimit Limit

    ctypedef cudaResourceDesc ResourceDesc

    ctypedef cudaMemcpy3DParms Memcpy3DParms

    ctypedef cudaTextureDesc TextureDesc

    ctypedef cudaIpcMemHandle_t IpcMemHandle

    ctypedef cudaIpcEventHandle_t IpcEventHandle

    ctypedef cudaUUID_t cudaUUID

    ctypedef cudaMemPool_t MemPool

    ctypedef cudaMemPoolAttr MemPoolAttr

    ctypedef cudaMemPoolProps _MemPoolProps

    ctypedef cudaPointerAttributes _PointerAttributes

    ctypedef cudaDeviceProp DeviceProp

    ctypedef cudaStreamCaptureStatus StreamCaptureStatus
    ctypedef cudaStreamCaptureMode StreamCaptureMode
    ctypedef cudaGraph_t Graph
    ctypedef cudaGraphExec_t GraphExec
    ctypedef cudaGraphNode_t GraphNode

    ctypedef cudaMemAllocationType MemAllocationType
    ctypedef cudaMemAllocationHandleType MemAllocationHandleType
    ctypedef cudaMemLocationType MemLocationType

ELSE:
    include "_runtime_typedef.pxi"
    from cupy_backends.cuda.api._runtime_enum cimport *


# For backward compatibility, keep APIs not prefixed with "cuda".
cpdef enum:
    memcpyHostToHost = 0
    memcpyHostToDevice = 1
    memcpyDeviceToHost = 2
    memcpyDeviceToDevice = 3
    memcpyDefault = 4

    hostAllocDefault = 0
    hostAllocPortable = 1
    hostAllocMapped = 2
    hostAllocWriteCombined = 4

    # cudaStream flags
    streamDefault = 0
    streamNonBlocking = 1

    # cudaStream handles
    streamLegacy = 1
    streamPerThread = 2

    eventDefault = 0
    eventBlockingSync = 1
    eventDisableTiming = 2
    eventInterprocess = 4

    # cudaStreamCaptureMode
    streamCaptureModeGlobal = 0
    streamCaptureModeThreadLocal = 1
    streamCaptureModeRelaxed = 2

    # cudaStreamCaptureStatus
    streamCaptureStatusNone = 0
    streamCaptureStatusActive = 1
    streamCaptureStatusInvalidated = 2

    # cudaMemoryType
    memoryTypeUnregistered = 0
    memoryTypeHost = 1
    memoryTypeDevice = 2
    memoryTypeManaged = 3


###############################################################################
# Constants
###############################################################################

# TODO(kmaehashi): Deprecate these aliases and use `cuda*`.
cdef int errorMemoryAllocation
cdef int errorInvalidValue
cdef int errorPeerAccessAlreadyEnabled
cdef int errorContextIsDestroyed
cdef int errorInvalidResourceHandle

cdef int deviceAttributeComputeCapabilityMajor
cdef int deviceAttributeComputeCapabilityMinor


###############################################################################
# Constants (CuPy)
###############################################################################

cdef bint _is_hip_environment


###############################################################################
# Error handling
###############################################################################

cpdef check_status(int status)


###############################################################################
# Initialization
###############################################################################

cpdef int driverGetVersion() except? -1
cpdef int runtimeGetVersion() except? -1
cpdef int _getCUDAMajorVersion() except? -1


###############################################################################
# Device and context operations
###############################################################################

cpdef int getDevice() except? -1
cpdef int deviceGetAttribute(int attrib, int device) except? -1
cpdef int deviceGetByPCIBusId(str pci_bus_id) except? -1
cpdef str deviceGetPCIBusId(int device)
cpdef int getDeviceCount() except? -1
cpdef setDevice(int device)
cpdef deviceSynchronize()
cpdef getDeviceProperties(int device)

cpdef int deviceCanAccessPeer(int device, int peerDevice) except? -1
cpdef deviceEnablePeerAccess(int peerDevice)
cpdef _deviceEnsurePeerAccess(int peerDevice)

cpdef size_t deviceGetLimit(int limit) except? -1
cpdef deviceSetLimit(int limit, size_t value)


###############################################################################
# Memory management
###############################################################################

cpdef intptr_t malloc(size_t size) except? 0
cpdef intptr_t mallocManaged(size_t size, unsigned int flags=*) except? 0
cpdef intptr_t malloc3DArray(intptr_t desc, size_t width, size_t height,
                             size_t depth, unsigned int flags=*) except? 0
cpdef intptr_t mallocArray(intptr_t desc, size_t width, size_t height,
                           unsigned int flags=*) except? 0
cpdef intptr_t mallocAsync(size_t size, intptr_t stream) except? 0
cpdef intptr_t mallocFromPoolAsync(size_t, intptr_t, intptr_t) except? 0
cpdef intptr_t hostAlloc(size_t size, unsigned int flags) except? 0
cpdef hostRegister(intptr_t ptr, size_t size, unsigned int flags)
cpdef hostUnregister(intptr_t ptr)
cpdef free(intptr_t ptr)
cpdef freeHost(intptr_t ptr)
cpdef freeArray(intptr_t ptr)
cpdef freeAsync(intptr_t ptr, intptr_t stream)
cpdef memGetInfo()
cpdef memcpy(intptr_t dst, intptr_t src, size_t size, int kind)
cpdef memcpyAsync(intptr_t dst, intptr_t src, size_t size, int kind,
                  intptr_t stream)
cpdef memcpyPeer(intptr_t dst, int dstDevice, intptr_t src, int srcDevice,
                 size_t size)
cpdef memcpyPeerAsync(intptr_t dst, int dstDevice,
                      intptr_t src, int srcDevice,
                      size_t size, intptr_t stream)
cpdef memcpy2D(intptr_t dst, size_t dpitch, intptr_t src, size_t spitch,
               size_t width, size_t height, MemoryKind kind)
cpdef memcpy2DAsync(intptr_t dst, size_t dpitch, intptr_t src, size_t spitch,
                    size_t width, size_t height, MemoryKind kind,
                    intptr_t stream)
cpdef memcpy2DFromArray(intptr_t dst, size_t dpitch, intptr_t src,
                        size_t wOffset, size_t hOffset, size_t width,
                        size_t height, int kind)
cpdef memcpy2DFromArrayAsync(intptr_t dst, size_t dpitch, intptr_t src,
                             size_t wOffset, size_t hOffset, size_t width,
                             size_t height, int kind, intptr_t stream)
cpdef memcpy2DToArray(intptr_t dst, size_t wOffset, size_t hOffset,
                      intptr_t src, size_t spitch, size_t width, size_t height,
                      int kind)
cpdef memcpy2DToArrayAsync(intptr_t dst, size_t wOffset, size_t hOffset,
                           intptr_t src, size_t spitch, size_t width,
                           size_t height, int kind, intptr_t stream)
cpdef memcpy3D(intptr_t Memcpy3DParmsPtr)
cpdef memcpy3DAsync(intptr_t Memcpy3DParmsPtr, intptr_t stream)
cpdef memset(intptr_t ptr, int value, size_t size)
cpdef memsetAsync(intptr_t ptr, int value, size_t size, intptr_t stream)
cpdef memPrefetchAsync(intptr_t devPtr, size_t count, int dstDevice,
                       intptr_t stream)
cpdef memAdvise(intptr_t devPtr, size_t count, int advice, int device)
cpdef PointerAttributes pointerGetAttributes(intptr_t ptr)
cpdef intptr_t deviceGetDefaultMemPool(int) except? 0
cpdef intptr_t deviceGetMemPool(int) except? 0
cpdef deviceSetMemPool(int, intptr_t)
cpdef intptr_t memPoolCreate(MemPoolProps) except? 0
cpdef memPoolDestroy(intptr_t)
cpdef memPoolTrimTo(intptr_t, size_t)
cpdef memPoolGetAttribute(intptr_t, int)
cpdef memPoolSetAttribute(intptr_t, int, object)


###############################################################################
# Stream and Event
###############################################################################

cpdef intptr_t streamCreate() except? 0
cpdef intptr_t streamCreateWithFlags(unsigned int flags) except? 0
cpdef streamDestroy(intptr_t stream)
cpdef streamSynchronize(intptr_t stream)
cpdef streamAddCallback(intptr_t stream, callback, intptr_t arg,
                        unsigned int flags=*)
cpdef launchHostFunc(intptr_t stream, callback, intptr_t arg)
cpdef streamQuery(intptr_t stream)
cpdef streamWaitEvent(intptr_t stream, intptr_t event, unsigned int flags=*)
cpdef streamBeginCapture(intptr_t stream, int mode=*)
cpdef intptr_t streamEndCapture(intptr_t stream) except? 0
cpdef bint streamIsCapturing(intptr_t stream) except*
cpdef intptr_t eventCreate() except? 0
cpdef intptr_t eventCreateWithFlags(unsigned int flags) except? 0
cpdef eventDestroy(intptr_t event)
cpdef float eventElapsedTime(intptr_t start, intptr_t end) except? 0
cpdef eventQuery(intptr_t event)
cpdef eventRecord(intptr_t event, intptr_t stream)
cpdef eventSynchronize(intptr_t event)


##############################################################################
# util
##############################################################################

cdef _ensure_context()


##############################################################################
# Texture
##############################################################################

cpdef uintmax_t createTextureObject(
    intptr_t ResDesc, intptr_t TexDesc) except? 0
cpdef destroyTextureObject(uintmax_t texObject)
cdef ChannelFormatDesc getChannelDesc(intptr_t array) except*
cdef ResourceDesc getTextureObjectResourceDesc(uintmax_t texobj) except*
cdef TextureDesc getTextureObjectTextureDesc(uintmax_t texobj) except*
cdef Extent make_Extent(size_t w, size_t h, size_t d) except*
cdef Pos make_Pos(size_t x, size_t y, size_t z) except*
cdef PitchedPtr make_PitchedPtr(
    intptr_t d, size_t p, size_t xsz, size_t ysz) except*

cpdef uintmax_t createSurfaceObject(intptr_t ResDesc) except? 0
cpdef destroySurfaceObject(uintmax_t surfObject)
# TODO(leofang): add cudaGetSurfaceObjectResourceDesc


##############################################################################
# Graph
##############################################################################

cpdef graphDestroy(intptr_t graph)
cpdef graphExecDestroy(intptr_t graphExec)
cpdef intptr_t graphInstantiate(intptr_t graph) except? 0
cpdef graphLaunch(intptr_t graphExec, intptr_t stream)
cpdef graphUpload(intptr_t graphExec, intptr_t stream)


##############################################################################
# Profiler
##############################################################################

cpdef profilerStart()
cpdef profilerStop()
