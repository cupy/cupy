"""Thin wrapper of CUDA Runtime API.

There are four differences compared to the original C API.

1. Not all functions are ported.
2. Errors are translated into CUDARuntimeError exceptions.
3. The 'cuda' prefix of each API is omitted and the next character is set to
   lower case.
4. The resulting values are returned directly instead of references.

"""
import threading

cimport cpython  # NOQA
cimport cython  # NOQA

from cupy_backends.cuda.api cimport driver


cdef class PointerAttributes:

    def __init__(self, int device, intptr_t devicePointer,
                 intptr_t hostPointer):
        self.device = device
        self.devicePointer = devicePointer
        self.hostPointer = hostPointer


###############################################################################
# Thread-local storage
###############################################################################

cdef object _thread_local = threading.local()


cdef class _ThreadLocal:

    cdef list context_initialized

    def __init__(self):
        cdef int i
        self.context_initialized = [False for i in range(getDeviceCount())]

    @staticmethod
    cdef _ThreadLocal get():
        try:
            tls = _thread_local.tls
        except AttributeError:
            tls = _thread_local.tls = _ThreadLocal()
        return <_ThreadLocal>tls


###############################################################################
# Extern
###############################################################################
cdef extern from *:
    ctypedef int DeviceAttr 'cudaDeviceAttr'
    ctypedef int MemoryAdvise 'cudaMemoryAdvise'

    ctypedef void StreamCallbackDef(
        driver.Stream stream, Error status, void* userData)
    ctypedef StreamCallbackDef* StreamCallback 'cudaStreamCallback_t'

    ctypedef void HostFnDef(void* userData)
    ctypedef HostFnDef* HostFn 'cudaHostFn_t'


cdef extern from '../../cupy_backend_runtime.h' nogil:

    # Types
    ctypedef struct _PointerAttributes 'cudaPointerAttributes':
        int device
        void* devicePointer
        void* hostPointer

    # Error handling
    const char* cudaGetErrorName(Error error)
    const char* cudaGetErrorString(Error error)
    int cudaGetLastError()

    # Initialization
    int cudaDriverGetVersion(int* driverVersion)
    int cudaRuntimeGetVersion(int* runtimeVersion)

    # Device operations
    int cudaGetDevice(int* device)
    int cudaDeviceGetAttribute(int* value, DeviceAttr attr, int device)
    int cudaDeviceGetByPCIBusId(int* device, const char* pciBusId)
    int cudaDeviceGetPCIBusId(char* pciBusId, int len, int device)
    int cudaGetDeviceProperties(DeviceProp* prop, int device)
    int cudaGetDeviceCount(int* count)
    int cudaSetDevice(int device)
    int cudaDeviceSynchronize()

    int cudaDeviceCanAccessPeer(int* canAccessPeer, int device,
                                int peerDevice)
    int cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags)

    int cudaDeviceGetLimit(size_t* value, Limit limit)
    int cudaDeviceSetLimit(Limit limit, size_t value)

    # IPC
    int cudaIpcCloseMemHandle(void* devPtr)
    int cudaIpcGetEventHandle(IpcEventHandle* handle, driver.Event event)
    int cudaIpcGetMemHandle(IpcMemHandle*, void* devPtr)
    int cudaIpcOpenEventHandle(driver.Event* event, IpcEventHandle handle)
    int cudaIpcOpenMemHandle(void** devPtr, IpcMemHandle handle,
                             unsigned int  flags)

    # Memory management
    int cudaMalloc(void** devPtr, size_t size)
    int cudaMallocManaged(void** devPtr, size_t size, unsigned int flags)
    int cudaMalloc3DArray(Array* array, const ChannelFormatDesc* desc,
                          Extent extent, unsigned int flags)
    int cudaMallocArray(Array* array, const ChannelFormatDesc* desc,
                        size_t width, size_t height, unsigned int flags)
    int cudaMallocAsync(void**, size_t, driver.Stream)
    int cudaHostAlloc(void** ptr, size_t size, unsigned int flags)
    int cudaHostRegister(void *ptr, size_t size, unsigned int flags)
    int cudaHostUnregister(void *ptr)
    int cudaFree(void* devPtr)
    int cudaFreeHost(void* ptr)
    int cudaFreeArray(Array array)
    int cudaFreeAsync(void*, driver.Stream)
    int cudaMemGetInfo(size_t* free, size_t* total)
    int cudaMemcpy(void* dst, const void* src, size_t count,
                   MemoryKind kind)
    int cudaMemcpyAsync(void* dst, const void* src, size_t count,
                        MemoryKind kind, driver.Stream stream)
    int cudaMemcpyPeer(void* dst, int dstDevice, const void* src,
                       int srcDevice, size_t count)
    int cudaMemcpyPeerAsync(void* dst, int dstDevice, const void* src,
                            int srcDevice, size_t count,
                            driver.Stream stream)
    int cudaMemcpy2DFromArray(void* dst, size_t dpitch, Array src,
                              size_t wOffset, size_t hOffset, size_t width,
                              size_t height, MemoryKind kind)
    int cudaMemcpy2DFromArrayAsync(void* dst, size_t dpitch, Array src,
                                   size_t wOffset, size_t hOffset,
                                   size_t width, size_t height,
                                   MemoryKind kind, driver.Stream stream)
    int cudaMemcpy2DToArray(Array dst, size_t wOffset, size_t hOffset,
                            const void* src, size_t spitch, size_t width,
                            size_t height, MemoryKind kind)
    int cudaMemcpy2DToArrayAsync(Array dst, size_t wOffset, size_t hOffset,
                                 const void* src, size_t spitch, size_t width,
                                 size_t height, MemoryKind kind,
                                 driver.Stream stream)
    int cudaMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch,
                     size_t width, size_t height, MemoryKind kind)
    int cudaMemcpy2DAsync(void* dst, size_t dpitch, const void* src,
                          size_t spitch, size_t width, size_t height,
                          MemoryKind kind, driver.Stream stream)
    int cudaMemcpy3D(Memcpy3DParms* Memcpy3DParmsPtr)
    int cudaMemcpy3DAsync(Memcpy3DParms* Memcpy3DParmsPtr,
                          driver.Stream stream)
    int cudaMemset(void* devPtr, int value, size_t count)
    int cudaMemsetAsync(void* devPtr, int value, size_t count,
                        driver.Stream stream)
    int cudaMemPrefetchAsync(const void *devPtr, size_t count, int dstDevice,
                             driver.Stream stream)
    int cudaMemAdvise(const void *devPtr, size_t count,
                      MemoryAdvise advice, int device)
    int cudaDeviceGetDefaultMemPool(MemPool*, int)
    int cudaDeviceGetMemPool(MemPool*, int)
    int cudaDeviceSetMemPool(int, MemPool)
    int cudaMemPoolTrimTo(MemPool, size_t)
    int cudaPointerGetAttributes(_PointerAttributes* attributes,
                                 const void* ptr)
    Extent make_cudaExtent(size_t w, size_t h, size_t d)
    Pos make_cudaPos(size_t x, size_t y, size_t z)
    PitchedPtr make_cudaPitchedPtr(void* d, size_t p, size_t xsz, size_t ysz)

    # Stream and Event
    int cudaStreamCreate(driver.Stream* pStream)
    int cudaStreamCreateWithFlags(driver.Stream* pStream,
                                  unsigned int flags)
    int cudaStreamDestroy(driver.Stream stream)
    int cudaStreamSynchronize(driver.Stream stream)
    int cudaStreamAddCallback(driver.Stream stream, StreamCallback callback,
                              void* userData, unsigned int flags)
    int cudaLaunchHostFunc(driver.Stream stream, HostFn fn, void* userData)
    int cudaStreamQuery(driver.Stream stream)
    int cudaStreamWaitEvent(driver.Stream stream, driver.Event event,
                            unsigned int flags)
    int cudaEventCreate(driver.Event* event)
    int cudaEventCreateWithFlags(driver.Event* event, unsigned int flags)
    int cudaEventDestroy(driver.Event event)
    int cudaEventElapsedTime(float* ms, driver.Event start,
                             driver.Event end)
    int cudaEventQuery(driver.Event event)
    int cudaEventRecord(driver.Event event, driver.Stream stream)
    int cudaEventSynchronize(driver.Event event)

    # Texture
    int cudaCreateTextureObject(TextureObject* pTexObject,
                                const ResourceDesc* pResDesc,
                                const TextureDesc* pTexDesc,
                                const ResourceViewDesc* pResViewDesc)
    int cudaDestroyTextureObject(TextureObject texObject)
    int cudaGetChannelDesc(ChannelFormatDesc* desc, Array array)
    int cudaGetTextureObjectResourceDesc(ResourceDesc* desc, TextureObject obj)
    int cudaGetTextureObjectTextureDesc(TextureDesc* desc, TextureObject obj)

    # Surface
    int cudaCreateSurfaceObject(SurfaceObject* pSurObject,
                                const ResourceDesc* pResDesc)
    int cudaDestroySurfaceObject(SurfaceObject surObject)

    bint hip_environment
    int cudaDevAttrComputeCapabilityMajor
    int cudaDevAttrComputeCapabilityMinor

    # Error code
    int cudaErrorMemoryAllocation
    int cudaErrorInvalidValue
    int cudaErrorPeerAccessAlreadyEnabled
    int cudaErrorContextIsDestroyed
    int cudaErrorInvalidResourceHandle


_is_hip_environment = hip_environment  # for runtime being cimport'd
is_hip = hip_environment  # for runtime being import'd
deviceAttributeComputeCapabilityMajor = cudaDevAttrComputeCapabilityMajor
deviceAttributeComputeCapabilityMinor = cudaDevAttrComputeCapabilityMinor


###############################################################################
# Error codes
###############################################################################

errorInvalidValue = cudaErrorInvalidValue
errorMemoryAllocation = cudaErrorMemoryAllocation
errorPeerAccessAlreadyEnabled = cudaErrorPeerAccessAlreadyEnabled
errorContextIsDestroyed = cudaErrorContextIsDestroyed
errorInvalidResourceHandle = cudaErrorInvalidResourceHandle


###############################################################################
# Error handling
###############################################################################

class CUDARuntimeError(RuntimeError):

    def __init__(self, status):
        self.status = status
        cdef bytes name = cudaGetErrorName(<Error>status)
        cdef bytes msg = cudaGetErrorString(<Error>status)
        super(CUDARuntimeError, self).__init__(
            '%s: %s' % (name.decode(), msg.decode()))

    def __reduce__(self):
        return (type(self), (self.status,))


@cython.profile(False)
cpdef inline check_status(int status):
    if status != 0:
        # to reset error status
        cudaGetLastError()
        raise CUDARuntimeError(status)


###############################################################################
# Initialization
###############################################################################

cpdef int driverGetVersion() except? -1:
    cdef int version
    status = cudaDriverGetVersion(&version)
    check_status(status)
    return version


cpdef int runtimeGetVersion() except? -1:
    cdef int version
    status = cudaRuntimeGetVersion(&version)
    check_status(status)
    return version


###############################################################################
# Device and context operations
###############################################################################

cpdef int getDevice() except? -1:
    cdef int device
    status = cudaGetDevice(&device)
    check_status(status)
    return device

cpdef int deviceGetAttribute(int attrib, int device) except? -1:
    cdef int ret
    status = cudaDeviceGetAttribute(&ret, <DeviceAttr>attrib, device)
    check_status(status)
    return ret

cpdef getDeviceProperties(int device):
    cdef DeviceProp props
    cdef int status = cudaGetDeviceProperties(&props, device)
    check_status(status)

    cdef dict properties = {'name': b'UNAVAILABLE'}  # for RTD

    # Common properties to CUDA 9.0, 9.2, 10.x, 11.x, and HIP
    IF CUDA_VERSION > 0 or HIP_VERSION > 0:
        properties = {
            'name': props.name,
            'totalGlobalMem': props.totalGlobalMem,
            'sharedMemPerBlock': props.sharedMemPerBlock,
            'regsPerBlock': props.regsPerBlock,
            'warpSize': props.warpSize,
            'maxThreadsPerBlock': props.maxThreadsPerBlock,
            'maxThreadsDim': tuple(props.maxThreadsDim),
            'maxGridSize': tuple(props.maxGridSize),
            'clockRate': props.clockRate,
            'totalConstMem': props.totalConstMem,
            'major': props.major,
            'minor': props.minor,
            'textureAlignment': props.textureAlignment,
            'texturePitchAlignment': props.texturePitchAlignment,
            'multiProcessorCount': props.multiProcessorCount,
            'kernelExecTimeoutEnabled': props.kernelExecTimeoutEnabled,
            'integrated': props.integrated,
            'canMapHostMemory': props.canMapHostMemory,
            'computeMode': props.computeMode,
            'maxTexture1D': props.maxTexture1D,
            'maxTexture2D': tuple(props.maxTexture2D),
            'maxTexture3D': tuple(props.maxTexture3D),
            'concurrentKernels': props.concurrentKernels,
            'ECCEnabled': props.ECCEnabled,
            'pciBusID': props.pciBusID,
            'pciDeviceID': props.pciDeviceID,
            'pciDomainID': props.pciDomainID,
            'tccDriver': props.tccDriver,
            'memoryClockRate': props.memoryClockRate,
            'memoryBusWidth': props.memoryBusWidth,
            'l2CacheSize': props.l2CacheSize,
            'maxThreadsPerMultiProcessor': props.maxThreadsPerMultiProcessor,
            'isMultiGpuBoard': props.isMultiGpuBoard,
            'cooperativeLaunch': props.cooperativeLaunch,
            'cooperativeMultiDeviceLaunch': props.cooperativeMultiDeviceLaunch,
        }
    IF CUDA_VERSION >= 9020:
        properties['deviceOverlap'] = props.deviceOverlap
        properties['maxTexture1DMipmap'] = props.maxTexture1DMipmap
        properties['maxTexture1DLinear'] = props.maxTexture1DLinear
        properties['maxTexture1DLayered'] = tuple(props.maxTexture1DLayered)
        properties['maxTexture2DMipmap'] = tuple(props.maxTexture2DMipmap)
        properties['maxTexture2DLinear'] = tuple(props.maxTexture2DLinear)
        properties['maxTexture2DLayered'] = tuple(props.maxTexture2DLayered)
        properties['maxTexture2DGather'] = tuple(props.maxTexture2DGather)
        properties['maxTexture3DAlt'] = tuple(props.maxTexture3DAlt)
        properties['maxTextureCubemap'] = props.maxTextureCubemap
        properties['maxTextureCubemapLayered'] = tuple(
            props.maxTextureCubemapLayered)
        properties['maxSurface1D'] = props.maxSurface1D
        properties['maxSurface1DLayered'] = tuple(props.maxSurface1DLayered)
        properties['maxSurface2D'] = tuple(props.maxSurface2D)
        properties['maxSurface2DLayered'] = tuple(props.maxSurface2DLayered)
        properties['maxSurface3D'] = tuple(props.maxSurface3D)
        properties['maxSurfaceCubemap'] = props.maxSurfaceCubemap
        properties['maxSurfaceCubemapLayered'] = tuple(
            props.maxSurfaceCubemapLayered)
        properties['surfaceAlignment'] = props.surfaceAlignment
        properties['asyncEngineCount'] = props.asyncEngineCount
        properties['unifiedAddressing'] = props.unifiedAddressing
        properties['streamPrioritiesSupported'] = (
            props.streamPrioritiesSupported)
        properties['globalL1CacheSupported'] = props.globalL1CacheSupported
        properties['localL1CacheSupported'] = props.localL1CacheSupported
        properties['sharedMemPerMultiprocessor'] = (
            props.sharedMemPerMultiprocessor)
        properties['regsPerMultiprocessor'] = props.regsPerMultiprocessor
        properties['managedMemory'] = props.managedMemory
        properties['multiGpuBoardGroupID'] = props.multiGpuBoardGroupID
        properties['hostNativeAtomicSupported'] = (
            props.hostNativeAtomicSupported)
        properties['singleToDoublePrecisionPerfRatio'] = (
            props.singleToDoublePrecisionPerfRatio)
        properties['pageableMemoryAccess'] = props.pageableMemoryAccess
        properties['concurrentManagedAccess'] = props.concurrentManagedAccess
        properties['computePreemptionSupported'] = (
            props.computePreemptionSupported)
        properties['canUseHostPointerForRegisteredMem'] = (
            props.canUseHostPointerForRegisteredMem)
        properties['sharedMemPerBlockOptin'] = props.sharedMemPerBlockOptin
        properties['pageableMemoryAccessUsesHostPageTables'] = (
            props.pageableMemoryAccessUsesHostPageTables)
        properties['directManagedMemAccessFromHost'] = (
            props.directManagedMemAccessFromHost)
    IF CUDA_VERSION >= 10000:
        properties['uuid'] = props.uuid.bytes
        properties['luid'] = props.luid
        properties['luidDeviceNodeMask'] = props.luidDeviceNodeMask
    IF CUDA_VERSION >= 11000:
        properties['persistingL2CacheMaxSize'] = props.persistingL2CacheMaxSize
        properties['maxBlocksPerMultiProcessor'] = (
            props.maxBlocksPerMultiProcessor)
        properties['accessPolicyMaxWindowSize'] = (
            props.accessPolicyMaxWindowSize)
        properties['reservedSharedMemPerBlock'] = (
            props.reservedSharedMemPerBlock)
    IF HIP_VERSION > 0:  # HIP-only props
        properties['clockInstructionRate'] = props.clockInstructionRate
        properties['maxSharedMemoryPerMultiProcessor'] = (
            props.maxSharedMemoryPerMultiProcessor)
        properties['gcnArch'] = props.gcnArch
        properties['hdpMemFlushCntl'] = <intptr_t>(props.hdpMemFlushCntl)
        properties['hdpRegFlushCntl'] = <intptr_t>(props.hdpRegFlushCntl)
        properties['memPitch'] = props.memPitch
        properties['cooperativeMultiDeviceUnmatchedFunc'] = (
            props.cooperativeMultiDeviceUnmatchedFunc)
        properties['cooperativeMultiDeviceUnmatchedGridDim'] = (
            props.cooperativeMultiDeviceUnmatchedGridDim)
        properties['cooperativeMultiDeviceUnmatchedBlockDim'] = (
            props.cooperativeMultiDeviceUnmatchedBlockDim)
        properties['cooperativeMultiDeviceUnmatchedSharedMem'] = (
            props.cooperativeMultiDeviceUnmatchedSharedMem)
        properties['isLargeBar'] = props.isLargeBar

        cdef dict arch = {}  # for hipDeviceArch_t
        arch['hasGlobalInt32Atomics'] = props.arch.hasGlobalInt32Atomics
        arch['hasGlobalFloatAtomicExch'] = props.arch.hasGlobalFloatAtomicExch
        arch['hasSharedInt32Atomics'] = props.arch.hasSharedInt32Atomics
        arch['hasSharedFloatAtomicExch'] = props.arch.hasSharedFloatAtomicExch
        arch['hasFloatAtomicAdd'] = props.arch.hasFloatAtomicAdd
        arch['hasGlobalInt64Atomics'] = props.arch.hasGlobalInt64Atomics
        arch['hasSharedInt64Atomics'] = props.arch.hasSharedInt64Atomics
        arch['hasDoubles'] = props.arch.hasDoubles
        arch['hasWarpVote'] = props.arch.hasWarpVote
        arch['hasWarpBallot'] = props.arch.hasWarpBallot
        arch['hasWarpShuffle'] = props.arch.hasWarpShuffle
        arch['hasFunnelShift'] = props.arch.hasFunnelShift
        arch['hasThreadFenceSystem'] = props.arch.hasThreadFenceSystem
        arch['hasSyncThreadsExt'] = props.arch.hasSyncThreadsExt
        arch['hasSurfaceFuncs'] = props.arch.hasSurfaceFuncs
        arch['has3dGrid'] = props.arch.has3dGrid
        arch['hasDynamicParallelism'] = props.arch.hasDynamicParallelism
        properties['arch'] = arch
    IF HIP_VERSION >= 310:
        properties['gcnArchName'] = props.gcnArchName
        properties['asicRevision'] = props.asicRevision
        properties['managedMemory'] = props.managedMemory
        properties['directManagedMemAccessFromHost'] = (
            props.directManagedMemAccessFromHost)
        properties['concurrentManagedAccess'] = props.concurrentManagedAccess
        properties['pageableMemoryAccess'] = props.pageableMemoryAccess
        properties['pageableMemoryAccessUsesHostPageTables'] = (
            props.pageableMemoryAccessUsesHostPageTables)
    return properties

cpdef int deviceGetByPCIBusId(str pci_bus_id) except? -1:
    # Encode the python string before passing to native code
    byte_pci_bus_id = pci_bus_id.encode('ascii')
    cdef const char* c_pci_bus_id = byte_pci_bus_id

    cdef int device = -1
    cdef int status
    status = cudaDeviceGetByPCIBusId(&device, c_pci_bus_id)
    check_status(status)
    # on ROCm, it might fail silently, so we also need to check if the
    # device is meaningful or not
    if hip_environment and device == -1:
        check_status(cudaErrorInvalidValue)
    return device

cpdef str deviceGetPCIBusId(int device):
    # The PCI Bus ID string must be able to store 13 characters including the
    # NULL-terminator according to the CUDA documentation.
    # https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html
    cdef char pci_bus_id[13]
    status = cudaDeviceGetPCIBusId(pci_bus_id, 13, device)
    check_status(status)
    return pci_bus_id.decode('ascii')

cpdef int getDeviceCount() except? -1:
    cdef int count
    status = cudaGetDeviceCount(&count)
    check_status(status)
    return count

cpdef setDevice(int device):
    status = cudaSetDevice(device)
    check_status(status)

cpdef deviceSynchronize():
    with nogil:
        status = cudaDeviceSynchronize()
    check_status(status)

cpdef int deviceCanAccessPeer(int device, int peerDevice) except? -1:
    cpdef int ret
    status = cudaDeviceCanAccessPeer(&ret, device, peerDevice)
    check_status(status)
    return ret

cpdef deviceEnablePeerAccess(int peerDevice):
    status = cudaDeviceEnablePeerAccess(peerDevice, 0)
    check_status(status)

cpdef size_t deviceGetLimit(int limit) except? -1:
    cdef size_t value
    status = cudaDeviceGetLimit(&value, <Limit>limit)
    check_status(status)
    return value

cpdef deviceSetLimit(int limit, size_t value):
    status = cudaDeviceSetLimit(<Limit>limit, value)
    check_status(status)


###############################################################################
# IPC operations
###############################################################################
cpdef ipcCloseMemHandle(intptr_t devPtr):
    status = cudaIpcCloseMemHandle(<void*>devPtr)
    check_status(status)


cpdef ipcGetEventHandle(intptr_t event):
    cdef IpcEventHandle handle
    status = cudaIpcGetEventHandle(&handle, <driver.Event>event)
    check_status(status)
    # We need to do this due to a bug in Cython that
    # cuts out the 0 bytes in an array of chars when
    # constructing the python object
    # resulting in different sizes assignment errors
    # when recreating the struct from the python
    # array of bytes
    reserved = [<unsigned char>handle.reserved[i] for i in range(64)]
    return bytes(reserved)

cpdef ipcGetMemHandle(intptr_t devPtr):
    cdef IpcMemHandle handle
    status = cudaIpcGetMemHandle(&handle, <void*>devPtr)
    check_status(status)
    # We need to do this due to a bug in Cython that
    # when converting an array of chars in C to a python object
    # it discards the data after the first 0 value
    # resulting in a loss of data, as this is not a string
    # but a buffer of bytes
    reserved = [<unsigned char>handle.reserved[i] for i in range(64)]
    return bytes(reserved)

cpdef ipcOpenEventHandle(bytes handle):
    cdef driver.Event event
    cdef IpcEventHandle handle_
    handle_.reserved = handle
    status = cudaIpcOpenEventHandle(&event, handle_)
    check_status(status)
    return <intptr_t>event

cpdef ipcOpenMemHandle(bytes handle,
                       unsigned int flags=cudaIpcMemLazyEnablePeerAccess):
    cdef void* devPtr
    cdef IpcMemHandle handle_
    handle_.reserved = handle
    status = cudaIpcOpenMemHandle(&devPtr, handle_, flags)
    check_status(status)
    return <intptr_t>devPtr


###############################################################################
# Memory management
###############################################################################

cpdef intptr_t malloc(size_t size) except? 0:
    cdef void* ptr
    with nogil:
        status = cudaMalloc(&ptr, size)
    check_status(status)
    return <intptr_t>ptr


cpdef intptr_t mallocManaged(
        size_t size, unsigned int flags=cudaMemAttachGlobal) except? 0:
    cdef void* ptr
    with nogil:
        status = cudaMallocManaged(&ptr, size, flags)
    check_status(status)
    return <intptr_t>ptr


cpdef intptr_t malloc3DArray(intptr_t descPtr, size_t width, size_t height,
                             size_t depth, unsigned int flags=0) except? 0:
    cdef Array ptr
    cdef Extent extent = make_cudaExtent(width, height, depth)
    with nogil:
        status = cudaMalloc3DArray(&ptr, <ChannelFormatDesc*>descPtr, extent,
                                   flags)
    check_status(status)
    return <intptr_t>ptr


cpdef intptr_t mallocArray(intptr_t descPtr, size_t width, size_t height,
                           unsigned int flags=0) except? 0:
    cdef Array ptr
    with nogil:
        status = cudaMallocArray(&ptr, <ChannelFormatDesc*>descPtr, width,
                                 height, flags)
    check_status(status)
    return <intptr_t>ptr


cpdef intptr_t mallocAsync(size_t size, intptr_t stream) except? 0:
    cdef void* ptr
    if CUDA_VERSION < 11020:
        raise RuntimeError('mallocAsync is supported since CUDA 11.2')
    with nogil:
        status = cudaMallocAsync(&ptr, size, <driver.Stream>stream)
    check_status(status)
    return <intptr_t>ptr


cpdef intptr_t hostAlloc(size_t size, unsigned int flags) except? 0:
    cdef void* ptr
    with nogil:
        status = cudaHostAlloc(&ptr, size, flags)
    check_status(status)
    return <intptr_t>ptr


cpdef hostRegister(intptr_t ptr, size_t size, unsigned int flags):
    with nogil:
        status = cudaHostRegister(<void*>ptr, size, flags)
    check_status(status)


cpdef hostUnregister(intptr_t ptr):
    with nogil:
        status = cudaHostUnregister(<void*>ptr)
    check_status(status)


cpdef free(intptr_t ptr):
    with nogil:
        status = cudaFree(<void*>ptr)
    check_status(status)


cpdef freeHost(intptr_t ptr):
    with nogil:
        status = cudaFreeHost(<void*>ptr)
    check_status(status)


cpdef freeArray(intptr_t ptr):
    with nogil:
        status = cudaFreeArray(<Array>ptr)
    check_status(status)


cpdef freeAsync(intptr_t ptr, intptr_t stream):
    if CUDA_VERSION < 11020:
        raise RuntimeError('freeAsync is supported since CUDA 11.2')
    with nogil:
        status = cudaFreeAsync(<void*>ptr, <driver.Stream>stream)
    check_status(status)


cpdef memGetInfo():
    cdef size_t free, total
    status = cudaMemGetInfo(&free, &total)
    check_status(status)
    return free, total


cpdef memcpy(intptr_t dst, intptr_t src, size_t size, int kind):
    with nogil:
        status = cudaMemcpy(<void*>dst, <void*>src, size, <MemoryKind>kind)
    check_status(status)


cpdef memcpyAsync(intptr_t dst, intptr_t src, size_t size, int kind,
                  intptr_t stream):
    with nogil:
        status = cudaMemcpyAsync(
            <void*>dst, <void*>src, size, <MemoryKind>kind,
            <driver.Stream>stream)
    check_status(status)


cpdef memcpyPeer(intptr_t dst, int dstDevice, intptr_t src, int srcDevice,
                 size_t size):
    with nogil:
        status = cudaMemcpyPeer(<void*>dst, dstDevice, <void*>src, srcDevice,
                                size)
    check_status(status)


cpdef memcpyPeerAsync(intptr_t dst, int dstDevice, intptr_t src, int srcDevice,
                      size_t size, intptr_t stream):
    with nogil:
        status = cudaMemcpyPeerAsync(<void*>dst, dstDevice, <void*>src,
                                     srcDevice, size, <driver.Stream> stream)
    check_status(status)

cpdef memcpy2D(intptr_t dst, size_t dpitch, intptr_t src, size_t spitch,
               size_t width, size_t height, MemoryKind kind):
    with nogil:
        status = cudaMemcpy2D(<void*>dst, dpitch, <void*>src, spitch, width,
                              height, kind)
    check_status(status)

cpdef memcpy2DAsync(intptr_t dst, size_t dpitch, intptr_t src, size_t spitch,
                    size_t width, size_t height, MemoryKind kind,
                    intptr_t stream):
    with nogil:
        status = cudaMemcpy2DAsync(<void*>dst, dpitch, <void*>src, spitch,
                                   width, height, kind, <driver.Stream>stream)
    check_status(status)

cpdef memcpy2DFromArray(intptr_t dst, size_t dpitch, intptr_t src,
                        size_t wOffset, size_t hOffset, size_t width,
                        size_t height, int kind):
    with nogil:
        status = cudaMemcpy2DFromArray(<void*>dst, dpitch, <Array>src, wOffset,
                                       hOffset, width, height,
                                       <MemoryKind>kind)
    check_status(status)


cpdef memcpy2DFromArrayAsync(intptr_t dst, size_t dpitch, intptr_t src,
                             size_t wOffset, size_t hOffset, size_t width,
                             size_t height, int kind, intptr_t stream):
    with nogil:
        status = cudaMemcpy2DFromArrayAsync(<void*>dst, dpitch, <Array>src,
                                            wOffset, hOffset, width, height,
                                            <MemoryKind>kind,
                                            <driver.Stream>stream)
    check_status(status)


cpdef memcpy2DToArray(intptr_t dst, size_t wOffset, size_t hOffset,
                      intptr_t src, size_t spitch, size_t width, size_t height,
                      int kind):
    with nogil:
        status = cudaMemcpy2DToArray(<Array>dst, wOffset, hOffset, <void*>src,
                                     spitch, width, height, <MemoryKind>kind)
    check_status(status)


cpdef memcpy2DToArrayAsync(intptr_t dst, size_t wOffset, size_t hOffset,
                           intptr_t src, size_t spitch, size_t width,
                           size_t height, int kind, intptr_t stream):
    with nogil:
        status = cudaMemcpy2DToArrayAsync(<Array>dst, wOffset, hOffset,
                                          <void*>src, spitch, width, height,
                                          <MemoryKind>kind,
                                          <driver.Stream>stream)
    check_status(status)


cpdef memcpy3D(intptr_t Memcpy3DParmsPtr):
    with nogil:
        status = cudaMemcpy3D(<Memcpy3DParms*>Memcpy3DParmsPtr)
    check_status(status)


cpdef memcpy3DAsync(intptr_t Memcpy3DParmsPtr, intptr_t stream):
    with nogil:
        status = cudaMemcpy3DAsync(<Memcpy3DParms*>Memcpy3DParmsPtr,
                                   <driver.Stream> stream)
    check_status(status)


cpdef memset(intptr_t ptr, int value, size_t size):
    with nogil:
        status = cudaMemset(<void*>ptr, value, size)
    check_status(status)


cpdef memsetAsync(intptr_t ptr, int value, size_t size, intptr_t stream):
    with nogil:
        status = cudaMemsetAsync(<void*>ptr, value, size,
                                 <driver.Stream> stream)
    check_status(status)

cpdef memPrefetchAsync(intptr_t devPtr, size_t count, int dstDevice,
                       intptr_t stream):
    with nogil:
        status = cudaMemPrefetchAsync(<void*>devPtr, count, dstDevice,
                                      <driver.Stream> stream)
    check_status(status)

cpdef memAdvise(intptr_t devPtr, size_t count, int advice, int device):
    with nogil:
        status = cudaMemAdvise(<void*>devPtr, count,
                               <MemoryAdvise>advice, device)
    check_status(status)


cpdef PointerAttributes pointerGetAttributes(intptr_t ptr):
    cdef _PointerAttributes attrs
    status = cudaPointerGetAttributes(&attrs, <void*>ptr)
    check_status(status)
    return PointerAttributes(
        attrs.device,
        <intptr_t>attrs.devicePointer,
        <intptr_t>attrs.hostPointer)


cpdef intptr_t deviceGetDefaultMemPool(int device) except? 0:
    '''Get the default mempool on the current device.'''
    if CUDA_VERSION < 11020:
        raise RuntimeError('deviceGetDefaultMemPool is supported since '
                           'CUDA 11.2')
    cdef MemPool pool
    with nogil:
        status = cudaDeviceGetDefaultMemPool(&pool, device)
    check_status(status)
    return <intptr_t>(pool)


cpdef intptr_t deviceGetMemPool(int device) except? 0:
    '''Get the current mempool on the current device.'''
    if CUDA_VERSION < 11020:
        raise RuntimeError('deviceGetMemPool is supported since '
                           'CUDA 11.2')
    cdef MemPool pool
    with nogil:
        status = cudaDeviceGetMemPool(&pool, device)
    check_status(status)
    return <intptr_t>(pool)


cpdef deviceSetMemPool(int device, intptr_t pool):
    '''Set the current mempool on the current device to pool.'''
    if CUDA_VERSION < 11020:
        raise RuntimeError('deviceSetMemPool is supported since '
                           'CUDA 11.2')
    with nogil:
        status = cudaDeviceSetMemPool(device, <MemPool>pool)
    check_status(status)


cpdef memPoolTrimTo(intptr_t pool, size_t size):
    if CUDA_VERSION < 11020:
        raise RuntimeError('memPoolTrimTo is supported since CUDA 11.2')
    with nogil:
        status = cudaMemPoolTrimTo(<MemPool>pool, size)
    check_status(status)


###############################################################################
# Stream and Event
###############################################################################

cpdef intptr_t streamCreate() except? 0:
    cdef driver.Stream stream
    status = cudaStreamCreate(&stream)
    check_status(status)
    return <intptr_t>stream


cpdef intptr_t streamCreateWithFlags(unsigned int flags) except? 0:
    cdef driver.Stream stream
    status = cudaStreamCreateWithFlags(&stream, flags)
    check_status(status)
    return <intptr_t>stream


cpdef streamDestroy(intptr_t stream):
    status = cudaStreamDestroy(<driver.Stream>stream)
    check_status(status)


cpdef streamSynchronize(intptr_t stream):
    with nogil:
        status = cudaStreamSynchronize(<driver.Stream>stream)
    check_status(status)


cdef _streamCallbackFunc(driver.Stream hStream, int status,
                         void* func_arg) with gil:
    obj = <object>func_arg
    func, arg = obj
    func(<intptr_t>hStream, status, arg)
    cpython.Py_DECREF(obj)


# Use Cython macro to suppress compiler warning
IF CUDA_VERSION >= 10000:
    cdef _HostFnFunc(void* func_arg) with gil:
        obj = <object>func_arg
        func, arg = obj
        func(arg)
        cpython.Py_DECREF(obj)


cpdef streamAddCallback(intptr_t stream, callback, intptr_t arg,
                        unsigned int flags=0):
    if _is_hip_environment and stream == 0:
        raise RuntimeError('HIP does not allow adding callbacks to the '
                           'default (null) stream')
    func_arg = (callback, arg)
    cpython.Py_INCREF(func_arg)
    with nogil:
        status = cudaStreamAddCallback(
            <driver.Stream>stream, <StreamCallback>_streamCallbackFunc,
            <void*>func_arg, flags)
    check_status(status)


cpdef launchHostFunc(intptr_t stream, callback, intptr_t arg):
    if _is_hip_environment:
        raise RuntimeError('This feature is not supported on HIP')
    if CUDA_VERSION < 10000:
        raise RuntimeError('This feature is only supported on CUDA 10.0+')

    func_arg = (callback, arg)
    cpython.Py_INCREF(func_arg)
    with nogil:
        status = cudaLaunchHostFunc(
            <driver.Stream>stream, <HostFn>_HostFnFunc,
            <void*>func_arg)
    check_status(status)


cpdef streamQuery(intptr_t stream):
    return cudaStreamQuery(<driver.Stream>stream)


cpdef streamWaitEvent(intptr_t stream, intptr_t event, unsigned int flags=0):
    with nogil:
        status = cudaStreamWaitEvent(<driver.Stream>stream,
                                     <driver.Event>event, flags)
    check_status(status)


cpdef intptr_t eventCreate() except? 0:
    cdef driver.Event event
    status = cudaEventCreate(&event)
    check_status(status)
    return <intptr_t>event

cpdef intptr_t eventCreateWithFlags(unsigned int flags) except? 0:
    cdef driver.Event event
    status = cudaEventCreateWithFlags(&event, flags)
    check_status(status)
    return <intptr_t>event


cpdef eventDestroy(intptr_t event):
    status = cudaEventDestroy(<driver.Event>event)
    check_status(status)


cpdef float eventElapsedTime(intptr_t start, intptr_t end) except? 0:
    cdef float ms
    status = cudaEventElapsedTime(&ms, <driver.Event>start, <driver.Event>end)
    check_status(status)
    return ms


cpdef eventQuery(intptr_t event):
    return cudaEventQuery(<driver.Event>event)


cpdef eventRecord(intptr_t event, intptr_t stream):
    status = cudaEventRecord(<driver.Event>event, <driver.Stream>stream)
    check_status(status)


cpdef eventSynchronize(intptr_t event):
    with nogil:
        status = cudaEventSynchronize(<driver.Event>event)
    check_status(status)


##############################################################################
# util
##############################################################################

cdef _ensure_context():
    """Ensure that CUcontext bound to the calling host thread exists.

    See discussion on https://github.com/cupy/cupy/issues/72 for details.
    """
    tls = _ThreadLocal.get()
    cdef int dev = getDevice()
    if not tls.context_initialized[dev]:
        # Call Runtime API to establish context on this host thread.
        memGetInfo()
        tls.context_initialized[dev] = True


##############################################################################
# Texture
##############################################################################

cpdef uintmax_t createTextureObject(intptr_t ResDescPtr, intptr_t TexDescPtr):
    cdef uintmax_t texobj = 0
    with nogil:
        status = cudaCreateTextureObject(<TextureObject*>(&texobj),
                                         <ResourceDesc*>ResDescPtr,
                                         <TextureDesc*>TexDescPtr,
                                         <ResourceViewDesc*>NULL)
    check_status(status)
    return texobj

cpdef destroyTextureObject(uintmax_t texObject):
    with nogil:
        status = cudaDestroyTextureObject(<TextureObject>texObject)
    check_status(status)

cpdef uintmax_t createSurfaceObject(intptr_t ResDescPtr):
    cdef uintmax_t surfobj = 0
    with nogil:
        status = cudaCreateSurfaceObject(<SurfaceObject*>(&surfobj),
                                         <ResourceDesc*>ResDescPtr)
    check_status(status)
    return surfobj

cpdef destroySurfaceObject(uintmax_t surfObject):
    with nogil:
        status = cudaDestroySurfaceObject(<SurfaceObject>surfObject)
    check_status(status)

cdef ChannelFormatDesc getChannelDesc(intptr_t array):
    cdef ChannelFormatDesc desc
    with nogil:
        status = cudaGetChannelDesc(&desc, <Array>array)
    check_status(status)
    return desc

cdef ResourceDesc getTextureObjectResourceDesc(uintmax_t obj):
    cdef ResourceDesc desc
    with nogil:
        status = cudaGetTextureObjectResourceDesc(&desc, <TextureObject>obj)
    check_status(status)
    return desc

cdef TextureDesc getTextureObjectTextureDesc(uintmax_t obj):
    cdef TextureDesc desc
    with nogil:
        status = cudaGetTextureObjectTextureDesc(&desc, <TextureObject>obj)
    check_status(status)
    return desc

cdef Extent make_Extent(size_t w, size_t h, size_t d):
    return make_cudaExtent(w, h, d)

cdef Pos make_Pos(size_t x, size_t y, size_t z):
    return make_cudaPos(x, y, z)

cdef PitchedPtr make_PitchedPtr(intptr_t d, size_t p, size_t xsz, size_t ysz):
    return make_cudaPitchedPtr(<void*>d, p, xsz, ysz)
