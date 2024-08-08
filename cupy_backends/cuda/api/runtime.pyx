"""Thin wrapper of CUDA Runtime API.

There are four differences compared to the original C API.

1. Not all functions are ported.
2. Errors are translated into CUDARuntimeError exceptions.
3. The 'cuda' prefix of each API is omitted and the next character is set to
   lower case.
4. The resulting values are returned directly instead of references.

"""
from libc.stdint cimport uint64_t
from libc.string cimport memset as c_memset

import threading as _threading

cimport cpython  # NOQA
cimport cython  # NOQA

from cupy_backends.cuda.api cimport driver  # NOQA
from cupy_backends.cuda.libs cimport nvrtc  # no-cython-lint


###############################################################################
# Classes
###############################################################################

cdef class PointerAttributes:

    def __init__(self, int device, intptr_t devicePointer,
                 intptr_t hostPointer, int type=-1):
        self.type = type
        self.device = device
        self.devicePointer = devicePointer
        self.hostPointer = hostPointer

cdef class MemPoolProps:

    def __init__(
            self, int allocType, int handleType, int locationType, int devId):
        self.allocType = allocType
        self.handleType = handleType
        self.locationType = locationType
        self.devId = devId


###############################################################################
# Thread-local storage
###############################################################################

cdef object _thread_local = _threading.local()


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

include '_runtime_softlink.pxi'

IF CUPY_USE_CUDA_PYTHON:
    from cuda.ccudart cimport *
ELSE:
    include '_runtime_extern.pxi'
    pass  # for cython-lint

cdef extern from '../../cupy_backend_runtime.h' nogil:
    bint hip_environment


###############################################################################
# Constants
###############################################################################

# TODO(kmaehashi): Deprecate these aliases and use `cuda*`.
errorInvalidValue = cudaErrorInvalidValue
errorMemoryAllocation = cudaErrorMemoryAllocation
errorPeerAccessAlreadyEnabled = cudaErrorPeerAccessAlreadyEnabled
errorContextIsDestroyed = cudaErrorContextIsDestroyed
errorInvalidResourceHandle = cudaErrorInvalidResourceHandle
deviceAttributeComputeCapabilityMajor = cudaDevAttrComputeCapabilityMajor
deviceAttributeComputeCapabilityMinor = cudaDevAttrComputeCapabilityMinor


# Provide access to constants from Python.
# TODO(kmaehashi): Deprecate aliases above so that we can just do:
# from cupy_backends.cuda.api._runtime_enum import *
def _export_enum():
    import sys
    import cupy_backends.cuda.api._runtime_enum as _runtime_enum
    this = sys.modules[__name__]
    for key in dir(_runtime_enum):
        if not key.startswith('_'):
            setattr(this, key, getattr(_runtime_enum, key))


_export_enum()


###############################################################################
# Constants (CuPy)
###############################################################################

_is_hip_environment = hip_environment  # for runtime being cimport'd
is_hip = hip_environment  # for runtime being import'd


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
    """
    Returns the version of the CUDA Runtime statically linked to CuPy.

    .. seealso:: :meth:`cupy.cuda.get_local_runtime_version`
    """

    cdef int version
    IF CUPY_USE_CUDA_PYTHON:
        # Workarounds an issue that cuda-python returns its version instead of
        # the real runtime version.
        # https://github.com/NVIDIA/cuda-python/issues/16
        cdef int major, minor
        (major, minor) = nvrtc.getVersion()
        version = major * 1000 + minor * 10
    ELSE:
        status = cudaRuntimeGetVersion(&version)
        check_status(status)
    return version


cpdef int _getCUDAMajorVersion() except? -1:
    cdef int major = 0
    IF 0 < CUPY_CUDA_VERSION:
        major = runtimeGetVersion() // 1000
    return major


cpdef int _getLocalRuntimeVersion() except? -1:
    cdef int version
    initialize()
    status = DYN_cudaRuntimeGetVersion(&version)
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
    IF CUPY_CUDA_VERSION > 0 or CUPY_HIP_VERSION > 0:
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
    IF CUPY_USE_CUDA_PYTHON or CUPY_CUDA_VERSION >= 9020:
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
    if CUPY_USE_CUDA_PYTHON or CUPY_CUDA_VERSION >=10000:
        properties['uuid'] = props.uuid.bytes
        properties['luid'] = props.luid
        properties['luidDeviceNodeMask'] = props.luidDeviceNodeMask
    if CUPY_USE_CUDA_PYTHON or CUPY_CUDA_VERSION >= 11000:
        properties['persistingL2CacheMaxSize'] = props.persistingL2CacheMaxSize
        properties['maxBlocksPerMultiProcessor'] = (
            props.maxBlocksPerMultiProcessor)
        properties['accessPolicyMaxWindowSize'] = (
            props.accessPolicyMaxWindowSize)
        properties['reservedSharedMemPerBlock'] = (
            props.reservedSharedMemPerBlock)
    IF CUPY_HIP_VERSION > 0:  # HIP-only props
        properties['clockInstructionRate'] = props.clockInstructionRate
        properties['maxSharedMemoryPerMultiProcessor'] = (
            props.maxSharedMemoryPerMultiProcessor)
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
    IF CUPY_HIP_VERSION < 600: # removed in HIP 6.0.0
        properties['gcnArch'] = props.gcnArch
    IF CUPY_HIP_VERSION >= 310:
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
    cdef int ret
    status = cudaDeviceCanAccessPeer(&ret, device, peerDevice)
    check_status(status)
    return ret

cpdef deviceEnablePeerAccess(int peerDevice):
    status = cudaDeviceEnablePeerAccess(peerDevice, 0)
    check_status(status)

cpdef deviceDisablePeerAccess(int peerDevice):
    status = cudaDeviceDisablePeerAccess(peerDevice)
    check_status(status)

cpdef _deviceEnsurePeerAccess(int peerDevice):
    status = cudaDeviceEnablePeerAccess(peerDevice, 0)
    if status == 0:
        return
    elif status == errorPeerAccessAlreadyEnabled:
        cudaGetLastError()  # clear error status
        return
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
    if 0 < CUPY_HIP_VERSION < 40300000:
        raise RuntimeError('Managed memory requires ROCm 4.3+')
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
    if _is_hip_environment:
        raise RuntimeError('HIP does not support mallocAsync')
    with nogil:
        status = cudaMallocAsync(&ptr, size, <driver.Stream>stream)
    check_status(status)
    return <intptr_t>ptr

cpdef intptr_t mallocFromPoolAsync(
        size_t size, intptr_t pool, intptr_t stream) except? 0:
    cdef void* ptr
    if _is_hip_environment:
        raise RuntimeError('HIP does not support mallocFromPoolAsync')
    with nogil:
        status = cudaMallocFromPoolAsync(
            &ptr, size, <MemPool>pool, <driver.Stream>stream)
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
    if _is_hip_environment:
        raise RuntimeError('HIP does not support freeAsync')
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
    if 0 < CUPY_HIP_VERSION < 40300000:
        raise RuntimeError('Managed memory requires ROCm 4.3+')
    with nogil:
        status = cudaMemPrefetchAsync(<void*>devPtr, count, dstDevice,
                                      <driver.Stream> stream)
    check_status(status)

cpdef memAdvise(intptr_t devPtr, size_t count, int advice, int device):
    if 0 < CUPY_HIP_VERSION < 40300000:
        raise RuntimeError('Managed memory requires ROCm 4.3+')
    with nogil:
        status = cudaMemAdvise(<void*>devPtr, count,
                               <MemoryAdvise>advice, device)
    check_status(status)

cpdef PointerAttributes pointerGetAttributes(intptr_t ptr):
    cdef _PointerAttributes attrs
    status = cudaPointerGetAttributes(&attrs, <void*>ptr)
    check_status(status)
    IF CUPY_CUDA_VERSION > 0:
        return PointerAttributes(
            attrs.device,
            <intptr_t>attrs.devicePointer,
            <intptr_t>attrs.hostPointer,
            attrs.type)
    ELIF 0 < CUPY_HIP_VERSION < 600:
        return PointerAttributes(
            attrs.device,
            <intptr_t>attrs.devicePointer,
            <intptr_t>attrs.hostPointer,
            attrs.memoryType)
    ELIF CUPY_HIP_VERSION >= 600:
        return PointerAttributes(
            attrs.device,
            <intptr_t>attrs.devicePointer,
            <intptr_t>attrs.hostPointer,
            attrs.type)
    ELSE:  # for RTD
        return None

cpdef intptr_t deviceGetDefaultMemPool(int device) except? 0:
    '''Get the default mempool on the current device.'''
    if _is_hip_environment:
        raise RuntimeError('HIP does not support deviceGetDefaultMemPool')
    cdef MemPool pool
    with nogil:
        status = cudaDeviceGetDefaultMemPool(&pool, device)
    check_status(status)
    return <intptr_t>(pool)

cpdef intptr_t deviceGetMemPool(int device) except? 0:
    '''Get the current mempool on the current device.'''
    if _is_hip_environment:
        raise RuntimeError('HIP does not support deviceGetMemPool')
    cdef MemPool pool
    with nogil:
        status = cudaDeviceGetMemPool(&pool, device)
    check_status(status)
    return <intptr_t>(pool)

cpdef deviceSetMemPool(int device, intptr_t pool):
    '''Set the current mempool on the current device to pool.'''
    if _is_hip_environment:
        raise RuntimeError('HIP does not support deviceSetMemPool')
    with nogil:
        status = cudaDeviceSetMemPool(device, <MemPool>pool)
    check_status(status)

cpdef intptr_t memPoolCreate(MemPoolProps props) except? 0:
    if _is_hip_environment:
        raise RuntimeError('HIP does not support memPoolCreate')

    cdef MemPool pool
    cdef _MemPoolProps props_c
    c_memset(&props_c, 0, sizeof(_MemPoolProps))
    props_c.allocType = <MemAllocationType>props.allocType
    props_c.handleTypes = <MemAllocationHandleType>props.handleType
    props_c.location.type = <MemLocationType>props.locationType
    props_c.location.id = props.devId

    with nogil:
        status = cudaMemPoolCreate(&pool, &props_c)
    check_status(status)
    return <intptr_t>pool

cpdef memPoolDestroy(intptr_t pool):
    if _is_hip_environment:
        raise RuntimeError('HIP does not support memPoolDestroy')
    with nogil:
        status = cudaMemPoolDestroy(<MemPool>pool)
    check_status(status)

cpdef memPoolTrimTo(intptr_t pool, size_t size):
    if _is_hip_environment:
        raise RuntimeError('HIP does not support memPoolTrimTo')
    with nogil:
        status = cudaMemPoolTrimTo(<MemPool>pool, size)
    check_status(status)

cpdef memPoolGetAttribute(intptr_t pool, int attr):
    if _is_hip_environment:
        raise RuntimeError('HIP does not support memPoolGetAttribute')
    cdef int val1
    cdef uint64_t val2
    cdef void* out
    # TODO(leofang): check this hack when more cudaMemPoolAttr are added!
    out = <void*>(&val1) if attr <= 0x3 else <void*>(&val2)
    with nogil:
        status = cudaMemPoolGetAttribute(<MemPool>pool, <MemPoolAttr>attr, out)
    check_status(status)
    # TODO(leofang): check this hack when more cudaMemPoolAttr are added!
    # cast to Python int regardless of C types
    return val1 if attr <= 0x3 else val2

cpdef memPoolSetAttribute(intptr_t pool, int attr, object value):
    if _is_hip_environment:
        raise RuntimeError('HIP does not support memPoolSetAttribute')
    cdef int val1
    cdef uint64_t val2
    cdef void* out
    # TODO(leofang): check this hack when more cudaMemPoolAttr are added!
    if attr <= 0x3:
        val1 = value
        out = <void*>(&val1)
    else:
        val2 = value
        out = <void*>(&val2)
    with nogil:
        status = cudaMemPoolSetAttribute(<MemPool>pool, <MemPoolAttr>attr, out)
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


cpdef intptr_t streamCreateWithPriority(unsigned int flags,
                                        int priority) except? 0:
    cdef driver.Stream stream
    status = cudaStreamCreateWithPriority(&stream, flags, priority)
    check_status(status)
    return <intptr_t>stream


cpdef unsigned int streamGetFlags(intptr_t stream) except? 0:
    cdef unsigned int flags
    status = cudaStreamGetFlags(<driver.Stream>stream, &flags)
    check_status(status)
    return flags


cpdef int streamGetPriority(intptr_t stream) except? 0:
    cdef int priority
    status = cudaStreamGetPriority(<driver.Stream>stream, &priority)
    check_status(status)
    return priority


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


cpdef streamBeginCapture(intptr_t stream, int mode=streamCaptureModeRelaxed):
    if _is_hip_environment:
        raise RuntimeError('streamBeginCapture is not supported in ROCm')
    # TODO(leofang): check and raise if stream == 0?
    with nogil:
        status = cudaStreamBeginCapture(<driver.Stream>stream,
                                        <StreamCaptureMode>mode)
    check_status(status)


cpdef intptr_t streamEndCapture(intptr_t stream) except? 0:
    # TODO(leofang): check and raise if stream == 0?
    cdef Graph g
    if _is_hip_environment:
        raise RuntimeError('streamEndCapture is not supported in ROCm')
    with nogil:
        status = cudaStreamEndCapture(<driver.Stream>stream, &g)
    check_status(status)
    return <intptr_t>g


cpdef bint streamIsCapturing(intptr_t stream) except*:
    cdef StreamCaptureStatus s
    if _is_hip_environment:
        raise RuntimeError('streamIsCapturing is not supported in ROCm')
    with nogil:
        status = cudaStreamIsCapturing(<driver.Stream>stream, &s)
    check_status(status)  # cudaErrorStreamCaptureImplicit could be raised here
    if s == <StreamCaptureStatus>streamCaptureStatusInvalidated:
        raise RuntimeError('the stream was capturing, but an error has '
                           'invalidated the capture sequence')
    return <bint>s


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

cpdef uintmax_t createTextureObject(
        intptr_t ResDescPtr, intptr_t TexDescPtr) except? 0:
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

cpdef uintmax_t createSurfaceObject(intptr_t ResDescPtr) except? 0:
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

cdef ChannelFormatDesc getChannelDesc(intptr_t array) except*:
    cdef ChannelFormatDesc desc
    with nogil:
        status = cudaGetChannelDesc(&desc, <Array>array)
    check_status(status)
    return desc

cdef ResourceDesc getTextureObjectResourceDesc(uintmax_t obj) except*:
    cdef ResourceDesc desc
    with nogil:
        status = cudaGetTextureObjectResourceDesc(&desc, <TextureObject>obj)
    check_status(status)
    return desc

cdef TextureDesc getTextureObjectTextureDesc(uintmax_t obj) except*:
    cdef TextureDesc desc
    with nogil:
        status = cudaGetTextureObjectTextureDesc(&desc, <TextureObject>obj)
    check_status(status)
    return desc

cdef Extent make_Extent(size_t w, size_t h, size_t d) except*:
    return make_cudaExtent(w, h, d)

cdef Pos make_Pos(size_t x, size_t y, size_t z) except*:
    return make_cudaPos(x, y, z)

cdef PitchedPtr make_PitchedPtr(
        intptr_t d, size_t p, size_t xsz, size_t ysz) except*:
    return make_cudaPitchedPtr(<void*>d, p, xsz, ysz)


##############################################################################
# Graph
##############################################################################

cpdef graphDestroy(intptr_t graph):
    with nogil:
        status = cudaGraphDestroy(<Graph>graph)
    check_status(status)

cpdef graphExecDestroy(intptr_t graphExec):
    with nogil:
        status = cudaGraphExecDestroy(<GraphExec>graphExec)
    check_status(status)

cpdef intptr_t graphInstantiate(intptr_t graph) except? 0:
    # TODO(leofang): support reporting error log?
    cdef GraphExec ge
    with nogil:
        status = cudaGraphInstantiate(<GraphExec*>(&ge), <Graph>graph,
                                      NULL, NULL, 0)
    check_status(status)
    return <intptr_t>ge

cpdef graphLaunch(intptr_t graphExec, intptr_t stream):
    with nogil:
        status = cudaGraphLaunch(<GraphExec>(graphExec), <driver.Stream>stream)
    check_status(status)

cpdef graphUpload(intptr_t graphExec, intptr_t stream):
    with nogil:
        status = cudaGraphUpload(<GraphExec>(graphExec), <driver.Stream>stream)
    check_status(status)


##############################################################################
# Profiler
##############################################################################

cpdef profilerStart():
    """Enable profiling.

    A user can enable CUDA profiling. When an error occurs, it raises an
    exception.

    See the CUDA document for detail.
    """
    status = cudaProfilerStart()
    check_status(status)


cpdef profilerStop():
    """Disable profiling.

    A user can disable CUDA profiling. When an error occurs, it raises an
    exception.

    See the CUDA document for detail.
    """
    status = cudaProfilerStop()
    check_status(status)
