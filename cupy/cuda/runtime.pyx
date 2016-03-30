"""Thin wrapper of CUDA Runtime API.

There are four differences compared to the original C API.

1. Not all functions are ported.
2. Errors are translated into CUDARuntimeError exceptions.
3. The 'cuda' prefix of each API is omitted and the next character is set to
   lower case.
4. The resulting values are returned directly instead of references.

"""
cimport cython


cdef class PointerAttributes:

    def __init__(self, int device, size_t devicePointer, size_t hostPointer,
                 int isManaged, int memoryType):
        self.device = device
        self.devicePointer = devicePointer
        self.hostPointer = hostPointer
        self.isManaged = isManaged
        self.memoryType = memoryType


###############################################################################
# Extern
###############################################################################
cdef extern from "cupy_cuda.h":
    # Types
    struct _PointerAttributes 'cudaPointerAttributes':
        int device
        void* devicePointer
        void* hostPointer
        int isManaged
        int memoryType

    # Error handling
    const char* cudaGetErrorName(Error error) nogil
    const char* cudaGetErrorString(Error error) nogil

    # Initialization
    int cudaDriverGetVersion(int* driverVersion ) nogil

    # Device operations
    int cudaGetDevice(int* device) nogil
    int cudaDeviceGetAttribute(int* value, DeviceAttr attr, int device) nogil
    int cudaGetDeviceCount(int* count) nogil
    int cudaSetDevice(int device) nogil
    int cudaDeviceSynchronize() nogil

    int cudaDeviceCanAccessPeer(int* canAccessPeer, int device,
                                int peerDevice) nogil
    int cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags) nogil

    # Memory management
    int cudaMalloc(void** devPtr, size_t size) nogil
    int cudaFree(void* devPtr) nogil
    int cudaMemGetInfo(size_t* free, size_t* total) nogil
    int cudaMemcpy(void* dst, const void* src, size_t count,
                   MemoryKind kind) nogil
    int cudaMemcpyAsync(void* dst, const void* src, size_t count,
                        MemoryKind kind, Stream stream) nogil
    int cudaMemcpyPeer(void* dst, int dstDevice, const void* src,
                       int srcDevice, size_t count) nogil
    int cudaMemcpyPeerAsync(void* dst, int dstDevice, const void* src,
                       int srcDevice, size_t count, Stream stream) nogil
    int cudaMemset(void* devPtr, int value, size_t count) nogil
    int cudaMemsetAsync(void* devPtr, int value, size_t count,
                        Stream stream) nogil
    int cudaPointerGetAttributes(_PointerAttributes* attributes,
                                 const void* ptr) nogil

    # Stream and Event
    int cudaStreamCreate(Stream* pStream) nogil
    int cudaStreamCreateWithFlags(Stream* pStream, unsigned int flags) nogil
    int cudaStreamDestroy(Stream stream) nogil
    int cudaStreamSynchronize(Stream stream) nogil
    int cudaStreamAddCallback(Stream stream, StreamCallback callback,
                              void* userData, unsigned int flags) nogil
    int cudaStreamQuery(Stream stream) nogil
    int cudaStreamWaitEvent(Stream stream, Event event,
                            unsigned int flags) nogil
    int cudaEventCreate(Event* event) nogil
    int cudaEventCreateWithFlags(Event* event, unsigned int flags) nogil
    int cudaEventDestroy(Event event) nogil
    int cudaEventElapsedTime(float* ms, Event start, Event end) nogil
    int cudaEventQuery(Event event) nogil
    int cudaEventRecord(Event event, Stream stream) nogil
    int cudaEventSynchronize(Event event) nogil


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


@cython.profile(False)
cpdef inline check_status(int status):
    if status != 0:
        raise CUDARuntimeError(status)


###############################################################################
# Initialization
###############################################################################

cpdef int driverGetVersion() except *:
    cdef int version
    status = cudaDriverGetVersion(&version)
    check_status(status)
    return version


###############################################################################
# Device and context operations
###############################################################################

cpdef int getDevice() except *:
    cdef int device
    status = cudaGetDevice(&device)
    check_status(status)
    return device


cpdef int deviceGetAttribute(int attrib, int device) except *:
    cdef int ret
    status = cudaDeviceGetAttribute(&ret, <DeviceAttr>attrib, device)
    check_status(status)
    return ret


cpdef int getDeviceCount() except *:
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


cpdef int deviceCanAccessPeer(int device, int peerDevice) except *:
    cpdef int ret
    status = cudaDeviceCanAccessPeer(&ret, device, peerDevice)
    check_status(status)
    return ret


cpdef deviceEnablePeerAccess(int peerDevice):
    status = cudaDeviceEnablePeerAccess(peerDevice, 0)
    check_status(status)


###############################################################################
# Memory management
###############################################################################

cpdef size_t malloc(size_t size) except *:
    cdef void* ptr
    with nogil:
        status = cudaMalloc(&ptr, size)
    check_status(status)
    return <size_t>ptr


cpdef free(size_t ptr):
    with nogil:
        status = cudaFree(<void*>ptr)
    check_status(status)


cpdef memGetInfo():
    cdef size_t free, total
    status = cudaMemGetInfo(&free, &total)
    check_status(status)
    return free, total


cpdef memcpy(size_t dst, size_t src, size_t size, int kind):
    with nogil:
        status = cudaMemcpy(<void*>dst, <void*>src, size, <MemoryKind>kind)
    check_status(status)


cpdef memcpyAsync(size_t dst, size_t src, size_t size, int kind,
                  size_t stream):
    status = cudaMemcpyAsync(
        <void*>dst, <void*>src, size, <MemoryKind>kind, <Stream>stream)
    check_status(status)


cpdef memcpyPeer(size_t dst, int dstDevice, size_t src, int srcDevice,
               size_t size):
    with nogil:
        status = cudaMemcpyPeer(<void*>dst, dstDevice, <void*>src, srcDevice,
                                size)
    check_status(status)


cpdef memcpyPeerAsync(size_t dst, int dstDevice,
                      size_t src, int srcDevice,
                      size_t size, size_t stream):
    status = cudaMemcpyPeerAsync(<void*>dst, dstDevice,
                                 <void*>src, srcDevice, size, <Stream> stream)
    check_status(status)


cpdef memset(size_t ptr, int value, size_t size):
    with nogil:
        status = cudaMemset(<void*>ptr, value, size)
    check_status(status)


cpdef memsetAsync(size_t ptr, int value, size_t size, size_t stream):
    status = cudaMemsetAsync(<void*>ptr, value, size, <Stream> stream)
    check_status(status)


cpdef PointerAttributes pointerGetAttributes(size_t ptr):
    cdef _PointerAttributes attrs
    status = cudaPointerGetAttributes(&attrs, <void*>ptr)
    check_status(status)
    return PointerAttributes(
        attrs.device, <size_t>attrs.devicePointer, <size_t>attrs.hostPointer,
        attrs.isManaged, attrs.memoryType)


###############################################################################
# Stream and Event
###############################################################################

cpdef size_t streamCreate() except *:
    cdef Stream stream
    status = cudaStreamCreate(&stream)
    check_status(status)
    return <size_t>stream


cpdef size_t streamCreateWithFlags(unsigned int flags) except *:
    cdef Stream stream
    status = cudaStreamCreateWithFlags(&stream, flags)
    check_status(status)
    return <size_t>stream


cpdef streamDestroy(size_t stream):
    status = cudaStreamDestroy(<Stream>stream)
    check_status(status)


cpdef streamSynchronize(size_t stream):
    status = cudaStreamSynchronize(<Stream>stream)
    check_status(status)


cdef _streamCallbackFunc(Stream hStream, int status, void* userData):
    func, data = <tuple>userData
    func(<size_t>hStream, status, data)

cpdef streamAddCallback(size_t stream, callback, size_t arg,
                        unsigned int flags=0):
    func_arg = (callback, arg)
    status = cudaStreamAddCallback(
        <Stream>stream, <StreamCallback>_streamCallbackFunc,
        <void*>func_arg, flags)
    check_status(status)


cpdef streamQuery(size_t stream):
    return cudaStreamQuery(<Stream>stream)


cpdef streamWaitEvent(size_t stream, size_t event, unsigned int flags=0):
    status = cudaStreamWaitEvent(<Stream>stream, <Event>event, flags)
    check_status(status)


cpdef size_t eventCreate() except *:
    cdef Event event
    status = cudaEventCreate(&event)
    check_status(status)
    return <size_t>event

cpdef size_t eventCreateWithFlags(unsigned int flags) except *:
    cdef Event event
    status = cudaEventCreateWithFlags(&event, flags)
    check_status(status)
    return <size_t>event


cpdef eventDestroy(size_t event):
    status = cudaEventDestroy(<Event>event)
    check_status(status)


cpdef float eventElapsedTime(size_t start, size_t end) except *:
    cdef float ms
    status = cudaEventElapsedTime(&ms, <Event>start, <Event>end)
    check_status(status)
    return ms


cpdef eventQuery(size_t event):
    return cudaEventQuery(<Event>event)


cpdef eventRecord(size_t event, size_t stream):
    status = cudaEventRecord(<Event>event, <Stream>stream)
    check_status(status)


cpdef eventSynchronize(size_t event):
    with nogil:
        status = cudaEventSynchronize(<Event>event)
    check_status(status)
