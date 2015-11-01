"""Thin wrapper of CUDA Runtime API.

There are four differences compared to the original C API.

1. Not all functions are ported.
2. Errors are translated into CUDARuntimeError exceptions.
3. The 'cuda' prefix of each API is omitted and the next character is set to
   lower case.
4. The resulting values are returned directly instead of references.

"""

###############################################################################
# Extern
###############################################################################

cdef extern from "cuda_runtime.h":
    # Error handling
    char* cudaGetErrorName(int error)
    char* cudaGetErrorString(int error)

    # Initialization
    int cudaDriverGetVersion(int* driverVersion )

    # Device operations
    int cudaGetDevice(Device* device)
    int cudaDeviceGetAttribute(int* value, int attr, int device )
    int cudaGetDeviceCount(int* const)
    int cudaSetDevice(Device device)
    int cudaDeviceSynchronize()

    # Memory management
    int cudaMalloc(void** devPtr, size_t size)
    int cudaFree(void* devPtr)
    int cudaMemGetInfo(size_t* free, size_t* total)
    int cudaMemcpy(void* dst, const void* src, size_t count, int kind)
    int cudaMemcpyAsync(void* dst, const void* src, size_t count, int kind,
                        Stream stream)
    int cudaMemcpyPeer(void* dst, int dstDevice, const void* src,
                       int srcDevice, size_t count)
    int cudaMemcpyPeerAsync(void* dst, int dstDevice, const void* src,
                       int srcDevice, size_t count, Stream stream)
    int cudaMemset(void* devPtr, int value, size_t count)
    int cudaMemsetAsync(void* devPtr, int value, size_t count, Stream stream)
    int cudaPointerGetAttributes(cudaPointerAttributes* attributes,
                                 const void* ptr)

    # Stream and Event
    int cudaStreamCreate(Stream* pStream)
    int cudaStreamCreateWithFlags(Stream* pStream, unsigned int flags)
    int cudaStreamDestroy(Stream stream)
    int cudaStreamSynchronize(Stream stream)
    int cudaStreamAddCallback(Stream stream, StreamCallback callback,
                              void* userData, unsigned int  flags)
    int cudaStreamQuery(Stream stream)
    int cudaStreamWaitEvent(Stream stream, Event event, unsigned int flags)
    int cudaEventCreate(Event* event)
    int cudaEventCreateWithFlags(Event* event, unsigned int flags)
    int cudaEventDestroy(Event event)
    int cudaEventElapsedTime(float* ms, Event start, Event end)
    int cudaEventQuery(Event event)
    int cudaEventRecord(Event event, Stream stream)
    int cudaEventSynchronize(Event event)

###############################################################################
# Enum
###############################################################################

memcpyHostToHost = 0
memcpyHostToDevice = 1
memcpyDeviceToHost = 2
memcpyDeviceToDevice = 3
memcpyDefault = 4

cudaMemoryTypeHost = 1
cudaMemoryTypeDevice = 2

streamDefault = 0
streamNonBlocking = 1

eventDefault = 0
eventBlockingSync = 1
eventDisableTiming = 2
eventInterprocess = 4

###############################################################################
# Error handling
###############################################################################

class CUDARuntimeError(RuntimeError):

    def __init__(self, status):
        self.status = status
        self.status = status
        cdef bytes name = cudaGetErrorName(status)
        cdef bytes msg = cudaGetErrorString(status)
        super(CUDARuntimeError, self).__init__(
            '%s: %s' % (name.decode(), msg.decode()))


cpdef check_status(int status):
    if status != 0:
        raise CUDARuntimeError(status)


###############################################################################
# Initialization
###############################################################################

cpdef driverGetVersion():
    cdef int version
    status = cudaDriverGetVersion(&version)
    check_status(status)
    return version


###############################################################################
# Device and context operations
###############################################################################

cpdef Device getDevice():
    cdef Device device
    status = cudaGetDevice(&device)
    check_status(status)
    return device


cpdef int deviceGetAttribute(attrib, device):
    cdef int ret
    status = cudaDeviceGetAttribute(&ret, attrib, device)
    check_status(status)
    return ret


cpdef int getDeviceCount():
    cdef int count
    status = cudaGetDeviceCount(&count)
    check_status(status)
    return count


cpdef setDevice(Device device):
    status = cudaSetDevice(device)
    check_status(status)


cpdef deviceSynchronize():
    status = cudaDeviceSynchronize()
    check_status(status)


###############################################################################
# Memory management
###############################################################################

cpdef size_t malloc(size_t size):
    cdef void* ptr
    status = cudaMalloc(&ptr, size)
    check_status(status)
    return <size_t>ptr


cpdef free(size_t ptr):
    status = cudaFree(<void*>ptr)
    check_status(status)


cpdef memGetInfo():
    cdef size_t free, total
    status = cudaMemGetInfo(&free, &total)
    check_status(status)
    return free, total


cpdef memcpy(size_t dst, size_t src, size_t size, int kind):
    status = cudaMemcpy(<void*>dst, <void*>src, size, kind)
    check_status(status)


cpdef memcpyAsync(size_t dst, size_t src, size_t size, int kind,
                  size_t stream):
    status = cudaMemcpyAsync(
        <void*>dst, <void*>src, size, kind, <Stream>stream)
    check_status(status)


cpdef memcpyPeer(size_t dst, Device dstDevice, size_t src, Device srcDevice,
               size_t size):
    status = cudaMemcpyPeer(<void*>dst, dstDevice, <void*>src, srcDevice, size)
    check_status(status)


cpdef memcpyPeerAsync(size_t dst, Device dstDevice,
                      size_t src, Device srcDevice,
                      size_t size, size_t stream):
    status = cudaMemcpyPeerAsync(<void*>dst, dstDevice,
                                 <void*>src, srcDevice, size, <Stream> stream)
    check_status(status)


cpdef memset(size_t ptr, int value, size_t size):
    status = cudaMemset(<void*>ptr, value, size)
    check_status(status)


cpdef memsetAsync(size_t ptr, int value, size_t size, size_t stream):
    status = cudaMemsetAsync(<void*>ptr, value, size, <Stream> stream)
    check_status(status)


cdef class PointerAttributes:
    cdef:
        public int device
        public size_t devicePointer
        public size_t hostPointer
        public int isManaged
        public int memoryType

    cdef _init(self, cudaPointerAttributes* attrs):
        self.device = attrs.device
        self.devicePointer = <size_t>(attrs.devicePointer)
        self.hostPointer = <size_t>(attrs.hostPointer)
        self.isManaged = attrs.isManaged
        self.memoryType = attrs.memoryType


cpdef PointerAttributes pointerGetAttributes(size_t ptr):
    cdef cudaPointerAttributes attrs
    status = cudaPointerGetAttributes(&attrs, <void*>ptr)
    check_status(status)
    ret = PointerAttributes()
    ret._init(&attrs)
    return ret


###############################################################################
# Stream and Event
###############################################################################

cpdef size_t streamCreate():
    cdef Stream stream
    status = cudaStreamCreate(&stream)
    check_status(status)
    return <size_t>stream


cpdef size_t streamCreateWithFlags(unsigned int flags):
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


def streamWaitEvent(size_t stream, size_t event, unsigned int flags=0):
    status = cudaStreamWaitEvent(<Stream>stream, <Event>event, flags)
    check_status(status)


cpdef size_t eventCreate():
    cdef Event event
    status = cudaEventCreate(&event)
    check_status(status)
    return <size_t>event

cpdef size_t eventCreateWithFlags(flags):
    cdef Event event
    status = cudaEventCreateWithFlags(&event, flags)
    check_status(status)
    return <size_t>event


cpdef eventDestroy(size_t event):
    status = cudaEventDestroy(<Event>event)
    check_status(status)


cpdef float eventElapsedTime(size_t start, size_t end):
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
    status = cudaEventSynchronize(<Event>event)
    check_status(status)
