###############################################################################
# Types
###############################################################################

from cupy.cuda.driver cimport Event
from cupy.cuda.driver cimport Stream


cdef class PointerAttributes:
    cdef:
        public int device
        public size_t devicePointer
        public size_t hostPointer
        public int isManaged
        public int memoryType


IF CUPY_USE_CUDA:
    cdef extern from *:
        ctypedef int Error 'cudaError_t'
        ctypedef int DeviceAttr 'enum cudaDeviceAttr'
        ctypedef int MemoryKind 'enum cudaMemcpyKind'

        ctypedef size_t _Pointer 'void*'

        ctypedef void (*StreamCallbackDef)(
            Stream stream, Error status, void* userData)
        ctypedef StreamCallbackDef StreamCallback 'cudaStreamCallback_t'

ELSE:
    cdef:
        ctypedef int Error
        ctypedef int DeviceAttr
        ctypedef int MemoryKind

        ctypedef size_t _Pointer

        ctypedef void (*StreamCallbackDef)(
            Stream stream, Error status, void* userData)
        ctypedef StreamCallbackDef StreamCallback


###############################################################################
# Enum
###############################################################################

cpdef enum:
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
# Initialization
###############################################################################

cpdef driverGetVersion()


###############################################################################
# Device and context operations
###############################################################################

cpdef int getDevice()
cpdef int deviceGetAttribute(int attrib, int device)
cpdef int getDeviceCount()
cpdef setDevice(int device)
cpdef deviceSynchronize()


###############################################################################
# Memory management
###############################################################################

cpdef size_t malloc(size_t size)
cpdef free(size_t ptr)
cpdef memGetInfo()
cpdef memcpy(size_t dst, size_t src, size_t size, int kind)
cpdef memcpyAsync(size_t dst, size_t src, size_t size, int kind,
                  size_t stream)
cpdef memcpyPeer(size_t dst, int dstDevice, size_t src, int srcDevice,
               size_t size)
cpdef memcpyPeerAsync(size_t dst, int dstDevice,
                      size_t src, int srcDevice,
                      size_t size, size_t stream)
cpdef memset(size_t ptr, int value, size_t size)
cpdef memsetAsync(size_t ptr, int value, size_t size, size_t stream)
cpdef PointerAttributes pointerGetAttributes(size_t ptr)


###############################################################################
# Stream and Event
###############################################################################

cpdef size_t streamCreate()
cpdef size_t streamCreateWithFlags(unsigned int flags)
cpdef streamDestroy(size_t stream)
cpdef streamSynchronize(size_t stream)
cpdef streamAddCallback(size_t stream, callback, size_t arg,
                        unsigned int flags=*)
cpdef streamQuery(size_t stream)
cpdef streamWaitEvent(size_t stream, size_t event, unsigned int flags=*)
cpdef size_t eventCreate()
cpdef size_t eventCreateWithFlags(unsigned int flags)
cpdef eventDestroy(size_t event)
cpdef float eventElapsedTime(size_t start, size_t end)
cpdef eventQuery(size_t event)
cpdef eventRecord(size_t event, size_t stream)
cpdef eventSynchronize(size_t event)
