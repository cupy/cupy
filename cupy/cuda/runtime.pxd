###############################################################################
# Types
###############################################################################

cdef class PointerAttributes:
    cdef:
        public int device
        public size_t devicePointer
        public size_t hostPointer
        public int isManaged
        public int memoryType


cdef extern from *:
    ctypedef int Error 'cudaError_t'
    ctypedef int DataType 'cudaDataType'


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

    hostAllocDefault = 0
    hostAllocPortable = 1
    hostAllocMapped = 2
    hostAllocWriteCombined = 4

    streamDefault = 0
    streamNonBlocking = 1

    eventDefault = 0
    eventBlockingSync = 1
    eventDisableTiming = 2
    eventInterprocess = 4

    CUDA_R_32F = 0  # 32 bit real
    CUDA_R_64F = 1  # 64 bit real
    CUDA_R_16F = 2  # 16 bit real
    CUDA_R_8I = 3  # 8 bit real as a signed integer
    CUDA_C_32F = 4  # 32 bit complex
    CUDA_C_64F = 5  # 64 bit complex
    CUDA_C_16F = 6  # 16 bit complex
    CUDA_C_8I = 7  # 8 bit complex as a pair of signed integers
    CUDA_R_8U = 8  # 8 bit real as a signed integer
    CUDA_C_8U = 9  # 8 bit complex as a pair of signed integers

    errorMemoryAllocation = 2


###############################################################################
# Error handling
###############################################################################

cpdef check_status(int status)


###############################################################################
# Initialization
###############################################################################

cpdef int driverGetVersion() except *
cpdef int runtimeGetVersion() except *


###############################################################################
# Device and context operations
###############################################################################

cpdef int getDevice() except *
cpdef int deviceGetAttribute(int attrib, int device) except *
cpdef int getDeviceCount() except *
cpdef setDevice(int device)
cpdef deviceSynchronize()

cpdef int deviceCanAccessPeer(int device, int peerDevice) except *
cpdef deviceEnablePeerAccess(int peerDevice)


###############################################################################
# Memory management
###############################################################################

cpdef size_t malloc(size_t size) except *
cpdef size_t hostAlloc(size_t size, unsigned int flags) except *
cpdef free(size_t ptr)
cpdef freeHost(size_t ptr)
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

cpdef size_t streamCreate() except *
cpdef size_t streamCreateWithFlags(unsigned int flags) except *
cpdef streamDestroy(size_t stream)
cpdef streamSynchronize(size_t stream)
cpdef streamAddCallback(size_t stream, callback, size_t arg,
                        unsigned int flags=*)
cpdef streamQuery(size_t stream)
cpdef streamWaitEvent(size_t stream, size_t event, unsigned int flags=*)
cpdef size_t eventCreate() except *
cpdef size_t eventCreateWithFlags(unsigned int flags) except *
cpdef eventDestroy(size_t event)
cpdef float eventElapsedTime(size_t start, size_t end) except *
cpdef eventQuery(size_t event)
cpdef eventRecord(size_t event, size_t stream)
cpdef eventSynchronize(size_t event)
