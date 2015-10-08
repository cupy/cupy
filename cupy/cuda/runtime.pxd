###############################################################################
# Types
###############################################################################

ctypedef int Device
ctypedef void* Event
ctypedef void* Stream
#ctypedef void* Context
#ctypedef void* Function
#ctypedef void* Module
ctypedef void (*StreamCallback)(Stream hStream, int status, void* userData)

cdef struct cudaPointerAttributes:
    int device
    void* devicePointer
    void* hostPointer
    int isManaged
    int memoryType

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


cpdef int getDevice()
cpdef setDevice(int device)

