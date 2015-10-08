###############################################################################
# Types
###############################################################################
ctypedef int Device
ctypedef void* Event
ctypedef void* Stream
ctypedef void* Context
ctypedef void* Function
ctypedef void* Module
ctypedef void* Deviceptr
ctypedef void (*StreamCallback)(Stream hStream, int status, void* userData)

###############################################################################
# Extern
###############################################################################

cdef extern from "cuda.h":
    pass
    int cuGetErrorName(int error, const char** pStr)
    int cuGetErrorString(int error, const char** pStr)

    # Initialization

    int cuInit(unsigned int Flags)

    # Device and context operations

    int cuDriverGetVersion(int* driverVersion)
    int cuDeviceGet(Device* device, int ordinal)
    int cuDeviceGetAttribute(int* pi, int attrib, Device dev)
    int cuDeviceGetCount(int* count)
    int cuDeviceTotalMem(size_t* bytes, Device dev)
    int cuCtxCreate(Context* pctx, unsigned int flags, Device dev)
    int cuCtxDestroy(Context ctx)
    int cuCtxGetApiVersion(Context ctx, unsigned int* version)
    int cuCtxGetCurrent(Context* pctx)
    int cuCtxGetDevice(Device* device)
    int cuCtxPopCurrent(Context* pctx)
    int cuCtxPushCurrent(Context ctx)
    int cuCtxSetCurrent(Context ctx)
    int cuCtxSynchronize()

    # Module load and kernel execution

    int cuModuleLoad(Module* module, char* fname)
    int cuModuleLoadData(Module* module, void* image)
    int cuModuleUnload(Module hmod)
    int cuModuleGetFunction(Function* hfunc, Module hmod, char* name)
    int cuModuleGetGlobal(Deviceptr* dptr, size_t* bytes, Module hmod, char* name)
    int cuLaunchKernel (
        Function f,
        unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
        unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
        unsigned int sharedMemBytes, Stream hStream,
        void** kernelParams, void** extra)


    # Memory management

    int cuMemAlloc(Deviceptr* dptr, size_t bytesize)
    int cuMemFree(Deviceptr dptr)
    int cuMemGetInfo(size_t* free, size_t* total)
    int cuMemcpy(Deviceptr dst, Deviceptr src, size_t ByteCount)
    int cuMemcpyAsync(Deviceptr dst, Deviceptr src, size_t ByteCount, Stream hStream)
    int cuMemcpyDtoD(Deviceptr dstDevice, Deviceptr srcDevice, size_t ByteCount)
    int cuMemcpyDtoDAsync(Deviceptr dstDevice, Deviceptr srcDevice, size_t ByteCount, Stream hStream)
    int cuMemcpyDtoH(void* dstHost, Deviceptr srcDevice, size_t ByteCount)
    int cuMemcpyDtoHAsync(void* dstHost, Deviceptr srcDevice, size_t ByteCount, Stream hStream)
    int cuMemcpyHtoD(Deviceptr dstDevice, void* srcHost, size_t ByteCount)
    int cuMemcpyHtoDAsync(Deviceptr dstDevice, void* srcHost, size_t ByteCount, Stream hStream)
    int cuMemcpyPeer(Deviceptr dstDevice, Context dstContext, Deviceptr srcDevice, Context srcContext, size_t ByteCount)
    int cuMemcpyPeerAsync(Deviceptr dstDevice, Context dstContext, Deviceptr srcDevice, Context srcContext, size_t ByteCount, Stream hStream)
    int cuMemsetD32(Deviceptr dstDevice, unsigned int ui, size_t N)
    int cuMemsetD32Async(Deviceptr dstDevice, unsigned int ui, size_t N, Stream hStream)
    int cuPointerGetAttribute(void* data, int attribute, Deviceptr ptr)

    # Stream and Event

    int cuStreamCreate(Stream* phStream, unsigned int Flags)
    int cuStreamDestroy(Stream hStream)
    int cuStreamSynchronize(Stream hStream)
    int cuStreamAddCallback(Stream hStream, StreamCallback callback, void* userData, unsigned int flags)
    int cuEventCreate(Event* phEvent, unsigned int Flags)
    int cuEventDestroy(Event hEvent)
    int cuEventRecord(Event hEvent, Stream hStream)
    int cuEventSynchronize(Event hEvent)

###############################################################################
# Error handling
###############################################################################

cpdef check_status(int status)
