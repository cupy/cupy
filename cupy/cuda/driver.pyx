"""Thin wrapper of CUDA Driver API.

There are four differences compared to the original C API.

1. Not all functions are ported.
2. Errors are translated into CUDADriverError exceptions.
3. The 'cu' prefix of each API is omitted and the next character is set to
   lower case.
4. The resulting values are returned directly instead of references.

"""


###############################################################################
# Extern
###############################################################################

cdef extern from "cuda.h":
    # Error handling
    int cuGetErrorName(Result error, const char** pStr)
    int cuGetErrorString(Result error, const char** pStr)

    # Initialization
    int cuInit(unsigned int Flags)

    # Device and context operations
    int cuDriverGetVersion(int* driverVersion)
    int cuDeviceGet(Device* device, int ordinal)
    int cuDeviceGetAttribute(int* pi, DeviceAttribute attrib, Device dev)
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
    int cuModuleGetGlobal(Deviceptr* dptr, size_t* bytes, Module hmod,
                          char* name)
    int cuLaunchKernel(
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
    int cuMemcpyAsync(Deviceptr dst, Deviceptr src, size_t ByteCount,
                      Stream hStream)
    int cuMemcpyDtoD(Deviceptr dstDevice, Deviceptr srcDevice,
                     size_t ByteCount)
    int cuMemcpyDtoDAsync(Deviceptr dstDevice, Deviceptr srcDevice,
                          size_t ByteCount, Stream hStream)
    int cuMemcpyDtoH(void* dstHost, Deviceptr srcDevice, size_t ByteCount)
    int cuMemcpyDtoHAsync(void* dstHost, Deviceptr srcDevice, size_t ByteCount,
                          Stream hStream)
    int cuMemcpyHtoD(Deviceptr dstDevice, void* srcHost, size_t ByteCount)
    int cuMemcpyHtoDAsync(Deviceptr dstDevice, void* srcHost, size_t ByteCount,
                          Stream hStream)
    int cuMemcpyPeer(Deviceptr dstDevice, Context dstContext,
                     Deviceptr srcDevice, Context srcContext, size_t ByteCount)
    int cuMemcpyPeerAsync(Deviceptr dstDevice, Context dstContext,
                          Deviceptr srcDevice, Context srcContext,
                          size_t ByteCount, Stream hStream)
    int cuMemsetD32(Deviceptr dstDevice, unsigned int ui, size_t N)
    int cuMemsetD32Async(Deviceptr dstDevice, unsigned int ui, size_t N,
                         Stream hStream)
    int cuPointerGetAttribute(void* data, PointerAttribute attribute,
                              Deviceptr ptr)

    # Stream and Event
    int cuStreamCreate(Stream* phStream, unsigned int Flags)
    int cuStreamDestroy(Stream hStream)
    int cuStreamSynchronize(Stream hStream)
    int cuStreamAddCallback(Stream hStream, StreamCallback callback,
                            void* userData, unsigned int flags)
    int cuEventCreate(Event* phEvent, unsigned int Flags)
    int cuEventDestroy(Event hEvent)
    int cuEventRecord(Event hEvent, Stream hStream)
    int cuEventSynchronize(Event hEvent)


###############################################################################
# Error handling
###############################################################################

class CUDADriverError(RuntimeError):

    def __init__(self, Result status):
        self.status = status
        cdef const char *name
        cdef const char *msg
        cuGetErrorName(status, &name)
        cuGetErrorString(status, &msg)
        cdef bytes s_name = name, s_msg = msg
        super(CUDADriverError, self).__init__(
            '%s: %s' % (s_name.decode(), s_msg.decode()))


cpdef check_status(int status):
    if status != 0:
        raise CUDADriverError(status)


###############################################################################
# Initialization
###############################################################################

cpdef init():
    status = cuInit(0)
    check_status(status)


cpdef int driverGetVersion():
    cdef int version
    status = cuDriverGetVersion(&version)
    check_status(status)
    return version

###############################################################################
# Device and context operations
###############################################################################

cpdef Device deviceGet(int device_id):
    cdef Device device
    status = cuDeviceGet(&device, device_id)
    check_status(status)
    return device


cpdef int deviceGetAttribute(int attrib, Device device):
    cdef int ret
    status = cuDeviceGetAttribute(&ret, <DeviceAttribute>attrib, device)
    check_status(status)
    return ret


cpdef int deviceGetCount():
    cdef int count
    status = cuDeviceGetCount(&count)
    check_status(status)
    return count


cpdef size_t deviceTotalMem(Device device):
    cdef size_t mem
    status = cuDeviceTotalMem(&mem, device)
    check_status(status)
    return mem


cpdef size_t ctxCreate(unsigned int flag, Device device):
    cdef Context ctx
    status = cuCtxCreate(&ctx, flag, device)
    check_status(status)
    return <size_t>ctx


cpdef ctxDestroy(size_t ctx):
    status = cuCtxDestroy(<Context>ctx)
    check_status(status)


cpdef unsigned int ctxGetApiVersion(size_t ctx):
    cdef unsigned int version
    status = cuCtxGetApiVersion(<Context>ctx, &version)
    check_status(status)
    return version


cpdef size_t ctxGetCurrent():
    cdef Context ctx
    status = cuCtxGetCurrent(&ctx)
    check_status(status)
    return <size_t>ctx


cpdef Device ctxGetDevice():
    cdef Device device
    status = cuCtxGetDevice(&device)
    check_status(status)
    return device


cpdef size_t ctxPopCurrent():
    cdef Context ctx
    status = cuCtxPopCurrent(&ctx)
    check_status(status)
    return <size_t>ctx


cpdef ctxPushCurrent(size_t ctx):
    status = cuCtxPushCurrent(<Context>ctx)
    check_status(status)


cpdef ctxSetCurrent(size_t ctx):
    status = cuCtxSetCurrent(<Context>ctx)
    check_status(status)


cpdef ctxSynchronize():
    status = cuCtxSynchronize()
    check_status(status)


###############################################################################
# Module load and kernel execution
###############################################################################

cpdef size_t moduleLoad(str filename):
    cdef Module module
    cdef bytes b_filename = filename.encode()
    status = cuModuleLoad(&module, b_filename)
    check_status(status)
    return <size_t>module


cpdef size_t moduleLoadData(bytes image):
    cdef Module module
    status = cuModuleLoadData(&module, <char*>image)
    check_status(status)
    return <size_t>module


cpdef moduleUnload(size_t module):
    status = cuModuleUnload(<Module>module)
    check_status(status)


cpdef size_t moduleGetFunction(size_t module, str funcname):
    cdef Function func
    cdef bytes b_funcname = funcname.encode()
    status = cuModuleGetFunction(&func, <Module>module, <char*>b_funcname)
    check_status(status)
    return <size_t>func


cpdef size_t moduleGetGlobal(size_t module, str varname):
    cdef Deviceptr var
    cdef size_t size
    cdef bytes b_varname = varname.encode()
    status = cuModuleGetGlobal(&var, &size, <Module>module, <char*>b_varname)
    check_status(status)
    return <size_t>var


cpdef launchKernel(
        size_t f, unsigned int grid_dim_x, unsigned int grid_dim_y,
        unsigned int grid_dim_z, unsigned int block_dim_x,
        unsigned int block_dim_y, unsigned int block_dim_z,
        unsigned int shared_mem_bytes, size_t stream, size_t kernel_params,
        size_t extra):
    status = cuLaunchKernel(
        <Function>f, grid_dim_x, grid_dim_y, grid_dim_z,
        block_dim_x, block_dim_y, block_dim_z,
        shared_mem_bytes, <Stream>stream,
        <void**>kernel_params, <void**>extra)
    check_status(status)


###############################################################################
# Memory management
###############################################################################

cpdef size_t memAlloc(size_t size):
    cdef Deviceptr ptr
    status = cuMemAlloc(&ptr, size)
    check_status(status)
    return <size_t>ptr


cpdef memFree(size_t ptr):
    status = cuMemFree(<Deviceptr>ptr)
    check_status(status)


cpdef tuple memGetinfo():
    cdef size_t free, total
    status = cuMemGetInfo(&free, &total)
    check_status(status)
    return free, total


cpdef memcpy(size_t dst, size_t src, size_t size):
    status = cuMemcpy(<Deviceptr>dst, <Deviceptr>src, size)
    check_status(status)


cpdef memcpyAsync(size_t dst, size_t src, size_t size, size_t stream):
    status = cuMemcpyAsync(
        <Deviceptr>dst, <Deviceptr>src, size, <Stream>stream)
    check_status(status)


cpdef memcpyDtoD(size_t dst, size_t src, size_t size):
    status = cuMemcpyDtoD(<Deviceptr>dst, <Deviceptr>src, size)
    check_status(status)


cpdef memcpyDtoDAsync(size_t dst, size_t src, size_t size, size_t stream):
    status = cuMemcpyDtoDAsync(
        <Deviceptr>dst, <Deviceptr>src, size, <Stream>stream)
    check_status(status)


cpdef memcpyDtoH(size_t dst, size_t src, size_t size):
    status = cuMemcpyDtoH(<void*>dst, <Deviceptr>src, size)
    check_status(status)


cpdef memcpyDtoHAsync(size_t dst, size_t src, size_t size, size_t stream):
    status = cuMemcpyDtoHAsync(
        <void*>dst, <Deviceptr>src, size, <Stream>stream)
    check_status(status)


cpdef memcpyHtoD(size_t dst, size_t src, size_t size):
    status = cuMemcpyHtoD(<Deviceptr>dst, <void*>src, size)
    check_status(status)


cpdef memcpyHtoDAsync(size_t dst, size_t src, size_t size, size_t stream):
    status = cuMemcpyHtoDAsync(
        <Deviceptr>dst, <void*>src, size, <Stream>stream)
    check_status(status)


cpdef memcpyPeer(size_t dst, size_t dst_ctx, size_t src, size_t src_ctx,
                 size_t size):
    status = cuMemcpyPeer(
        <Deviceptr>dst, <Context>dst_ctx, <Deviceptr>src, <Context>src_ctx,
        size)
    check_status(status)


cpdef memcpyPeerAsync(size_t dst, size_t dst_ctx, size_t src, size_t src_ctx,
                      size_t size, size_t stream):
    status = cuMemcpyPeerAsync(
        <Deviceptr>dst, <Context>dst_ctx, <Deviceptr>src, <Context>src_ctx,
        size, <Stream>stream)
    check_status(status)


cpdef memsetD32(size_t ptr, unsigned int value, size_t size):
    status = cuMemsetD32(<Deviceptr>ptr, value, size)
    check_status(status)


cpdef memsetD32Async(size_t ptr, unsigned int value, size_t size,
                     size_t stream):
    status = cuMemsetD32Async(<Deviceptr>ptr, value, size, <Stream>stream)
    check_status(status)


cpdef size_t pointerGetAttribute(int attribute, size_t ptr):
    assert attribute == 0  # Currently only context query is supported

    cdef Context ctx
    status = cuPointerGetAttribute(
        &ctx, <PointerAttribute>attribute, <Deviceptr>ptr)
    check_status(status)
    return <size_t>ctx


###############################################################################
# Stream and Event
###############################################################################

cpdef size_t streamCreate(unsigned int flag=0):
    cdef Stream stream
    status = cuStreamCreate(&stream, flag)
    check_status(status)
    return <size_t>stream


cpdef streamDestroy(size_t stream):
    status = cuStreamDestroy(<Stream>stream)
    check_status(status)


cpdef streamSynchronize(size_t stream):
    status = cuStreamSynchronize(<Stream>stream)
    check_status(status)


cdef _streamCallbackFunc(Stream hStream, Result status, void *userData):
    func, data = <tuple>userData
    func(<size_t>hStream, <int>status, data)


cpdef streamAddCallback(size_t stream, object callback, size_t arg,
                        unsigned int flags=0):
    func_arg = (callback, arg)
    status = cuStreamAddCallback(
        <Stream>stream, <StreamCallback>_streamCallbackFunc,
        <void*>func_arg, flags)
    check_status(status)


cpdef size_t eventCreate(unsigned int flag):
    cdef Event event
    status = cuEventCreate(&event, flag)
    check_status(status)
    return <size_t>event


cpdef eventDestroy(size_t event):
    status = cuEventDestroy(<Event>event)
    check_status(status)


cpdef eventRecord(size_t event, size_t stream):
    status = cuEventRecord(<Event>event, <Stream>stream)
    check_status(status)


cpdef eventSynchronize(size_t event):
    status = cuEventSynchronize(<Event>event)
    check_status(status)
