"""Thin wrapper of CUDA Driver API.

There are four differences compared to the original C API.

1. Not all functions are ported.
2. Errors are translated into CUDADriverError exceptions.
3. The 'cu' prefix of each API is omitted and the next character is set to
   lower case.
4. The resulting values are returned directly instead of references.

"""

###############################################################################
# Enum
###############################################################################
CU_POINTER_ATTRIBUTE_CONTEXT = 1
CU_POINTER_ATTRIBUTE_MEMORY_TYPE = 2
CU_POINTER_ATTRIBUTE_DEVICE_POINTER = 3
CU_POINTER_ATTRIBUTE_HOST_POINTER = 4
CU_POINTER_ATTRIBUTE_P2P_TOKENS = 5
CU_POINTER_ATTRIBUTE_SYNC_MEMOPS = 6
CU_POINTER_ATTRIBUTE_BUFFER_ID = 7
CU_POINTER_ATTRIBUTE_IS_MANAGED = 8

EVENT_DEFAULT = 0
EVENT_BLOCKING_SYNC = 1
EVENT_DISABLE_TIMING = 2
EVENT_INTERPROCESS = 4


###############################################################################
# Error handling
###############################################################################

class CUDADriverError(RuntimeError):

    def __init__(self, int status):
        self.status = status
        cdef char *name
        cdef char *msg
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

cpdef Device deviceGet(Device device_id):
    cdef Device device
    status = cuDeviceGet(&device, device_id)
    check_status(status)
    return device


cpdef int deviceGetAttribute(attrib, Device device):
    cdef int ret
    status = cuDeviceGetAttribute(&ret, attrib, device)
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


cpdef size_t ctxCreate(flag, Device device):
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
    cdef void* var
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


cpdef size_t memAlloc(size):
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


cpdef memcpy(size_t dst, size_t src, size):
    status = cuMemcpy(<Deviceptr>dst, <Deviceptr>src, size)
    check_status(status)


cpdef memcpyAsync(size_t dst, size_t src, size, size_t stream):
    status = cuMemcpyAsync(
        <Deviceptr>dst, <Deviceptr>src, size, <Stream>stream)
    check_status(status)


cpdef memcpyDtoD(size_t dst, size_t src, size):
    status = cuMemcpyDtoD(<Deviceptr>dst, <Deviceptr>src, size)
    check_status(status)


cpdef memcpyDtoDAsync(size_t dst, size_t src, size, size_t stream):
    status = cuMemcpyDtoDAsync(
        <Deviceptr>dst, <Deviceptr>src, size, <Stream>stream)
    check_status(status)


cpdef memcpyDtoH(size_t dst, size_t src, size):
    status = cuMemcpyDtoH(<Deviceptr>dst, <Deviceptr>src, size)
    check_status(status)


cpdef memcpyDtoHAsync(size_t dst, size_t src, size, size_t stream):
    status = cuMemcpyDtoHAsync(
        <Deviceptr>dst, <Deviceptr>src, size, <Stream>stream)
    check_status(status)


cpdef memcpyHtoD(size_t dst, size_t src, size):
    status = cuMemcpyHtoD(<Deviceptr>dst, <Deviceptr>src, size)
    check_status(status)


cpdef memcpyHtoDAsync(size_t dst, size_t src, size, size_t stream):
    status = cuMemcpyHtoDAsync(
        <Deviceptr>dst, <Deviceptr>src, size, <Stream>stream)
    check_status(status)


cpdef memcpyPeer(size_t dst, size_t dst_ctx, size_t src, size_t src_ctx, size):
    status = cuMemcpyPeer(
        <Deviceptr>dst, <Context>dst_ctx, <Deviceptr>src, <Context>src_ctx,
        size)
    check_status(status)


cpdef memcpyPeerAsync(size_t dst, size_t dst_ctx, size_t src, size_t src_ctx,
                      size, size_t stream):
    status = cuMemcpyPeerAsync(
        <Deviceptr>dst, <Context>dst_ctx, <Deviceptr>src, <Context>src_ctx,
        size, <Stream>stream)
    check_status(status)


cpdef memsetD32(size_t ptr, value, size):
    status = cuMemsetD32(<Deviceptr>ptr, value, size)
    check_status(status)


cpdef memsetD32Async(size_t ptr, value, size, size_t stream):
    status = cuMemsetD32Async(<Deviceptr>ptr, value, size, <Stream>stream)
    check_status(status)


cpdef size_t pointerGetAttribute(attribute, ptr):
    assert attribute == 0  # Currently only context query is supported

    cdef Context ctx
    status = cuPointerGetAttribute(&ctx, attribute, <Deviceptr>ptr)
    check_status(status)
    return <size_t>ctx


###############################################################################
# Stream and Event
###############################################################################

cpdef size_t streamCreate(flag=0):
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


cdef _streamCallbackFunc(Stream hStream, int status, void *userData):
    func, data = <tuple>userData
    func(<size_t>hStream, status, data)

cpdef streamAddCallback(size_t stream, callback, size_t arg,
                        unsigned int flags=0):
    func_arg = (callback, arg)
    status = cuStreamAddCallback(
        <Stream>stream, <StreamCallback>_streamCallbackFunc,
        <void*>func_arg, flags)
    check_status(status)


cpdef size_t eventCreate(flag):
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
