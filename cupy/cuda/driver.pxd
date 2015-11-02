###############################################################################
# Types
###############################################################################

IF CUPY_USE_CUDA:
    cdef extern from *:
        ctypedef int Device 'CUdevice'
        ctypedef int DeviceAttribute 'CUdevice_attribute'
        ctypedef int PointerAttribute 'CUpointer_attribute'
        ctypedef int Result 'CUresult'

        ctypedef void* Context 'struct CUctx_st*'
        ctypedef void* Deviceptr 'CUdeviceptr'
        ctypedef void* Event 'struct CUevent_st*'
        ctypedef void* Function 'struct CUfunc_st*'
        ctypedef void* Module 'struct CUmod_st*'
        ctypedef void* Stream 'struct CUstream_st*'

        ctypedef void (*StreamCallbackDef)(
            Stream hStream, Result status, void* userData)
        ctypedef StreamCallbackDef StreamCallback 'CUstreamCallback'

ELSE:
    cdef:
        ctypedef int Device
        ctypedef int DeviceAttribute
        ctypedef int PointerAttribute
        ctypedef int Result

        ctypedef void* Context
        ctypedef void* Deviceptr
        ctypedef void* Event
        ctypedef void* Function
        ctypedef void* Module
        ctypedef void* Stream

        ctypedef void (*StreamCallbackDef)(
            Stream hStream, Result status, void* userData)
        ctypedef StreamCallbackDef StreamCallback


###############################################################################
# Enum
###############################################################################

cpdef enum:
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
# Initialization
###############################################################################

cpdef init()
cpdef int driverGetVersion()


###############################################################################
# Device and context operations
###############################################################################

cpdef Device deviceGet(int device_id)
cpdef int deviceGetAttribute(int attrib, Device device)
cpdef int deviceGetCount()
cpdef size_t deviceTotalMem(Device device)
cpdef size_t ctxCreate(unsigned int flag, Device device)
cpdef ctxDestroy(size_t ctx)
cpdef unsigned int ctxGetApiVersion(size_t ctx)
cpdef size_t ctxGetCurrent()
cpdef Device ctxGetDevice()
cpdef size_t ctxPopCurrent()
cpdef ctxPushCurrent(size_t ctx)
cpdef ctxSetCurrent(size_t ctx)
cpdef ctxSynchronize()


###############################################################################
# Module load and kernel execution
###############################################################################

cpdef size_t moduleLoad(str filename)
cpdef size_t moduleLoadData(bytes image)
cpdef moduleUnload(size_t module)
cpdef size_t moduleGetFunction(size_t module, str funcname)
cpdef size_t moduleGetGlobal(size_t module, str varname)
cpdef launchKernel(
        size_t f, unsigned int grid_dim_x, unsigned int grid_dim_y,
        unsigned int grid_dim_z, unsigned int block_dim_x,
        unsigned int block_dim_y, unsigned int block_dim_z,
        unsigned int shared_mem_bytes, size_t stream, size_t kernel_params,
        size_t extra)


###############################################################################
# Memory management
###############################################################################

cpdef size_t memAlloc(size_t size)
cpdef memFree(size_t ptr)
cpdef tuple memGetinfo()
cpdef memcpy(size_t dst, size_t src, size_t size)
cpdef memcpyAsync(size_t dst, size_t src, size_t size, size_t stream)
cpdef memcpyDtoD(size_t dst, size_t src, size_t size)
cpdef memcpyDtoDAsync(size_t dst, size_t src, size_t size, size_t stream)
cpdef memcpyDtoH(size_t dst, size_t src, size_t size)
cpdef memcpyDtoHAsync(size_t dst, size_t src, size_t size, size_t stream)
cpdef memcpyHtoD(size_t dst, size_t src, size_t size)
cpdef memcpyHtoDAsync(size_t dst, size_t src, size_t size, size_t stream)
cpdef memcpyPeer(size_t dst, size_t dst_ctx, size_t src, size_t src_ctx,
                 size_t size)
cpdef memcpyPeerAsync(size_t dst, size_t dst_ctx, size_t src, size_t src_ctx,
                      size_t size, size_t stream)
cpdef memsetD32(size_t ptr, unsigned int value, size_t size)
cpdef memsetD32Async(size_t ptr, unsigned int value, size_t size,
                     size_t stream)
cpdef size_t pointerGetAttribute(int attribute, size_t ptr)


###############################################################################
# Stream and Event
###############################################################################

cpdef size_t streamCreate(unsigned int flag=*)
cpdef streamDestroy(size_t stream)
cpdef streamSynchronize(size_t stream)
cdef _streamCallbackFunc(Stream hStream, Result status, void *userData)
cpdef streamAddCallback(size_t stream, object callback, size_t arg,
                        unsigned int flags=*)
cpdef size_t eventCreate(unsigned int flag)
cpdef eventDestroy(size_t event)
cpdef eventRecord(size_t event, size_t stream)
cpdef eventSynchronize(size_t event)
