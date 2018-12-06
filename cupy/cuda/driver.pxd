###############################################################################
# Types
###############################################################################

cdef class FuncAttributes:
    cdef:
        public size_t sharedSizeBytes
        public size_t constSizeBytes
        public size_t localSizeBytes
        public int maxThreadsPerBlock
        public int numRegs
        public int ptxVersion
        public int binaryVersion
        public int cacheModeCA
        public int maxDynamicSharedSizeBytes
        public int preferredShmemCarveout


cdef extern from *:
    ctypedef int Device 'CUdevice'
    ctypedef int Result 'CUresult'

    ctypedef void* Context 'CUcontext'
    ctypedef void* Deviceptr 'CUdeviceptr'
    ctypedef void* Event 'struct CUevent_st*'
    ctypedef void* Function 'struct CUfunc_st*'
    ctypedef void* Module 'struct CUmod_st*'
    ctypedef void* Stream 'struct CUstream_st*'
    ctypedef void* LinkState 'CUlinkState'

    ctypedef int CUjit_option 'CUjit_option'
    ctypedef int CUjitInputType 'CUjitInputType'
    ctypedef int CUfunction_attribute 'CUfunction_attribute'


cpdef enum:
    CU_JIT_INPUT_CUBIN = 0
    CU_JIT_INPUT_PTX = 1
    CU_JIT_INPUT_FATBINARY = 2
    CU_JIT_INPUT_OBJECT = 3
    CU_JIT_INPUT_LIBRARY = 4

    CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0
    CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1
    CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES = 2
    CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3
    CU_FUNC_ATTRIBUTE_NUM_REGS = 4
    CU_FUNC_ATTRIBUTE_PTX_VERSION = 5
    CU_FUNC_ATTRIBUTE_BINARY_VERSION = 6
    CU_FUNC_ATTRIBUTE_CACHE_MODE_CA = 7
    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8
    CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 9


###############################################################################
# Primary context management
###############################################################################

cpdef devicePrimaryCtxRelease(Device dev)

###############################################################################
# Context management
###############################################################################

cpdef size_t ctxGetCurrent() except? 0
cpdef ctxSetCurrent(size_t ctx)
cpdef size_t ctxCreate(Device dev) except? 0
cpdef ctxDestroy(size_t ctx)

###############################################################################
# Module load and kernel execution
###############################################################################

cpdef size_t linkCreate() except? 0
cpdef linkAddData(size_t state, int input_type, bytes data, unicode name)
cpdef bytes linkComplete(size_t state)
cpdef linkDestroy(size_t state)
cpdef size_t moduleLoad(str filename) except? 0
cpdef size_t moduleLoadData(bytes image) except? 0
cpdef moduleUnload(size_t module)
cpdef size_t moduleGetFunction(size_t module, str funcname) except? 0
cpdef size_t moduleGetGlobal(size_t module, str varname) except? 0
cpdef launchKernel(
    size_t f, unsigned int grid_dim_x, unsigned int grid_dim_y,
    unsigned int grid_dim_z, unsigned int block_dim_x,
    unsigned int block_dim_y, unsigned int block_dim_z,
    unsigned int shared_mem_bytes, size_t stream, size_t kernel_params,
    size_t extra)

###############################################################################
# Kernel attributes
###############################################################################

cpdef FuncAttributes funcGetAttributes(size_t func)
