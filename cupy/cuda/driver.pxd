from cupy.cuda.driver_types cimport Device


###############################################################################
# Types
###############################################################################

cpdef enum:
    CU_JIT_INPUT_CUBIN = 0
    CU_JIT_INPUT_PTX = 1
    CU_JIT_INPUT_FATBINARY = 2
    CU_JIT_INPUT_OBJECT = 3
    CU_JIT_INPUT_LIBRARY = 4


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
