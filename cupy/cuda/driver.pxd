###############################################################################
# Types
###############################################################################

cdef extern from *:
    ctypedef int Device 'CUdevice'
    ctypedef int Result 'CUresult'

    ctypedef void* Deviceptr 'CUdeviceptr'
    ctypedef void* Event 'struct CUevent_st*'
    ctypedef void* Function 'struct CUfunc_st*'
    ctypedef void* Module 'struct CUmod_st*'
    ctypedef void* Stream 'struct CUstream_st*'
    ctypedef void* LinkState 'CUlinkState'

    ctypedef int CUjit_option 'CUjit_option'
    ctypedef int CUjitInputType 'CUjitInputType'

cpdef enum:
    CU_JIT_INPUT_CUBIN = 0
    CU_JIT_INPUT_PTX = 1
    CU_JIT_INPUT_FATBINARY = 2
    CU_JIT_INPUT_OBJECT = 3
    CU_JIT_INPUT_LIBRARY = 4


###############################################################################
# Module load and kernel execution
###############################################################################

cpdef size_t linkCreate() except *
cpdef linkAddData(size_t state, int input_type, bytes data, str name)
cpdef bytes linkComplete(size_t state)
cpdef linkDestroy(size_t state)
cpdef size_t moduleLoad(str filename) except *
cpdef size_t moduleLoadData(bytes image) except *
cpdef moduleUnload(size_t module)
cpdef size_t moduleGetFunction(size_t module, str funcname) except *
cpdef size_t moduleGetGlobal(size_t module, str varname) except *
cpdef launchKernel(
    size_t f, unsigned int grid_dim_x, unsigned int grid_dim_y,
    unsigned int grid_dim_z, unsigned int block_dim_x,
    unsigned int block_dim_y, unsigned int block_dim_z,
    unsigned int shared_mem_bytes, size_t stream, size_t kernel_params,
    size_t extra)
