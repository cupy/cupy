# distutils: language = c++

"""Thin wrapper of CUDA Driver API.

There are four differences compared to the original C API.

1. Not all functions are ported.
2. Errors are translated into CUDADriverError exceptions.
3. The 'cu' prefix of each API is omitted and the next character is set to
   lower case.
4. The resulting values are returned directly instead of references.

"""
cimport cython


###############################################################################
# Extern
###############################################################################

cdef extern from "cupy_cuda.h" nogil:
    # Error handling
    int cuGetErrorName(Result error, const char** pStr)
    int cuGetErrorString(Result error, const char** pStr)

    # Module load and kernel execution
    int cuLinkCreate(unsigned int numOptions, CUjit_option* options,
                     void** optionValues, LinkState* stateOut)
    int cuLinkAddData(LinkState state, CUjitInputType type, void* data,
                      size_t size, const char* name, unsigned int  numOptions,
                      CUjit_option* options, void** optionValues)
    int cuLinkComplete(LinkState state, void** cubinOut, size_t* sizeOut)
    int cuLinkDestroy(LinkState state)
    int cuModuleLoad(Module* module, char* fname)
    int cuModuleLoadData(Module* module, void* image)
    int cuModuleUnload(Module hmod)
    int cuModuleGetFunction(Function* hfunc, Module hmod,
                            char* name)
    int cuModuleGetGlobal(Deviceptr* dptr, size_t* bytes, Module hmod,
                          char* name)
    int cuLaunchKernel(
        Function f, unsigned int gridDimX, unsigned int gridDimY,
        unsigned int gridDimZ, unsigned int blockDimX,
        unsigned int blockDimY, unsigned int blockDimZ,
        unsigned int sharedMemBytes, Stream hStream,
        void** kernelParams, void** extra)


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


@cython.profile(False)
cpdef inline check_status(int status):
    if status != 0:
        raise CUDADriverError(status)


###############################################################################
# Module load and kernel execution
###############################################################################

cpdef size_t linkCreate() except *:
    cpdef LinkState state
    with nogil:
        status = cuLinkCreate(0, <CUjit_option*>0, <void**>0, &state)
    check_status(status)
    return <size_t>state


cpdef linkAddData(size_t state, int input_type, bytes data, unicode name):
    cdef const char* data_ptr = data
    cdef size_t data_size = len(data) + 1
    cdef bytes b_name = name.encode()
    cdef const char* b_name_ptr = b_name
    with nogil:
        status = cuLinkAddData(
            <LinkState>state, <CUjitInputType>input_type, <void*>data_ptr,
            data_size, b_name_ptr, 0, <CUjit_option*>0, <void**>0)
    check_status(status)


cpdef bytes linkComplete(size_t state):
    cdef void* cubinOut
    cdef size_t sizeOut
    with nogil:
        status = cuLinkComplete(<LinkState>state, &cubinOut, &sizeOut)
    check_status(status)
    return bytes((<char*>cubinOut)[:sizeOut])


cpdef linkDestroy(size_t state):
    with nogil:
        status = cuLinkDestroy(<LinkState>state)
    check_status(status)


cpdef size_t moduleLoad(str filename) except *:
    cdef Module module
    cdef bytes b_filename = filename.encode()
    cdef char* b_filename_ptr = b_filename
    with nogil:
        status = cuModuleLoad(&module, b_filename_ptr)
    check_status(status)
    return <size_t>module


cpdef size_t moduleLoadData(bytes image) except *:
    cdef Module module
    cdef char* image_ptr = image
    with nogil:
        status = cuModuleLoadData(&module, image_ptr)
    check_status(status)
    return <size_t>module


cpdef moduleUnload(size_t module):
    with nogil:
        status = cuModuleUnload(<Module>module)
    check_status(status)


cpdef size_t moduleGetFunction(size_t module, str funcname) except *:
    cdef Function func
    cdef bytes b_funcname = funcname.encode()
    cdef char* b_funcname_ptr = b_funcname
    with nogil:
        status = cuModuleGetFunction(&func, <Module>module, b_funcname_ptr)
    check_status(status)
    return <size_t>func


cpdef size_t moduleGetGlobal(size_t module, str varname) except *:
    cdef Deviceptr var
    cdef size_t size
    cdef bytes b_varname = varname.encode()
    cdef char* b_varname_ptr = b_varname
    with nogil:
        status = cuModuleGetGlobal(&var, &size, <Module>module, b_varname_ptr)
    check_status(status)
    return <size_t>var


cpdef launchKernel(
        size_t f, unsigned int grid_dim_x, unsigned int grid_dim_y,
        unsigned int grid_dim_z, unsigned int block_dim_x,
        unsigned int block_dim_y, unsigned int block_dim_z,
        unsigned int shared_mem_bytes, size_t stream, size_t kernel_params,
        size_t extra):
    with nogil:
        status = cuLaunchKernel(
            <Function>f, grid_dim_x, grid_dim_y, grid_dim_z,
            block_dim_x, block_dim_y, block_dim_z,
            shared_mem_bytes, <Stream>stream,
            <void**>kernel_params, <void**>extra)
    check_status(status)
