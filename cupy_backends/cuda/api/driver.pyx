# distutils: language = c++

"""Thin wrapper of CUDA Driver API.

There are four differences compared to the original C API.

1. Not all functions are ported.
2. Errors are translated into CUDADriverError exceptions.
3. The 'cu' prefix of each API is omitted and the next character is set to
   lower case.
4. The resulting values are returned directly instead of references.

"""
cimport cython  # NOQA
from libc.stdint cimport intptr_t


###############################################################################
# Extern and Constants
###############################################################################

IF CUPY_USE_CUDA_PYTHON:
    from cuda.ccuda cimport *
    cdef inline void initialize():
        return
ELSE:
    include '_driver_extern.pxi'
    pass  # for cython-lint

cdef extern from '../../cupy_backend.h' nogil:
    # Build-time version
    # Note: CUDA_VERSION is defined either in CUDA Python or _driver_extern.pxi
    enum: HIP_VERSION

# Provide access to constants from Python.
from cupy_backends.cuda.api._driver_enum import *


###############################################################################
# Error handling
###############################################################################

class CUDADriverError(RuntimeError):

    def __init__(self, Result status):
        initialize()
        self.status = status
        cdef const char *name
        cdef const char *msg
        cuGetErrorName(status, &name)
        cuGetErrorString(status, &msg)
        cdef bytes s_name = name, s_msg = msg
        super(CUDADriverError, self).__init__(
            '%s: %s' % (s_name.decode(), s_msg.decode()))

    def __reduce__(self):
        return (type(self), (self.status,))


@cython.profile(False)
cpdef inline check_status(int status):
    if status != 0:
        raise CUDADriverError(status)


@cython.profile(False)
cdef inline void check_attribute_status(int status, int* pi) except *:
    # set attribute to -1 on older versions of CUDA where it was undefined
    if status == CUDA_ERROR_INVALID_VALUE:
        pi[0] = -1
    elif status != 0:
        raise CUDADriverError(status)


###############################################################################
# Build-time version
###############################################################################

cpdef get_build_version():
    """Returns the CUDA_VERSION / HIP_VERSION constant.

    Note that when built with CUDA Python support, CUDA_VERSION will become a
    constant:

    https://github.com/NVIDIA/cuda-python/blob/v11.4.0/cuda/ccuda.pxd#L2268

    In CuPy codebase, use `runtime.runtimeGetVersion()` instead of
    this function to change the behavior based on the target CUDA version.
    """

    # The versions are mutually exclusive
    if CUPY_CUDA_VERSION > 0:
        return CUDA_VERSION
    elif CUPY_HIP_VERSION > 0:
        return HIP_VERSION
    else:
        return 0


cpdef bint _is_cuda_python():
    return CUPY_USE_CUDA_PYTHON


###############################################################################
# Primary context management
###############################################################################

cpdef devicePrimaryCtxRelease(Device dev):
    initialize()
    with nogil:
        status = cuDevicePrimaryCtxRelease(dev)
    check_status(status)

###############################################################################
# Context management
###############################################################################

cpdef intptr_t ctxGetCurrent() except? 0:
    initialize()
    cdef Context ctx
    with nogil:
        status = cuCtxGetCurrent(&ctx)
    check_status(status)
    return <intptr_t>ctx

cpdef ctxSetCurrent(intptr_t ctx):
    initialize()
    with nogil:
        status = cuCtxSetCurrent(<Context>ctx)
    check_status(status)

cpdef intptr_t ctxCreate(Device dev) except? 0:
    initialize()
    cdef Context ctx
    cdef unsigned int flags = 0
    with nogil:
        status = cuCtxCreate(&ctx, flags, dev)
    check_status(status)
    return <intptr_t>ctx

cpdef ctxDestroy(intptr_t ctx):
    initialize()
    with nogil:
        status = cuCtxDestroy(<Context>ctx)
    check_status(status)

cpdef int ctxGetDevice() except? -1:
    initialize()
    cdef Device dev
    with nogil:
        status = cuCtxGetDevice(&dev)
    check_status(status)
    return dev


###############################################################################
# Module load and kernel execution
###############################################################################

cpdef intptr_t linkCreate() except? 0:
    initialize()
    cdef LinkState state
    with nogil:
        status = cuLinkCreate(0, <CUjit_option*>0, <void**>0, &state)
    check_status(status)
    return <intptr_t>state


cpdef linkAddData(intptr_t state, int input_type, bytes data, unicode name):
    initialize()
    cdef const char* data_ptr = data
    cdef size_t data_size = len(data) + 1
    cdef bytes b_name = name.encode()
    cdef const char* b_name_ptr = b_name
    with nogil:
        status = cuLinkAddData(
            <LinkState>state, <CUjitInputType>input_type, <void*>data_ptr,
            data_size, b_name_ptr, 0, <CUjit_option*>0, <void**>0)
    check_status(status)


cpdef linkAddFile(intptr_t state, int input_type, unicode path):
    initialize()
    cdef bytes b_path = path.encode()
    cdef const char* b_path_ptr = b_path
    with nogil:
        status = cuLinkAddFile(<LinkState>state, <CUjitInputType>input_type,
                               b_path_ptr, 0, <CUjit_option*>0, <void**>0)
    check_status(status)


cpdef bytes linkComplete(intptr_t state):
    initialize()
    cdef void* cubinOut
    cdef size_t sizeOut
    with nogil:
        status = cuLinkComplete(<LinkState>state, &cubinOut, &sizeOut)
    check_status(status)
    return bytes((<char*>cubinOut)[:sizeOut])


cpdef linkDestroy(intptr_t state):
    initialize()
    with nogil:
        status = cuLinkDestroy(<LinkState>state)
    check_status(status)


cpdef intptr_t moduleLoad(str filename) except? 0:
    initialize()
    cdef Module module
    cdef bytes b_filename = filename.encode()
    cdef char* b_filename_ptr = b_filename
    with nogil:
        status = cuModuleLoad(&module, b_filename_ptr)
    check_status(status)
    return <intptr_t>module


cpdef intptr_t moduleLoadData(bytes image) except? 0:
    initialize()
    cdef Module module
    cdef char* image_ptr = image
    with nogil:
        status = cuModuleLoadData(&module, image_ptr)
    check_status(status)
    return <intptr_t>module


cpdef moduleUnload(intptr_t module):
    initialize()
    with nogil:
        status = cuModuleUnload(<Module>module)
    check_status(status)


cpdef intptr_t moduleGetFunction(intptr_t module, str funcname) except? 0:
    initialize()
    cdef Function func
    cdef bytes b_funcname = funcname.encode()
    cdef char* b_funcname_ptr = b_funcname
    with nogil:
        status = cuModuleGetFunction(&func, <Module>module, b_funcname_ptr)
    check_status(status)
    return <intptr_t>func


cpdef intptr_t moduleGetGlobal(intptr_t module, str varname) except? 0:
    initialize()
    cdef Deviceptr var
    cdef size_t size
    cdef bytes b_varname = varname.encode()
    cdef char* b_varname_ptr = b_varname
    with nogil:
        status = cuModuleGetGlobal(&var, &size, <Module>module, b_varname_ptr)
    check_status(status)
    return <intptr_t>var


cpdef launchKernel(
        intptr_t f, unsigned int grid_dim_x, unsigned int grid_dim_y,
        unsigned int grid_dim_z, unsigned int block_dim_x,
        unsigned int block_dim_y, unsigned int block_dim_z,
        unsigned int shared_mem_bytes, intptr_t stream, intptr_t kernel_params,
        intptr_t extra):
    initialize()
    with nogil:
        status = cuLaunchKernel(
            <Function>f, grid_dim_x, grid_dim_y, grid_dim_z,
            block_dim_x, block_dim_y, block_dim_z,
            shared_mem_bytes, <Stream>stream,
            <void**>kernel_params, <void**>extra)
    check_status(status)


cpdef launchCooperativeKernel(
        intptr_t f, unsigned int grid_dim_x, unsigned int grid_dim_y,
        unsigned int grid_dim_z, unsigned int block_dim_x,
        unsigned int block_dim_y, unsigned int block_dim_z,
        unsigned int shared_mem_bytes, intptr_t stream,
        intptr_t kernel_params):
    initialize()
    with nogil:
        status = cuLaunchCooperativeKernel(
            <Function>f, grid_dim_x, grid_dim_y, grid_dim_z,
            block_dim_x, block_dim_y, block_dim_z,
            shared_mem_bytes, <Stream>stream, <void**>kernel_params)
    check_status(status)


###############################################################################
# Function attributes
###############################################################################

# -1 is reserved by check_attribute_status
cpdef int funcGetAttribute(int attribute, intptr_t f) except? -2:
    initialize()
    cdef int pi
    with nogil:
        status = cuFuncGetAttribute(
            &pi,
            <CUfunction_attribute> attribute,
            <Function> f)
    check_attribute_status(status, &pi)
    return pi


cpdef funcSetAttribute(intptr_t f, int attribute, int value):
    initialize()
    with nogil:
        status = cuFuncSetAttribute(
            <Function> f,
            <CUfunction_attribute> attribute,
            value)
    check_status(status)


###############################################################################
# Occupancy
###############################################################################

cpdef int occupancyMaxActiveBlocksPerMultiprocessor(
        intptr_t func, int blockSize, size_t dynamicSMemSize):
    initialize()
    cdef int numBlocks
    with nogil:
        status = cuOccupancyMaxActiveBlocksPerMultiprocessor(
            &numBlocks, <Function> func, blockSize, dynamicSMemSize)
    check_status(status)
    return numBlocks


cpdef occupancyMaxPotentialBlockSize(intptr_t func, size_t dynamicSMemSize,
                                     int blockSizeLimit):
    initialize()
    # CUoccupancyB2DSize is set to NULL as there is no way to pass in a
    # unary function from Python.
    cdef int minGridSize, blockSize
    with nogil:
        status = cuOccupancyMaxPotentialBlockSize(
            &minGridSize, &blockSize, <Function> func, <CUoccupancyB2DSize>
            NULL, dynamicSMemSize, blockSizeLimit)
    check_status(status)
    return minGridSize, blockSize


###############################################################################
# Stream management
###############################################################################

cpdef intptr_t streamGetCtx(intptr_t stream) except? 0:
    initialize()
    cdef Context ctx
    with nogil:
        status = cuStreamGetCtx(<Stream>stream, &ctx)
    check_status(status)
    return <intptr_t>ctx
