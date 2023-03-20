from libc.stdint cimport intptr_t


###############################################################################
# Types and Enums
###############################################################################

IF CUPY_USE_CUDA_PYTHON:
    from cuda.ccuda cimport *
    # Aliases for compatibillity with existing CuPy codebase.
    # Keep in sync with names defined in `_driver_typedef.pxi`.
    # TODO(kmaehashi): Remove these aliases.
    ctypedef CUdevice Device
    ctypedef CUresult Result
    ctypedef CUcontext Context
    ctypedef CUdeviceptr Deviceptr
    ctypedef CUevent Event
    ctypedef CUfunction Function
    ctypedef CUmodule Module
    ctypedef CUstream Stream
    ctypedef CUlinkState LinkState
    ctypedef CUarray_st* Array
    ctypedef CUarray_format Array_format
    ctypedef CUDA_ARRAY_DESCRIPTOR Array_desc
    ctypedef CUaddress_mode Address_mode
    ctypedef CUfilter_mode Filter_mode
ELSE:
    include "_driver_typedef.pxi"
    from cupy_backends.cuda.api._driver_enum cimport *


###############################################################################
# Build-time version
###############################################################################

cpdef get_build_version()

###############################################################################
# Primary context management
###############################################################################

cpdef devicePrimaryCtxRelease(Device dev)

###############################################################################
# Context management
###############################################################################

cpdef intptr_t ctxGetCurrent() except? 0
cpdef ctxSetCurrent(intptr_t ctx)
cpdef intptr_t ctxCreate(Device dev) except? 0
cpdef ctxDestroy(intptr_t ctx)
cpdef int ctxGetDevice() except? -1

###############################################################################
# Module load and kernel execution
###############################################################################

cpdef intptr_t linkCreate() except? 0
cpdef linkAddData(intptr_t state, int input_type, bytes data, unicode name)
cpdef linkAddFile(intptr_t state, int input_type, unicode path)
cpdef bytes linkComplete(intptr_t state)
cpdef linkDestroy(intptr_t state)
cpdef intptr_t moduleLoad(str filename) except? 0
cpdef intptr_t moduleLoadData(bytes image) except? 0
cpdef moduleUnload(intptr_t module)
cpdef intptr_t moduleGetFunction(intptr_t module, str funcname) except? 0
cpdef intptr_t moduleGetGlobal(intptr_t module, str varname) except? 0
cpdef launchKernel(
    intptr_t f, unsigned int grid_dim_x, unsigned int grid_dim_y,
    unsigned int grid_dim_z, unsigned int block_dim_x,
    unsigned int block_dim_y, unsigned int block_dim_z,
    unsigned int shared_mem_bytes, intptr_t stream, intptr_t kernel_params,
    intptr_t extra)
cpdef launchCooperativeKernel(
    intptr_t f, unsigned int grid_dim_x, unsigned int grid_dim_y,
    unsigned int grid_dim_z, unsigned int block_dim_x,
    unsigned int block_dim_y, unsigned int block_dim_z,
    unsigned int shared_mem_bytes, intptr_t stream, intptr_t kernel_params)

###############################################################################
# Kernel attributes
###############################################################################

cpdef int funcGetAttribute(int attribute, intptr_t func) except? -2
cpdef funcSetAttribute(intptr_t func, int attribute, int value)

###############################################################################
# Occupancy
###############################################################################

cpdef int occupancyMaxActiveBlocksPerMultiprocessor(
    intptr_t func, int blockSize, size_t dynamicSMemSize)

cpdef occupancyMaxPotentialBlockSize(intptr_t func, size_t dynamicSMemSize,
                                     int blockSizeLimit)

###############################################################################
# Stream management
###############################################################################

cpdef intptr_t streamGetCtx(intptr_t stream) except? 0
