from libc.stdint cimport intptr_t


###############################################################################
# Types
###############################################################################

cdef extern from *:
    ctypedef int Device 'CUdevice'
    ctypedef int Result 'CUresult'

    ctypedef void* Context 'CUcontext'
    ctypedef void* Deviceptr 'CUdeviceptr'
    ctypedef void* Event 'cudaEvent_t'
    ctypedef void* Function 'CUfunction'
    ctypedef void* Module 'CUmodule'
    ctypedef void* Stream 'cudaStream_t'
    ctypedef void* LinkState 'CUlinkState'
    ctypedef void* TexRef 'CUtexref_st*'

    ctypedef int CUjit_option 'CUjit_option'
    ctypedef int CUjitInputType 'CUjitInputType'
    ctypedef int CUfunction_attribute 'CUfunction_attribute'

    ctypedef size_t(*CUoccupancyB2DSize)(int)

    # For Texture Reference
    ctypedef void* Array 'CUarray_st*'  # = cupy.cuda.runtime.Array
    ctypedef int Array_format 'CUarray_format'
    ctypedef struct Array_desc 'CUDA_ARRAY_DESCRIPTOR':
        Array_format Format
        size_t Height
        unsigned int NumChannels
        size_t Width
    ctypedef int Address_mode 'CUaddress_mode'
    ctypedef int Filter_mode 'CUfilter_mode'


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

    CUDA_ERROR_INVALID_VALUE = 1

    # CUarray_format
    CU_AD_FORMAT_UNSIGNED_INT8 = 0x01
    CU_AD_FORMAT_UNSIGNED_INT16 = 0x02
    CU_AD_FORMAT_UNSIGNED_INT32 = 0x03
    CU_AD_FORMAT_SIGNED_INT8 = 0x08
    CU_AD_FORMAT_SIGNED_INT16 = 0x09
    CU_AD_FORMAT_SIGNED_INT32 = 0x0a
    CU_AD_FORMAT_HALF = 0x10
    CU_AD_FORMAT_FLOAT = 0x20

    # CUaddress_mode
    CU_TR_ADDRESS_MODE_WRAP = 0
    CU_TR_ADDRESS_MODE_CLAMP = 1
    CU_TR_ADDRESS_MODE_MIRROR = 2
    CU_TR_ADDRESS_MODE_BORDER = 3

    # CUfilter_mode
    CU_TR_FILTER_MODE_POINT = 0
    CU_TR_FILTER_MODE_LINEAR = 1

    CU_TRSA_OVERRIDE_FORMAT = 0x01

    CU_TRSF_READ_AS_INTEGER = 0x01
    CU_TRSF_NORMALIZED_COORDINATES = 0x02
    CU_TRSF_SRGB = 0x10


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
cpdef intptr_t moduleGetTexRef(intptr_t module, str texrefname) except? 0
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
# Texture reference
###############################################################################

cpdef size_t texRefSetAddress(intptr_t texref, intptr_t dptr, size_t nbytes)
cpdef texRefSetAddress2D(intptr_t texref, intptr_t desc, intptr_t dptr,
                         size_t Pitch)
cpdef texRefSetAddressMode(intptr_t texref, int dim, int am)
cpdef texRefSetArray(intptr_t texref, intptr_t array)
cpdef texRefSetBorderColor(intptr_t texref, pBorderColor)
cpdef texRefSetFilterMode(intptr_t texref, int fm)
cpdef texRefSetFlags(intptr_t texref, unsigned int Flags)
cpdef texRefSetFormat(intptr_t texref, int fmt, int NumPackedComponents)
cpdef texRefSetMaxAnisotropy(intptr_t texref, unsigned int maxAniso)

###############################################################################
# Occupancy
###############################################################################

cpdef int occupancyMaxActiveBlocksPerMultiprocessor(
    intptr_t func, int blockSize, size_t dynamicSMemSize)

cpdef occupancyMaxPotentialBlockSize(intptr_t func, size_t dynamicSMemSize,
                                     int blockSizeLimit)
