# Keep in sync with typenames exported in `driver.pxd`.

cdef extern from *:
    ctypedef int Device 'CUdevice'
    ctypedef int Result 'CUresult'

    ctypedef void* Context 'CUcontext'
    ctypedef void* Deviceptr 'CUdeviceptr'
    ctypedef void* Event 'CUevent'
    ctypedef void* Function 'CUfunction'
    ctypedef void* Module 'CUmodule'
    ctypedef void* Stream 'CUstream'
    ctypedef void* LinkState 'CUlinkState'

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
