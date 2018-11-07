###############################################################################
# Types
###############################################################################

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
