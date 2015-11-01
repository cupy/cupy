###############################################################################
# Types
###############################################################################
ctypedef int Device
ctypedef void* Event
ctypedef void* Stream
ctypedef void* Context
ctypedef void* Function
ctypedef void* Module
ctypedef void* Deviceptr
ctypedef void (*StreamCallback)(Stream hStream, int status, void* userData)

###############################################################################
# Error handling
###############################################################################

cpdef check_status(int status)
