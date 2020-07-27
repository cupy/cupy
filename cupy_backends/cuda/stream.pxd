from libc.stdint cimport intptr_t

cdef bint enable_current_stream
cdef intptr_t get_current_stream_ptr()
cdef set_current_stream_ptr(intptr_t ptr)
