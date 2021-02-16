from libc.stdint cimport intptr_t

cdef intptr_t get_current_stream_ptr()
cdef set_current_stream_ptr(intptr_t ptr)
cdef bint is_ptds_enabled()
