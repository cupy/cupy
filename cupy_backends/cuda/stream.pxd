from libc.stdint cimport intptr_t


cdef bint enable_current_stream
cdef intptr_t get_current_stream_ptr()
cpdef get_current_stream()
