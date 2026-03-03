from libc.stdint cimport intptr_t


cdef intptr_t get_current_stream_ptr() except? -1
cpdef get_current_stream(int device_id=*)
