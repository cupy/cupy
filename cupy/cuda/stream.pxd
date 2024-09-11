from libc.stdint cimport intptr_t


cdef intptr_t get_current_stream_ptr()
cpdef get_current_stream(int device_id=*)
cpdef set_current_cublas_workspace(
        intptr_t workspace, size_t size, int device_id=*)
