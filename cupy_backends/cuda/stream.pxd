from libc.stdint cimport intptr_t

cdef intptr_t get_current_stream_ptr()
cdef intptr_t get_stream_ptr(int device_id)
cdef set_current_stream_ptr(intptr_t ptr, int device_id=*)
cpdef intptr_t get_default_stream_ptr()
cdef bint is_ptds_enabled()
cdef intptr_t get_cublas_workspace_ptr()
cdef size_t get_cublas_workspace_size()
