from libc.stdint cimport intptr_t

# Uses Python threading
cdef intptr_t get_current_stream_ptr() except? -1
cdef intptr_t get_stream_ptr(int device_id) except? -1
cdef set_current_stream_ptr(intptr_t ptr, int device_id=*)
cpdef intptr_t get_default_stream_ptr() noexcept
cdef bint is_ptds_enabled() noexcept
