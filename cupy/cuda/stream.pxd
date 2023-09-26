from libc.stdint cimport intptr_t


cdef intptr_t get_current_stream_ptr()
cpdef get_current_stream()


cdef class _BaseStream:
    cdef:
        public intptr_t ptr
        public int device_id


cdef class ExternalStream(_BaseStream):
    pass
