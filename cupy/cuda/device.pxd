cpdef int get_device_id() except *
cpdef size_t get_cublas_handle() except *
cpdef size_t get_cusolver_handle() except *
cpdef size_t get_cusparse_handle() except *
cpdef str get_compute_capability()

cdef class Device:
    cdef:
        public int id
        list _device_stack

    cpdef use(self)
    cpdef synchronize(self)
