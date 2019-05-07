cpdef int get_device_id() except? -1
cpdef size_t get_cublas_handle() except? 0
cpdef size_t get_cusolver_handle() except? 0
cpdef size_t get_cusparse_handle() except? 0
cpdef str get_compute_capability()

cdef class Handle:
    cdef:
        public size_t handle
        object _destroy_func

cdef class Device:
    cdef:
        public int id
        list _device_stack

    cpdef use(self)
    cpdef synchronize(self)
