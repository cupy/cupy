cpdef int get_device_id()
cpdef get_cublas_handle()

cdef class Device:
    cdef:
        public int id
        list _device_stack

    cpdef use(self)
    cpdef synchronize(self)
