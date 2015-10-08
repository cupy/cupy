cdef class Device:
    cdef public int id
    cdef list _device_stack
    cpdef use(self)
    cpdef synchronize(self)
