cdef class SoftLink:
    cdef:
        object _cdll
        str _prefix
        void* get_func(self, str name)
