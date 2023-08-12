ctypedef int (*func_ptr)(...) nogil  # NOQA

cdef class SoftLink:
    cdef:
        object error
        object _cdll
        str _prefix
        func_ptr get(self, str name)
