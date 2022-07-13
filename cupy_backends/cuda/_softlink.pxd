ctypedef int (*func_ptr)(...) nogil

cdef class SoftLink:
    cdef:
        object _cdll
        str _prefix
        func_ptr get(self, str name)
