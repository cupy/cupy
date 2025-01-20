ctypedef int (*func_ptr)(...) nogil  # NOQA

cdef class SoftLink:
    cdef:
        object error
        str prefix
        str _libname
        object _cdll
        func_ptr get(self, str name)
