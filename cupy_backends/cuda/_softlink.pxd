ctypedef int (*F_t)(...) nogil  # NOQA

cdef class SoftLink:
    cdef:
        bint available
        object _cdll
        str _prefix
        F_t get(self, str name)
