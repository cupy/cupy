ctypedef int (*F_t)(...) nogil  # NOQA

cdef class SoftLink:
    cdef:
        object _cdll
        str _prefix
        F_t get(self, str name)
