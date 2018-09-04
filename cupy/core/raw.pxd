cdef class RawKernel:

    cdef:
        readonly str code
        readonly str name
        readonly tuple options
