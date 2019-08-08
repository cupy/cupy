cdef class RawKernel:

    cdef:
        readonly str code
        readonly str name
        readonly tuple options
        object _kernel


cdef class RawModule:

    cdef:
        readonly str code
        readonly tuple options
        public dict kernels
        object module
