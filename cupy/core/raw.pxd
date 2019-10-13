cdef class RawKernel:

    cdef:
        readonly str code
        readonly str name
        readonly tuple options
        object _kernel
        readonly str backend


cdef class RawModule:

    cdef:
        readonly str code
        readonly str cubin_path
        readonly tuple options
        dict kernels
        readonly str backend
        object module
