cdef class RawKernel:

    cdef:
        readonly str code
        readonly str name
        readonly tuple options
        object _kernel
        readonly str backend
        bint translate_cucomplex
        readonly bint enable_cooperative_groups


cdef class RawModule:

    cdef:
        readonly str code
        readonly str cubin_path
        readonly tuple options
        dict kernels
        readonly str backend
        object module
        bint translate_cucomplex
        readonly bint enable_cooperative_groups
