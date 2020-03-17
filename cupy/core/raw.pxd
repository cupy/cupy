cdef class RawKernel:

    cdef:
        readonly str code
        readonly str name
        readonly tuple options
        readonly str backend
        readonly bint enable_cooperative_groups
        bint translate_cucomplex
        dict kernels


cdef class RawModule:

    cdef:
        readonly str code
        readonly str cubin_path
        readonly tuple options
        readonly str backend
        readonly bint enable_cooperative_groups
        bint translate_cucomplex
        dict kernels
        dict modules

        _load_from_path(self, str)
        _compile_from_source(self, str, tuple, str, bint, bint)
