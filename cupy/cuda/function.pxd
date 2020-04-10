from libc.stdint cimport intptr_t

from cupy.core cimport core


cdef class CPointer:
    cdef void* ptr


cdef class Function:

    cdef:
        public Module module
        public intptr_t ptr

    cpdef linear_launch(self, size_t size, args, size_t shared_mem=*,
                        size_t block_max_size=*, stream=*)


cdef class Module:

    cdef:
        public intptr_t ptr

    cpdef load_file(self, filename)
    cpdef load(self, bytes cubin)
    cpdef get_global_var(self, name)
    cpdef get_function(self, name)
    cpdef get_texref(self, name)


cdef class LinkState:

    cdef:
        public intptr_t ptr

    cpdef add_ptr_data(self, unicode data, unicode name)
    cpdef add_ptr_file(self, unicode path)
    cpdef bytes complete(self)


cdef enum _ArgKind:
    ARG_KIND_NDARRAY = 1
    ARG_KIND_INDEXER
    ARG_KIND_SCALAR
    ARG_KIND_POINTER


cdef class Arg:

    cdef:
        object obj
        # The following arguments are returned by get_immutable_key() and thus
        # read-only.
        readonly _ArgKind arg_kind
        readonly type type
        object dtype
        readonly int ndim
        readonly bint c_contiguous

    cdef _init_fast_base(self, object obj, _ArgKind arg_kind, object typ,
                         object dtype, int ndim, bint c_contiguous)

    @staticmethod
    cdef Arg from_obj(object obj)

    @staticmethod
    cdef NdarrayArg from_ndarray(core.ndarray arr)

    @staticmethod
    cdef IndexerArg from_indexer(tuple shape)

    cdef tuple get_immutable_key(self)

    cdef bint is_ndarray(self)

    cdef bint is_scalar(self)

    cdef CPointer get_pointer(self)


cdef class IndexerArg(Arg):

    cdef:
        readonly tuple shape

    cdef _init_fast(self, tuple shape)

    cdef CPointer get_pointer(self)


cdef class NdarrayArg(Arg):

    cdef _init_fast(self, core.ndarray obj, int ndim, bint c_contiguous)

    cdef CPointer get_pointer(self)

    cdef copy_in_arg_if_needed(self, list out_args)

    cdef NdarrayArg as_ndim(self, int ndim)


cdef class ScalarArg(Arg):
    cdef:
        readonly object _numpy_scalar
        readonly object _dtype
        bint _dtype_applied

    cdef _init_fast(self, object obj)

    cdef object get_min_scalar_type(self)

    cdef apply_dtype(self, object dtype)

    cdef CPointer get_pointer(self)


cdef class PointerArg(Arg):

    cdef _init_fast(self, intptr_t ptr)

    cdef CPointer get_pointer(self)
