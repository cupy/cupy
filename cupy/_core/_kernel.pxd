from libcpp cimport vector

from cupy._core cimport _carray
from cupy._core cimport _scalar
from cupy._core._carray cimport shape_t
from cupy._core.core cimport ndarray
from cupy.cuda cimport memory
from cupy.cuda cimport texture


cdef class ParameterInfo:
    cdef:
        readonly str name
        readonly object dtype
        readonly str ctype
        readonly bint raw
        readonly bint is_const


cdef enum _ArgKind:
    ARG_KIND_NDARRAY = 1
    ARG_KIND_INDEXER
    ARG_KIND_SCALAR
    ARG_KIND_POINTER
    ARG_KIND_TEXTURE


cdef class _ArgInfo:
    # Holds metadata of an argument.
    # This class is immutable and used as a part of hash keys.

    cdef:
        readonly _ArgKind arg_kind
        readonly type type
        readonly object dtype
        readonly int ndim
        readonly bint c_contiguous
        readonly bint index_32_bits

    cdef _ArgInfo _init(
        self,
        _ArgKind arg_kind,
        type typ,
        object dtype,
        int ndim,
        bint c_contiguous,
        bint index_32_bits)

    @staticmethod
    cdef _ArgInfo from_arg(object arg)

    @staticmethod
    cdef _ArgInfo from_ndarray(ndarray arg)

    @staticmethod
    cdef _ArgInfo from_scalar(_scalar.CScalar arg)

    @staticmethod
    cdef _ArgInfo from_indexer(_carray.Indexer arg)

    @staticmethod
    cdef _ArgInfo from_memptr(memory.MemoryPointer arg)

    @staticmethod
    cdef _ArgInfo from_texture(texture.TextureObject arg)

    cdef _ArgInfo as_ndarray_with_ndim(self, int ndim)

    cdef bint is_ndarray(self)

    cdef bint is_scalar(self)

    cdef str get_c_type(self)

    cdef str get_param_c_type(self, ParameterInfo p)

    cdef str get_c_var_name(self, ParameterInfo p)


cdef class _TypeMap:
    # Typedef mapping between C types.
    # This class is immutable.

    cdef:
        tuple _pairs

    cdef str get_typedef_code(self)


cdef class _Op:
    """Simple data structure that represents a kernel routine with single \
concrete dtype mapping.
    """

    cdef:
        readonly tuple in_types
        readonly tuple out_types
        readonly int nin
        readonly int nout
        readonly object routine
        # If the type combination specified by in_types and out_types is
        # disallowed, error_func must be set instead of routine.
        # It's called by check_valid() method.
        readonly object error_func

    @staticmethod
    cdef _Op _from_type_and_routine_or_error_func(
        str typ, object routine, object error_func)

    # Creates an op instance parsing a dtype mapping.
    @staticmethod
    cdef _Op from_type_and_routine(str typ, routine)

    cpdef tuple get_in_dtypes(self)

    cpdef tuple get_out_dtypes(self)

    # Creates an op instance parsing a dtype mapping with given error function.
    @staticmethod
    cdef _Op from_type_and_error_func(str typ, error_func)

    # Raises an error if error_func is given.
    cdef check_valid(self)


cdef class _Ops:
    """A kernel routine representation with various dtype mappings.
    """

    cdef:
        readonly tuple ops
        readonly int nin
        readonly int nout

    @staticmethod
    cdef _Ops from_tuples(object ops, routine)

    # Queries a single op from input arguments.
    cpdef _Op guess_routine(
        self, str name, dict cache, list in_args, dtype, _Ops out_ops)

    cpdef _Op _guess_routine_from_in_types(
        self, tuple in_types, object can_cast=*)

    cpdef _Op _guess_routine_from_dtype(self, object dtype)


cpdef create_ufunc(name, ops, routine=*, preamble=*, doc=*,
                   default_casting=*, loop_prep=*, out_ops=*)

cdef tuple _get_arginfos(list args)

cdef str _get_kernel_params(tuple params, tuple arginfos)

cdef list _broadcast(list args, tuple params, bint use_size, shape_t& shape)

cdef list _get_out_args(list out_args, tuple out_types,
                        const shape_t& out_shape, casting)

cdef list _get_out_args_with_params(
    list out_args, tuple out_types,
    const shape_t& out_shape, tuple out_params, bint is_size_specified)

cdef _check_array_device_id(ndarray arr, int device_id)

cdef list _preprocess_args(int dev_id, args, bint use_c_scalar)

cdef shape_t _reduce_dims(list args, tuple params, const shape_t& shape)
