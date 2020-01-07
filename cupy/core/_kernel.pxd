from cupy.core cimport _carray
from cupy.core cimport _scalar
from cupy.core.core cimport ndarray


cdef class ParameterInfo:
    cdef:
        readonly str name
        readonly object dtype
        readonly str ctype
        readonly bint raw
        readonly bint is_const


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
    cdef _Op guess_routine(
        self, str name, dict cache, list in_args, dtype, _Ops out_ops)

    cdef _Op _guess_routine_from_in_types(self, tuple in_types)

    cdef _Op _guess_routine_from_dtype(self, object dtype)


cpdef create_ufunc(name, ops, routine=*, preamble=*, doc=*,
                   default_casting=*, loop_prep=*, out_ops=*)

cpdef str _get_kernel_params(tuple params, tuple arginfos)

cdef tuple _broadcast(list args, tuple params, bint use_size)

cdef list _get_out_args(list out_args, tuple out_types, tuple out_shape,
                        casting)

cdef list _get_out_args_with_params(
    list out_args, tuple out_types, tuple out_shape, tuple out_params,
    bint is_size_specified)


cdef _check_array_device_id(ndarray arr, int device_id)

cdef list _preprocess_args(int dev_id, args)

cdef tuple _reduce_dims(list args, tuple params, tuple shape)
