from cupy.core cimport _carray
from cupy.core cimport _scalar
from cupy.core.core cimport ndarray


cdef class _Args:
    # This class encapsulates kernel arguments and provides various operations
    # on them that are necessary in preparation for a kernel launch.
    # Not immutable.

    cdef:
        readonly object in_params
        readonly object in_args
        readonly object out_params
        readonly list out_args  # can be None

    # Returns whether the specified input argument is an ndarray.
    cdef bint is_in_ndarray(self, int index)

    cdef list all_args(self)

    cdef tuple all_params(self)

    # Preprocesses arguments for kernel invocation.
    # - Converts arguments to _Arg.
    # - Checks device compatibility for ndarrays
    @staticmethod
    cdef list _preprocess(list objs, int device_id)

    # Copies in_args if their memory ranges are overlapping with out_args.
    # Items in in_args are updated in-place.
    @staticmethod
    cdef _copy_in_args_if_needed(list in_args, list out_args)

    # Broadcasts the arguments.
    # self.in_args and self.out_args will be updated with the resulted arrays.
    cdef broadcast(self)

    cdef set_scalar_dtypes(self, in_types)

    # Assigns new out_args.
    # Existing out_args, if any, will be overwritten.
    cdef set_out_args(self, out_args)

    cdef tuple get_out_arrays(self)

    # Squashes dimensions of arrays and returns the resulted shape.
    # in_args, out_args are updated in-place.
    cdef tuple reduce_dims(self, shape)

    # Returns a list of arguments that are directly passed to the kernel
    # function. The indexer argument is returned separately.
    cdef tuple get_kernel_args(self, shape)


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
