from cupy.core cimport _kernel
from cupy.core.core cimport ndarray
from cupy.cuda cimport function


cdef class _AbstractReductionKernel:

    cdef:
        readonly str name
        public str identity
        readonly tuple in_params
        readonly tuple out_params
        readonly tuple _params

    cpdef ndarray _call(
        self,
        list in_args, list out_args,
        tuple a_shape, axis, dtype,
        bint keepdims, bint reduce_dims,
        stream)

    cdef tuple _get_expressions_and_types(
        self, list in_args, list out_args, dtype)

    cdef list _get_out_args(
        self, list out_args, tuple out_types, tuple out_shape)

    cdef function.Function _get_function(
        self,
        tuple params, tuple args_info, _kernel._TypeMap type_map,
        str map_expr, str reduce_expr, str post_map_expr, str reduce_type,
        Py_ssize_t block_size, int device_id)


cdef class ReductionKernel(_AbstractReductionKernel):

    cdef:
        readonly int nin
        readonly int nout
        readonly int nargs
        readonly tuple params
        readonly str reduce_expr
        readonly str map_expr
        readonly str post_map_expr
        readonly object options
        readonly bint reduce_dims
        readonly object reduce_type
        readonly str preamble


cpdef create_reduction_func(name, ops, routine=*, identity=*, preamble=*)
