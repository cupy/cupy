from cupy.core._carray cimport shape_t
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
        const shape_t& a_shape, axis, dtype,
        bint keepdims, bint reduce_dims, int device_id,
        stream)

    cdef void _launch(
        self, out_block_num, block_size, block_stride,
        in_args, out_args, in_shape, out_shape, types,
        map_expr, reduce_expr, post_map_expr, reduce_type,
        stream)

    cdef tuple _get_expressions_and_types(
        self, list in_args, list out_args, dtype)

    cdef list _get_out_args(
        self, list out_args, tuple out_types, const shape_t& out_shape)

    cdef function.Function _get_function(
        self,
        tuple params, tuple arginfos, _kernel._TypeMap types,
        str map_expr, str reduce_expr, str post_map_expr, str reduce_type,
        Py_ssize_t block_size)


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
