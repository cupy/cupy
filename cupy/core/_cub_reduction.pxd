from cupy.core cimport _kernel
from cupy.core cimport _scalar
from cupy.cuda cimport function


cdef tuple _can_use_cub_block_reduction(
    list in_args, list out_args, tuple reduce_axis, tuple out_axis)


cdef (Py_ssize_t, Py_ssize_t) _get_cub_block_specs(  # NOQA
    Py_ssize_t contiguous_size)


cdef _scalar.CScalar _cub_convert_to_c_scalar(
    Py_ssize_t segment_size, Py_ssize_t value)


cdef _cub_two_pass_launch(
    str name, Py_ssize_t block_size, Py_ssize_t segment_size,
    Py_ssize_t items_per_thread, str reduce_type, tuple params,
    list in_args, list out_args,
    str identity, str pre_map_expr, str reduce_expr, str post_map_expr,
    _kernel._TypeMap type_map, str input_expr, str output_expr,
    str preamble, tuple options, stream)
