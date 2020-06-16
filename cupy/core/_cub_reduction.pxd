from cupy.core cimport _kernel
from cupy.core cimport _scalar
from cupy.cuda cimport function


cdef tuple _can_use_cub_block_reduction(
    list in_args, list out_args, tuple reduce_axis, tuple out_axis)


cdef (Py_ssize_t, Py_ssize_t) _get_cub_block_specs(  # NOQA
    Py_ssize_t contiguous_size)


cdef _launch_cub(
    self, out_block_num, block_size, block_stride,
    in_args, out_args, in_shape, out_shape, type_map,
    map_expr, reduce_expr, post_map_expr, reduce_type,
    stream, params, cub_params)
