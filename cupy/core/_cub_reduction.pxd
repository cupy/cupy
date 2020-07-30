from cupy.core._carray cimport shape_t
from cupy.core.core cimport ndarray


cdef bint _try_to_call_cub_reduction(
    self, list in_args, list out_args, const shape_t& a_shape,
    stream, optimize_context, tuple key,
    map_expr, reduce_expr, post_map_expr, reduce_type, type_map,
    tuple reduce_axis, tuple out_axis, const shape_t& out_shape,
    ndarray ret) except *
