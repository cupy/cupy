from cupy._core._carray cimport shape_t
from cupy._core._kernel cimport _TypeMap
from cupy._core.core cimport ndarray


cdef bint _try_to_call_cub_reduction(
    self, list in_args, list out_args, const shape_t& a_shape,
    stream, optimize_context, tuple key,
    map_expr, reduce_expr, post_map_expr,
    reduce_type, _TypeMap type_map,
    tuple reduce_axis, tuple out_axis, const shape_t& out_shape,
    ndarray ret) except *
