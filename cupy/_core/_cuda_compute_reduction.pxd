from cupy._core.core cimport _ndarray_base


cdef bint _try_to_call_cuda_compute_reduction(
    self, list in_args, list out_args, stream, str map_expr,
    str reduce_expr, str post_map_expr, str reduce_type, type_map,
    tuple reduce_axis, tuple out_axis, _ndarray_base ret) except *
