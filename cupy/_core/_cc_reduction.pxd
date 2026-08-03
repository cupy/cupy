from cupy._core._carray cimport shape_t
from cupy._core.core cimport _ndarray_base


cdef bint _try_to_call_cc_reduction(
    self, list in_args, list out_args, const shape_t& a_shape,
    stream, tuple reduce_axis, tuple out_axis, const shape_t& out_shape,
    _ndarray_base ret) except *
