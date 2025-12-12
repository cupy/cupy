from cupy._core.core cimport _ndarray_base
from cupy._core._carray cimport shape_t, strides_t

from libcpp.pair cimport pair


cdef get_range(
    Py_ssize_t itemsize, shape_t& shape, strides_t& strides,
    Py_ssize_t& out_left, Py_ssize_t &out_right)
cpdef pair[Py_ssize_t, Py_ssize_t] get_bound(_ndarray_base array)
cpdef bint may_share_bounds(_ndarray_base a, _ndarray_base b)
