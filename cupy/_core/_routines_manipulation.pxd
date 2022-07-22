from libcpp cimport vector

from cupy._core._carray cimport shape_t
from cupy._core._carray cimport strides_t
from cupy._core.core cimport _ndarray_base


cdef class broadcast:
    cdef:
        readonly tuple values
        readonly tuple shape
        readonly Py_ssize_t size
        readonly Py_ssize_t nd


cdef _ndarray_shape_setter(_ndarray_base self, newshape)
cdef _ndarray_base _ndarray_reshape(_ndarray_base self, tuple shape, order)
cdef _ndarray_base _ndarray_transpose(_ndarray_base self, tuple axes)
cdef _ndarray_base _ndarray_swapaxes(
    _ndarray_base self, Py_ssize_t axis1, Py_ssize_t axis2)
cdef _ndarray_base _ndarray_flatten(_ndarray_base self, order)
cdef _ndarray_base _ndarray_ravel(_ndarray_base self, order)
cdef _ndarray_base _ndarray_squeeze(_ndarray_base self, axis)
cdef _ndarray_base _ndarray_repeat(_ndarray_base self, repeats, axis)

cpdef _ndarray_base _expand_dims(_ndarray_base a, tuple axis)
cpdef _ndarray_base moveaxis(_ndarray_base a, source, destination)
cpdef _ndarray_base _move_single_axis(
    _ndarray_base a, Py_ssize_t source, Py_ssize_t destination)
cpdef _ndarray_base rollaxis(
    _ndarray_base a, Py_ssize_t axis, Py_ssize_t start=*)
cpdef _ndarray_base broadcast_to(_ndarray_base array, shape)
cpdef _ndarray_base _reshape(_ndarray_base self, const shape_t &shape_spec)
cpdef _ndarray_base _T(_ndarray_base self)
cpdef _ndarray_base _transpose(
    _ndarray_base self, const vector.vector[Py_ssize_t] &axes)
cpdef _ndarray_base _concatenate(
    list arrays, Py_ssize_t axis, tuple shape, _ndarray_base out, str casting)
cpdef _ndarray_base concatenate_method(
    tup, int axis, _ndarray_base out=*, dtype=*, casting=*)
