from libcpp cimport vector

from cupy.core.core cimport ndarray


cdef class broadcast:
    cdef:
        readonly tuple values
        readonly tuple shape
        readonly Py_ssize_t size
        readonly Py_ssize_t nd

cdef tuple _broadcast_core(arrays)
cdef _ndarray_shape_setter(ndarray self, newshape)
cdef ndarray _ndarray_reshape(ndarray self, tuple shape, order)
cdef ndarray _ndarray_transpose(ndarray self, tuple axes)
cdef ndarray _ndarray_swapaxes(
    ndarray self, Py_ssize_t axis1, Py_ssize_t axis2)
cdef ndarray _ndarray_flatten(ndarray self)
cdef ndarray _ndarray_ravel(ndarray self, order)
cdef ndarray _ndarray_squeeze(ndarray self, axis)
cdef ndarray _ndarray_repeat(ndarray self, repeats, axis)

cpdef ndarray moveaxis(ndarray a, source, destination)
cpdef ndarray rollaxis(ndarray a, Py_ssize_t axis, Py_ssize_t start=*)
cpdef ndarray broadcast_to(ndarray array, shape)
cpdef ndarray _reshape(ndarray self,
                       const vector.vector[Py_ssize_t] &shape_spec)
cpdef ndarray _T(ndarray self)
cpdef ndarray _transpose(ndarray self, const vector.vector[Py_ssize_t] &axes)
cpdef ndarray _concatenate(list arrays, Py_ssize_t axis, tuple shape, dtype)
cpdef ndarray concatenate_method(tup, int axis)
