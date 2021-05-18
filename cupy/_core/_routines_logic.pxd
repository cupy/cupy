from cupy._core.core cimport ndarray


cdef ndarray _ndarray_all(ndarray self, axis, out, keepdims)
cdef ndarray _ndarray_any(ndarray self, axis, out, keepdims)
cdef ndarray _ndarray_greater(ndarray self, other)
cdef ndarray _ndarray_greater_equal(ndarray self, other)
cdef ndarray _ndarray_less(ndarray self, other)
cdef ndarray _ndarray_less_equal(ndarray self, other)
cdef ndarray _ndarray_equal(ndarray self, other)
cdef ndarray _ndarray_not_equal(ndarray self, other)
