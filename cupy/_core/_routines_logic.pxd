from cupy._core.core cimport _ndarray_base


cdef _ndarray_base _ndarray_all(_ndarray_base self, axis, out, keepdims)
cdef _ndarray_base _ndarray_any(_ndarray_base self, axis, out, keepdims)
cdef _ndarray_base _ndarray_greater(_ndarray_base self, other)
cdef _ndarray_base _ndarray_greater_equal(_ndarray_base self, other)
cdef _ndarray_base _ndarray_less(_ndarray_base self, other)
cdef _ndarray_base _ndarray_less_equal(_ndarray_base self, other)
cdef _ndarray_base _ndarray_equal(_ndarray_base self, other)
cdef _ndarray_base _ndarray_not_equal(_ndarray_base self, other)
