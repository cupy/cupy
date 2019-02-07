from cupy.core.core cimport ndarray


cdef ndarray _ndarray_all(ndarray self, axis, out, keepdims)
cdef ndarray _ndarray_any(ndarray self, axis, out, keepdims)
