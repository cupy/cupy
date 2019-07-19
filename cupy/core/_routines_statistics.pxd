from cupy.core.core cimport ndarray


cdef ndarray _ndarray_max(ndarray self, axis, out, dtype, keepdims)
cdef ndarray _ndarray_min(ndarray self, axis, out, dtype, keepdims)
cdef ndarray _ndarray_argmax(ndarray self, axis, out, dtype, keepdims)
cdef ndarray _ndarray_nanargmax(ndarray self, axis, out, dtype, keepdims)
cdef ndarray _ndarray_argmin(ndarray self, axis, out, dtype, keepdims)
cdef ndarray _ndarray_nanargmin(ndarray self, axis, out, dtype, keepdims)
cdef ndarray _ndarray_mean(ndarray self, axis, dtype, out, keepdims)
cdef ndarray _ndarray_var(ndarray self, axis, dtype, out, ddof, keepdims)
cdef ndarray _ndarray_std(ndarray self, axis, dtype, out, ddof, keepdims)
