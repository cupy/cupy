from cupy._core.core cimport ndarray


# TODO(niboshi): Move {nan,}arg{min,max} to sorting


cdef ndarray _ndarray_max(ndarray self, axis, out, dtype, keepdims)
cdef ndarray _ndarray_min(ndarray self, axis, out, dtype, keepdims)
cdef ndarray _ndarray_ptp(ndarray self, axis, out, keepdims)
cdef ndarray _ndarray_argmax(ndarray self, axis, out, dtype, keepdims)
cdef ndarray _ndarray_argmin(ndarray self, axis, out, dtype, keepdims)
cdef ndarray _ndarray_mean(ndarray self, axis, dtype, out, keepdims)
cdef ndarray _ndarray_var(ndarray self, axis, dtype, out, ddof, keepdims)
cdef ndarray _ndarray_std(ndarray self, axis, dtype, out, ddof, keepdims)

cpdef ndarray _median(ndarray a, axis, out, overwrite_input, keepdims)

cpdef ndarray _nanmean(ndarray a, axis, dtype, out, keepdims)
cpdef ndarray _nanvar(ndarray a, axis, dtype, out, ddof, keepdims)
cpdef ndarray _nanstd(ndarray a, axis, dtype, out, ddof, keepdims)


cpdef ndarray _nanargmin(ndarray a, axis, out, dtype, keepdims)
cpdef ndarray _nanargmax(ndarray a, axis, out, dtype, keepdims)
