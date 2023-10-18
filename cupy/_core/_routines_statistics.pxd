from cupy._core.core cimport _ndarray_base


# TODO(niboshi): Move {nan,}arg{min,max} to sorting


cdef _ndarray_base _ndarray_max(_ndarray_base self, axis, out, dtype, keepdims)
cdef _ndarray_base _ndarray_min(_ndarray_base self, axis, out, dtype, keepdims)
cdef _ndarray_base _ndarray_ptp(_ndarray_base self, axis, out, keepdims)
cdef _ndarray_base _ndarray_argmax(
    _ndarray_base self, axis, out, dtype, keepdims)
cdef _ndarray_base _ndarray_argmin(
    _ndarray_base self, axis, out, dtype, keepdims)
cdef _ndarray_base _ndarray_mean(
    _ndarray_base self, axis, dtype, out, keepdims)
cdef _ndarray_base _ndarray_var(
    _ndarray_base self, axis, dtype, out, ddof, keepdims)
cdef _ndarray_base _ndarray_std(
    _ndarray_base self, axis, dtype, out, ddof, keepdims)

cpdef _ndarray_base _median(
    _ndarray_base a, axis, out, overwrite_input, keepdims)

cpdef _ndarray_base _nanmean(_ndarray_base a, axis, dtype, out, keepdims)
cpdef _ndarray_base _nanvar(_ndarray_base a, axis, dtype, out, ddof, keepdims)
cpdef _ndarray_base _nanstd(_ndarray_base a, axis, dtype, out, ddof, keepdims)


cpdef _ndarray_base _nanargmin(_ndarray_base a, axis, out, dtype, keepdims)
cpdef _ndarray_base _nanargmax(_ndarray_base a, axis, out, dtype, keepdims)
