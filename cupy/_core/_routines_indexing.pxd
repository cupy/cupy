from cupy._core.core cimport _ndarray_base


cpdef _ndarray_base _ndarray_argwhere(_ndarray_base self)
cdef _ndarray_base _ndarray_getitem(_ndarray_base self, slices)
cdef _ndarray_setitem(_ndarray_base self, slices, value)
cdef tuple _ndarray_nonzero(_ndarray_base self)
cdef _ndarray_scatter_add(_ndarray_base self, slices, value)
cdef _ndarray_scatter_max(_ndarray_base self, slices, value)
cdef _ndarray_scatter_min(_ndarray_base self, slices, value)
cdef _ndarray_base _ndarray_take(_ndarray_base self, indices, axis, out)
cdef _ndarray_base _ndarray_put(_ndarray_base self, indices, values, mode)
cdef _ndarray_base _ndarray_choose(_ndarray_base self, choices, out, mode)
cdef _ndarray_base _ndarray_compress(_ndarray_base self, condition, axis, out)
cdef _ndarray_base _ndarray_diagonal(_ndarray_base self, offset, axis1, axis2)
