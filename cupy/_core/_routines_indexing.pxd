from cupy._core.core cimport ndarray


cpdef ndarray _ndarray_argwhere(ndarray self)
cdef ndarray _ndarray_getitem(ndarray self, slices)
cdef _ndarray_setitem(ndarray self, slices, value)
cdef tuple _ndarray_nonzero(ndarray self)
cdef _ndarray_scatter_add(ndarray self, slices, value)
cdef _ndarray_scatter_max(ndarray self, slices, value)
cdef _ndarray_scatter_min(ndarray self, slices, value)
cdef ndarray _ndarray_take(ndarray self, indices, axis, out)
cdef ndarray _ndarray_put(ndarray self, indices, values, mode)
cdef ndarray _ndarray_choose(ndarray self, choices, out, mode)
cdef ndarray _ndarray_compress(ndarray self, condition, axis, out)
cdef ndarray _ndarray_diagonal(ndarray self, offset, axis1, axis2)

cdef ndarray _simple_getitem(ndarray a, list slice_list)
