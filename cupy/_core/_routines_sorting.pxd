from cupy._core.core cimport ndarray


cdef _ndarray_sort(ndarray self, int axis)
cdef ndarray _ndarray_argsort(ndarray self, axis)
cdef _ndarray_partition(ndarray self, kth, int axis)
cdef ndarray _ndarray_argpartition(self, kth, axis)
