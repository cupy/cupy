from cupy._core.core cimport _ndarray_base


cdef _ndarray_sort(_ndarray_base self, int axis)
cdef _ndarray_base _ndarray_argsort(_ndarray_base self, axis)
cdef _ndarray_partition(_ndarray_base self, kth, int axis)
cdef _ndarray_base _ndarray_argpartition(self, kth, axis)
