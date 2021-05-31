from cupy._core.core cimport ndarray

cpdef object toDlpack(array) except +
cpdef ndarray fromDlpack(object dltensor) except +
cdef object _cupy_ndarray_to_dlpack(ndarray array) except+
cpdef from_dlpack(array)
