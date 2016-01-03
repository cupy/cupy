from libcpp cimport vector

cpdef Py_ssize_t prod(args, Py_ssize_t init=*) except *

cpdef Py_ssize_t prod_ssize_t(
        vector.vector[Py_ssize_t]& arr, Py_ssize_t init=*)

cpdef tuple get_size(object size)

cpdef bint vector_equal(
        vector.vector[Py_ssize_t]& x, vector.vector[Py_ssize_t]& y)

cdef void get_reduced_dims(
        vector.vector[Py_ssize_t]& shape, vector.vector[Py_ssize_t]& strides,
        Py_ssize_t itemsize, vector.vector[Py_ssize_t]& reduced_shape,
        vector.vector[Py_ssize_t]& reduced_strides)

cpdef vector.vector[Py_ssize_t] get_contiguous_strides(
        vector.vector[Py_ssize_t]& shape, Py_ssize_t itemsize) except *

cpdef bint get_c_contiguity(
        vector.vector[Py_ssize_t]& shape, vector.vector[Py_ssize_t]& strides,
        Py_ssize_t itemsize) except *

cpdef vector.vector[Py_ssize_t] infer_unknown_dimension(
        vector.vector[Py_ssize_t]& shape, Py_ssize_t size) except *

cpdef slice complete_slice(slice slc, Py_ssize_t dim)
