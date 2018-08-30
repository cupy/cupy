from libcpp cimport vector
from libc.stdint cimport uint16_t


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
    vector.vector[Py_ssize_t]& shape, Py_ssize_t itemsize,
    bint is_c_contiguous) except *

cdef set_contiguous_strides(
    vector.vector[Py_ssize_t]& shape, vector.vector[Py_ssize_t]& strides,
    Py_ssize_t itemsize, bint is_c_contiguous)

cpdef bint get_c_contiguity(
    vector.vector[Py_ssize_t]& shape, vector.vector[Py_ssize_t]& strides,
    Py_ssize_t itemsize) except *

cpdef vector.vector[Py_ssize_t] infer_unknown_dimension(
    vector.vector[Py_ssize_t]& shape, Py_ssize_t size) except *

cpdef slice complete_slice(slice slc, Py_ssize_t dim)

cpdef tuple complete_slice_list(list slice_list, Py_ssize_t ndim)

cpdef size_t clp2(size_t x)

ctypedef unsigned short _float16

cpdef uint16_t to_float16(float f)

cpdef float from_float16(uint16_t v)
