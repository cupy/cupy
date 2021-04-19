from libcpp cimport bool as cpp_bool
from libcpp cimport vector
from libc.stdint cimport uint16_t

from cupy._core._carray cimport shape_t
from cupy._core._carray cimport strides_t


cpdef Py_ssize_t prod(const vector.vector[Py_ssize_t]& args)

cpdef Py_ssize_t prod_sequence(object args)

cpdef bint is_in(const vector.vector[Py_ssize_t]& args, Py_ssize_t x)

cpdef tuple get_size(object size)

cpdef bint vector_equal(
    const vector.vector[Py_ssize_t]& x, const vector.vector[Py_ssize_t]& y)

cdef void get_reduced_dims(
    shape_t& shape, strides_t& strides,
    Py_ssize_t itemsize, shape_t& reduced_shape,
    strides_t& reduced_strides)

# Computes the contiguous strides given a shape and itemsize.
# Returns the size (total number of elements).
cdef Py_ssize_t get_contiguous_strides_inplace(
    const shape_t& shape, strides_t& strides,
    Py_ssize_t itemsize, bint is_c_contiguous)

cpdef bint get_c_contiguity(
    shape_t& shape, strides_t& strides, Py_ssize_t itemsize)

cpdef shape_t infer_unknown_dimension(
    const shape_t& shape, Py_ssize_t size) except *

cpdef slice complete_slice(slice slc, Py_ssize_t dim)

cpdef tuple complete_slice_list(list slice_list, Py_ssize_t ndim)

cpdef size_t clp2(size_t x)

ctypedef unsigned short _float16

cpdef uint16_t to_float16(float f)

cpdef float from_float16(uint16_t v)

cdef int _normalize_order(order, cpp_bool allow_k=*) except? 0

cdef _broadcast_core(list arrays, shape_t& shape)

cpdef bint _contig_axes(tuple axes)

cpdef Py_ssize_t _normalize_axis_index(
    Py_ssize_t axis, Py_ssize_t ndim) except -1

cpdef tuple _normalize_axis_indices(
    axes, Py_ssize_t ndim, cpp_bool sort_axes=*)

cpdef strides_t _get_strides_for_order_K(x, dtype, shape=*)

cpdef int _update_order_char(
    bint is_c_contiguous, bint is_f_contiguous, int order_char)

cpdef tuple _broadcast_shapes(shapes)
