cimport cython  # NOQA
from libc.stdint cimport intptr_t
from libcpp cimport vector

from cupy.cuda cimport function


ctypedef vector.vector[Py_ssize_t] shape_t
ctypedef vector.vector[Py_ssize_t] strides_t

# this matches NPY_MAXDIMS
# Note: we make it an enum to work around cython/cython#4369
cdef enum: MAX_NDIM = 64


# This is now excluding shape/strides only for getting the compiler-time size
cdef struct _CArray:
    void* data
    Py_ssize_t size


@cython.final
cdef class mdspan(function.CPointer):

    cdef int init(
        self, void* data_ptr, int itemsize,
        const shape_t& shape, const strides_t& strides,
        int index_itemsize, bint allow_unsafe) except?-1


@cython.final
cdef class CArray(function.CPointer):

    cdef int ndim

    cdef _CArray* _allocate(
        self, void* data_ptr, Py_ssize_t element_count, size_t ndim,
    ) except NULL

    @staticmethod
    cdef CArray create(
        void* data_ptr, Py_ssize_t element_count,
        const shape_t& shape, const strides_t& strides)

    @staticmethod
    cdef CArray from_packed_layout(
        void* data_ptr, Py_ssize_t element_count,
        const shape_t& shape, const strides_t& strides,
        const shape_t& loop_shape, const vector.vector[int]& kept_axes,
        int core_ndim,
    )


# This is now excluding shape/strides only for getting the compiler-time size
cdef struct _CIndexer:
    Py_ssize_t size


cdef class CIndexer(function.CPointer):

    cdef int ndim

    @staticmethod
    cdef CIndexer create(const shape_t& shape)

    cdef Py_ssize_t size(self) noexcept
    cdef bint _index_32_bits(self) noexcept
