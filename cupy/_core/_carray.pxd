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
cdef class CArray(function.CPointer):

    cdef void init(
        self, void* data_ptr, Py_ssize_t data_size,
        const shape_t& shape, const strides_t& strides) except*


# This is now excluding shape/strides only for getting the compiler-time size
cdef struct _CIndexer:
    Py_ssize_t size


cdef class CIndexer(function.CPointer):

    cdef void init(self, Py_ssize_t size, const shape_t &shape) except*


cdef class Indexer:
    cdef:
        readonly Py_ssize_t size
        readonly shape_t shape
        readonly bint _index_32_bits

    cdef void init(self, const shape_t& shape)

    cdef function.CPointer get_pointer(self)


cdef Indexer _indexer_init(const shape_t& shape)
