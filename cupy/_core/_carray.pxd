cimport cython  # NOQA
from libcpp cimport vector

from cupy.cuda cimport function


ctypedef vector.vector[Py_ssize_t] shape_t
ctypedef vector.vector[Py_ssize_t] strides_t

# this matches NPY_MAXDIMS
# Note: we make it an enum to work around cython/cython#4369
cdef enum: MAX_NDIM = 32


cdef struct _CArray:
    void* data
    Py_ssize_t size
    Py_ssize_t shape_and_strides[MAX_NDIM * 2]


@cython.final
cdef class CArray(function.CPointer):

    cdef:
        _CArray val

    cdef void init(
        self, void* data_ptr, Py_ssize_t data_size,
        const shape_t& shape, const strides_t& strides) except*


cdef struct _CIndexer:
    Py_ssize_t size
    Py_ssize_t shape_and_index[MAX_NDIM * 2]


cdef class CIndexer(function.CPointer):
    cdef:
        _CIndexer val

    cdef void init(self, Py_ssize_t size, const shape_t &shape) except*


cdef class Indexer:
    cdef:
        readonly Py_ssize_t size
        readonly shape_t shape
        readonly bint _index_32_bits

    cdef void init(self, const shape_t& shape)

    cdef function.CPointer get_pointer(self)


cdef Indexer _indexer_init(const shape_t& shape)
