cimport cython  # NOQA
from libcpp cimport vector

from cupy.cuda cimport function


DEF MAX_NDIM = 25


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
        const vector.vector[Py_ssize_t]& shape,
        const vector.vector[Py_ssize_t]& strides)


cdef struct _CIndexer:
    Py_ssize_t size
    Py_ssize_t shape_and_index[MAX_NDIM * 2]


cdef class CIndexer(function.CPointer):
    cdef:
        _CIndexer val


cdef class Indexer:
    cdef:
        readonly Py_ssize_t size
        readonly tuple shape

    cdef function.CPointer get_pointer(self)
