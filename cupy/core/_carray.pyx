import os

from libcpp cimport vector

from cupy.cuda cimport function


cdef class CArray(function.CPointer):

    cdef void init(
            self, void* data_ptr, Py_ssize_t data_size,
            const vector.vector[Py_ssize_t]& shape,
            const vector.vector[Py_ssize_t]& strides):
        cdef size_t ndim = shape.size()
        cdef Py_ssize_t* shape_and_strides = (
            self.val.shape_and_strides)
        cdef size_t i

        self.val.data = data_ptr
        self.val.size = data_size
        for i in range(ndim):
            shape_and_strides[i] = shape[i]
            shape_and_strides[i + ndim] = strides[i]
        self.ptr = <void*>&self.val


cdef class CIndexer(function.CPointer):

    def __init__(self, Py_ssize_t size, tuple shape):
        self.val.size = size
        cdef Py_ssize_t i
        for i in range(len(shape)):
            self.val.shape_and_index[i] = shape[i]
        self.ptr = <void*>&self.val


cdef class Indexer:

    def __init__(self, tuple shape):
        cdef Py_ssize_t size = 1
        for s in shape:
            size *= s
        self.shape = shape
        self.size = size

    @property
    def ndim(self):
        return len(self.shape)

    cdef function.CPointer get_pointer(self):
        return CIndexer(self.size, self.shape)
