from libc.stdint cimport intptr_t

from cupy.cuda cimport function
from cupy._core cimport internal

import numpy as _numpy


cdef dict _carray_dtype_cache = {}
cdef dict _cindexer_dtype_cache = {}


# make it cpdef for easier testing
cpdef carray_cindexer_dtype_factory(size_t ndim, str kind):
    cache = _carray_dtype_cache if kind == "CArray" else _cindexer_dtype_cache
    if ndim in cache:
        return cache[ndim]

    if kind == "CArray":
        dtype = _numpy.dtype([
            ('data', _numpy.intp,),
            ('size', _numpy.uintp,),
            ('shape', _numpy.uintp, (ndim,)),
            ('strides', _numpy.uintp, (ndim,)),
            ], align=True
        )
        assert dtype.itemsize == \
            (sizeof(_CArray) + ndim * 2 * sizeof(Py_ssize_t))
    else:
        dtype = _numpy.dtype([
            ('size', _numpy.uintp,),
            ('shape', _numpy.uintp, (ndim,)),
            ('index', _numpy.uintp, (ndim,)),
            ], align=True
        )
        assert dtype.itemsize == \
            (sizeof(_CIndexer) + ndim * 2 * sizeof(Py_ssize_t))
    cache[ndim] = dtype
    return dtype


cdef class CArray(function.CPointer):

    cdef void init(
            self, void* data_ptr, Py_ssize_t data_size,
            const shape_t& shape, const strides_t& strides) except*:
        cdef size_t ndim = shape.size()
        assert ndim == strides.size()
        assert ndim <= MAX_NDIM
        cdef size_t i

        dtype = carray_cindexer_dtype_factory(ndim, "CArray")
        val = _numpy.empty(1, dtype=dtype)
        val["data"] = <intptr_t>(data_ptr)
        val["size"] = data_size
        shape_obj = val["shape"][0]
        strides_obj = val["strides"][0]
        for i in range(ndim):
            shape_obj[i] = shape[i]
            strides_obj[i] = strides[i]
        self.val = val
        self.ptr = <void*><intptr_t>(val.ctypes.data)


cdef class CIndexer(function.CPointer):

    cdef void init(self, Py_ssize_t size, const shape_t &shape) except*:
        cdef size_t ndim = shape.size()
        assert ndim <= MAX_NDIM
        cdef size_t i

        dtype = carray_cindexer_dtype_factory(ndim, "CIndexer")
        val = _numpy.empty(1, dtype=dtype)
        val["size"] = size
        shape_obj = val["shape"][0]
        for i in range(ndim):
            shape_obj[i] = shape[i]
        self.val = val
        self.ptr = <void*><intptr_t>(val.ctypes.data)


cdef class Indexer:

    cdef void init(self, const shape_t& shape):
        self.shape = shape
        self.size = internal.prod(shape)
        self._index_32_bits = self.size <= (1 << 31)

    @property
    def ndim(self):
        return self.shape.size()

    cdef function.CPointer get_pointer(self):
        cdef CIndexer indexer = CIndexer.__new__(CIndexer)
        indexer.init(self.size, self.shape)
        return indexer


cdef inline Indexer _indexer_init(const shape_t& shape):
    cdef Indexer indexer = Indexer.__new__(Indexer)
    indexer.init(shape)
    return indexer
