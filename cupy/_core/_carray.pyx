from libc.string cimport memcpy
from cpython.mem cimport PyMem_Malloc, PyMem_Free

from cupy.cuda cimport function
from cupy._core cimport internal


cdef fused index_t:
    int
    long long


cdef inline int populate_shape_strides(
        index_t val, size_t ndim, bint allow_unsafe,
        void* data, size_t& offset, int itemsize,
        const shape_t& shape, const strides_t& strides) except?-1:
    cdef int i

    for i in range(ndim):
        if not allow_unsafe and shape[i] == 0:
            raise RuntimeError(
                f"{i}-th dimension has size zero")
        (<index_t*>(<char*>(data) + offset))[0] = <index_t>shape[i]
        offset += sizeof(index_t)

    for i in range(ndim):
        if not allow_unsafe and strides[i] <= 0:
            raise RuntimeError(
                f"{i}-th dimension has non-positive stride")
        (<index_t*>(<char*>(data) + offset))[0] = (
            <index_t>(strides[i] // itemsize)
        )
        offset += sizeof(index_t)

    return 0


cdef class mdspan(function.CPointer):

    cdef int init(
            self, void* data_ptr, int itemsize,
            const shape_t& shape, const strides_t& strides,
            int index_itemsize, bint allow_unsafe) except?-1:
        cdef size_t ndim = shape.size()
        assert ndim == strides.size()
        assert ndim <= MAX_NDIM

        cdef size_t total_size = \
            sizeof(void*) + ndim * 2 * index_itemsize
        cdef void* data = PyMem_Malloc(total_size)
        if data == NULL:
            raise MemoryError
        self.ptr = <intptr_t>data

        cdef size_t offset = 0
        (<void**>(<char*>(data) + offset))[0] = data_ptr
        offset += sizeof(data_ptr)
        if ndim != 0:
            try:
                if index_itemsize == 4:
                    populate_shape_strides(
                        <int>(0),  # dummy
                        ndim, allow_unsafe, data, offset, itemsize,
                        shape, strides)
                elif index_itemsize == 8:
                    populate_shape_strides(
                        <long long>(0),  # dummy
                        ndim, allow_unsafe, data, offset, itemsize,
                        shape, strides)
                else:
                    raise ValueError(
                        f"Unsupported index_itemsize: {index_itemsize}"
                    )
            except Exception:
                PyMem_Free(data)
                self.ptr = 0
                raise

        assert offset == total_size

    def __cinit__(self):
        self.ptr = 0

    def __dealloc__(self):
        if self.ptr != 0:
            PyMem_Free(<void*>(self.ptr))
            self.ptr = 0


cdef class CArray(function.CPointer):

    cdef void init(
            self, void* data_ptr, Py_ssize_t data_size,
            const shape_t& shape, const strides_t& strides) except*:
        cdef size_t ndim = shape.size()
        assert ndim == strides.size()
        assert ndim <= MAX_NDIM

        cdef size_t total_size = \
            sizeof(_CArray) + ndim * 2 * sizeof(Py_ssize_t)
        cdef void* data = PyMem_Malloc(total_size)
        if data == NULL:
            raise MemoryError
        self.ptr = <intptr_t>data

        cdef size_t offset = 0
        (<void **>(<char*>(data) + offset))[0] = data_ptr
        offset += sizeof(data_ptr)
        (<Py_ssize_t*>(<char*>(data) + offset))[0] = data_size
        offset += sizeof(data_size)

        if ndim != 0:
            memcpy(<char*>(data) + offset,
                   shape.data(),
                   sizeof(Py_ssize_t) * ndim)
            offset += sizeof(Py_ssize_t) * ndim
            memcpy(<char*>(data) + offset,
                   strides.data(),
                   sizeof(Py_ssize_t) * ndim)
            offset += sizeof(Py_ssize_t) * ndim
        assert offset == total_size

    def __cinit__(self):
        self.ptr = 0

    def __dealloc__(self):
        if self.ptr != 0:
            PyMem_Free(<void*>(self.ptr))
            self.ptr = 0


cdef class CIndexer(function.CPointer):

    cdef void init(self, Py_ssize_t size, const shape_t &shape) except*:
        cdef size_t ndim = shape.size()
        assert ndim <= MAX_NDIM

        cdef size_t total_size = \
            sizeof(_CIndexer) + ndim * 2 * sizeof(Py_ssize_t)
        cdef void* data = PyMem_Malloc(total_size)
        if data == NULL:
            raise MemoryError
        self.ptr = <intptr_t>data

        cdef size_t offset = 0
        (<Py_ssize_t*>(<char*>(data) + offset))[0] = size
        offset += sizeof(size)
        if ndim != 0:
            memcpy(<char*>(data) + offset,
                   shape.data(),
                   sizeof(Py_ssize_t) * ndim)
            offset += sizeof(Py_ssize_t) * ndim
        assert offset + sizeof(Py_ssize_t) * ndim == total_size

    def __cinit__(self):
        self.ptr = 0

    def __dealloc__(self):
        if self.ptr != 0:
            PyMem_Free(<void*>(self.ptr))
            self.ptr = 0


cdef class Indexer:

    cdef void init(self, const shape_t& shape) noexcept:
        self.shape = shape
        self.size = internal.prod(shape)
        self._index_32_bits = self.size <= <Py_ssize_t>(1 << 31)

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
