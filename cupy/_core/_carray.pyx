from libc.string cimport memcpy
from cpython.mem cimport PyMem_Malloc, PyMem_Free

from cupy.cuda cimport function
from cupy._core cimport internal


cdef class mdspan(function.CPointer):

    # TODO: add a casting arg to choose the indexint type
    cdef int init(
            self, void* data_ptr, int itemsize,
            const shape_t& shape, const strides_t& strides) except?-1:
        cdef size_t ndim = shape.size()
        assert ndim == strides.size()
        assert ndim <= MAX_NDIM

        cdef size_t total_size = \
            sizeof(void*) + ndim * 2 * sizeof(size_t)
        cdef void* data = PyMem_Malloc(total_size)
        if data == NULL:
            raise MemoryError
        self.ptr = <intptr_t>data

        cdef size_t offset = 0
        cdef int i
        memcpy(<char*>(data) + offset, &data_ptr, sizeof(data_ptr))
        offset += sizeof(data_ptr)
        if ndim != 0:
            # FIXME: we used size_t for experiment only
            for i in range(ndim):
                (<size_t*>(<char*>(data) + offset))[0] = shape[i]
                offset += sizeof(size_t)

            for i in range(ndim):
                print("stride", i, "=", strides[i] // itemsize)
                (<size_t*>(<char*>(data) + offset))[0] = strides[i] // itemsize
                offset += sizeof(size_t)
        assert offset == total_size
        print("mdspan self.ptr =", self.ptr, hex(self.ptr), total_size)

        print(<intptr_t>((<void**><char*>(self.ptr))[0]), <intptr_t>data_ptr)
        print((<size_t*><char*>(self.ptr + 8))[0])
        print((<size_t*><char*>(self.ptr + 16))[0])

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
        memcpy(<char*>(data) + offset, &data_ptr, sizeof(data_ptr))
        offset += sizeof(data_ptr)
        memcpy(<char*>(data) + offset, &data_size, sizeof(data_size))
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
        memcpy(<char*>(data) + offset, &size, sizeof(size))
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
