from cupy.cuda cimport function
from cupy._core cimport internal


cdef class CArray(function.CPointer):

    cdef void init(
            self, void* data_ptr, Py_ssize_t data_size,
            const shape_t& shape, const strides_t& strides) except*:
        cdef size_t ndim = shape.size()
        assert ndim == strides.size()
        assert ndim <= MAX_NDIM
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

    cdef void init(self, Py_ssize_t size, const shape_t &shape) except*:
        cdef size_t ndim = shape.size()
        assert ndim <= MAX_NDIM
        self.val.size = size
        cdef Py_ssize_t i
        for i in range(<Py_ssize_t>shape.size()):
            self.val.shape_and_index[i] = shape[i]
        self.ptr = <void*>&self.val


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
