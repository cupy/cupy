import os

from cupy import cuda

from cupy.cuda cimport function


cdef struct _CArray:
    void* data
    int size
    int shape_and_strides[MAX_NDIM * 2]


cdef class CArray(cupy.cuda.function.CPointer):

    cdef:
        _CArray val

    def __init__(self, ndarray arr):
        cdef int i, ndim = arr.ndim
        self.val.data = <void*>arr.data.ptr
        self.val.size = arr.size
        for i in range(ndim):
            self.val.shape_and_strides[i] = arr._shape[i]
            self.val.shape_and_strides[i + ndim] = arr._strides[i]
        self.ptr = <void*>&self.val


cdef struct _CIndexer:
    int size
    int shape_and_index[MAX_NDIM * 2]


cdef class CIndexer(cupy.cuda.function.CPointer):
    cdef:
        _CIndexer val

    def __init__(self, Py_ssize_t size, tuple shape):
        self.val.size = size
        cdef int i
        for i in range(len(shape)):
            self.val.shape_and_index[i] = shape[i]
        self.ptr = <void*>&self.val


@cython.profile(False)
cpdef inline CIndexer to_cindexer(Py_ssize_t size, tuple shape):
    return CIndexer(size, shape)


cdef class Indexer:
    cdef:
        public Py_ssize_t size
        public tuple shape

    def __init__(self, tuple shape):
        cdef Py_ssize_t size = 1
        for s in shape:
            size *= s
        self.shape = shape
        self.size = size

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def cstruct(self):
        return to_cindexer(self.size, self.shape)


cdef str _header_source = None


cpdef str _get_header_source():
    global _header_source
    if _header_source is None:
        header_path = os.path.join(os.path.dirname(__file__), 'carray.cuh')
        with open(header_path) as header_file:
            _header_source = header_file.read()
    return _header_source


cpdef function.Module compile_with_cache(
        str source, tuple options=(), arch=None, cachd_dir=None):
    source = _get_header_source() + source
    return cuda.compile_with_cache(source, options, arch, cachd_dir)
