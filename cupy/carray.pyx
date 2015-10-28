import os

from cupy import cuda
cimport cupy.cuda.module

DEF MAX_NDIM = 25

cdef:
    struct _CArray:
        void* data
        int size
        int shape_and_strides[MAX_NDIM * 2]

    class CArray(cupy.cuda.module.CPointer):
        cdef _CArray val

        def __init__(self, size_t data, Py_ssize_t size, tuple shape,
                     tuple strides):
            self.val.data = <void*>data
            self.val.size = size
            cdef int i, ndim = len(shape)
            for i in range(ndim):
                self.val.shape_and_strides[i] = shape[i]
                self.val.shape_and_strides[i + ndim] = strides[i]
            self.ptr = <void*>&self.val

    struct _CIndexer:
        int size
        int shape_and_index[MAX_NDIM * 2]

    class CIndexer(cupy.cuda.module.CPointer):
        cdef _CIndexer val

        def __init__(self, Py_ssize_t size, tuple shape):
            self.val.size = size
            cdef int i, ndim = len(shape)
            for i in range(ndim):
                self.val.shape_and_index[i] = shape[i]
            self.ptr = <void*>&self.val


cpdef CArray to_carray(data, Py_ssize_t size, tuple shape, tuple strides):
    return CArray(data, size, shape, strides)


cpdef CIndexer to_cindexer(Py_ssize_t size, tuple shape):
    return CIndexer(size, shape)


cdef class Indexer:
    cdef:
        public Py_ssize_t size
        public tuple shape

    def __init__(self, shape):
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

_header_source = None


def _get_header_source():
    global _header_source
    if _header_source is None:
        header_path = os.path.join(os.path.dirname(__file__), 'carray.cuh')
        with open(header_path) as header_file:
            _header_source = header_file.read()
    return _header_source


def compile_with_cache(source, options=(), arch=None, cachd_dir=None):
    source = _get_header_source() + source
    return cuda.compile_with_cache(source, options, arch, cachd_dir)
