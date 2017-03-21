from libcpp cimport vector
from cupy.cuda cimport memory

from cupy.cuda.function cimport CPointer


cdef class ndarray:
    cdef:
        readonly Py_ssize_t size
        public vector.vector[Py_ssize_t] _shape
        public vector.vector[Py_ssize_t] _strides
        readonly bint _c_contiguous
        readonly bint _f_contiguous
        readonly object dtype
        readonly memory.MemoryPointer data
        readonly ndarray base

    cpdef tolist(self)
    cpdef tofile(self, fid, sep=*, format=*)
    cpdef dump(self, file)
    cpdef dumps(self)
    cpdef ndarray astype(self, dtype, copy=*)
    cpdef ndarray copy(self, order=*)
    cpdef ndarray view(self, dtype=*)
    cpdef fill(self, value)
    cpdef ndarray _reshape(self, vector.vector[Py_ssize_t] shape)
    cpdef ndarray _transpose(self, vector.vector[Py_ssize_t] axes)
    cpdef ndarray swapaxes(self, Py_ssize_t axis1, Py_ssize_t axis2)
    cpdef ndarray flatten(self)
    cpdef ndarray ravel(self)
    cpdef ndarray squeeze(self, axis=*)
    cpdef ndarray take(self, indices, axis=*, out=*)
    cpdef repeat(self, repeats, axis=*)
    cpdef choose(self, choices, out=*, mode=*)
    cpdef ndarray diagonal(self, offset=*, axis1=*, axis2=*)
    cpdef ndarray max(self, axis=*, out=*, dtype=*, keepdims=*)
    cpdef ndarray argmax(self, axis=*, out=*, dtype=*,
                         keepdims=*)
    cpdef ndarray min(self, axis=*, out=*, dtype=*, keepdims=*)
    cpdef ndarray argmin(self, axis=*, out=*, dtype=*,
                         keepdims=*)
    cpdef ndarray clip(self, a_min, a_max, out=*)

    cpdef ndarray trace(self, offset=*, axis1=*, axis2=*, dtype=*,
                        out=*)
    cpdef ndarray sum(self, axis=*, dtype=*, out=*, keepdims=*)

    cpdef ndarray mean(self, axis=*, dtype=*, out=*, keepdims=*)
    cpdef ndarray var(self, axis=*, dtype=*, out=*, ddof=*,
                      keepdims=*)
    cpdef ndarray std(self, axis=*, dtype=*, out=*, ddof=*,
                      keepdims=*)
    cpdef ndarray prod(self, axis=*, dtype=*, out=*, keepdims=*)
    cpdef ndarray all(self, axis=*, out=*, keepdims=*)
    cpdef ndarray any(self, axis=*, out=*, keepdims=*)
    cpdef get(self, stream=*)
    cpdef set(self, arr, stream=*)
    cpdef ndarray reduced_view(self, dtype=*)
    cpdef _update_c_contiguity(self)
    cpdef _update_f_contiguity(self)
    cpdef _update_contiguity(self)
    cpdef _set_shape_and_strides(self, vector.vector[Py_ssize_t]& shape,
                                 vector.vector[Py_ssize_t]& strides,
                                 bint update_c_contiguity=*)
    cdef CPointer get_pointer(self)


cdef class Indexer:
    cdef:
        public Py_ssize_t size
        public tuple shape
    cdef CPointer get_pointer(self)
