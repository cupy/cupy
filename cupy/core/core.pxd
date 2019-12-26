from libcpp cimport vector
from cupy.cuda cimport memory

from cupy.cuda.function cimport CPointer
from cupy.cuda.function cimport Module

cdef class ndarray:
    cdef:
        object __weakref__
        readonly Py_ssize_t size
        public vector.vector[Py_ssize_t] _shape
        public vector.vector[Py_ssize_t] _strides
        readonly bint _c_contiguous
        readonly bint _f_contiguous
        readonly object dtype
        readonly memory.MemoryPointer data
        # TODO(niboshi): Return arbitrary owner object as `base` if the
        # underlying memory is UnownedMemory.
        readonly ndarray base

    cdef _init_fast(self, const vector.vector[Py_ssize_t]& shape, dtype,
                    bint c_order)
    cpdef item(self)
    cpdef tolist(self)
    cpdef bytes tobytes(self, order=*)
    cpdef tofile(self, fid, sep=*, format=*)
    cpdef dump(self, file)
    cpdef bytes dumps(self)
    cpdef ndarray astype(self, dtype, order=*, casting=*, subok=*, copy=*)
    cpdef ndarray copy(self, order=*)
    cpdef ndarray view(self, dtype=*)
    cpdef fill(self, value)
    cpdef ndarray swapaxes(self, Py_ssize_t axis1, Py_ssize_t axis2)
    cpdef ndarray flatten(self)
    cpdef ndarray ravel(self, order=*)
    cpdef ndarray squeeze(self, axis=*)
    cpdef ndarray take(self, indices, axis=*, out=*)
    cpdef put(self, indices, values, mode=*)
    cpdef repeat(self, repeats, axis=*)
    cpdef choose(self, choices, out=*, mode=*)
    cpdef sort(self, int axis=*)
    cpdef ndarray argsort(self, axis=*)
    cpdef partition(self, kth, int axis=*)
    cpdef ndarray argpartition(self, kth, axis=*)
    cpdef tuple nonzero(self)
    cpdef ndarray diagonal(self, offset=*, axis1=*, axis2=*)
    cpdef ndarray max(self, axis=*, out=*, keepdims=*)
    cpdef ndarray argmax(self, axis=*, out=*, dtype=*,
                         keepdims=*)
    cpdef ndarray min(self, axis=*, out=*, keepdims=*)
    cpdef ndarray argmin(self, axis=*, out=*, dtype=*,
                         keepdims=*)
    cpdef ndarray clip(self, a_min=*, a_max=*, out=*)
    cpdef ndarray round(self, decimals=*, out=*)

    cpdef ndarray trace(self, offset=*, axis1=*, axis2=*, dtype=*,
                        out=*)
    cpdef ndarray sum(self, axis=*, dtype=*, out=*, keepdims=*)
    cpdef ndarray cumsum(self, axis=*, dtype=*, out=*)
    cpdef ndarray mean(self, axis=*, dtype=*, out=*, keepdims=*)
    cpdef ndarray var(self, axis=*, dtype=*, out=*, ddof=*,
                      keepdims=*)
    cpdef ndarray std(self, axis=*, dtype=*, out=*, ddof=*,
                      keepdims=*)
    cpdef ndarray prod(self, axis=*, dtype=*, out=*, keepdims=*)
    cpdef ndarray cumprod(self, axis=*, dtype=*, out=*)
    cpdef ndarray all(self, axis=*, out=*, keepdims=*)
    cpdef ndarray any(self, axis=*, out=*, keepdims=*)
    cpdef ndarray conj(self)
    cpdef get(self, stream=*, order=*, out=*)
    cpdef set(self, arr, stream=*)
    cpdef ndarray reduced_view(self, dtype=*)
    cpdef _update_c_contiguity(self)
    cpdef _update_f_contiguity(self)
    cpdef _update_contiguity(self)
    cpdef _set_shape_and_strides(self, const vector.vector[Py_ssize_t]& shape,
                                 const vector.vector[Py_ssize_t]& strides,
                                 bint update_c_contiguity,
                                 bint update_f_contiguity)
    cdef ndarray _view(self, const vector.vector[Py_ssize_t]& shape,
                       const vector.vector[Py_ssize_t]& strides,
                       bint update_c_contiguity,
                       bint update_f_contiguity)
    cpdef _set_contiguous_strides(
        self, Py_ssize_t itemsize, bint is_c_contiguous)
    cdef CPointer get_pointer(self)
    cpdef object toDlpack(self)


cpdef ndarray _internal_ascontiguousarray(ndarray a)
cpdef ndarray _internal_asfortranarray(ndarray a)
cpdef ndarray ascontiguousarray(ndarray a, dtype=*)
cpdef ndarray asfortranarray(ndarray a, dtype=*)

cpdef Module compile_with_cache(str source, tuple options=*, arch=*,
                                cachd_dir=*, prepend_cupy_headers=*,
                                backend=*, translate_cucomplex=*)


# TODO(niboshi): Move to _routines_creation.pyx
cpdef ndarray array(obj, dtype=*, bint copy=*, order=*, bint subok=*,
                    Py_ssize_t ndmin=*)
cpdef ndarray _convert_object_with_cuda_array_interface(a)

cdef ndarray _ndarray_init(const vector.vector[Py_ssize_t]& shape, dtype)
