from libcpp cimport vector
from cupy.cuda cimport memory

from cupy.cuda.function cimport CPointer
from cupy.cuda.function cimport Module
from cupy._core._carray cimport shape_t
from cupy._core._carray cimport strides_t


cdef class _ndarray_base:
    cdef:
        object __weakref__
        readonly Py_ssize_t size
        public shape_t _shape
        public strides_t _strides
        readonly bint _c_contiguous
        readonly bint _f_contiguous
        # Whether the array memory can be addressed with 32bit signed
        # integers. To do fast indexing in the CArray class.
        readonly bint _index_32_bits
        readonly object dtype
        readonly memory.MemoryPointer data
        # TODO(niboshi): Return arbitrary owner object as `base` if the
        # underlying memory is UnownedMemory.
        readonly _ndarray_base base

    cdef _init_fast(self, const shape_t& shape, dtype, bint c_order)
    cdef inline bint is_host_accessible(self) except*
    cpdef item(self)
    cpdef tolist(self)
    cpdef bytes tobytes(self, order=*)
    cpdef tofile(self, fid, sep=*, format=*)
    cpdef dump(self, file)
    cpdef bytes dumps(self)
    cpdef _ndarray_base _astype(
        self, dtype, order=*, casting=*, subok=*, copy=*)
    cpdef _ndarray_base astype(
        self, dtype, order=*, casting=*, subok=*, copy=*)
    cpdef _ndarray_base copy(self, order=*)
    cpdef _ndarray_base view(self, dtype=*, array_class=*)
    cpdef fill(self, value)
    cpdef _ndarray_base swapaxes(self, Py_ssize_t axis1, Py_ssize_t axis2)
    cpdef _ndarray_base flatten(self, order=*)
    cpdef _ndarray_base ravel(self, order=*)
    cpdef _ndarray_base squeeze(self, axis=*)
    cpdef _ndarray_base take(self, indices, axis=*, out=*)
    cpdef put(self, indices, values, mode=*)
    cpdef repeat(self, repeats, axis=*)
    cpdef choose(self, choices, out=*, mode=*)
    cpdef sort(self, int axis=*, kind=*)
    cpdef _ndarray_base argsort(self, axis=*, kind=*)
    cpdef partition(self, kth, int axis=*)
    cpdef _ndarray_base argpartition(self, kth, axis=*)
    cpdef tuple nonzero(self)
    cpdef _ndarray_base compress(self, condition, axis=*, out=*)
    cpdef _ndarray_base diagonal(self, offset=*, axis1=*, axis2=*)
    cpdef _ndarray_base max(self, axis=*, out=*, keepdims=*)
    cpdef _ndarray_base argmax(self, axis=*, out=*, dtype=*, keepdims=*)
    cpdef _ndarray_base min(self, axis=*, out=*, keepdims=*)
    cpdef _ndarray_base argmin(self, axis=*, out=*, dtype=*, keepdims=*)
    cpdef _ndarray_base ptp(self, axis=*, out=*, keepdims=*)
    cpdef _ndarray_base clip(self, min=*, max=*, out=*)
    cpdef _ndarray_base round(self, decimals=*, out=*)

    cpdef _ndarray_base trace(self, offset=*, axis1=*, axis2=*, dtype=*, out=*)
    cpdef _ndarray_base sum(self, axis=*, dtype=*, out=*, keepdims=*)
    cpdef _ndarray_base cumsum(self, axis=*, dtype=*, out=*)
    cpdef _ndarray_base mean(self, axis=*, dtype=*, out=*, keepdims=*)
    cpdef _ndarray_base var(self, axis=*, dtype=*, out=*, ddof=*, keepdims=*)
    cpdef _ndarray_base std(self, axis=*, dtype=*, out=*, ddof=*, keepdims=*)
    cpdef _ndarray_base prod(self, axis=*, dtype=*, out=*, keepdims=*)
    cpdef _ndarray_base cumprod(self, axis=*, dtype=*, out=*)
    cpdef _ndarray_base _add_reduceat(self, indices, axis, dtype, out)
    cpdef _ndarray_base all(self, axis=*, out=*, keepdims=*)
    cpdef _ndarray_base any(self, axis=*, out=*, keepdims=*)
    cpdef _ndarray_base conj(self)
    cpdef _ndarray_base conjugate(self)
    cpdef get(self, stream=*, order=*, out=*, blocking=*)
    cpdef set(self, arr, stream=*)
    cpdef _ndarray_base reduced_view(self, dtype=*)
    cdef _update_c_contiguity(self)
    cdef _update_f_contiguity(self)
    cdef _update_contiguity(self)
    cpdef _set_shape_and_strides(self, const shape_t& shape,
                                 const strides_t& strides,
                                 bint update_c_contiguity,
                                 bint update_f_contiguity)
    cdef _ndarray_base _view(self, subtype, const shape_t& shape,
                             const strides_t& strides,
                             bint update_c_contiguity,
                             bint update_f_contiguity, obj)
    cdef _set_contiguous_strides(
        self, Py_ssize_t itemsize, bint is_c_contiguous)
    cdef CPointer get_pointer(self)
    cpdef object toDlpack(self)


cpdef _ndarray_base _internal_ascontiguousarray(_ndarray_base a)
cpdef _ndarray_base _internal_asfortranarray(_ndarray_base a)
cpdef _ndarray_base ascontiguousarray(_ndarray_base a, dtype=*)
cpdef _ndarray_base asfortranarray(_ndarray_base a, dtype=*)

cpdef Module compile_with_cache(str source, tuple options=*, arch=*,
                                prepend_cupy_headers=*,
                                backend=*, translate_cucomplex=*,
                                enable_cooperative_groups=*,
                                name_expressions=*, log_stream=*,
                                bint jitify=*)


# TODO(niboshi): Move to _routines_creation.pyx

cpdef _ndarray_base array(
    obj, dtype=*, copy=*, order=*, bint subok=*, Py_ssize_t ndmin=*,
    bint blocking=*)
cpdef _ndarray_base _convert_object_with_cuda_array_interface(a)

cdef _ndarray_base _ndarray_init(subtype, const shape_t& shape, dtype, obj)

cdef _ndarray_base _create_ndarray_from_shape_strides(
    subtype, const shape_t& shape, const strides_t& strides, dtype, obj)


cdef inline _ndarray_base _convert_from_cupy_like(
        obj, error):
    """Inline function to convert cupy array-likes to cupy ndarray.

    Includes a check to allow passing a cupy.ndarray itself and converts the
    protocols `__cuda_array_interface__` and `__cupy_get_ndarray__`.
    Used where general GPU arrays should be supported but CPU arrays are not
    supported or handled differently.

    If `error` is False no error will be raised if conversion if the type
    is not cupy-like and `None` is returned instead.
    """
    if isinstance(obj, _ndarray_base):
        return <_ndarray_base>obj
    elif hasattr(obj, "__cuda_array_interface__"):
        return _convert_object_with_cuda_array_interface(obj)
    elif hasattr(obj, "__cupy_get_ndarray__"):
        return obj.__cupy_get_ndarray__()

    if error is not False:
        raise TypeError(
            f'{error} has incorrect type '
            f'(expected {_ndarray_base}, got {type(obj)}).'
        )

    return None
