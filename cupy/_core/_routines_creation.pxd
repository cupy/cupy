from cupy._core.core cimport _ndarray_base
from cupy._core.core cimport shape_t, strides_t

cpdef _ndarray_base array(
    obj, dtype=*, copy=*, order=*, bint subok=*, Py_ssize_t ndmin=*,
    bint blocking=*)
cpdef _ndarray_base _convert_object_with_cuda_array_interface(a)

cdef _ndarray_base _ndarray_init(subtype, const shape_t& shape, dtype, obj)

cdef _ndarray_base _create_ndarray_from_shape_strides(
    subtype, const shape_t& shape, const strides_t& strides, dtype, obj)

cpdef _ndarray_base _internal_ascontiguousarray(_ndarray_base a)
cpdef _ndarray_base _internal_asfortranarray(_ndarray_base a)
cpdef _ndarray_base ascontiguousarray(_ndarray_base a, dtype=*)
cpdef _ndarray_base asfortranarray(_ndarray_base a, dtype=*)
