from cupy._core.core cimport _ndarray_base
from cupy._core.core cimport shape_t, strides_t

cpdef _ndarray_base array(
    obj, dtype=*, copy=*, order=*, bint subok=*, Py_ssize_t ndmin=*,
    bint blocking=*)
cpdef _ndarray_base _convert_object_with_cuda_array_interface(a)
cdef tuple _array_info_from_nested_sequence(obj)
cdef _ndarray_base _ndarray_init(subtype, const shape_t& shape, dtype, obj)

cdef _ndarray_base _create_ndarray_from_shape_strides(
    subtype, const shape_t& shape, const strides_t& strides, dtype, obj)

cpdef _ndarray_base _internal_ascontiguousarray(_ndarray_base a)
cpdef _ndarray_base _internal_asfortranarray(_ndarray_base a)
cpdef _ndarray_base ascontiguousarray(_ndarray_base a, dtype=*)
cpdef _ndarray_base asfortranarray(_ndarray_base a, dtype=*)

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
