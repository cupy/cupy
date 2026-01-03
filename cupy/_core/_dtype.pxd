from cpython cimport Py_buffer

cimport numpy as cnp

cdef str all_type_chars
cdef bytes all_type_chars_b

cdef bint check_supported_dtype(cnp.dtype dtype, bint error) except -1
cpdef get_dtype(t)
cpdef tuple get_dtype_with_itemsize(t, bint check_support)
cpdef int to_cuda_dtype(dtype, bint is_half_allowed=*) except -1

cpdef void _raise_if_invalid_cast(
    from_dt,
    to_dt,
    str casting,
    argname=*
) except *

cdef void populate_format(Py_buffer* buf, str dtype) except*


cdef inline normalize_dtype(dtype):
    """Given an existing NumPy dtype normalize it for cupy.

    Right now, this is a small helper to ensure little-endian.
    Use this when converting from NumPy/CPU arrays where cupy helps the
    user by byte-swapping automatically.
    """
    return dtype.newbyteorder("<")
