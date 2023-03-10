cpdef get_dtype(t)
cpdef tuple get_dtype_with_itemsize(t)
cpdef int to_cuda_dtype(dtype, bint is_half_allowed=*) except -1

cpdef void _raise_if_invalid_cast(
    from_dt,
    to_dt,
    str casting,
    argname=*
) except *

cpdef _resolve_dtype_cast(from_dt, to)
