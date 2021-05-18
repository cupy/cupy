from cupy._core.core cimport ndarray


cdef ndarray _ndarray_conj(ndarray self)
cdef ndarray _ndarray_real_getter(ndarray self)
cdef ndarray _ndarray_real_setter(ndarray self, value)
cdef ndarray _ndarray_imag_getter(ndarray self)
cdef ndarray _ndarray_imag_setter(ndarray self, value)
cdef ndarray _ndarray_prod(ndarray self, axis, dtype, out, keepdims)
cdef ndarray _ndarray_sum(ndarray self, axis, dtype, out, keepdims)
cdef ndarray _ndarray_cumsum(ndarray self, axis, dtype, out)
cdef ndarray _ndarray_cumprod(ndarray self, axis, dtype, out)
cdef ndarray _ndarray_clip(ndarray self, a_min, a_max, out)

cpdef ndarray _nansum(ndarray a, axis, dtype, out, keepdims)
cpdef ndarray _nanprod(ndarray a, axis, dtype, out, keepdims)

cpdef enum scan_op:
    SCAN_SUM = 0
    SCAN_PROD = 1

cdef ndarray scan(ndarray a, op, dtype=*, ndarray out=*, incomplete=*,
                  chunk_size=*)
cdef object _sum_auto_dtype
cdef object _add
cdef object _conj
cdef object _angle
cdef object _real
cdef object _imag
cdef object _negative
cdef object _multiply
cdef object _divide
cdef object _power
cdef object _subtract
cdef object _true_divide
cdef object _floor_divide
cdef object _remainder
cdef object _absolute
cdef object _sqrt
