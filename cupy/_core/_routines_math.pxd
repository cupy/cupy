from cupy._core.core cimport _ndarray_base


cdef _ndarray_base _ndarray_conj(_ndarray_base self)
cdef _ndarray_base _ndarray_real_getter(_ndarray_base self)
cdef _ndarray_base _ndarray_real_setter(_ndarray_base self, value)
cdef _ndarray_base _ndarray_imag_getter(_ndarray_base self)
cdef _ndarray_base _ndarray_imag_setter(_ndarray_base self, value)
cdef _ndarray_base _ndarray_prod(
    _ndarray_base self, axis, dtype, out, keepdims)
cdef _ndarray_base _ndarray_sum(_ndarray_base self, axis, dtype, out, keepdims)
cdef _ndarray_base _ndarray_cumsum(_ndarray_base self, axis, dtype, out)
cdef _ndarray_base _ndarray_cumprod(_ndarray_base self, axis, dtype, out)
cdef _ndarray_base _ndarray_clip(_ndarray_base self, a_min, a_max, out)

cpdef _ndarray_base _nansum(_ndarray_base a, axis, dtype, out, keepdims)
cpdef _ndarray_base _nanprod(_ndarray_base a, axis, dtype, out, keepdims)

cpdef enum scan_op:
    SCAN_SUM = 0
    SCAN_PROD = 1

cdef _ndarray_base scan(_ndarray_base a, op, dtype=*, _ndarray_base out=*,
                        incomplete=*, chunk_size=*)
cdef object _sum_auto_dtype
cdef object _add
cdef object _conj
cdef object _angle
cdef object _positive
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
