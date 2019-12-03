cimport cython  # NOQA

from libc.stdint cimport int8_t
from libc.stdint cimport int32_t

from cupy.cuda.function cimport CPointer


@cython.final
cdef class CScalar(CPointer):

    cdef:
        char kind
        int8_t size

    cpdef apply_dtype(self, dtype)
    cpdef get_numpy_type(self)


cdef CScalar CScalar_from_int32(int32_t value)

cpdef str get_typename(dtype)
cpdef get_scalar_from_numpy(x, dtype)
cpdef convert_scalar(x, bint use_c_scalar)
