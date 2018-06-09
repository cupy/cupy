from libc.stdint cimport int8_t
from libc.stdint cimport int16_t
from libc.stdint cimport int32_t
from libc.stdint cimport int64_t
from libc.stdint cimport uint8_t
from libc.stdint cimport uint16_t
from libc.stdint cimport uint32_t
from libc.stdint cimport uint64_t

from cupy.cuda.function cimport CPointer


cdef union Scalar:
    bint bool_
    int8_t int8_
    int16_t int16_
    int32_t int32_
    int64_t int64_
    uint8_t uint8_
    uint16_t uint16_
    uint32_t uint32_
    uint64_t uint64_
    float float32_
    double float64_
    float complex complex64_
    double complex complex128_


cdef class CScalar(CPointer):

    cdef:
        Scalar val
        char kind
        int8_t size

    cpdef apply_dtype(self, dtype)


cpdef str get_typename(dtype)
cpdef convert_scalar(x, bint use_c_scalar)
