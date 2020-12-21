cimport cython  # NOQA

from libc.stdint cimport int8_t
from libc.stdint cimport int32_t

from cupy.cuda.function cimport CPointer


@cython.final
cdef class CScalar(CPointer):

    cdef:
        char kind
        int8_t size

    @staticmethod
    cdef CScalar from_int32(int32_t value)

    @staticmethod
    cdef CScalar from_numpy_scalar_with_dtype(object x, object dtype)

    @staticmethod
    cdef CScalar _from_python_scalar(object x)

    @staticmethod
    cdef CScalar _from_numpy_scalar(object x)

    cpdef apply_dtype(self, dtype)
    cpdef get_numpy_type(self)


cpdef str get_typename(dtype)

cdef set scalar_type_set
cdef CScalar scalar_to_c_scalar(object x)
cdef object scalar_to_numpy_scalar(object x)
