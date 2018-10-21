from libc.stdint cimport int8_t

from cupy.cuda.function cimport CPointer


cdef class CScalar(CPointer):

    cdef:
        char kind
        int8_t size

    cpdef apply_dtype(self, dtype)
    cpdef get_numpy_type(self)


cpdef str get_typename(dtype)
cpdef get_scalar_from_numpy(x, dtype)
cpdef convert_scalar(x, bint use_c_scalar)
