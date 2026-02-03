cimport cython  # NOQA

cimport numpy as cnp

from libc.stdint cimport int8_t
from libc.stdint cimport int32_t

from cupy.cuda.function cimport CPointer


@cython.final
cdef class CScalar(CPointer):

    cdef:
        cnp.clongdouble_t _data[2]  # assume largest alignment
        readonly object value
        readonly cnp.dtype descr  # C-level dtype
        readonly object weak_t

    @staticmethod
    cdef CScalar from_int32(int32_t value)

    cdef _store_c_value(self)
    cpdef apply_dtype(self, dtype)
    cpdef get_numpy_type(self)


cpdef str get_typename(dtype, type_headers=*)
cdef Py_ssize_t get_cuda_alignment(dtype) except -1

cdef set scalar_type_set
cpdef str _get_cuda_scalar_repr(obj, dtype)
