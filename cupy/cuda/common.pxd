# distutils: language = c++

from libcpp cimport bool as bool_t

cdef extern from '../cuda/cupy_common.h':  # thru parent to import in core
    ctypedef char cpy_byte
    ctypedef unsigned char cpy_ubyte
    ctypedef short cpy_short
    ctypedef unsigned short cpy_ushort
    ctypedef int cpy_int
    ctypedef unsigned int cpy_uint
    ctypedef long long cpy_long
    ctypedef unsigned long long cpy_ulong
    ctypedef float cpy_float
    ctypedef double cpy_double
    ctypedef struct cpy_complex64 'cuComplex'
    ctypedef struct cpy_complex128 'cuDoubleComplex'
    ctypedef bool_t cpy_bool 'bool'
