# distutils: language = c++

"""Thin wrapper of Thrust implementations for CuPy API."""

from cupy.cuda cimport common


###############################################################################
# Extern
###############################################################################

cdef extern from "../cuda/cupy_thrust.h" namespace "cupy::thrust":
    void stable_sort[T](T *first, T *last)


###############################################################################
# Python interfaces
###############################################################################

cpdef stable_sort_ubyte(size_t first, size_t last):
    stable_sort[common.cpy_ubyte](<common.cpy_ubyte *>first,
                                  <common.cpy_ubyte *>last)

cpdef stable_sort_byte(size_t first, size_t last):
    stable_sort[common.cpy_byte](<common.cpy_byte *>first,
                                 <common.cpy_byte *>last)

cpdef stable_sort_ushort(size_t first, size_t last):
    stable_sort[common.cpy_ushort](<common.cpy_ushort *>first,
                                   <common.cpy_ushort *>last)

cpdef stable_sort_short(size_t first, size_t last):
    stable_sort[common.cpy_short](<common.cpy_short *>first,
                                  <common.cpy_short *>last)

cpdef stable_sort_uint(size_t first, size_t last):
    stable_sort[common.cpy_uint](<common.cpy_uint *>first,
                                 <common.cpy_uint *>last)

cpdef stable_sort_int(size_t first, size_t last):
    stable_sort[common.cpy_int](<common.cpy_int *>first,
                                <common.cpy_int *>last)

cpdef stable_sort_ulong(size_t first, size_t last):
    stable_sort[common.cpy_ulong](<common.cpy_ulong *>first,
                                  <common.cpy_ulong *>last)

cpdef stable_sort_long(size_t first, size_t last):
    stable_sort[common.cpy_long](<common.cpy_long *>first,
                                 <common.cpy_long *>last)

cpdef stable_sort_float(size_t first, size_t last):
    stable_sort[common.cpy_float](<common.cpy_float *>first,
                                  <common.cpy_float *>last)

cpdef stable_sort_double(size_t first, size_t last):
    stable_sort[common.cpy_double](<common.cpy_double *>first,
                                   <common.cpy_double *>last)
