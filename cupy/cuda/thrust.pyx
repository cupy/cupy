"""Thin wrapper of CuPy Thrust wrapper."""
# NOTE: This wrapper does not cover all APIs of Thrust.


###############################################################################
# Extern
###############################################################################

cdef extern from "cupy_thrust.h" namespace "cupy::thrust":
    void stable_sort_byte(cpy_byte *first, cpy_byte *last)
    void stable_sort_ubyte(cpy_ubyte *first, cpy_ubyte *last)
    void stable_sort_short(cpy_short *first, cpy_short *last)
    void stable_sort_ushort(cpy_ushort *first, cpy_ushort *last)
    void stable_sort_int(cpy_int *first, cpy_int *last)
    void stable_sort_uint(cpy_uint *first, cpy_uint *last)
    void stable_sort_long(cpy_long *first, cpy_long *last)
    void stable_sort_ulong(cpy_ulong *first, cpy_ulong *last)
    void stable_sort_float(cpy_float *first, cpy_float *last)
    void stable_sort_double(cpy_double *first, cpy_double *last)
