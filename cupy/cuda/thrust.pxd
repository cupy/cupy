from cupy.cuda.common cimport *

cdef void stable_sort_byte(cpy_byte *first, cpy_byte *last)
cdef void stable_sort_ubyte(cpy_ubyte *first, cpy_ubyte *last)
cdef void stable_sort_short(cpy_short *first, cpy_short *last)
cdef void stable_sort_ushort(cpy_ushort *first, cpy_ushort *last)
cdef void stable_sort_int(cpy_int *first, cpy_int *last)
cdef void stable_sort_uint(cpy_uint *first, cpy_uint *last)
cdef void stable_sort_long(cpy_long *first, cpy_long *last)
cdef void stable_sort_ulong(cpy_ulong *first, cpy_ulong *last)
cdef void stable_sort_float(cpy_float *first, cpy_float *last)
cdef void stable_sort_double(cpy_double *first, cpy_double *last)
