from cupy.cuda.common cimport *

cdef void stable_sort_byte(void *start, ssize_t num)
cdef void stable_sort_ubyte(void *start, ssize_t num)
cdef void stable_sort_short(void *start, ssize_t num)
cdef void stable_sort_ushort(void *start, ssize_t num)
cdef void stable_sort_int(void *start, ssize_t num)
cdef void stable_sort_uint(void *start, ssize_t num)
cdef void stable_sort_long(void *start, ssize_t num)
cdef void stable_sort_ulong(void *start, ssize_t num)
cdef void stable_sort_float(void *start, ssize_t num)
cdef void stable_sort_double(void *start, ssize_t num)
