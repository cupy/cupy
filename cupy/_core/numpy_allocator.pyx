# distutils: language = c++

from libc cimport errno
from libc cimport stdlib
from libc.stdint cimport intptr_t
from libc.string cimport memset, memcpy


IF UNAME_SYSNAME == "Windows":
    cdef extern from "stdlib.h" nogil:
        void * _aligned_malloc(size_t size, size_t alignment)
        void _aligned_free(void * memblock)

    cdef inline void * aligned_alloc(size_t alignment, size_t size) nogil:
        return _aligned_malloc(size, alignment)

    cdef inline void aligned_free(void * ptr) nogil:
        _aligned_free(ptr)
ELSE:
    cdef extern from * nogil:
        void * aligned_alloc(size_t alignment, size_t size)

    cdef inline void aligned_free(void * ptr) nogil:
        stdlib.free(ptr)


# CuPy mempool requirement, see ALLOCATION_UNIT_SIZE in cupy/cuda/memory.pyx
DEF ALIGNMENT = 512


cdef extern void* _calloc(size_t nmemb, size_t size) nogil:
    errno.errno = 0
    cdef void* buf = aligned_alloc(ALIGNMENT, nmemb * size)
    if buf and errno.errno == 0:
        buf = memset(buf, 0, nmemb * size)

    return buf


cdef extern void* _malloc(size_t size) nogil:
    errno.errno = 0
    return aligned_alloc(ALIGNMENT, size)


cdef extern void* _realloc(void *ptr, size_t size) nogil:
    errno.errno = 0
    cdef void* buf = stdlib.realloc(ptr, size)
    cdef void* tmp

    if buf and errno.errno == 0 and <intptr_t>(buf) % ALIGNMENT != 0:
        tmp = buf
        errno.errno = 0
        buf = aligned_alloc(ALIGNMENT, size)
        if buf and errno.errno == 0:
            buf = memcpy(buf, tmp, size)
            stdlib.free(tmp)

    return buf


cdef extern void _free(void* ptr) nogil:
    aligned_free(ptr)
