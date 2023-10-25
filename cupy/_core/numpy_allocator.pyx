# distutils: language = c++

from libc cimport errno
from libc cimport stdlib
from libc.stdint cimport intptr_t
from libc.string cimport memset, memcpy


# TODO(leofang): move _calloc, _malloc, _realloc to inline header so that we
# have true nogil
cdef extern from * nogil:
    # available since C11
    void* aligned_alloc(size_t alignment, size_t size)


# CuPy mempool requirement, see ALLOCATION_UNIT_SIZE in cupy/cuda/memory.pyx
DEF ALIGNMENT = 512


cdef public void* _calloc(size_t nmemb, size_t size) nogil:
    errno.errno = 0
    cdef void* buf = aligned_alloc(ALIGNMENT, nmemb * size)
    if buf and errno.errno == 0:
        buf = memset(buf, 0, nmemb * size)

    return buf


cdef public void* _malloc(size_t size) nogil:
    errno.errno = 0
    return aligned_alloc(ALIGNMENT, size)


cdef public void* _realloc(void *ptr, size_t size) nogil:
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


# def get_aligned_host_allocator():
#     try:
#         import numpy_allocator
#     except ImportError as e:
#         raise RuntimeError('numpy_allocator must be available') from e
# 
#     import ctypes
#     lib = ctypes.CDLL(__file__)
#     class AlignedHostAllocator(metaclass=numpy_allocator.type):
#         # Note: we cannot just do this:
#         #   _malloc_ = <intptr_t>(_malloc)
#         # but need ctypes because the pointer addresses are relocated (I think)
#         _calloc_ = ctypes.addressof(lib._calloc)
#         _malloc_ = ctypes.addressof(lib._malloc)
#         _realloc_ = ctypes.addressof(lib._realloc)
#     return AlignedHostAllocator
