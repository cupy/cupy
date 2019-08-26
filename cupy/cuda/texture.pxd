from libc.stdint cimport intptr_t


cdef class CUDAArray:
    cdef:
        readonly intptr_t ptr


cdef class TextureObject:
    cdef:
        readonly intptr_t ptr
