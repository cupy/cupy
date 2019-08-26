from libc.stdint cimport intptr_t

from cupy.cuda cimport runtime
from cupy.cuda.memory cimport BaseMemory


cdef class CUDAArray(BaseMemory):
    pass


cdef class TextureObject:
    cdef:
        readonly runtime.TextureObject ptr
