from cupy.cuda.memory cimport BaseMemory


cdef class CUDAArray(BaseMemory):
    cdef:
        readonly object desc
        readonly size_t width
        readonly size_t height
        readonly size_t depth
        readonly unsigned int flags


cdef class TextureObject:
    cdef:
        readonly unsigned long long ptr  # type: cudaTextureObject_t
