from cupy.cuda.memory cimport BaseMemory


cdef class CUDAArray(BaseMemory):
    cdef:
        readonly object desc
        readonly size_t width
        readonly size_t height
        readonly size_t depth
        readonly unsigned int flags
        readonly int ndim
        int _get_kind(self, src, dst)
        void* _make_cudaMemcpy3DParms(self, src, dst)


cdef class TextureObject:
    cdef:
        readonly object ResDesc, TexDesc
        readonly unsigned long long ptr  # type: cudaTextureObject_t
