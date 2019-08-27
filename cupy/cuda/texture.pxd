from cupy.cuda cimport runtime
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
        _make_cudaMemcpy2DParams(self, src, dst)
        runtime.Memcpy3DParms* _make_cudaMemcpy3DParms(self, src, dst)
        _print_param(self, runtime.Memcpy3DParms* param)


cdef class TextureObject:
    cdef:
        readonly unsigned long long ptr  # type: cudaTextureObject_t
