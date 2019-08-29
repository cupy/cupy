from libc.stdint cimport intptr_t

from cupy.cuda.memory cimport BaseMemory


cdef class ChannelFormatDescriptor:
    cdef:
        readonly intptr_t ptr


cdef class ResourceDescriptor:
    cdef:
        readonly intptr_t ptr
        readonly ChannelFormatDescriptor chDesc


cdef class TextureDescriptor:
    cdef:
        readonly intptr_t ptr


cdef class CUDAArray(BaseMemory):
    cdef:
        readonly ChannelFormatDescriptor desc
        readonly size_t width
        readonly size_t height
        readonly size_t depth
        readonly unsigned int flags
        readonly int ndim
        int _get_kind(self, src, dst)
        void* _make_cudaMemcpy3DParms(self, src, dst)


cdef class TextureObject:
    cdef:
        readonly ResourceDescriptor ResDesc
        readonly TextureDescriptor TexDesc
        readonly unsigned long long ptr  # type: cudaTextureObject_t
