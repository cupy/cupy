from libc.stdint cimport intptr_t


cdef class ChannelFormatDescriptor:
    cdef:
        readonly intptr_t ptr


cdef class ResourceDescriptor:
    cdef:
        readonly intptr_t ptr
        readonly ChannelFormatDescriptor chDesc
        readonly object cuArr, arr


cdef class TextureDescriptor:
    cdef:
        readonly intptr_t ptr


cdef class CUDAArray:
    cdef:
        readonly intptr_t ptr
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
        readonly intptr_t ptr
        readonly ResourceDescriptor ResDesc
        readonly TextureDescriptor TexDesc
