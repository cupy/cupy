from libc.stdint cimport intptr_t, uintmax_t

from cupy._core.core cimport ndarray


cdef class ChannelFormatDescriptor:
    cdef:
        readonly intptr_t ptr


cdef class ResourceDescriptor:
    cdef:
        readonly intptr_t ptr
        readonly ChannelFormatDescriptor chDesc
        readonly CUDAarray cuArr
        readonly ndarray arr


cdef class TextureDescriptor:
    cdef:
        readonly intptr_t ptr


cdef class CUDAarray:
    cdef:
        readonly intptr_t ptr
        readonly ChannelFormatDescriptor desc
        readonly size_t width, height, depth
        readonly unsigned int flags
        readonly int ndim

        int _get_memory_kind(self, src, dst)
        void* _make_cudaMemcpy3DParms(self, src, dst)


cdef class TextureObject:
    cdef:
        readonly uintmax_t ptr
        readonly ResourceDescriptor ResDesc
        readonly TextureDescriptor TexDesc


cdef class SurfaceObject:
    cdef:
        readonly uintmax_t ptr
        readonly ResourceDescriptor ResDesc


cdef class TextureReference:
    cdef:
        readonly intptr_t texref
        readonly ResourceDescriptor ResDesc
        readonly TextureDescriptor TexDesc

        _get_format(self, dict ch_format)
