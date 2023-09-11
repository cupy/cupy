from libc.stdint cimport intptr_t, uintmax_t

from cupy._core.core cimport _ndarray_base


cdef class ChannelFormatDescriptor:
    cdef:
        readonly intptr_t ptr


cdef class ResourceDescriptor:
    cdef:
        readonly intptr_t ptr
        readonly ChannelFormatDescriptor chDesc
        readonly CUDAarray cuArr
        readonly _ndarray_base arr


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
        void _prepare_copy(self, arr, stream, direction) except*


cdef class TextureObject:
    cdef:
        readonly uintmax_t ptr
        readonly ResourceDescriptor ResDesc
        readonly TextureDescriptor TexDesc


cdef class SurfaceObject:
    cdef:
        readonly uintmax_t ptr
        readonly ResourceDescriptor ResDesc
