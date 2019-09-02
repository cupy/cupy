from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.string cimport memset as c_memset

import numpy

from cupy.core.core cimport ndarray
from cupy.cuda cimport device
from cupy.cuda cimport runtime
from cupy.cuda.runtime cimport Array, ChannelFormatDesc, ChannelFormatKind,\
    Memcpy3DParms, MemoryKind, PitchedPtr, ResourceDesc, ResourceType, \
    TextureAddressMode, TextureDesc, TextureFilterMode, TextureReadMode
from cupy.cuda.runtime import CUDARuntimeError


cdef class ChannelFormatDescriptor:
    def __init__(self, int x, int y, int z, int w, int f):
        # We don't call cudaCreateChannelDesc() here for two reasons: 1. to
        # avoid out of scope; 2. it doesn't do input verification for us.
        #
        # WARNING: don't use [0] or cython.operator.dereference to dereference
        # a pointer to struct for writing to its members !!! (But read is OK.)
        # Turns out while there's no arrow operator '->' in Cython, one can
        # just treat the ptr as a real object and access the struct attributes.
        # This applies to several classes below.
        self.ptr = <intptr_t>PyMem_Malloc(sizeof(ChannelFormatDesc))
        cdef ChannelFormatDesc* desc = (<ChannelFormatDesc*>self.ptr)
        desc.x = x
        desc.y = y
        desc.z = z
        desc.w = w
        desc.f = <ChannelFormatKind>f

    def __dealloc__(self):
        PyMem_Free(<ChannelFormatDesc*>self.ptr)
        self.ptr = 0

    def get_channel_format(self):
        cdef dict desc = {}
        desc['x'] = (<ChannelFormatDesc*>self.ptr).x
        desc['y'] = (<ChannelFormatDesc*>self.ptr).y
        desc['z'] = (<ChannelFormatDesc*>self.ptr).z
        desc['w'] = (<ChannelFormatDesc*>self.ptr).w
        desc['f'] = (<ChannelFormatDesc*>self.ptr).f
        return desc


cdef class ResourceDescriptor:
    def __init__(self, int restype, CUDAarray cuArr=None, ndarray arr=None,
                 ChannelFormatDescriptor chDesc=None, size_t sizeInBytes=0,
                 size_t width=0, size_t height=0, size_t pitchInBytes=0):
        '''
        Args:
        '''
        cdef ResourceType resType = <ResourceType>restype
        if resType == runtime.cudaResourceTypeMipmappedArray:
            # TODO(leofang): support this?
            raise NotImplementedError('cudaResourceTypeMipmappedArray is '
                                      'currently not supported.')

        self.ptr = <intptr_t>PyMem_Malloc(sizeof(ResourceDesc))
        cdef ResourceDesc* desc = (<ResourceDesc*>self.ptr)
        c_memset(desc, 0, sizeof(ResourceDesc))

        desc.resType = resType
        if resType == runtime.cudaResourceTypeArray:
            desc.res.array.array = <Array>(cuArr.ptr)
        elif resType == runtime.cudaResourceTypeLinear:
            desc.res.linear.devPtr = <void*>(arr.data.ptr)
            desc.res.linear.desc = (<ChannelFormatDesc*>chDesc.ptr)[0]
            desc.res.linear.sizeInBytes = sizeInBytes
        elif resType == runtime.cudaResourceTypePitch2D:
            desc.res.pitch2D.devPtr = <void*>(arr.data.ptr)
            desc.res.pitch2D.desc = (<ChannelFormatDesc*>chDesc.ptr)[0]
            desc.res.pitch2D.width = width
            desc.res.pitch2D.height = height
            desc.res.pitch2D.pitchInBytes = pitchInBytes

        self.chDesc = chDesc
        self.cuArr = cuArr
        self.arr = arr

    def __dealloc__(self):
        PyMem_Free(<ResourceDesc*>self.ptr)
        self.ptr = 0

    def get_resource_desc(self):
        cdef dict desc = {}
        cdef intptr_t ptr
        cdef size_t size, pitch, w, h

        desc['resType'] = (<ResourceDesc*>self.ptr).resType
        ptr = <intptr_t>((<ResourceDesc*>self.ptr).res.array.array)
        desc['array'] = {'array': ptr}
        # TODO(leofang): add linear, pitch2D
        # ptr = <intptr_t>((<ResourceDesc*>self.ptr).res.linear.devPtr)
        # desc['linear'] = {'devPtr': ptr,
        #                   'desc': None,
        #                   'sizeInBytes': <ResourceDesc*>self.ptr)\
        #                        .res.linear.sizeInBytes)
        # desc[
        return desc


cdef class TextureDescriptor:
    def __init__(self, addressModes=None, int filterMode=0, int readMode=0,
                 sRGB=None, borderColors=None, normalizedCoords=None,
                 maxAnisotropy=None):
        self.ptr = <intptr_t>PyMem_Malloc(sizeof(TextureDesc))
        cdef TextureDesc* desc = (<TextureDesc*>self.ptr)
        c_memset(desc, 0, sizeof(TextureDesc))

        if addressModes is not None:
            assert len(addressModes) <= 3
            for i, mode in enumerate(addressModes):
                desc.addressMode[i] = <TextureAddressMode>mode
        desc.filterMode = <TextureFilterMode>filterMode
        desc.readMode = <TextureReadMode>readMode
        if normalizedCoords is not None:
            desc.normalizedCoords = normalizedCoords
        if sRGB is not None:
            desc.sRGB = sRGB
        if borderColors is not None:
            assert len(borderColors) <= 4
            for i, color in enumerate(borderColors):
                desc.borderColor[i] = color
        if maxAnisotropy is not None:
            desc.maxAnisotropy = maxAnisotropy
        # TODO(leofang): support mipmap?

    def __dealloc__(self):
        PyMem_Free(<TextureDesc*>self.ptr)
        self.ptr = 0

    def get_texture_desc(self):
        cdef dict desc = {}
        cdef TextureDesc* ptr = <TextureDesc*>(self.ptr)
        desc['addressMode'] = (ptr.addressMode[0],
                               ptr.addressMode[1],
                               ptr.addressMode[2])
        desc['filterMode'] = ptr.filterMode
        desc['readMode'] = ptr.readMode
        desc['sRGB'] = ptr.sRGB
        desc['borderColor']= (ptr.borderColor[0],
                              ptr.borderColor[1],
                              ptr.borderColor[2],
                              ptr.borderColor[3])
        desc['normalizedCoords']= ptr.normalizedCoords
        desc['maxAnisotropy'] = ptr.maxAnisotropy
        # TODO(leofang): support these?
        # desc['mipmapFilterMode'] = ptr.mipmapFilterMode
        # desc['mipmapLevelBias'] = ptr.mipmapLevelBias
        # desc['minMipmapLevelClamp'] = ptr.minMipmapLevelClamp
        # desc['maxMipmapLevelClamp'] = ptr.maxMipmapLevelClamp
        return desc


cdef class CUDAarray:
    # TODO(leofang): perhaps this wrapper is not needed when cupy.ndarray
    # can be backed by texture memory/CUDA arrays?
    def __init__(self, ChannelFormatDescriptor desc, size_t width,
                 size_t height, size_t depth=0, unsigned int flags=0):
        if width == 0:
            raise ValueError('To create a CUDA array, width must be nonzero.')
        elif height == 0 and depth > 0:
            raise ValueError('To create a 2D CUDA array, height must be '
                             'nonzero.')
        else:
            # malloc3DArray handles all possibilities (1D, 2D, 3D)
            self.ptr = runtime.malloc3DArray(desc.ptr, width, height, depth,
                                             flags)

        # bookkeeping
        self.desc = desc
        self.width = width
        self.height = height
        self.depth = depth
        self.flags = flags
        self.ndim = 3 if depth > 0 else 2 if height > 0 else 1

    def __dealloc__(self):
        runtime.freeArray(self.ptr)
        self.ptr = 0

    cdef int _get_memory_kind(self, src, dst):
        cdef int kind
        if isinstance(src, ndarray) and dst is self:
            kind = runtime.memcpyDeviceToDevice
        elif src is self and isinstance(dst, ndarray):
            kind = runtime.memcpyDeviceToDevice
        elif isinstance(src, numpy.ndarray) and dst is self:
            kind = runtime.memcpyHostToDevice
        elif src is self and isinstance(dst, numpy.ndarray):
            kind = runtime.memcpyDeviceToHost
        else:
            raise
        return kind

    cdef void* _make_cudaMemcpy3DParms(self, src, dst):
        '''Private helper for data transfer. Supports all dimensions.'''
        cdef Memcpy3DParms* param = \
            <Memcpy3DParms*>PyMem_Malloc(sizeof(Memcpy3DParms))
        c_memset(param, 0, sizeof(Memcpy3DParms))
        cdef PitchedPtr srcPitchedPtr, dstPitchedPtr
        cdef intptr_t ptr

        # get kind
        param.kind = <MemoryKind>self._get_memory_kind(src, dst)

        # get src
        if src is self:
            # Important: cannot convert from src.ptr!
            param.srcArray = <Array>(self.ptr)
            param.extent = runtime.make_Extent(self.width, self.height,
                                               self.depth)
        else:
            width = src.shape[-1]
            if src.ndim >= 2:
                height = src.shape[-2]
            else:
                height = 1  # same "one-stride" trick here

            if isinstance(src, ndarray):
                ptr = src.data.ptr
            else:  # numpy.ndarray
                ptr = src.ctypes.data

            srcPitchedPtr = runtime.make_PitchedPtr(
                ptr, width*src.dtype.itemsize, width, height)
            param.srcPtr = srcPitchedPtr

        # get dst
        if dst is self:
            # Important: cannot convert from dst.ptr!
            param.dstArray = <Array>(self.ptr)
            param.extent = runtime.make_Extent(self.width, self.height,
                                               self.depth)
        else:
            width = dst.shape[-1]
            if dst.ndim >= 2:
                height = dst.shape[-2]
            else:
                height = 1  # same "one-stride" trick here

            if isinstance(dst, ndarray):
                ptr = dst.data.ptr
            else:  # numpy.ndarray
                ptr = dst.ctypes.data

            dstPitchedPtr = runtime.make_PitchedPtr(
                ptr, width*dst.dtype.itemsize, width, height)
            param.dstPtr = dstPitchedPtr

        return <void*>param

    def _prepare_copy(self, arr, stream, direction):
        # sanity checks:
        # - check shape
        if self.ndim == 3:
            if arr.shape != (self.depth, self.height, self.width):
                raise ValueError("shape mismatch")
        elif self.ndim == 2:
            if arr.shape != (self.height, self.width):
                raise ValueError("shape mismatch")
        else:  # ndim = 1
            if arr.shape != (self.width,):
                raise ValueError("shape mismatch")

        # - check dtype
        # TODO(leofang): we should also check channel bit size vs dtype
        # itemsize, but can we assume it's always single channel?
        ch_kind = self.desc.get_channel_format()['f']
        if ch_kind == runtime.cudaChannelFormatKindSigned:
            if arr.dtype not in (numpy.int8, numpy.int16, numpy.int32):
                raise ValueError("dtype mismatch")
        elif ch_kind == runtime.cudaChannelFormatKindUnsigned:
            if arr.dtype not in (numpy.uint8, numpy.uint16, numpy.uint32):
                raise ValueError("dtype mismatch")
        elif ch_kind == runtime.cudaChannelFormatKindFloat:
            if arr.dtype not in (numpy.float16, numpy.float32):
                raise ValueError("dtype mismatch")
        else:
            raise ValueError("dtype not supported")

        cdef Memcpy3DParms* param = NULL

        # Trick: For 1D or 2D CUDA arrays, we need to "fool" memcpy3D so that
        # at least one stride gets copied. This is not properly documented in
        # Runtime API unfortunately. See, e.g.,
        # https://stackoverflow.com/a/39217379/2344149
        if self.ndim == 1:
            self.height = 1
            self.depth = 1
        elif self.ndim == 2:
            self.depth = 1

        if direction == 'in':
            param = <Memcpy3DParms*>self._make_cudaMemcpy3DParms(arr, self)
        elif direction == 'out':
            param = <Memcpy3DParms*>self._make_cudaMemcpy3DParms(self, arr)
        try:
            if stream is None:
                runtime.memcpy3D(<intptr_t>param)
            else:
                runtime.memcpy3DAsync(<intptr_t>param, stream.ptr)
        except CUDARuntimeError as ex:
            raise ex
        finally:
            PyMem_Free(param)

            # restore old config
            if self.ndim == 1:
                self.height = 0
                self.depth = 0
            elif self.ndim == 2:
                self.depth = 0

    def copy_from(self, in_arr, stream=None):
        '''Copy data from device or host array to CUDA array.

        Args:
            arr (cupy.core.core.ndarray or numpy.ndarray)
            stream (cupy.cuda.Stream)
        '''
        self._prepare_copy(in_arr, stream, direction='in')

    def copy_to(self, out_arr, stream=None):
        '''Copy data from CUDA array to device or host array.

        Args:
            arr (cupy.core.core.ndarray or numpy.ndarray)
            stream (cupy.cuda.Stream)
        '''
        self._prepare_copy(out_arr, stream, direction='out')


cdef class TextureObject:
    def __init__(self, ResourceDescriptor ResDesc, TextureDescriptor TexDesc):
        self.ptr = runtime.createTextureObject(ResDesc.ptr, TexDesc.ptr)
        self.ResDesc = ResDesc
        self.TexDesc = TexDesc

    def __dealloc__(self):
        runtime.destroyTextureObject(self.ptr)
        self.ptr = 0
