from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.string cimport memset as c_memset

import numpy

from cupy._core.core cimport ndarray
from cupy_backends.cuda.api cimport driver
from cupy_backends.cuda.api cimport runtime
from cupy_backends.cuda.api.runtime cimport Array,\
    ChannelFormatDesc, ChannelFormatKind,\
    Memcpy3DParms, MemoryKind, PitchedPtr, ResourceDesc, ResourceType, \
    TextureAddressMode, TextureDesc, TextureFilterMode, TextureReadMode
from cupy.cuda cimport stream as stream_module

from cupy_backends.cuda.api.runtime import CUDARuntimeError


cdef extern from '../../cupy_backends/cupy_backend.h':
    pass

cdef extern from '../../cupy_backends/cupy_backend_runtime.h':
    pass


cdef class ChannelFormatDescriptor:
    '''A class that holds the channel format description. Equivalent to
    ``cudaChannelFormatDesc``.

    Args:
        x (int): the number of bits for the x channel.
        y (int): the number of bits for the y channel.
        z (int): the number of bits for the z channel.
        w (int): the number of bits for the w channel.
        f (int): the channel format. Use one of the values in ``cudaChannelFormat*``,
            such as :const:`cupy.cuda.runtime.cudaChannelFormatKindFloat`.

    .. seealso:: `cudaCreateChannelDesc()`_

    .. _cudaCreateChannelDesc():
        https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TEXTURE__OBJECT.html#group__CUDART__TEXTURE__OBJECT_1g39df9e3b6edc41cd6f189d2109672ca5
    '''  # noqa
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
        '''Returns a dict containing the input.'''
        cdef dict desc = {}
        desc['x'] = (<ChannelFormatDesc*>self.ptr).x
        desc['y'] = (<ChannelFormatDesc*>self.ptr).y
        desc['z'] = (<ChannelFormatDesc*>self.ptr).z
        desc['w'] = (<ChannelFormatDesc*>self.ptr).w
        desc['f'] = (<ChannelFormatDesc*>self.ptr).f
        return desc


cdef class ResourceDescriptor:
    '''A class that holds the resource description. Equivalent to
    ``cudaResourceDesc``.

    Args:
        restype (int): the resource type. Use one of the values in
            ``cudaResourceType*``, such as
            :const:`cupy.cuda.runtime.cudaResourceTypeArray`.
        cuArr (CUDAarray, optional): An instance of :class:`CUDAarray`,
            required if ``restype`` is set to
            :const:`cupy.cuda.runtime.cudaResourceTypeArray`.
        arr (cupy.ndarray, optional): An instance of :class:`~cupy.ndarray`,
            required if ``restype`` is set to
            :const:`cupy.cuda.runtime.cudaResourceTypeLinear` or
            :const:`cupy.cuda.runtime.cudaResourceTypePitch2D`.
        chDesc (ChannelFormatDescriptor, optional): an instance of
            :class:`ChannelFormatDescriptor`, required if ``restype`` is set to
            :const:`cupy.cuda.runtime.cudaResourceTypeLinear` or
            :const:`cupy.cuda.runtime.cudaResourceTypePitch2D`.
        sizeInBytes (int, optional): total bytes in the linear memory, required
            if ``restype`` is set to
            :const:`cupy.cuda.runtime.cudaResourceTypeLinear`.
        width (int, optional): the width (in elements) of the 2D array,
            required if ``restype`` is set to
            :const:`cupy.cuda.runtime.cudaResourceTypePitch2D`.
        height (int, optional): the height (in elements) of the 2D array,
            required if ``restype`` is set to
            :const:`cupy.cuda.runtime.cudaResourceTypePitch2D`.
        pitchInBytes (int, optional): the number of bytes per pitch-aligned row,
            required if ``restype`` is set to
            :const:`cupy.cuda.runtime.cudaResourceTypePitch2D`.

    .. note::
        A texture backed by `mipmap` arrays is currently not supported in CuPy.

    .. seealso:: `cudaCreateTextureObject()`_

    .. _cudaCreateTextureObject():
        https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TEXTURE__OBJECT.html#group__CUDART__TEXTURE__OBJECT_1g16ac75814780c3a16e4c63869feb9ad3
    '''  # noqa
    def __init__(self, int restype, CUDAarray cuArr=None, ndarray arr=None,
                 ChannelFormatDescriptor chDesc=None, size_t sizeInBytes=0,
                 size_t width=0, size_t height=0, size_t pitchInBytes=0):
        if restype == runtime.cudaResourceTypeMipmappedArray:
            # TODO(leofang): support this?
            raise NotImplementedError('cudaResourceTypeMipmappedArray is '
                                      'currently not supported.')

        self.ptr = <intptr_t>PyMem_Malloc(sizeof(ResourceDesc))
        cdef ResourceDesc* desc = (<ResourceDesc*>self.ptr)
        c_memset(desc, 0, sizeof(ResourceDesc))

        desc.resType = <ResourceType>restype
        if restype == runtime.cudaResourceTypeArray:
            desc.res.array.array = <Array>(cuArr.ptr)
        elif restype == runtime.cudaResourceTypeLinear:
            desc.res.linear.devPtr = <void*>(arr.data.ptr)
            desc.res.linear.desc = (<ChannelFormatDesc*>chDesc.ptr)[0]
            desc.res.linear.sizeInBytes = sizeInBytes
        elif restype == runtime.cudaResourceTypePitch2D:
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
        '''Returns a dict containing the input.'''
        cdef dict desc = {}
        cdef intptr_t ptr
        cdef ResourceDesc* resPtr = <ResourceDesc*>(self.ptr)
        cdef size_t size, pitch, w, h

        # For texture memory, print the underlying pointer address so that
        # it can be used for verification by the caller. Note that resPtr.res
        # is a union, so ptr is always there for every category, which could
        # be confusing and thus need a logic to make selections.
        desc['resType'] = resPtr.resType
        if resPtr.resType == 0:
            ptr = <intptr_t>(resPtr.res.array.array)
            desc['array'] = {'array': ptr}
        elif resPtr.resType == 2:
            ptr = <intptr_t>(resPtr.res.linear.devPtr)
            desc['linear'] = {'devPtr': ptr,
                              'desc': self.chDesc.get_channel_format(),
                              'sizeInBytes': resPtr.res.linear.sizeInBytes}
        elif resPtr.resType == 3:
            ptr = <intptr_t>(resPtr.res.pitch2D.devPtr)
            desc['pitch2D'] = {'devPtr': ptr,
                               'desc': self.chDesc.get_channel_format(),
                               'width': resPtr.res.pitch2D.width,
                               'height': resPtr.res.pitch2D.height,
                               'pitchInBytes': resPtr.res.pitch2D.pitchInBytes}
        return desc


cdef class TextureDescriptor:
    '''A class that holds the texture description. Equivalent to
    ``cudaTextureDesc``.

    Args:
        addressModes (tuple or list): an iterable with length up to 3, each
            element is one of the values in ``cudaAddressMode*``, such as
            :const:`cupy.cuda.runtime.cudaAddressModeWrap`.
        filterMode (int): the filter mode. Use one of the values in
            ``cudaFilterMode*``, such as
            :const:`cupy.cuda.runtime.cudaFilterModePoint`.
        readMode (int): the read mode. Use one of the values in
            ``cudaReadMode*``, such as
            :const:`cupy.cuda.runtime.cudaReadModeElementType`.
        normalizedCoords (int): whether coordinates are normalized or not.
        sRGB (int, optional)
        borderColors (tuple or list, optional): an iterable with length up to
            4.
        maxAnisotropy (int, optional)

    .. note::
        A texture backed by `mipmap` arrays is currently not supported in CuPy.

    .. seealso:: `cudaCreateTextureObject()`_

    .. _cudaCreateTextureObject():
        https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TEXTURE__OBJECT.html#group__CUDART__TEXTURE__OBJECT_1g16ac75814780c3a16e4c63869feb9ad3
    '''  # noqa
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
        '''Returns a dict containing the input.'''
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
    '''Allocate a CUDA array (`cudaArray_t`) that can be used as texture memory.
    Depending on the input, either 1D, 2D, or 3D CUDA array is returned.

    Args:
        desc (ChannelFormatDescriptor): an instance of
            :class:`ChannelFormatDescriptor`.
        width (int): the width (in elements) of the array.
        height (int, optional): the height (in elements) of the array.
        depth (int, optional): the depth (in elements) of the array.
        flags (int, optional): the flag for extensions. Use one of the values
            in ``cudaArray*``, such as
            :const:`cupy.cuda.runtime.cudaArrayDefault`.

    .. warning::
        The memory allocation of :class:`CUDAarray` is done outside of CuPy's
        memory management (enabled by default) due to CUDA's limitation. Users
        of :class:`CUDAarray` should be cautious about any out-of-memory
        possibilities.

    .. seealso:: `cudaMalloc3DArray()`_

    .. _cudaMalloc3DArray():
        https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g948143cf2423a072ac6a31fb635efd88
    '''  # noqa
    # TODO(leofang): perhaps this wrapper is not needed when cupy.ndarray
    # can be backed by texture memory/CUDA arrays?
    def __init__(self, ChannelFormatDescriptor desc, size_t width,
                 size_t height=0, size_t depth=0, unsigned int flags=0):
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
        cdef dict ch = self.desc.get_channel_format()

        # sanity checks:
        # - check shape
        cdef int num_channels = 0
        for key in ['x', 'y', 'z', 'w']:
            if ch[key] > 0:
                num_channels += 1

        if self.ndim == 3:
            if arr.shape != (self.depth, self.height, num_channels*self.width):
                raise ValueError("shape mismatch")
        elif self.ndim == 2:
            if arr.shape != (self.height, num_channels*self.width):
                raise ValueError("shape mismatch")
        else:  # ndim = 1
            if arr.shape != (num_channels*self.width,):
                raise ValueError("shape mismatch")

        # - check dtype
        ch_kind = ch['f']
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

        # we need to serialize on the current or given stream to ensure
        # data is ready
        cdef intptr_t stream_ptr
        try:
            if stream is None:
                stream_ptr = stream_module.get_current_stream_ptr()
            else:
                stream_ptr = <intptr_t>(stream.ptr)
            runtime.memcpy3DAsync(<intptr_t>param, stream_ptr)
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
            in_arr (cupy.ndarray or numpy.ndarray)
            stream (cupy.cuda.Stream): if not ``None``, an asynchronous copy is
                performed.

        .. note::
            For CUDA arrays with different dimensions, the requirements for the
            shape of the input array are given as follows:

                - 1D: ``(nch * width,)``
                - 2D: ``(height, nch * width)``
                - 3D: ``(depth, height, nch * width)``

            where ``nch`` is the number of channels specified in
            :attr:`~CUDAarray.desc`.
        '''
        self._prepare_copy(in_arr, stream, direction='in')

    def copy_to(self, out_arr, stream=None):
        '''Copy data from CUDA array to device or host array.

        Args:
            out_arr (cupy.ndarray or numpy.ndarray)
            stream (cupy.cuda.Stream): if not ``None``, an asynchronous copy is
                performed.

        .. note::
            For CUDA arrays with different dimensions, the requirements for the
            shape of the output array are given as follows:

                - 1D: ``(nch * width,)``
                - 2D: ``(height, nch * width)``
                - 3D: ``(depth, height, nch * width)``

            where ``nch`` is the number of channels specified in
            :attr:`~CUDAarray.desc`.
        '''
        self._prepare_copy(out_arr, stream, direction='out')


cdef class TextureObject:
    '''A class that holds a texture object. Equivalent to
    ``cudaTextureObject_t``. The returned :class:`TextureObject` instance can
    be passed as a argument when launching :class:`~cupy.RawKernel` or
    :class:`~cupy.ElementwiseKernel`.

    Args:
        ResDesc (ResourceDescriptor): an intance of the resource descriptor.
        TexDesc (TextureDescriptor): an instance of the texture descriptor.

    .. seealso:: `cudaCreateTextureObject()`_

    .. _cudaCreateTextureObject():
        https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TEXTURE__OBJECT.html#group__CUDART__TEXTURE__OBJECT_1g16ac75814780c3a16e4c63869feb9ad3
    '''  # noqa
    def __init__(self, ResourceDescriptor ResDesc, TextureDescriptor TexDesc):
        self.ptr = runtime.createTextureObject(ResDesc.ptr, TexDesc.ptr)
        self.ResDesc = ResDesc
        self.TexDesc = TexDesc

    def __dealloc__(self):
        runtime.destroyTextureObject(self.ptr)
        self.ptr = 0


cdef class SurfaceObject:
    '''A class that holds a surface object. Equivalent to
    ``cudaSurfaceObject_t``. The returned :class:`SurfaceObject` instance can
    be passed as a argument when launching :class:`~cupy.RawKernel`.

    Args:
        ResDesc (ResourceDescriptor): an intance of the resource descriptor.

    .. seealso:: `cudaCreateSurfaceObject()`_

    .. _cudaCreateSurfaceObject():
        https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__SURFACE__OBJECT.html#group__CUDART__SURFACE__OBJECT_1g958899474ab2c5f40d233b524d6c5a01
    '''  # noqa
    def __init__(self, ResourceDescriptor ResDesc):
        self.ptr = runtime.createSurfaceObject(ResDesc.ptr)
        self.ResDesc = ResDesc

    def __dealloc__(self):
        runtime.destroySurfaceObject(self.ptr)
        self.ptr = 0


cdef class TextureReference:
    '''A class that holds a texture reference. Equivalent to ``CUtexref`` (the
    driver API is used under the hood).

    Args:
        texref (intptr_t): a handle to the texture reference declared in the
            CUDA source code. This can be obtained by calling
            :meth:`~cupy.RawModule.get_texref`.
        ResDesc (ResourceDescriptor): an intance of the resource descriptor.
        TexDesc (TextureDescriptor): an instance of the texture descriptor.

    .. warning::
        As of CUDA Toolkit v10.1, the Texture Reference API (in both driver and
        runtime) is marked as deprecated. To help transition to the new Texture
        Object API, this class mimics the usage of
        :class:`~cupy.cuda.texture.TextureObject`. Users who have legacy CUDA
        codes that use texture references should consider migration to the new
        API.

        This CuPy interface is subject to removal once the offcial NVIDIA
        support is dropped in the future.

    .. seealso:: :class:`TextureObject`, `cudaCreateTextureObject()`_

    .. _cudaCreateTextureObject():
        https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TEXTURE__OBJECT.html#group__CUDART__TEXTURE__OBJECT_1g16ac75814780c3a16e4c63869feb9ad3
    '''  # noqa
    # Basically, this class translates from the Runtime API's descriptors
    # to the driver API calls.
    def __init__(self, intptr_t texref, ResourceDescriptor ResDesc,
                 TextureDescriptor TexDesc):
        cdef ResourceDesc* ResDescPtr = <ResourceDesc*>(ResDesc.ptr)
        cdef TextureDesc* TexDescPtr = <TextureDesc*>(TexDesc.ptr)
        cdef ChannelFormatDescriptor ChDesc
        cdef int arr_format, num_channels
        cdef driver.Array_desc arr_desc

        self.texref = texref
        self.ResDesc = ResDesc
        self.TexDesc = TexDesc

        if ResDescPtr.resType == runtime.cudaResourceTypeArray:
            driver.texRefSetArray(texref,
                                  <intptr_t>(ResDescPtr.res.array.array))
            ChDesc = ResDesc.cuArr.desc
            arr_format, num_channels = \
                self._get_format(ChDesc.get_channel_format())
            driver.texRefSetFormat(self.texref, arr_format, num_channels)
        elif ResDescPtr.resType == runtime.cudaResourceTypeLinear:
            driver.texRefSetAddress(texref,
                                    <intptr_t>(ResDescPtr.res.linear.devPtr),
                                    ResDescPtr.res.linear.sizeInBytes)
            ChDesc = ResDesc.chDesc
            arr_format, num_channels = \
                self._get_format(ChDesc.get_channel_format())
            driver.texRefSetFormat(self.texref, arr_format, num_channels)
        elif ResDescPtr.resType == runtime.cudaResourceTypePitch2D:
            ChDesc = ResDesc.chDesc
            arr_format, num_channels = \
                self._get_format(ChDesc.get_channel_format())
            arr_desc.Format = <driver.Array_format>arr_format
            arr_desc.NumChannels = num_channels
            arr_desc.Height = ResDescPtr.res.pitch2D.height
            arr_desc.Width = ResDescPtr.res.pitch2D.width
            driver.texRefSetAddress2D(
                texref, <intptr_t>(&arr_desc),
                <intptr_t>(ResDescPtr.res.pitch2D.devPtr),
                ResDescPtr.res.pitch2D.pitchInBytes)
            # don't call driver.texRefSetFormat() here!
        else:  # TODO(leofang): mipmap
            raise ValueError("mpimap array is not supported yet.")

        # For the following attributes, the constants in driver and runtime
        # are set to equal, so using the values set by runtime api is OK.
        for i in range(3):
            driver.texRefSetAddressMode(texref, i, TexDescPtr.addressMode[i])

        driver.texRefSetFilterMode(texref, TexDescPtr.filterMode)

        cdef int flag = 0x00
        if TexDescPtr.readMode == <int>(runtime.cudaReadModeElementType):
            flag = flag | driver.CU_TRSF_READ_AS_INTEGER
        if TexDescPtr.normalizedCoords:
            flag = flag | driver.CU_TRSF_NORMALIZED_COORDINATES
        if TexDescPtr.sRGB:
            flag = flag | driver.CU_TRSF_SRGB
        driver.texRefSetFlags(texref, flag)

        driver.texRefSetBorderColor(texref, TexDescPtr.borderColor)
        driver.texRefSetMaxAnisotropy(texref, TexDescPtr.maxAnisotropy)

    cdef _get_format(self, dict ch_format):
        cdef int arr_format
        cdef int num_channels = 0

        for key in ['x', 'y', 'z', 'w']:
            if ch_format[key] > 0:
                assert ch_format[key] == ch_format['x']
                num_channels += 1

        if ch_format['f'] == runtime.cudaChannelFormatKindSigned:
            if ch_format['x'] // 8 == 1:
                arr_format = driver.CU_AD_FORMAT_SIGNED_INT8
            elif ch_format['x'] // 8 == 2:
                arr_format = driver.CU_AD_FORMAT_SIGNED_INT16
            elif ch_format['x'] // 8 == 4:
                arr_format = driver.CU_AD_FORMAT_SIGNED_INT32
            else:
                raise ValueError("format mismatch")
        elif ch_format['f'] == runtime.cudaChannelFormatKindUnsigned:
            if ch_format['x'] // 8 == 1:
                arr_format = driver.CU_AD_FORMAT_UNSIGNED_INT8
            elif ch_format['x'] // 8 == 2:
                arr_format = driver.CU_AD_FORMAT_UNSIGNED_INT16
            elif ch_format['x'] // 8 == 4:
                arr_format = driver.CU_AD_FORMAT_UNSIGNED_INT32
            else:
                raise ValueError("format mismatch")
        elif ch_format['f'] == runtime.cudaChannelFormatKindFloat:
            if ch_format['x'] // 8 == 2:
                arr_format = driver.CU_AD_FORMAT_HALF
            elif ch_format['x'] // 8 == 4:
                arr_format = driver.CU_AD_FORMAT_FLOAT
            else:
                raise ValueError("format mismatch")
        else:
            raise ValueError("format not recognized")

        return (arr_format, num_channels)
