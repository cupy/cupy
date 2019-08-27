from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.stdint cimport intptr_t
from libc.string cimport memset as c_memset

import numpy

import cupy
from cupy.cuda cimport device
from cupy.cuda cimport runtime
from cupy.cuda.memory cimport BaseMemory
from cupy.cuda.runtime import CUDARuntimeError


cdef class CUDAArray(BaseMemory):
    # TODO(leofang): perhaps this wrapper is not needed when cupy.ndarray
    # can be backed by texture memory/CUDA arrays?
    def __init__(self, runtime.ChannelFormatDescriptor desc, size_t width,
                 size_t height, size_t depth=0, unsigned int flags=0):
        self.device_id = device.get_device_id()

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
        if self.ptr:
            runtime.freeArray(self.ptr)
            self.ptr = 0

    cdef int _get_kind(self, src, dst):
        cdef int kind
        if isinstance(src, cupy.core.core.ndarray) and dst is self:
            kind = runtime.memcpyDeviceToDevice
        elif src is self and isinstance(dst, cupy.core.core.ndarray):
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
        cdef runtime.Memcpy3DParms* param = \
            <runtime.Memcpy3DParms*>PyMem_Malloc(sizeof(runtime.Memcpy3DParms))
        c_memset(param, 0, sizeof(runtime.Memcpy3DParms))
        cdef runtime.PitchedPtr srcPitchedPtr, dstPitchedPtr
        cdef intptr_t ptr

        # get kind
        param.kind = <runtime.MemoryKind>self._get_kind(src, dst)

        # get src
        if src is self:
            # Important: cannot convert from src.ptr!
            param.srcArray = <runtime.Array>(self.ptr)
            param.extent = runtime.make_Extent(self.width, self.height,
                                               self.depth)
        else:
            width = src.shape[-1]
            if src.ndim >= 2:
                height = src.shape[-2]
            else:
                height = 1  # same "one-stride" trick here

            if isinstance(src, cupy.core.core.ndarray):
                ptr = src.data.ptr
            else:  # numpy.ndarray
                ptr = src.ctypes.data

            srcPitchedPtr = runtime.make_PitchedPtr(
                ptr, width*src.dtype.itemsize, width, height)
            param.srcPtr = srcPitchedPtr

        # get dst
        if dst is self:
            # Important: cannot convert from dst.ptr!
            param.dstArray = <runtime.Array>(self.ptr)
            param.extent = runtime.make_Extent(self.width, self.height,
                                               self.depth)
        else:
            width = dst.shape[-1]
            if dst.ndim >= 2:
                height = dst.shape[-2]
            else:
                height = 1  # same "one-stride" trick here

            if isinstance(dst, cupy.core.core.ndarray):
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
        # TODO(leofang): support signed and unsigned
        if arr.dtype != numpy.float32:
            raise ValueError("Currently only float32 is supported")

        cdef runtime.Memcpy3DParms* param = NULL

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
            param = <runtime.Memcpy3DParms*>self._make_cudaMemcpy3DParms(arr,
                                                                         self)
        elif direction == 'out':
            param = <runtime.Memcpy3DParms*>self._make_cudaMemcpy3DParms(self,
                                                                         arr)
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
    # GOAL: make this pass-able to RawKernel
    def __init__(self, runtime.ResourceDescriptor ResDesc,
                 runtime.TextureDescriptor TexDesc):
        self.ptr = runtime.createTextureObject(ResDesc.ptr, TexDesc.ptr)
        self.ResDesc = ResDesc
        self.TexDesc = TexDesc

    def __dealloc__(self):
        if self.ptr:
            runtime.destroyTextureObject(self.ptr)
            self.ptr = 0
