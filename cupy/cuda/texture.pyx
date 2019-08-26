from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.stdint cimport intptr_t
from libc.string cimport memset as c_memset

import numpy

import cupy
from cupy.cuda cimport device
from cupy.cuda cimport runtime
from cupy.cuda.memory cimport BaseMemory



cdef class CUDAArray(BaseMemory):
    # TODO(leofang): perhaps this wrapper is not needed when cupy.ndarray
    # can be backed by texture memory/CUDA arrays?
    def __init__(self, runtime.ChannelFormatDescriptor desc, size_t width,
                 size_t height, size_t depth=0, unsigned int flags=0):
        self.device_id = device.get_device_id()

        if width == 0:
            raise ValueError('To create a 1D or 2D CUDA array, width must be '
                             'nonzero.')
        elif height == 0 and depth > 0:
            raise ValueError
        else:
            self.ptr = runtime.malloc3DArray(desc.ptr, width, height, depth,
                                             flags)

        # bookkeeping
        self.desc = desc
        self.width = width
        self.height = height
        self.depth = depth
        self.flags = flags
        if self.depth > 0:
            self.ndim = 3
        elif self.height > 0:
            self.ndim = 2
        else:
            self.ndim = 1

    def __dealloc__(self):
        if self.ptr:
            runtime.freeArray(self.ptr)
            self.ptr = 0

    cdef runtime.Memcpy3DParms* _make_cudaMemcpy3DParms(self, src, dst):
        cdef runtime.Memcpy3DParms* param = \
            <runtime.Memcpy3DParms*>PyMem_Malloc(sizeof(runtime.Memcpy3DParms))
        c_memset(param, 0, sizeof(runtime.Memcpy3DParms))
        cdef runtime.Array srcArrayPtr, dstArrayPtr
        cdef runtime.PitchedPtr srcPitchedPtr, dstPitchedPtr
        cdef intptr_t ptr 

        # get kind
        if isinstance(src, cupy.core.core.ndarray) and isinstance(dst, type(self)):
            param.kind = <runtime.MemoryKind>runtime.memcpyDeviceToDevice
        elif isinstance(src, type(self)) and isinstance(dst, cupy.core.core.ndarray):
            param.kind = <runtime.MemoryKind>runtime.memcpyDeviceToDevice
        elif isinstance(src, type(self)) and isinstance(dst, numpy.ndarray):
            param.kind = <runtime.MemoryKind>runtime.memcpyDeviceToHost
        elif isinstance(src, numpy.ndarray) and isinstance(dst, type(self)):
            param.kind = <runtime.MemoryKind>runtime.memcpyHostToDevice
        else:
            raise ValueError("Either source or destination must be a CUDAArray instance.")

        # get src
        if isinstance(src, type(self)):
            srcArrayPtr = <runtime.Array>(src.ptr)
            param.srcArray = srcArrayPtr
            param.extent = runtime.make_Extent(src.width, src.height,
                                               src.depth)
            print("src ptr:", <intptr_t>srcArrayPtr)
        else:
            width = src.shape[0]
            if src.ndim >= 2:
                height = src.shape[1]
            else:
                height = 0

            if isinstance(src, cupy.core.core.ndarray):
                ptr = src.data.ptr
            else:  # numpy.ndarray
                ptr = src.ctypes.data
            print("src ptr:", ptr)

            srcPitchedPtr = runtime.make_PitchedPtr(
                ptr, width*src.dtype.itemsize, width, height)
            param.srcPtr = srcPitchedPtr
        param.srcPos = runtime.make_Pos(0, 0, 0)

        # get dst
        if isinstance(dst, type(self)):
            dstArrayPtr = <runtime.Array>(dst.ptr)
            param.dstArray = dstArrayPtr
            param.extent = runtime.make_Extent(dst.width, dst.height,
                                               dst.depth)
            print("dst ptr:", <intptr_t>dstArrayPtr)
        else:
            width = dst.shape[0]
            if dst.ndim >= 2:
                height = dst.shape[1]
            else:
                height = 0

            if isinstance(dst, cupy.core.core.ndarray):
                ptr = dst.data.ptr
            else:  # numpy.ndarray
                ptr = dst.ctypes.data
            print("dst ptr:", ptr)

            dstPitchedPtr = runtime.make_PitchedPtr(
                ptr, width*dst.dtype.itemsize, width, height)
            param.dstPtr = dstPitchedPtr
        param.dstPos = runtime.make_Pos(0, 0, 0)

        return param

    def _to_array(self, out_arr, stream):
        '''
        Args:
            out_arr (cupy.core.core.ndarray or numpy.ndarray)
        '''
        # sanity checks:
        # - check shape
        if self.depth > 0:
            if out_arr.shape != (self.width, self.height, self.depth):
                raise ValueError
        elif self.height > 0:
            if out_arr.shape != (self.width, self.height):
                raise ValueError
        else:
            if out_arr.shape != (self.width,):
                raise ValueError

        # - check dtype
        if out_arr.dtype != numpy.float32:
            raise ValueError

        print("prepare to make param")
        cdef runtime.Memcpy3DParms* param
        param = self._make_cudaMemcpy3DParms(self, out_arr)
        print("print param")
        self._print_param(param)

        runtime._ensure_context()
        try:
            if stream is None:
                runtime.memcpy3D(<intptr_t>param)
            else:
                runtime.memcpy3DAsync(<intptr_t>param, stream.ptr)
        except:
            raise
        finally:
            PyMem_Free(param)

    def _from_array(self, in_arr, stream):
        '''
        Args:
            in_arr (cupy.core.core.ndarray or numpy.ndarray)
        '''
        # sanity checks:
        # - check shape
        if self.depth > 0:
            if in_arr.shape != (self.width, self.height, self.depth):
                raise ValueError
        elif self.height > 0:
            if in_arr.shape != (self.width, self.height):
                raise ValueError
        else:
            if in_arr.shape != (self.width,):
                raise ValueError

        # - check dtype
        if in_arr.dtype != numpy.float32:
            raise ValueError

        print("prepare to make param")
        cdef runtime.Memcpy3DParms* param
        param = self._make_cudaMemcpy3DParms(in_arr, self)
        print("print param")
        self._print_param(param)

        runtime._ensure_context()
        try:
            if stream is None:
                runtime.memcpy3D(<intptr_t>param)
            else:
                runtime.memcpy3DAsync(<intptr_t>param, stream.ptr)
        except:
            raise
        finally:
            PyMem_Free(param)

    cdef _print_param(self, runtime.Memcpy3DParms* param):
        print(<intptr_t>param.srcArray)
        print(param.srcPos.x, param.srcPos.y, param.srcPos.z)
        print(<intptr_t>param.srcPtr.ptr, param.srcPtr.pitch, param.srcPtr.xsize, param.srcPtr.ysize)
        
        print(<intptr_t>param.dstArray)
        print(param.dstPos.x, param.dstPos.y, param.dstPos.z)
        print(<intptr_t>param.dstPtr.ptr, param.dstPtr.pitch, param.dstPtr.xsize, param.dstPtr.ysize)

        print(param.extent.width, param.extent.height, param.extent.depth)
        print(param.kind)
        print('\n', flush=True)

    def copy_from(self, in_arr, stream=None):
        self._from_array(in_arr, stream)

    def copy_to(self, out_arr, stream=None):
        self._to_array(out_arr, stream)


cdef class TextureObject:
    # GOAL: make this pass-able to RawKernel
    def __init__(self, runtime.ResourceDescriptor ResDesc,
                 runtime.TextureDescriptor TexDesc):
        self.ptr = runtime.createTextureObject(ResDesc.ptr, TexDesc.ptr)
        
    def __dealloc__(self):
        if self.ptr:
            runtime.destroyTextureObject(self.ptr)
            self.ptr = 0
