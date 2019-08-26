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
        elif width > 0:
            if depth > 0:
                self.ptr = runtime.malloc3DArray(desc, width, height, depth,
                                                 flags)
            else:
                self.ptr = runtime.mallocArray(desc, width, height, flags)

    def __dealloc__(self):
        if self.ptr:
            runtime.freeArray(self.ptr)


cdef class TextureObject:
    # GOAL: make this pass-able to RawKernel
    def __init__(self, runtime.ResourceDescriptor ResDesc,
                 runtime.TextureDescriptor TexDesc):
        self.ptr = runtime.createTextureObject(ResDesc, TexDesc)
        
    def __dealloc__(self):
        runtime.destroyTextureObject(self.ptr)
