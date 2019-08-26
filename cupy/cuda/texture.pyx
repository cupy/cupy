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
        elif width > 0:
            if depth > 0:
                self.ptr = runtime.malloc3DArray(desc.ptr, width, height,
                                                 depth, flags)
            else:
                self.ptr = runtime.mallocArray(desc.ptr, width, height, flags)

        # bookkeeping
        self.desc = desc
        self.width = width
        self.height = height
        self.depth = depth
        self.flags = flags

    def __dealloc__(self):
        if self.ptr:
            runtime.freeArray(self.ptr)
            self.ptr = 0

#    def to_array(self, out=None):
#        if out is None:
#            if self.depth > 0:
#                shape = (self.width, self.height, self.depth)
#            elif self.height > 0:
#                shape = (self.width, self.height)
#            else:
#                shape = (self.width,)
#            out = cupy.zeros(shape, dtype=cupy.float32)
#        
#        #runtime.memcpy2DToArray(self.ptr, 


cdef class TextureObject:
    # GOAL: make this pass-able to RawKernel
    def __init__(self, runtime.ResourceDescriptor ResDesc,
                 runtime.TextureDescriptor TexDesc):
        self.ptr = runtime.createTextureObject(ResDesc.ptr, TexDesc.ptr)
        
    def __dealloc__(self):
        if self.ptr:
            runtime.destroyTextureObject(self.ptr)
            self.ptr = 0
