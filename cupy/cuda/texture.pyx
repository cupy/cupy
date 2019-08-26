from cupy.cuda cimport runtime


cdef class CUDAArray:
    # TODO(leofang): perhaps this wrapper is not needed when cupy.ndarray
    # can be backed by texture memory/CUDA arrays?
    def __init__(self):
        self.ptr = 0


cdef class TextureObject:
    # GOAL: make this pass-able to RawKernel
    def __init__(self, runtime.ResourceDescriptor ResDesc,
                 runtime.TextureDescriptor TexDesc):
        # TextureObject is already an opaque type, no need to re-cast the ptr
        self.ptr = runtime.createTextureObject(ResDesc, TexDesc)
        
    def __dealloc__(self):
        runtime.destroyTextureObject(self.ptr)
