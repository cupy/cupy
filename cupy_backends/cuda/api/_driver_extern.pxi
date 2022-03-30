cdef extern from '../../cupy_backend.h' nogil:
    # Error handling
    int cuGetErrorName(Result error, const char** pStr)
    int cuGetErrorString(Result error, const char** pStr)

    # Primary context management
    int cuDevicePrimaryCtxRelease(Device dev)

    # Context management
    int cuCtxGetCurrent(Context* pctx)
    int cuCtxSetCurrent(Context ctx)
    int cuCtxCreate(Context* pctx, unsigned int flags, Device dev)
    int cuCtxDestroy(Context ctx)
    int cuCtxGetDevice(Device*)

    # Module load and kernel execution
    int cuLinkCreate(unsigned int numOptions, CUjit_option* options,
                     void** optionValues, LinkState* stateOut)
    int cuLinkAddData(LinkState state, CUjitInputType type, void* data,
                      size_t size, const char* name, unsigned int  numOptions,
                      CUjit_option* options, void** optionValues)
    int cuLinkAddFile(LinkState state, CUjitInputType type, const char* path,
                      unsigned int numOptions, CUjit_option* options, void**
                      optionValues)
    int cuLinkComplete(LinkState state, void** cubinOut, size_t* sizeOut)
    int cuLinkDestroy(LinkState state)
    int cuModuleLoad(Module* module, char* fname)
    int cuModuleLoadData(Module* module, void* image)
    int cuModuleUnload(Module hmod)
    int cuModuleGetFunction(Function* hfunc, Module hmod,
                            char* name)
    int cuModuleGetGlobal(Deviceptr* dptr, size_t* bytes, Module hmod,
                          char* name)
    int cuModuleGetTexRef(TexRef* pTexRef, Module hmod, const char* name)
    int cuLaunchKernel(
        Function f, unsigned int gridDimX, unsigned int gridDimY,
        unsigned int gridDimZ, unsigned int blockDimX,
        unsigned int blockDimY, unsigned int blockDimZ,
        unsigned int sharedMemBytes, Stream hStream,
        void** kernelParams, void** extra)

    int cuLaunchCooperativeKernel(
        Function f, unsigned int gridDimX, unsigned int gridDimY,
        unsigned int gridDimZ, unsigned int blockDimX,
        unsigned int blockDimY, unsigned int blockDimZ,
        unsigned int sharedMemBytes, Stream hStream,
        void** kernelParams)

    # Kernel attributes
    int cuFuncGetAttribute(int *pi, CUfunction_attribute attrib,
                           Function hfunc)

    int cuFuncSetAttribute(Function hfunc, CUfunction_attribute attrib,
                           int value)

    # Texture reference
    int cuTexRefSetAddress(size_t* ByteOffset, TexRef hTexRef, Deviceptr dptr,
                           size_t bytes)
    int cuTexRefSetAddress2D(TexRef hTexRef, const Array_desc* desc,
                             Deviceptr dptr, size_t Pitch)
    int cuTexRefSetAddressMode(TexRef hTexRef, int dim, Address_mode am)
    int cuTexRefSetArray(TexRef hTexRef, Array hArray, unsigned int Flags)
    int cuTexRefSetBorderColor(TexRef hTexRef, float* pBorderColor)
    int cuTexRefSetFilterMode(TexRef hTexRef, Filter_mode fm)
    int cuTexRefSetFlags(TexRef hTexRef, unsigned int Flags)
    int cuTexRefSetFormat(TexRef hTexRef, Array_format fmt,
                          int NumPackedComponents)
    int cuTexRefSetMaxAnisotropy(TexRef hTexRef, unsigned int maxAniso)

    # Occupancy
    int cuOccupancyMaxActiveBlocksPerMultiprocessor(
        int* numBlocks, Function func, int blockSize, size_t dynamicSMemSize)
    int cuOccupancyMaxPotentialBlockSize(
        int* minGridSize, int* blockSize, Function func, CUoccupancyB2DSize
        block2shmem, size_t dynamicSMemSize, int blockSizeLimit)

    # Stream
    int cuStreamGetCtx(Stream hStream, Context* pctx)

    # Build-time version
    enum: CUDA_VERSION
