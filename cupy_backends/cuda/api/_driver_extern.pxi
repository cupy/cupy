import sys as _sys

from cupy_backends.cuda._softlink cimport SoftLink


# Error handling
ctypedef int (*F_cuGetErrorName)(Result error, const char** pStr) nogil
cdef F_cuGetErrorName cuGetErrorName
ctypedef int (*F_cuGetErrorString)(Result error, const char** pStr) nogil
cdef F_cuGetErrorString cuGetErrorString

# Primary context management
ctypedef int (*F_cuDevicePrimaryCtxRelease)(Device dev) nogil
cdef F_cuDevicePrimaryCtxRelease cuDevicePrimaryCtxRelease

# Context management
ctypedef int (*F_cuCtxGetCurrent)(Context* pctx) nogil
cdef F_cuCtxGetCurrent cuCtxGetCurrent
ctypedef int (*F_cuCtxSetCurrent)(Context ctx) nogil
cdef F_cuCtxSetCurrent cuCtxSetCurrent
ctypedef int (*F_cuCtxCreate)(Context* pctx, unsigned int flags, Device dev) nogil  # NOQA
cdef F_cuCtxCreate cuCtxCreate
ctypedef int (*F_cuCtxDestroy)(Context ctx) nogil
cdef F_cuCtxDestroy cuCtxDestroy
ctypedef int (*F_cuCtxGetDevice)(Device*) nogil
cdef F_cuCtxGetDevice cuCtxGetDevice

# Module load and kernel execution
ctypedef int (*F_cuLinkCreate)(unsigned int numOptions, CUjit_option* options, void** optionValues, LinkState* stateOut) nogil  # NOQA
cdef F_cuLinkCreate cuLinkCreate
ctypedef int (*F_cuLinkAddData)(LinkState state, CUjitInputType type, void* data, size_t size, const char* name, unsigned int  numOptions, CUjit_option* options, void** optionValues) nogil  # NOQA
cdef F_cuLinkAddData cuLinkAddData
ctypedef int (*F_cuLinkAddFile)(LinkState state, CUjitInputType type, const char* path, unsigned int numOptions, CUjit_option* options, void** optionValues) nogil  # NOQA
cdef F_cuLinkAddFile cuLinkAddFile
ctypedef int (*F_cuLinkComplete)(LinkState state, void** cubinOut, size_t* sizeOut) nogil  # NOQA
cdef F_cuLinkComplete cuLinkComplete
ctypedef int (*F_cuLinkDestroy)(LinkState state) nogil
cdef F_cuLinkDestroy cuLinkDestroy
ctypedef int (*F_cuModuleLoad)(Module* module, char* fname) nogil
cdef F_cuModuleLoad cuModuleLoad
ctypedef int (*F_cuModuleLoadData)(Module* module, void* image) nogil
cdef F_cuModuleLoadData cuModuleLoadData
ctypedef int (*F_cuModuleUnload)(Module hmod) nogil
cdef F_cuModuleUnload cuModuleUnload
ctypedef int (*F_cuModuleGetFunction)(Function* hfunc, Module hmod, char* name) nogil  # NOQA
cdef F_cuModuleGetFunction cuModuleGetFunction
ctypedef int (*F_cuModuleGetGlobal)(Deviceptr* dptr, size_t* bytes, Module hmod, char* name) nogil  # NOQA
cdef F_cuModuleGetGlobal cuModuleGetGlobal
ctypedef int (*F_cuLaunchKernel)(Function f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, Stream hStream, void** kernelParams, void** extra) nogil  # NOQA
cdef F_cuLaunchKernel cuLaunchKernel
ctypedef int (*F_cuLaunchCooperativeKernel)(Function f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, Stream hStream, void** kernelParams) nogil  # NOQA
cdef F_cuLaunchCooperativeKernel cuLaunchCooperativeKernel

# Kernel attributes
ctypedef int (*F_cuFuncGetAttribute)(int *pi, CUfunction_attribute attrib, Function hfunc) nogil  # NOQA
cdef F_cuFuncGetAttribute cuFuncGetAttribute

ctypedef int (*F_cuFuncSetAttribute)(Function hfunc, CUfunction_attribute attrib, int value) nogil  # NOQA
cdef F_cuFuncSetAttribute cuFuncSetAttribute

# Occupancy
ctypedef int (*F_cuOccupancyMaxActiveBlocksPerMultiprocessor)(int* numBlocks, Function func, int blockSize, size_t dynamicSMemSize) nogil  # NOQA
cdef F_cuOccupancyMaxActiveBlocksPerMultiprocessor cuOccupancyMaxActiveBlocksPerMultiprocessor  # NOQA
ctypedef int (*F_cuOccupancyMaxPotentialBlockSize)(int* minGridSize, int* blockSize, Function func, CUoccupancyB2DSize block2shmem, size_t dynamicSMemSize, int blockSizeLimit) nogil  # NOQA
cdef F_cuOccupancyMaxPotentialBlockSize cuOccupancyMaxPotentialBlockSize

# Stream
ctypedef int (*F_cuStreamGetCtx)(Stream hStream, Context* pctx) nogil
cdef F_cuStreamGetCtx cuStreamGetCtx


cdef extern from '../../cupy_backend.h' nogil:
    # Build-time version
    enum: CUDA_VERSION


cdef SoftLink _L = None
cdef inline void initialize() except *:
    global _L
    if _L is not None:
        return
    _initialize()


cdef void _initialize() except *:
    global _L
    _L = _get_softlink()

    cdef str version = '_v2' if CUPY_CUDA_VERSION != 0 else ''

    global cuGetErrorName
    cuGetErrorName = <F_cuGetErrorName>_L.get('GetErrorName')
    global cuGetErrorString
    cuGetErrorString = <F_cuGetErrorString>_L.get('GetErrorString')
    global cuDevicePrimaryCtxRelease
    cuDevicePrimaryCtxRelease = <F_cuDevicePrimaryCtxRelease>_L.get(f'DevicePrimaryCtxRelease{version}')  # NOQA
    global cuCtxGetCurrent
    cuCtxGetCurrent = <F_cuCtxGetCurrent>_L.get('CtxGetCurrent')
    global cuCtxSetCurrent
    cuCtxSetCurrent = <F_cuCtxSetCurrent>_L.get('CtxSetCurrent')
    global cuCtxCreate
    cuCtxCreate = <F_cuCtxCreate>_L.get(f'CtxCreate{version}')
    global cuCtxDestroy
    cuCtxDestroy = <F_cuCtxDestroy>_L.get(f'CtxDestroy{version}')
    global cuCtxGetDevice
    cuCtxGetDevice = <F_cuCtxGetDevice>_L.get('CtxGetDevice')
    global cuLinkCreate
    cuLinkCreate = <F_cuLinkCreate>_L.get(f'LinkCreate{version}')
    global cuLinkAddData
    cuLinkAddData = <F_cuLinkAddData>_L.get(f'LinkAddData{version}')
    global cuLinkAddFile
    cuLinkAddFile = <F_cuLinkAddFile>_L.get(f'LinkAddFile{version}')
    global cuLinkComplete
    cuLinkComplete = <F_cuLinkComplete>_L.get('LinkComplete')
    global cuLinkDestroy
    cuLinkDestroy = <F_cuLinkDestroy>_L.get('LinkDestroy')
    global cuModuleLoad
    cuModuleLoad = <F_cuModuleLoad>_L.get('ModuleLoad')
    global cuModuleLoadData
    cuModuleLoadData = <F_cuModuleLoadData>_L.get('ModuleLoadData')
    global cuModuleUnload
    cuModuleUnload = <F_cuModuleUnload>_L.get('ModuleUnload')
    global cuModuleGetFunction
    cuModuleGetFunction = <F_cuModuleGetFunction>_L.get('ModuleGetFunction')
    global cuModuleGetGlobal
    cuModuleGetGlobal = <F_cuModuleGetGlobal>_L.get(f'ModuleGetGlobal{version}')  # NOQA
    global cuLaunchKernel
    cuLaunchKernel = <F_cuLaunchKernel>_L.get('LaunchKernel')
    global cuLaunchCooperativeKernel
    cuLaunchCooperativeKernel = <F_cuLaunchCooperativeKernel>_L.get('LaunchCooperativeKernel')  # NOQA
    global cuFuncGetAttribute
    cuFuncGetAttribute = <F_cuFuncGetAttribute>_L.get('FuncGetAttribute')
    global cuFuncSetAttribute
    cuFuncSetAttribute = <F_cuFuncSetAttribute>_L.get('FuncSetAttribute')
    global cuOccupancyMaxActiveBlocksPerMultiprocessor
    cuOccupancyMaxActiveBlocksPerMultiprocessor = <F_cuOccupancyMaxActiveBlocksPerMultiprocessor>_L.get('OccupancyMaxActiveBlocksPerMultiprocessor')  # NOQA
    global cuOccupancyMaxPotentialBlockSize
    cuOccupancyMaxPotentialBlockSize = <F_cuOccupancyMaxPotentialBlockSize>_L.get('OccupancyMaxPotentialBlockSize')  # NOQA
    global cuStreamGetCtx
    cuStreamGetCtx = <F_cuStreamGetCtx>_L.get('StreamGetCtx')


cdef SoftLink _get_softlink():
    cdef str prefix = 'cu'
    cdef object libname = None

    if CUPY_CUDA_VERSION != 0:
        if _sys.platform == 'linux':
            libname = 'libcuda.so.1'
        else:
            libname = 'nvcuda.dll'
    elif CUPY_HIP_VERSION != 0:
        # Use CUDA-to-HIP layer defined in the header.
        libname = __file__

    return SoftLink(libname, prefix, mandatory=True)
