import sys as _sys

from cupy_backends.cuda.api cimport runtime
from cupy_backends.cuda._softlink cimport SoftLink


ctypedef int nvrtcResult
ctypedef void* nvrtcProgram
# TODO(kmaehashi): Remove this alias.
ctypedef nvrtcProgram Program

ctypedef const char* (*F_nvrtcGetErrorString)(nvrtcResult result) nogil
cdef F_nvrtcGetErrorString nvrtcGetErrorString

ctypedef nvrtcResult (*F_nvrtcVersion)(int *major, int *minor) nogil
cdef F_nvrtcVersion nvrtcVersion

ctypedef nvrtcResult (*F_nvrtcCreateProgram)(
    nvrtcProgram* prog, const char* src, const char* name, int numHeaders,
    const char** headers, const char** includeNames) nogil
cdef F_nvrtcCreateProgram nvrtcCreateProgram

ctypedef nvrtcResult (*F_nvrtcDestroyProgram)(nvrtcProgram *prog) nogil
cdef F_nvrtcDestroyProgram nvrtcDestroyProgram

ctypedef nvrtcResult (*F_nvrtcCompileProgram)(
    nvrtcProgram prog, int numOptions, const char** options) nogil
cdef F_nvrtcCompileProgram nvrtcCompileProgram

ctypedef nvrtcResult (*F_nvrtcGetPTXSize)(nvrtcProgram prog, size_t *ptxSizeRet) nogil  # NOQA
cdef F_nvrtcGetPTXSize nvrtcGetPTXSize

ctypedef nvrtcResult (*F_nvrtcGetPTX)(nvrtcProgram prog, char *ptx) nogil
cdef F_nvrtcGetPTX nvrtcGetPTX

ctypedef nvrtcResult (*F_nvrtcGetCUBINSize)(nvrtcProgram prog, size_t *cubinSizeRet) nogil  # NOQA
cdef F_nvrtcGetCUBINSize nvrtcGetCUBINSize

ctypedef nvrtcResult (*F_nvrtcGetCUBIN)(nvrtcProgram prog, char *cubin) nogil
cdef F_nvrtcGetCUBIN nvrtcGetCUBIN

ctypedef nvrtcResult (*F_nvrtcGetProgramLogSize)(nvrtcProgram prog, size_t* logSizeRet) nogil  # NOQA
cdef F_nvrtcGetProgramLogSize nvrtcGetProgramLogSize

ctypedef nvrtcResult (*F_nvrtcGetProgramLog)(nvrtcProgram prog, char* log) nogil  # NOQA
cdef F_nvrtcGetProgramLog nvrtcGetProgramLog

ctypedef nvrtcResult (*F_nvrtcAddNameExpression)(nvrtcProgram, const char*) nogil  # NOQA
cdef F_nvrtcAddNameExpression nvrtcAddNameExpression

ctypedef nvrtcResult (*F_nvrtcGetLoweredName)(nvrtcProgram, const char*, const char**) nogil  # NOQA
cdef F_nvrtcGetLoweredName nvrtcGetLoweredName

ctypedef nvrtcResult (*F_nvrtcGetNumSupportedArchs)(int* numArchs) nogil
cdef F_nvrtcGetNumSupportedArchs nvrtcGetNumSupportedArchs

ctypedef nvrtcResult (*F_nvrtcGetSupportedArchs)(int* supportedArchs) nogil
cdef F_nvrtcGetSupportedArchs nvrtcGetSupportedArchs

ctypedef nvrtcResult (*F_nvrtcGetNVVMSize)(nvrtcProgram prog, size_t *nvvmSizeRet) nogil  # NOQA
cdef F_nvrtcGetNVVMSize nvrtcGetNVVMSize

ctypedef nvrtcResult (*F_nvrtcGetNVVM)(nvrtcProgram prog, char *nvvm) nogil
cdef F_nvrtcGetNVVM nvrtcGetNVVM


cdef SoftLink _L = None
cdef inline void initialize() except *:
    global _L
    if _L is not None:
        return
    _initialize()

cdef void _initialize() except *:
    global _L
    _L = _get_softlink()

    global nvrtcGetErrorString
    nvrtcGetErrorString = <F_nvrtcGetErrorString>_L.get('GetErrorString')
    global nvrtcVersion
    nvrtcVersion = <F_nvrtcVersion>_L.get('Version')
    global nvrtcCreateProgram
    nvrtcCreateProgram = <F_nvrtcCreateProgram>_L.get('CreateProgram')
    global nvrtcDestroyProgram
    nvrtcDestroyProgram = <F_nvrtcDestroyProgram>_L.get('DestroyProgram')
    global nvrtcCompileProgram
    nvrtcCompileProgram = <F_nvrtcCompileProgram>_L.get('CompileProgram')
    global nvrtcGetPTXSize
    nvrtcGetPTXSize = <F_nvrtcGetPTXSize>_L.get('GetPTXSize' if _L.prefix == 'nvrtc' else 'GetCodeSize')  # NOQA
    global nvrtcGetPTX
    nvrtcGetPTX = <F_nvrtcGetPTX>_L.get('GetPTX' if _L.prefix == 'nvrtc' else 'GetCode')  # NOQA
    global nvrtcGetCUBINSize
    nvrtcGetCUBINSize = <F_nvrtcGetCUBINSize>_L.get('GetCUBINSize')
    global nvrtcGetCUBIN
    nvrtcGetCUBIN = <F_nvrtcGetCUBIN>_L.get('GetCUBIN')
    global nvrtcGetProgramLogSize
    nvrtcGetProgramLogSize = <F_nvrtcGetProgramLogSize>_L.get('GetProgramLogSize')  # NOQA
    global nvrtcGetProgramLog
    nvrtcGetProgramLog = <F_nvrtcGetProgramLog>_L.get('GetProgramLog')
    global nvrtcAddNameExpression
    nvrtcAddNameExpression = <F_nvrtcAddNameExpression>_L.get('AddNameExpression')  # NOQA
    global nvrtcGetLoweredName
    nvrtcGetLoweredName = <F_nvrtcGetLoweredName>_L.get('GetLoweredName')
    global nvrtcGetNumSupportedArchs
    nvrtcGetNumSupportedArchs = <F_nvrtcGetNumSupportedArchs>_L.get('GetNumSupportedArchs')  # NOQA
    global nvrtcGetSupportedArchs
    nvrtcGetSupportedArchs = <F_nvrtcGetSupportedArchs>_L.get('GetSupportedArchs')  # NOQA
    global nvrtcGetNVVMSize
    nvrtcGetNVVMSize = <F_nvrtcGetNVVMSize>_L.get('GetNVVMSize')
    global nvrtcGetNVVM
    nvrtcGetNVVM = <F_nvrtcGetNVVM>_L.get('GetNVVM')


cdef SoftLink _get_softlink():
    cdef int runtime_version
    cdef str prefix = 'nvrtc'
    cdef object libname = None

    if CUPY_CUDA_VERSION != 0:
        runtime_version = runtime._getCUDAMajorVersion()
        if runtime_version == 11:
            # CUDA 11.x (11.2+)
            if _sys.platform == 'linux':
                libname = 'libnvrtc.so.11.2'
            else:
                libname = 'nvrtc64_112_0.dll'
        elif runtime_version == 12:
            # CUDA 12.x
            if _sys.platform == 'linux':
                libname = 'libnvrtc.so.12'
            else:
                libname = 'nvrtc64_120_0.dll'
    elif CUPY_HIP_VERSION != 0:
        runtime_version = runtime.runtimeGetVersion()
        prefix = 'hiprtc'
        if runtime_version < 5_00_00000:
            # ROCm 4.x
            libname = 'libamdhip64.so.4'
        elif runtime_version < 6_00_00000:
            # ROCm 5.x
            libname = 'libamdhip64.so.5'

    return SoftLink(libname, prefix, mandatory=True)
