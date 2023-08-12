import sys as _sys

from cupy_backends.cuda.api cimport runtime
from cupy_backends.cuda._softlink cimport SoftLink


cdef int _runtime_version = runtime.runtimeGetVersion()
cdef str _prefix = 'nvrtc'
cdef object _libname = None

if CUPY_CUDA_VERSION != 0:
    if 11020 <= _runtime_version < 12000:
        # CUDA 11.x (11.2+)
        if _sys.platform == 'linux':
            _libname = 'libnvrtc.so.11.2'
        else:
            _libname = 'nvrtc64_112_0.dll'
    elif 12000 <= _runtime_version < 13000:
        # CUDA 12.x
        if _sys.platform == 'linux':
            _libname = 'libnvrtc.so.12'
        else:
            _libname = 'nvrtc64_120_0.dll'
elif CUPY_HIP_VERSION != 0:
    _prefix = 'hiprtc'
    if _runtime_version < 5_00_00000:
        # ROCm 4.x
        _libname = 'libamdhip64.so.4'
    elif _runtime_version < 6_00_00000:
        # ROCm 5.x
        _libname = 'libamdhip64.so.5'

cdef SoftLink _L = SoftLink(_libname, _prefix)

ctypedef int nvrtcResult
ctypedef void* nvrtcProgram

# Aliases for compatibillity with existing CuPy codebase.
# TODO(kmaehashi): Remove these aliases.
ctypedef nvrtcProgram Program

ctypedef const char* (*F_nvrtcGetErrorString)(nvrtcResult result) nogil
cdef F_nvrtcGetErrorString nvrtcGetErrorString = <F_nvrtcGetErrorString>_L.get('GetErrorString')  # NOQA

ctypedef nvrtcResult (*F_nvrtcVersion)(int *major, int *minor) nogil
cdef F_nvrtcVersion nvrtcVersion = <F_nvrtcVersion>_L.get('Version')

ctypedef nvrtcResult (*F_nvrtcCreateProgram)(
    nvrtcProgram* prog, const char* src, const char* name, int numHeaders,
    const char** headers, const char** includeNames) nogil
cdef F_nvrtcCreateProgram nvrtcCreateProgram = <F_nvrtcCreateProgram>_L.get('CreateProgram')  # NOQA

ctypedef nvrtcResult (*F_nvrtcDestroyProgram)(nvrtcProgram *prog) nogil
cdef F_nvrtcDestroyProgram nvrtcDestroyProgram = <F_nvrtcDestroyProgram>_L.get('DestroyProgram')  # NOQA

ctypedef nvrtcResult (*F_nvrtcCompileProgram)(
    nvrtcProgram prog, int numOptions, const char** options) nogil
cdef F_nvrtcCompileProgram nvrtcCompileProgram = <F_nvrtcCompileProgram>_L.get('CompileProgram')  # NOQA

ctypedef nvrtcResult (*F_nvrtcGetPTXSize)(nvrtcProgram prog, size_t *ptxSizeRet) nogil  # NOQA
cdef F_nvrtcGetPTXSize nvrtcGetPTXSize = <F_nvrtcGetPTXSize>_L.get(
    'GetPTXSize' if _prefix == 'nvrtc' else 'GetCodeSize')

ctypedef nvrtcResult (*F_nvrtcGetPTX)(nvrtcProgram prog, char *ptx) nogil
cdef F_nvrtcGetPTX nvrtcGetPTX = <F_nvrtcGetPTX>_L.get(
    'GetPTX' if _prefix == 'nvrtc' else 'GetCode')

ctypedef nvrtcResult (*F_nvrtcGetCUBINSize)(nvrtcProgram prog, size_t *cubinSizeRet) nogil  # NOQA
cdef F_nvrtcGetCUBINSize nvrtcGetCUBINSize = <F_nvrtcGetCUBINSize>_L.get('GetCUBINSize')  # NOQA

ctypedef nvrtcResult (*F_nvrtcGetCUBIN)(nvrtcProgram prog, char *cubin) nogil
cdef F_nvrtcGetCUBIN nvrtcGetCUBIN = <F_nvrtcGetCUBIN>_L.get('GetCUBIN')

ctypedef nvrtcResult (*F_nvrtcGetProgramLogSize)(nvrtcProgram prog, size_t* logSizeRet) nogil  # NOQA
cdef F_nvrtcGetProgramLogSize nvrtcGetProgramLogSize = <F_nvrtcGetProgramLogSize>_L.get('GetProgramLogSize')  # NOQA

ctypedef nvrtcResult (*F_nvrtcGetProgramLog)(nvrtcProgram prog, char* log) nogil  # NOQA
cdef F_nvrtcGetProgramLog nvrtcGetProgramLog = <F_nvrtcGetProgramLog>_L.get('GetProgramLog')  # NOQA

ctypedef nvrtcResult (*F_nvrtcAddNameExpression)(nvrtcProgram, const char*) nogil  # NOQA
cdef F_nvrtcAddNameExpression nvrtcAddNameExpression = <F_nvrtcAddNameExpression>_L.get('AddNameExpression')  # NOQA

ctypedef nvrtcResult (*F_nvrtcGetLoweredName)(nvrtcProgram, const char*, const char**) nogil  # NOQA
cdef F_nvrtcGetLoweredName nvrtcGetLoweredName = <F_nvrtcGetLoweredName>_L.get('GetLoweredName')  # NOQA

ctypedef nvrtcResult (*F_nvrtcGetNumSupportedArchs)(int* numArchs) nogil
cdef F_nvrtcGetNumSupportedArchs nvrtcGetNumSupportedArchs = <F_nvrtcGetNumSupportedArchs>_L.get('GetNumSupportedArchs')  # NOQA

ctypedef nvrtcResult (*F_nvrtcGetSupportedArchs)(int* supportedArchs) nogil
cdef F_nvrtcGetSupportedArchs nvrtcGetSupportedArchs = <F_nvrtcGetSupportedArchs>_L.get('GetSupportedArchs')  # NOQA

ctypedef nvrtcResult (*F_nvrtcGetNVVMSize)(nvrtcProgram prog, size_t *nvvmSizeRet) nogil  # NOQA
cdef F_nvrtcGetNVVMSize nvrtcGetNVVMSize = <F_nvrtcGetNVVMSize>_L.get('GetNVVMSize')  # NOQA

ctypedef nvrtcResult (*F_nvrtcGetNVVM)(nvrtcProgram prog, char *nvvm) nogil
cdef F_nvrtcGetNVVM nvrtcGetNVVM = <F_nvrtcGetNVVM>_L.get('GetNVVM')
