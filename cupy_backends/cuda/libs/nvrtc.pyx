# distutils: language = c++

"""Thin wrapper of NVRTC API.

There are four differences compared to the original C API.

1. Not all functions are ported.
2. Errors are translated into NVRTCError exceptions.
3. The 'nvrtc' prefix of each API is omitted and the next character is set to
   lower case.
4. The resulting values are returned directly instead of references.

"""
cimport cython  # NOQA
from libcpp cimport vector

from cupy_backends.cuda.api cimport runtime


###############################################################################
# Extern
###############################################################################

cdef extern from '../../cupy_rtc.h' nogil:
    const char *nvrtcGetErrorString(Result result)
    int nvrtcVersion(int *major, int *minor)
    int nvrtcCreateProgram(
        Program* prog, const char* src, const char* name, int numHeaders,
        const char** headers, const char** includeNames)
    int nvrtcDestroyProgram(Program *prog)
    int nvrtcCompileProgram(Program prog, int numOptions,
                            const char** options)
    int nvrtcGetPTXSize(Program prog, size_t *ptxSizeRet)
    int nvrtcGetPTX(Program prog, char *ptx)
    int nvrtcGetCUBINSize(Program prog, size_t *cubinSizeRet)
    int nvrtcGetCUBIN(Program prog, char *cubin)
    int nvrtcGetProgramLogSize(Program prog, size_t* logSizeRet)
    int nvrtcGetProgramLog(Program prog, char* log)
    int nvrtcAddNameExpression(Program, const char*)
    int nvrtcGetLoweredName(Program, const char*, const char**)
    int nvrtcGetNumSupportedArchs(int* numArchs)
    int nvrtcGetSupportedArchs(int* supportedArchs)


###############################################################################
# Error handling
###############################################################################

class NVRTCError(RuntimeError):

    def __init__(self, status):
        self.status = status
        cdef bytes msg = nvrtcGetErrorString(<Result>status)
        super(NVRTCError, self).__init__(
            '{} ({})'.format(msg.decode(), status))

    def __reduce__(self):
        return (type(self), (self.status,))


@cython.profile(False)
cpdef inline check_status(int status):
    if status != 0:
        raise NVRTCError(status)


cpdef tuple getVersion():
    cdef int major, minor
    with nogil:
        status = nvrtcVersion(&major, &minor)
    check_status(status)
    return major, minor


cpdef tuple getSupportedArchs():
    cdef int status, num_archs
    cdef vector.vector[int] archs
    if CUDA_VERSION < 11020 or runtime._is_hip_environment:
        raise RuntimeError("getSupportedArchs is supported since CUDA 11.2")

    with nogil:
        status = nvrtcGetNumSupportedArchs(&num_archs)
        if status == 0:
            archs.resize(num_archs)
            status = nvrtcGetSupportedArchs(archs.data())
    check_status(status)
    return tuple(archs)


###############################################################################
# Program
###############################################################################

cpdef intptr_t createProgram(unicode src, unicode name, headers,
                             include_names) except? 0:
    cdef Program prog
    cdef bytes b_src = src.encode()
    cdef const char* src_ptr = b_src
    cdef bytes b_name = name.encode()
    cdef const char* name_ptr
    if len(name) > 0:
        name_ptr = b_name
    else:
        name_ptr = NULL
    cdef int num_headers = len(headers)
    cdef vector.vector[const char*] header_vec
    cdef vector.vector[const char*] include_name_vec
    cdef const char** header_vec_ptr = NULL
    cdef const char** include_name_vec_ptr = NULL
    assert num_headers == len(include_names)
    for i in headers:
        header_vec.push_back(<const char*>i)
    for i in include_names:
        include_name_vec.push_back(<const char*>i)
    if num_headers > 0:
        header_vec_ptr = header_vec.data()
        include_name_vec_ptr = include_name_vec.data()
    with nogil:
        status = nvrtcCreateProgram(
            &prog, src_ptr, name_ptr, num_headers, header_vec_ptr,
            include_name_vec_ptr)
    check_status(status)
    return <intptr_t>prog


cpdef destroyProgram(intptr_t prog):
    cdef Program p = <Program>prog
    with nogil:
        status = nvrtcDestroyProgram(&p)
    check_status(status)


cpdef compileProgram(intptr_t prog, options):
    cdef int option_num = len(options)
    cdef vector.vector[const char*] option_vec
    cdef option_list = [opt.encode() for opt in options]
    cdef const char** option_vec_ptr = NULL
    for i in option_list:
        option_vec.push_back(<const char*>i)
    if option_num > 0:
        option_vec_ptr = option_vec.data()
    with nogil:
        status = nvrtcCompileProgram(<Program>prog, option_num,
                                     option_vec_ptr)
    check_status(status)


cpdef bytes getPTX(intptr_t prog):
    cdef size_t ptxSizeRet
    cdef vector.vector[char] ptx
    cdef char* ptx_ptr = NULL
    with nogil:
        status = nvrtcGetPTXSize(<Program>prog, &ptxSizeRet)
    check_status(status)
    if ptxSizeRet == 0:
        return b''
    ptx.resize(ptxSizeRet)
    ptx_ptr = ptx.data()
    with nogil:
        status = nvrtcGetPTX(<Program>prog, ptx_ptr)
    check_status(status)

    # Strip the trailing NULL.
    return ptx_ptr[:ptxSizeRet-1]


cpdef bytes getCUBIN(intptr_t prog):
    cdef size_t cubinSizeRet = 0
    cdef vector.vector[char] cubin
    cdef char* cubin_ptr = NULL
    if CUDA_VERSION < 11010 or runtime._is_hip_environment:
        raise RuntimeError("getCUBIN is supported since CUDA 11.1")
    with nogil:
        status = nvrtcGetCUBINSize(<Program>prog, &cubinSizeRet)
    check_status(status)
    if cubinSizeRet <= 1:
        # On CUDA 11.1, cubinSizeRet=1 if -arch=compute_XX is used, but the
        # spec says it should be 0 in this case...
        raise RuntimeError('cubin is requested, but the real arch (sm_XX) is '
                           'not provided')
    cubin.resize(cubinSizeRet)
    cubin_ptr = cubin.data()
    with nogil:
        status = nvrtcGetCUBIN(<Program>prog, cubin_ptr)
    check_status(status)

    # Strip the trailing NULL.
    return cubin_ptr[:cubinSizeRet-1]


cpdef unicode getProgramLog(intptr_t prog):
    cdef size_t logSizeRet
    cdef vector.vector[char] log
    cdef char* log_ptr = NULL
    with nogil:
        status = nvrtcGetProgramLogSize(<Program>prog, &logSizeRet)
    check_status(status)
    if logSizeRet == 0:
        return ''
    log.resize(logSizeRet)
    log_ptr = log.data()
    with nogil:
        status = nvrtcGetProgramLog(<Program>prog, log_ptr)
    check_status(status)

    # Strip the trailing NULL.
    return log_ptr[:logSizeRet-1].decode('UTF-8')


cpdef addAddNameExpression(intptr_t prog, str name):
    cdef bytes b_name = name.encode()
    cdef const char* c_name = b_name
    with nogil:
        status = nvrtcAddNameExpression(<Program>prog, c_name)
    check_status(status)


cpdef str getLoweredName(intptr_t prog, str name):
    cdef bytes b_name = name.encode()
    cdef const char* c_name = b_name
    cdef const char* mangled_name
    with nogil:
        status = nvrtcGetLoweredName(<Program>prog, c_name, &mangled_name)
    check_status(status)
    b_name = mangled_name
    return b_name.decode('UTF-8')
