# distutils: language = c++

"""Thin wrapper of CUDA Driver API.

There are four differences compared to the original C API.

1. Not all functions are ported.
2. Errors are translated into CUDADriverError exceptions.
3. The 'cu' prefix of each API is omitted and the next character is set to
   lower case.
4. The resulting values are returned directly instead of references.

"""
cimport cython
from libcpp cimport vector


###############################################################################
# Extern
###############################################################################

cdef extern from "cupy_nvrtc.h" nogil:
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
    int nvrtcGetProgramLogSize(Program prog, size_t* logSizeRet)
    int nvrtcGetProgramLog(Program prog, char* log)


###############################################################################
# Error handling
###############################################################################

class NVRTCError(RuntimeError):

    def __init__(self, status):
        self.status = status
        cdef bytes msg = nvrtcGetErrorString(<Result>status)
        super(NVRTCError, self).__init__(msg.decode())


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


###############################################################################
# Program
###############################################################################

cpdef size_t createProgram(unicode src, unicode name, headers,
                           include_names) except *:
    cdef Program prog
    cdef bytes b_src = src.encode()
    cdef const char* src_ptr = b_src
    cdef bytes b_name = name.encode()
    cdef const char* name_ptr = b_name
    cdef int num_headers = len(headers)
    cdef vector.vector[const char*] header_vec
    cdef vector.vector[const char*] include_name_vec
    for i in headers:
        header_vec.push_back(<const char*>i)
    for i in include_names:
        include_name_vec.push_back(<const char*>i)

    with nogil:
        status = nvrtcCreateProgram(
            &prog, src_ptr, name_ptr, num_headers, &(header_vec[0]),
            &(include_name_vec[0]))
    check_status(status)
    return <size_t>prog


cpdef destroyProgram(size_t prog):
    cdef Program p = <Program>prog
    with nogil:
        status = nvrtcDestroyProgram(&p)
    check_status(status)


cpdef compileProgram(size_t prog, options):
    cdef int option_num = len(options)
    cdef vector.vector[const char*] option_vec
    cdef option_list = [opt.encode() for opt in options]
    for i in option_list:
        option_vec.push_back(<const char*>i)

    with nogil:
        status = nvrtcCompileProgram(<Program>prog, option_num,
                                     &(option_vec[0]))
    check_status(status)


cpdef unicode getPTX(size_t prog):
    cdef size_t ptxSizeRet
    cdef bytes ptx
    cdef char* ptx_ptr
    with nogil:
        status = nvrtcGetPTXSize(<Program>prog, &ptxSizeRet)
    check_status(status)
    ptx = b' ' * ptxSizeRet
    ptx_ptr = ptx
    with nogil:
        status = nvrtcGetPTX(<Program>prog, ptx_ptr)
    check_status(status)
    return ptx.decode('UTF-8')


cpdef unicode getProgramLog(size_t prog):
    cdef size_t logSizeRet
    cdef bytes log
    cdef char* log_ptr
    with nogil:
        status = nvrtcGetProgramLogSize(<Program>prog, &logSizeRet)
    check_status(status)
    log = b' ' * logSizeRet
    log_ptr = log
    with nogil:
        status = nvrtcGetProgramLog(<Program>prog, log_ptr)
    check_status(status)
    return log.decode('UTF-8')
