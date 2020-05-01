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
#from libcpp.string cimport string as cpp_str
#
#from cupy.cuda cimport common
#
#import numpy


###############################################################################
# Extern
###############################################################################

cdef extern from 'cupy_nvrtc.h' nogil:
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

    int nvrtcAddNameExpression(Program, const char*)
    int nvrtcGetLoweredName(Program, const char*, const char**)
    int nvrtcGetTypeName[T](cpp_str*)


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


###############################################################################
# Program
###############################################################################

cpdef intptr_t createProgram(unicode src, unicode name, headers,
                             include_names) except? 0:
    cdef Program prog
    cdef bytes b_src = src.encode()
    cdef const char* src_ptr = b_src
    cdef bytes b_name = name.encode()
    cdef const char* name_ptr = b_name
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


cpdef unicode getPTX(intptr_t prog):
    cdef size_t ptxSizeRet
    cdef vector.vector[char] ptx
    with nogil:
        status = nvrtcGetPTXSize(<Program>prog, &ptxSizeRet)
    check_status(status)
    ptx.resize(ptxSizeRet)
    with nogil:
        status = nvrtcGetPTX(<Program>prog, &ptx[0])
    check_status(status)

    # Strip the trailing NULL.
    return (&ptx[0])[:ptxSizeRet-1].decode('UTF-8')


cpdef unicode getProgramLog(intptr_t prog):
    cdef size_t logSizeRet
    cdef vector.vector[char] log
    with nogil:
        status = nvrtcGetProgramLogSize(<Program>prog, &logSizeRet)
    check_status(status)
    log.resize(logSizeRet)
    with nogil:
        status = nvrtcGetProgramLog(<Program>prog, &log[0])
    check_status(status)

    # Strip the trailing NULL.
    return (&log[0])[:logSizeRet-1].decode('UTF-8')


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
    cdef bytes b_mangled_name = mangled_name
    return b_mangled_name.decode('UTF-8')
#
#
#cpdef str getTypeName(dtype):
#    '''Convert NumPy dtype to NVRTC type name'''
#    cdef cpp_str cpp_name
#
#    if dtype == numpy.int8:
#        status = nvrtcGetTypeName[char](&cpp_name)
#    #elif dtype == numpy.uint8:
#    #    status = nvrtcGetTypeName[unsigned char](&cpp_name)
#    elif dtype == numpy.int16:
#        status = nvrtcGetTypeName[short](&cpp_name)
#    #elif dtype == numpy.uint16:
#    #    status = nvrtcGetTypeName[unsigned short](&cpp_name)
#    elif dtype == numpy.int32:
#        status = nvrtcGetTypeName[int](&cpp_name)
#    #elif dtype == numpy.uint32:
#    #    status = nvrtcGetTypeName[unsigned int](&cpp_name)
#    #elif dtype == numpy.int64:
#    #    status = nvrtcGetTypeName[long long](&cpp_name)
#    #elif dtype == numpy.uint64:
#    #    status = nvrtcGetTypeName[unsigned long long](&cpp_name)
#    elif dtype == numpy.float32:
#        status = nvrtcGetTypeName[float](&cpp_name)
#    elif dtype == numpy.float64:
#        status = nvrtcGetTypeName[double](&cpp_name)
#    #elif dtype == numpy.complex64:
#    #    #status = nvrtcGetTypeName[common.cpy_complex64](&cpp_name)
#    #elif dtype == numpy.complex128:
#    #    #status = nvrtcGetTypeName[common.cpy_complex128](&cpp_name)
#    #elif dtype == numpy.bool:
#    #    status = nvrtcGetTypeName[common.cpy_bool](&cpp_name)
#    else:
#        raise NotImplementedError('dtype is not supported')
#    check_status(status)
#
#    return cpp_name.decode('UTF-8')
