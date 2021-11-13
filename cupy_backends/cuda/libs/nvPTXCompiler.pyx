# distutils: language = c++

"""Thin wrapper of nvPTXCompiler API.

There are four differences compared to the original C API.

1. Not all functions are ported.
2. Errors are translated into nvPTXCompilerError exceptions.
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

cdef extern from '../../cupy_nvptx.h' nogil:
    int nvPTXCompilerGetVersion(unsigned int* major, unsigned int* minor)
    int nvPTXCompilerCreate(
        Handle* compiler, size_t srcLen, const char* src)
    int nvPTXCompilerDestroy(Handle* compiler)
    int nvPTXCompilerCompile(
        Handle compiler, int numOptions, const char** options)
    int nvPTXCompilerGetCompiledProgramSize(Handle compiler, size_t* size)
    int nvPTXCompilerGetCompiledProgram(Handle compiler, void* binary)
    int nvPTXCompilerGetErrorLogSize(Handle compiler, size_t* size)
    int nvPTXCompilerGetErrorLog(Handle compiler, char* log)
    int nvPTXCompilerGetInfoLogSize(Handle compiler, size_t* size)
    int nvPTXCompilerGetInfoLog(Handle compiler, char* log)


###############################################################################
# Error handling
###############################################################################

class nvPTXCompilerError(RuntimeError):

    def __init__(self, status):
        self.status = status
        cdef bytes msg = b''  # TODO: use error log? or custom error dict?
        super().__init__(
            '{} ({})'.format(msg.decode(), status))

    def __reduce__(self):
        return (type(self), (self.status,))


@cython.profile(False)
cpdef inline check_status(int status):
    if status != 0:
        raise nvPTXCompilerError(status)


###############################################################################
# APIs
###############################################################################

cpdef tuple getVersion():
    cdef unsigned int major, minor
    status = nvPTXCompilerGetVersion(&major, &minor)
    check_status(status)
    return major, minor


cpdef intptr_t create(str src) except? 0:
    cdef Handle compiler
    cdef bytes b_src = src.encode()
    cdef const char* src_ptr = b_src
    cdef size_t srcLen = len(b_src)
    with nogil:
        status = nvPTXCompilerCreate(&compiler, srcLen, src_ptr)
    check_status(status)
    return <intptr_t>compiler


cpdef destroy(intptr_t compiler):
    cdef Handle p = <Handle>compiler
    with nogil:
        status = nvPTXCompilerDestroy(&p)
    check_status(status)


cpdef compile(intptr_t compiler, options):
    cdef int option_num = len(options)
    cdef vector.vector[const char*] option_vec
    cdef option_list = [opt.encode() for opt in options]
    cdef const char** option_vec_ptr = NULL
    for i in option_list:
        option_vec.push_back(<const char*>i)
    if option_num > 0:
        option_vec_ptr = option_vec.data()
    with nogil:
        status = nvPTXCompilerCompile(
            <Handle>compiler, option_num, option_vec_ptr)
    check_status(status)


cpdef bytes getCompiledProgram(intptr_t compiler):
    cdef size_t binarySize
    cdef vector.vector[char] binary
    cdef char* binary_ptr = NULL
    with nogil:
        status = nvPTXCompilerGetCompiledProgramSize(<Handle>compiler, &binarySize)
    check_status(status)
    if binarySize == 0:
        return b''
    binary.resize(binarySize)
    binary_ptr = binary.data()
    with nogil:
        status = nvPTXCompilerGetCompiledProgram(<Handle>compiler, binary_ptr)
    check_status(status)

    # Strip the trailing NULL.
    return binary_ptr[:binarySize-1]


cpdef str getErrorLog(intptr_t compiler):
    cdef size_t logSize
    cdef vector.vector[char] log
    cdef char* log_ptr = NULL
    with nogil:
        status = nvPTXCompilerGetErrorLogSize(<Handle>compiler, &logSize)
    check_status(status)
    if logSize == 0:
        return ''
    log.resize(logSize)
    log_ptr = log.data()
    with nogil:
        status = nvPTXCompilerGetErrorLog(<Handle>compiler, log_ptr)
    check_status(status)

    # Strip the trailing NULL
    return log_ptr[:logSize-1].decode('UTF-8')


cpdef str getInfoLog(intptr_t compiler):
    cdef size_t logSize
    cdef vector.vector[char] log
    cdef char* log_ptr = NULL
    with nogil:
        status = nvPTXCompilerGetInfoLogSize(<Handle>compiler, &logSize)
    check_status(status)
    if logSize == 0:
        return ''
    log.resize(logSize)
    log_ptr = log.data()
    with nogil:
        status = nvPTXCompilerGetInfoLog(<Handle>compiler, log_ptr)
    check_status(status)

    # Strip the trailing NULL
    return log_ptr[:logSize-1].decode('UTF-8')
