from libc.stdint cimport intptr_t
from libcpp.vector cimport vector


###############################################################################
# Types
###############################################################################

IF CUPY_USE_CUDA_PYTHON:
    from cuda.cnvrtc cimport *
    # Aliases for compatibillity with existing CuPy codebase.
    # TODO(kmaehashi): Remove these aliases.
    ctypedef nvrtcProgram Program

cpdef check_status(int status)

cpdef tuple getVersion()
cpdef tuple getSupportedArchs()


cdef class ByteHolder:
    # We need a placeholder and not pass around bytes objects, because
    # sometimes the returned chars could go out of range (i.e. signed).

    cdef:
        vector[char] data

    cdef inline char* getData(self) nogil:
        return self.data.data()


###############################################################################
# Program
###############################################################################

cpdef intptr_t createProgram(unicode src, unicode name, headers,
                             include_names) except? 0
cpdef destroyProgram(intptr_t prog)
cpdef compileProgram(intptr_t prog, options)
cpdef bytes getPTX(intptr_t prog)
cpdef bytes getCUBIN(intptr_t prog)
cpdef bytes getNVVM(intptr_t prog)
cpdef ByteHolder getLTOIR(intptr_t prog)
cpdef unicode getProgramLog(intptr_t prog)
cpdef addNameExpression(intptr_t prog, str name)
cpdef str getLoweredName(intptr_t prog, str name)
