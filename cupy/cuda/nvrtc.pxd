
###############################################################################
# Types
###############################################################################

cdef extern from *:
    ctypedef int Result 'nvrtcResult'

    ctypedef void* Program 'nvrtcProgram'


cpdef check_status(int status)

cpdef tuple getVersion()

###############################################################################
# Program
###############################################################################

cpdef size_t createProgram(str src, str name, headers, include_names) except *
cpdef destroyProgram(size_t prog)
cpdef compileProgram(size_t prog, options)
cpdef str getPTX(size_t prog)
cpdef str getProgramLog(size_t prog)
