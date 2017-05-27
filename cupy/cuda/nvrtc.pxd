
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

cpdef size_t createProgram(unicode src, unicode name, headers,
                           include_names) except *
cpdef destroyProgram(size_t prog)
cpdef compileProgram(size_t prog, options)
cpdef unicode getPTX(size_t prog)
cpdef unicode getProgramLog(size_t prog)
