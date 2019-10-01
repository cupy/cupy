from libc.stdint cimport intptr_t


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

cpdef intptr_t createProgram(unicode src, unicode name, headers,
                             include_names) except? 0
cpdef destroyProgram(intptr_t prog)
cpdef compileProgram(intptr_t prog, options)
cpdef unicode getPTX(intptr_t prog)
cpdef unicode getProgramLog(intptr_t prog)
