cdef extern from *:
    ctypedef int OutputMode 'cudaOutputMode_t'


cpdef enum:
    cudaKeyValuePair = 0
    cudaCSV = 1

cpdef void initialize(
    str config_file, str output_file, int output_mode) except *
cpdef void start() except *
cpdef void stop() except *
