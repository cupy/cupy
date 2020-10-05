cdef extern from *:
    ctypedef int OutputMode 'cudaOutputMode_t'


cpdef enum:
    cudaKeyValuePair = 0
    cudaCSV = 1

cpdef initialize(str config_file, str output_file, int output_mode)
cpdef start()
cpdef stop()
