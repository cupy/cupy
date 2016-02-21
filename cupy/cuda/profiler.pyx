"""Thin wrapper of cuda profiler."""
cimport cython

from cupy.cuda cimport runtime
from cupy.cuda import runtime


cdef extern from "cupy_cuda.h":
    runtime.Error cudaProfilerInitialize(const char *configFile, 
                                         const char *outputFile, 
                                         int outputMode)
    runtime.Error cudaProfilerStart()
    runtime.Error cudaProfilerStop()


cpdef void initialize(str config_file,
                      str output_file,
                      int output_mode) except *:
    cdef bytes b_config_file = config_file.encode()
    cdef bytes b_output_file = output_file.encode()
    status = cudaProfilerInitialize(<const char*>b_config_file,
                                    <const char*>b_output_file,
                                    <OutputMode>output_mode)
    runtime.check_status(status)


cpdef void start() except *:
    status = cudaProfilerStart()
    runtime.check_status(status)


cpdef void stop() except *:
    status = cudaProfilerStop()
    runtime.check_status(status)
