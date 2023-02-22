# distutils: language = c++

"""Thin wrapper of cuda profiler."""
from cupy_backends.cuda.api cimport runtime


# TODO(kmaehashi): cudaProfilerInitialize is deprecated thus unsupported in
# cudapython.
cdef extern from '../../cupy_profiler.h' nogil:
    ctypedef int cudaError_t

    cudaError_t cudaProfilerInitialize(const char *configFile,
                                       const char *outputFile,
                                       int outputMode)
    cudaError_t cudaProfilerStart()
    cudaError_t cudaProfilerStop()


cpdef initialize(str config_file,
                 str output_file,
                 int output_mode):
    """Initialize the CUDA profiler.

    This function initialize the CUDA profiler. See the CUDA document for
    detail.

    Args:
        config_file (str): Name of the configuration file.
        output_file (str): Name of the output file.
        output_mode (int): ``cupy.cuda.profiler.cudaKeyValuePair`` or
            ``cupy.cuda.profiler.cudaCSV``.

    .. warning::
       This API is marked as deprecated in CUDA 11, and has been removed
       in CUDA 12.
       This CuPy interface is subject to removal in future releases.
    """
    if 12000 <= runtime.runtimeGetVersion():
        raise RuntimeError(
            'cudaProfilerInitialize no longer available in CUDA 12+')
    cdef bytes b_config_file = config_file.encode()
    cdef bytes b_output_file = output_file.encode()
    status = cudaProfilerInitialize(<const char*>b_config_file,
                                    <const char*>b_output_file,
                                    <OutputMode>output_mode)
    runtime.check_status(status)


cpdef start():
    """Enable profiling.

    A user can enable CUDA profiling. When an error occurs, it raises an
    exception.

    See the CUDA document for detail.
    """
    status = cudaProfilerStart()
    runtime.check_status(status)


cpdef stop():
    """Disable profiling.

    A user can disable CUDA profiling. When an error occurs, it raises an
    exception.

    See the CUDA document for detail.
    """
    status = cudaProfilerStop()
    runtime.check_status(status)
