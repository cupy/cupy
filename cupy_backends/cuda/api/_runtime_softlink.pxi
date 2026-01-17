import sys as _sys

from cupy_backends.cuda._softlink cimport SoftLink


"""
Load CUDA Runtime shared library via SoftLink to allow probing CUDA
version installed in users' environment.
"""

ctypedef int (*F_cudaRuntimeGetVersion)(int* runtimeVersion) nogil
cdef F_cudaRuntimeGetVersion DYN_cudaRuntimeGetVersion


cdef SoftLink _L = None
cdef inline void initialize() except *:
    global _L
    if _L is not None:
        return
    _L = _initialize()


cdef SoftLink _initialize():
    _L = _get_softlink()

    global DYN_cudaRuntimeGetVersion
    DYN_cudaRuntimeGetVersion = <F_cudaRuntimeGetVersion>_L.get('RuntimeGetVersion')  # noqa

    return _L


cdef SoftLink _get_softlink():
    cdef int runtime_version
    cdef str prefix = 'cuda'
    cdef object libname = None
    cdef object handle = 0

    if CUPY_CUDA_VERSION != 0:
        # We let libname be None here to avoid loading the library twice,
        # which could potentially be loading different versions of the library.
        from cuda import pathfinder
        loaded_dl = pathfinder.load_nvidia_dynamic_lib('cudart')
        handle = loaded_dl._handle_uint
    elif CUPY_HIP_VERSION != 0:
        # Use CUDA-to-HIP layer defined in the header.
        libname = __file__

    return SoftLink(libname, prefix, mandatory=True, handle=handle)
