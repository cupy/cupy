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
    _initialize()


cdef void _initialize() except *:
    global _L
    _L = _get_softlink()

    global DYN_cudaRuntimeGetVersion
    DYN_cudaRuntimeGetVersion = <F_cudaRuntimeGetVersion>_L.get('RuntimeGetVersion')  # noqa


cdef SoftLink _get_softlink():
    cdef int runtime_version
    cdef str prefix = 'cuda'
    cdef object libname = None

    if CUPY_CUDA_VERSION != 0:
        if 11020 <= CUPY_CUDA_VERSION < 12000:
            # CUDA 11.x (11.2+)
            if _sys.platform == 'linux':
                libname = 'libcudart.so.11.0'
            else:
                libname = 'cudart64_110.dll'
        elif 12000 <= CUPY_CUDA_VERSION < 13000:
            # CUDA 12.x
            if _sys.platform == 'linux':
                libname = 'libcudart.so.12'
            else:
                libname = 'cudart64_12.dll'
    elif CUPY_HIP_VERSION != 0:
        # Use CUDA-to-HIP layer defined in the header.
        libname = __file__

    return SoftLink(libname, prefix, mandatory=True)
