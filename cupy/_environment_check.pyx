import ctypes
import ctypes.util
import warnings


cdef extern from "cuda/cupy_cuda.h":
    int CUDA_VERSION


def _check_cuda_version():
    libpath = ctypes.util.find_library('cudart')
    if libpath is None:
        warnings.warn('''CuPy could not detect CUDA runtime.''')
        return

    cudart = ctypes.CDLL(libpath)
    if cudart is None:
        warnings.warn('''\
CuPy could not load CUDA runtime.
Library: {}'''.format(libpath))
        return

    if not hasattr(cudart, 'cudaRuntimeGetVersion'):
        warnings.warn('''
CuPy could not detect CUDA runtime version.
Library: {}'''.format(libpath))
        return

    runtime_version = ctypes.c_int()
    ret = cudart.cudaRuntimeGetVersion(ctypes.byref(runtime_version))
    if ret != 0:
        warnings.warn('''
CuPy could not retrieve CUDA runtime version.
Library: {}
Status: {}'''.format(libpath, ret))
        return

    runtime_version = runtime_version.value
    if (CUDA_VERSION // 100) != (runtime_version // 100):
        warnings.warn('''\
This CuPy is built for CUDA version {} but version {} is installed.
Library: {}'''.format(CUDA_VERSION, runtime_version, libpath))


def check():
    _check_cuda_version()
