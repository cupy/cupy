import os

from cupy import cuda

import warnings


_cupy_header_list = [
    'cupy/complex.cuh',
    'cupy/carray.cuh',
]
_cupy_header = ''.join(
    ['#include <%s>\n' % i for i in _cupy_header_list])

# This is indirect include header list.
# These header files are subject to a hash key.
_cupy_extra_header_list = [
    'cupy/complex/complex.h',
    'cupy/complex/math_private.h',
    'cupy/complex/complex_inl.h',
    'cupy/complex/arithmetic.h',
    'cupy/complex/cproj.h',
    'cupy/complex/cexp.h',
    'cupy/complex/cexpf.h',
    'cupy/complex/clog.h',
    'cupy/complex/clogf.h',
    'cupy/complex/cpow.h',
    'cupy/complex/ccosh.h',
    'cupy/complex/ccoshf.h',
    'cupy/complex/csinh.h',
    'cupy/complex/csinhf.h',
    'cupy/complex/ctanh.h',
    'cupy/complex/ctanhf.h',
    'cupy/complex/csqrt.h',
    'cupy/complex/csqrtf.h',
    'cupy/complex/catrig.h',
    'cupy/complex/catrigf.h',
]

_header_path_cache = None
_header_source = None


def _get_header_dir_path():
    global _header_path_cache
    if _header_path_cache is None:
        # Cython cannot use __file__ in global scope
        _header_path_cache = os.path.abspath(
            os.path.join(os.path.dirname(__file__), 'include'))
    return _header_path_cache


def _get_header_source():
    global _header_source
    if _header_source is None:
        source = []
        base_path = _get_header_dir_path()
        for file_path in _cupy_header_list + _cupy_extra_header_list:
            header_path = os.path.join(base_path, file_path)
            with open(header_path) as header_file:
                source.append(header_file.read())
        _header_source = '\n'.join(source)
    return _header_source


_cuda_path = None


def _get_cuda_path():
    global _cuda_path
    if _cuda_path is None:
        _cuda_path = os.getenv('CUDA_PATH', None)
        if _cuda_path is not None:
            return _cuda_path

        for p in os.getenv('PATH', '').split(os.pathsep):
            for cmd in ('nvcc', 'nvcc.exe'):
                nvcc_path = os.path.join(p, cmd)
                if not os.path.exists(nvcc_path):
                    continue
                nvcc_dir = os.path.dirname(
                    os.path.abspath(nvcc_path))
                _cuda_path = os.path.normpath(
                    os.path.join(nvcc_dir, '..'))
                return _cuda_path

        if os.path.exists('/usr/local/cuda'):
            _cuda_path = '/usr/local/cuda'

    return _cuda_path


def compile_with_cache(
        source, options=(), arch=None, cachd_dir=None):
    source = _cupy_header + source
    extra_source = _get_header_source()
    options += ('-I%s' % _get_header_dir_path(),)

    if cuda.get_runtime_version() >= 9000:
        cuda_path = _get_cuda_path()
        if cuda_path is None:
            warnings.warn('Please set the CUDA path ' +
                          'to environment variable `CUDA_PATH`')
        else:
            path = os.path.join(cuda_path, 'include')
            options += ('-I ' + path,)

    return cuda.compile_with_cache(source, options, arch, cachd_dir,
                                   extra_source)
