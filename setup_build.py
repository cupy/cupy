from __future__ import print_function
import ctypes
from distutils.command.build_ext import build_ext
import os
from os import path
import platform
import sys

from setuptools import Extension

MODULES = [
    {
        'name': 'cuda',
        'file': [
            'cupy.carray',
            'cupy.cuda.cublas',
            'cupy.cuda.curand',
            'cupy.cuda.device',
            'cupy.cuda.driver',
            'cupy.cuda.memory',
            'cupy.cuda.module',
            'cupy.cuda.runtime',
        ],
        'include': [
            'cublas_v2.h',
            'cuda.h',
            'cuda_runtime.h',
        ],
        'libraries': [
            'cublas',
            'cuda',
            'cudart',
            'curand',
        ],
    },
    {
        'name': 'cudnn',
        'file': [
            'cupy.cuda.cudnn',
        ],
        'include': [
            'cudnn.h',
        ],
        'libraries': [
            'cudnn',
        ],
    }
]


_support_cuda_versions = (75, 70, 65)


def get_compiler_setting():
    include_dirs = []
    library_dirs = []
    define_macros = []
    if sys.platform.startswith('win'):
        include_dirs = [localpath('windows')]
        library_dirs = []
        cuda_path = os.environ.get('CUDA_PATH', None)
        if cuda_path:
            include_dirs.append(path.join(cuda_path, 'include'))
            library_dirs.append(path.join(cuda_path, 'bin'))
            library_dirs.append(path.join(cuda_path, 'lib\\x64'))
    else:
        include_dirs = get_path('CPATH') + ['/usr/local/cuda/include']
        library_dirs = get_path('LD_LIBRARY_PATH') + [
            '/opt/local/lib', '/usr/local/lib']

    return {
        'include_dirs': include_dirs,
        'library_dirs': library_dirs,
        'define_macros': define_macros,
    }


def localpath(*args):
    return path.abspath(path.join(path.dirname(__file__), *args))


def get_path(key):
    splitter = ';' if sys.platform.startswith('win') else ':'
    return os.environ.get(key, "").split(splitter)


def load_library(names):
    if isinstance(names, str):
        names = [names]
    if 'linux' in sys.platform:
        template = 'lib%s.so'
        module = ctypes.cdll
    elif 'darwin' == sys.platform:
        template = 'lib%s.dylib'
        module = ctypes.cdll
    elif 'win32' == sys.platform:
        template = '%s.dll'
        module = ctypes.windll
    else:
        raise RuntimeError('Unsupported platform: %s' % sys.platform)

    names = [template % i for i in names]
    for name in names:
        try:
            return module.LoadLibrary(name)
        except OSError:
            pass
    else:
        raise OSError('library not found. %s' % names)


def get_windows_cuda_library_names(name):
    bit = 64 if platform.machine().endswith('64') else 32
    return ['%s%s_%s' % (name, bit, ver) for ver in _support_cuda_versions]


def check_include(dirs, file_path):
    return any(path.exists(path.join(dir, file_path)) for dir in dirs)


def make_extensions():

    """Produce a list of Extension instances which passed to cythonize()."""
    import numpy

    settings = get_compiler_setting()
    try:
        numpy_includes = numpy.get_include()
    except AttributeError:
        # if numpy is not installed get the headers from the .egg directory
        import numpy.core
        numpy_includes = os.path.join(
            os.path.dirname(numpy.core.__file__), 'include')

    settings['include_dirs'] += [numpy_includes]

    if os.name != 'nt':
        settings['runtime_library_dirs'] = settings['library_dirs']

    include_dirs = settings['include_dirs']
    modules = []
    for m in MODULES:
        include = [i for i in m['include']
                   if not check_include(include_dirs, i)]
        if include:
            print("Missing include files.", *include)
            continue
        s = settings.copy()
        s['libraries'] = m['libraries']
        modules.extend([
            Extension(f, [localpath(path.join(*f.split('.')) + '.pyx')], **s)
            for f in m['file']])
    return modules


class chainer_build_ext(build_ext):

    def run(self):

        """Distutils calls this method to run the command."""

        from Cython.Build import cythonize

        print("Executing cythonize()")
        self.extensions = cythonize(make_extensions(), force=True)
        build_ext.run(self)
