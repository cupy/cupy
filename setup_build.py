from __future__ import print_function
from distutils.command import build_ext
import os
from os import path
import sys

from setuptools import extension


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


def get_compiler_setting():
    include_dirs = []
    library_dirs = []
    define_macros = []
    if sys.platform == 'win32':
        include_dirs = [localpath('windows')]
        library_dirs = []
        cuda_path = os.environ.get('CUDA_PATH', None)
        if cuda_path:
            include_dirs.append(path.join(cuda_path, 'include'))
            library_dirs.append(path.join(cuda_path, 'bin'))
            library_dirs.append(path.join(cuda_path, 'lib', 'x64'))
    else:
        include_dirs = get_path('CPATH') + ['/usr/local/cuda/include']
        library_dirs = get_path('LD_LIBRARY_PATH') + [
            '/usr/local/cuda/lib64',
            '/opt/local/lib',
            '/usr/local/lib']

    return {
        'include_dirs': include_dirs,
        'library_dirs': library_dirs,
        'define_macros': define_macros,
    }


def localpath(*args):
    return path.abspath(path.join(path.dirname(__file__), *args))


def get_path(key):
    splitter = ';' if sys.platform == 'win32' else ':'
    return os.environ.get(key, "").split(splitter)


def check_include(dirs, file_path):
    return any(path.exists(path.join(dir, file_path)) for dir in dirs)


def make_extensions():

    """Produce a list of Extension instances which passed to cythonize()."""

    settings = get_compiler_setting()

    try:
        import numpy
        numpy_include = numpy.get_include()
    except AttributeError:
        # if numpy is not installed get the headers from the .egg directory
        import numpy.core
        numpy_include = path.join(
            path.dirname(numpy.core.__file__), 'include')

    include_dirs = settings['include_dirs']
    include_dirs.append(numpy_include)

    ret = []
    for module in MODULES:
        include = [i for i in module['include']
                   if not check_include(include_dirs, i)]
        if include:
            print("Missing include files.", *include)
            continue
        s = settings.copy()
        s['libraries'] = module['libraries']
        ret.extend([extension.Extension(
                        f, [localpath(path.join(*f.split('.')) + '.pyx')], **s)
                    for f in module['file']])
    return ret


class chainer_build_ext(build_ext.build_ext):

    def run(self):

        """Distutils calls this method to run the command."""

        from Cython.Build import cythonize

        print("Executing cythonize()")
        self.extensions = cythonize(make_extensions(), force=True)

        build_ext.build_ext.run(self)
