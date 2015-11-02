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
            'cupy.core',
            'cupy.cuda.cublas',
            'cupy.cuda.curand',
            'cupy.cuda.device',
            'cupy.cuda.driver',
            'cupy.cuda.memory',
            'cupy.cuda.module',
            'cupy.cuda.runtime',
            'cupy.util',
        ],
        'include': [
            'cublas_v2.h',
            'cuda.h',
            'cuda_runtime.h',
            'curand.h',
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
        'language': 'c++',
    }


def localpath(*args):
    return path.abspath(path.join(path.dirname(__file__), *args))


def get_path(key):
    splitter = ';' if sys.platform == 'win32' else ':'
    return os.environ.get(key, "").split(splitter)


def check_include(dirs, file_path):
    return any(path.exists(path.join(dir, file_path)) for dir in dirs)


def make_extensions(options):

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

    settings['include_dirs'] = [
        x for x in include_dirs if path.exists(x)]
    settings['library_dirs'] = [
        x for x in settings['library_dirs'] if path.exists(x)]

    if options['linetrace']:
        settings['define_macros'].append(('CYTHON_TRACE', '1'))
    nocuda = options['nocuda']

    ret = []
    for module in MODULES:
        print('Include directories:', settings['include_dirs'])
        print('Library directories:', settings['library_dirs'])

        include = [i for i in module['include']
                   if not check_include(include_dirs, i)]
        if not nocuda and include:
            print('Missing include files:', include)
            continue

        s = settings.copy()
        if not nocuda:
            s['libraries'] = module['libraries']
        ret.extend([
            extension.Extension(
                f, [localpath(path.join(*f.split('.')) + '.pyx')], **s)
            for f in module['file']])
    return ret


_arg_options = {}


def parse_args():
    global _arg_options
    _arg_options['profile'] = '--cupy-profile' in sys.argv
    if _arg_options['profile']:
        sys.argv.remove('--cupy-profile')

    _arg_options['linetrace'] = '--cupy-linetrace' in sys.argv
    if _arg_options['linetrace']:
        sys.argv.remove('--cupy-linetrace')

    _arg_options['nocuda'] = '--cupy-nocuda' in sys.argv
    if _arg_options['nocuda']:
        sys.argv.remove('--cupy-nocuda')
    if os.environ.get('READTHEDOCS', None):
        _arg_options['nocuda'] = True


class chainer_build_ext(build_ext.build_ext):

    def run(self):

        """Distutils calls this method to run the command."""

        from Cython.Build import cythonize

        print('Executing cythonize()')
        print('Options:', _arg_options)

        directive_keys = ('linetrace', 'profile')
        directives = {key: _arg_options[key] for key in directive_keys}
        compile_time_env = {
            'CUPY_USE_CUDA': not _arg_options['nocuda'],
        }

        self.extensions = cythonize(
            make_extensions(_arg_options),
            force=True,
            compiler_directives=directives,
            compile_time_env=compile_time_env)

        build_ext.build_ext.run(self)
