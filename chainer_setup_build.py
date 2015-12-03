from __future__ import print_function
import copy
import distutils
import os
from os import path
import pkg_resources
import shutil
import subprocess
import sys
import tempfile

import setuptools
from setuptools.command import build_ext


dummy_extension = setuptools.Extension('chainer', ['chainer.c'])

MODULES = [
    {
        'name': 'cuda',
        'file': [
            'cupy.core.core',
            'cupy.core.flags',
            'cupy.cuda.cublas',
            'cupy.cuda.curand',
            'cupy.cuda.device',
            'cupy.cuda.driver',
            'cupy.cuda.memory',
            'cupy.cuda.function',
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
    nvcc_path = search_on_path(('nvcc', 'nvcc.exe'))
    cuda_path_default = None
    if nvcc_path is None:
        print('**************************************************************')
        print('*** WARNING: nvcc not in path.')
        print('*** WARNING: Please set path to nvcc.')
        print('**************************************************************')
    else:
        cuda_path_default = path.normpath(
            path.join(path.dirname(nvcc_path), '..'))

    cuda_path = os.environ.get('CUDA_PATH', '')  # Nvidia default on Windows
    if len(cuda_path) > 0 and cuda_path != cuda_path_default:
        print('**************************************************************')
        print('*** WARNING: nvcc path != CUDA_PATH')
        print('*** WARNING: nvcc path: %s', cuda_path_default)
        print('*** WARNING: CUDA_PATH: %s', cuda_path)
        print('**************************************************************')

    if not path.exists(cuda_path):
        cuda_path = cuda_path_default

    if not cuda_path and path.exists('/usr/local/cuda'):
        cuda_path = '/usr/local/cuda'

    include_dirs = []
    library_dirs = []
    define_macros = []

    if cuda_path:
        include_dirs.append(path.join(cuda_path, 'include'))
        if sys.platform == 'win32':
            library_dirs.append(path.join(cuda_path, 'bin'))
            library_dirs.append(path.join(cuda_path, 'lib', 'x64'))
        else:
            library_dirs.append(path.join(cuda_path, 'lib64'))
            library_dirs.append(path.join(cuda_path, 'lib'))
    if sys.platform == 'darwin':
        library_dirs.append('/usr/local/cuda/lib')

    return {
        'include_dirs': include_dirs,
        'library_dirs': library_dirs,
        'define_macros': define_macros,
        'language': 'c++',
    }


def localpath(*args):
    return path.abspath(path.join(path.dirname(__file__), *args))


def get_path(key):
    return os.environ.get(key, '').split(os.pathsep)


def search_on_path(filenames):
    for p in get_path('PATH'):
        for filename in filenames:
            full = path.join(p, filename)
            if path.exists(full):
                return path.abspath(full)


def check_include(dirs, file_path):
    return any(path.exists(path.join(dir, file_path)) for dir in dirs)


def check_readthedocs_environment():
    return os.environ.get('READTHEDOCS', None) == 'True'


def check_library(compiler, includes=[], libraries=[],
                  include_dirs=[], library_dirs=[]):
    temp_dir = tempfile.mkdtemp()

    try:
        source = '''
        int main(int argc, char* argv[]) {
          return 0;
        }
        '''
        fname = os.path.join(temp_dir, 'a.cpp')
        with open(fname, 'w') as f:
            for header in includes:
                f.write('#include <%s>\n' % header)
            f.write(source)

        try:
            objects = compiler.compile([fname], output_dir=temp_dir,
                                       include_dirs=include_dirs)
        except distutils.errors.CompileError:
            return False

        try:
            compiler.link_shared_lib(objects,
                                     os.path.join(temp_dir, 'a'),
                                     libraries=libraries,
                                     library_dirs=library_dirs)
        except (distutils.errors.LinkError, TypeError):
            return False

        return True

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def make_extensions(options, compiler):

    """Produce a list of Extension instances which passed to cythonize()."""

    no_cuda = options['no_cuda']
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
    if sys.platform != 'win32':
        settings['runtime_library_dirs'] = settings['library_dirs']

    if options['linetrace']:
        settings['define_macros'].append(('CYTHON_TRACE', '1'))
        settings['define_macros'].append(('CYTHON_TRACE_NOGIL', '1'))
    if no_cuda:
        settings['define_macros'].append(('CUPY_NO_CUDA', '1'))

    ret = []
    for module in MODULES:
        print('Include directories:', settings['include_dirs'])
        print('Library directories:', settings['library_dirs'])

        if not no_cuda:
            if not check_library(compiler,
                                 includes=module['include'],
                                 include_dirs=settings['include_dirs']):
                print('**************************************************')
                print('*** Include files not found: %s' % module['include'])
                print('*** Skip installing %s support' % module['name'])
                print('*** Check your CPATH environment variable')
                print('**************************************************')
                continue

            if not check_library(compiler,
                                 libraries=module['libraries'],
                                 library_dirs=settings['library_dirs']):
                print('**************************************************')
                print('*** Cannot link libraries: %s' % module['libraries'])
                print('*** Skip installing %s support' % module['name'])
                print('*** Check your LIBRARY_PATH environment variable')
                print('**************************************************')
                continue

        s = settings.copy()
        if not no_cuda:
            s['libraries'] = module['libraries']
        ret.extend([
            setuptools.Extension(f, [path.join(*f.split('.')) + '.pyx'], **s)
            for f in module['file']])
    return ret


_arg_options = {}


def parse_args():
    global _arg_options
    _arg_options['profile'] = '--cupy-profile' in sys.argv
    if _arg_options['profile']:
        sys.argv.remove('--cupy-profile')

    cupy_coverage = '--cupy-coverage' in sys.argv
    if cupy_coverage:
        sys.argv.remove('--cupy-coverage')
    _arg_options['linetrace'] = cupy_coverage
    _arg_options['annotate'] = cupy_coverage

    _arg_options['no_cuda'] = '--cupy-no-cuda' in sys.argv
    if _arg_options['no_cuda']:
        sys.argv.remove('--cupy-no-cuda')
    if check_readthedocs_environment():
        _arg_options['no_cuda'] = True


def cythonize(extensions, force=False, annotate=False, compiler_directives={}):
    cython_pkg = pkg_resources.get_distribution('cython')
    cython_path = path.join(cython_pkg.location, 'cython.py')
    print("cython path:%s" % cython_pkg.location)
    cython_cmdbase = [sys.executable, cython_path]
    subprocess.check_call(cython_cmdbase + ['--version'])

    cython_cmdbase.extend(['--fast-fail', '--verbose', '--cplus'])
    ret = []
    for ext in extensions:
        cmd = list(cython_cmdbase)
        for i in compiler_directives.items():
            cmd.append('--directive')
            cmd.append('%s=%s' % i)
        cpp_files = [path.splitext(f)[0] + ".cpp" for f in ext.sources]
        cmd += ext.sources
        subprocess.check_call(cmd)
        ext = copy.copy(ext)
        ext.sources = cpp_files
        ret.append(ext)
    return ret


class chainer_build_ext(build_ext.build_ext):

    """`build_ext` command for cython files."""

    def finalize_options(self):
        ext_modules = self.distribution.ext_modules
        if dummy_extension in ext_modules:
            print('Executing cythonize()')
            print('Options:', _arg_options)

            directive_keys = ('linetrace', 'profile')
            directives = {key: _arg_options[key] for key in directive_keys}

            cythonize_option_keys = ('annotate',)
            cythonize_options = {
                key: _arg_options[key] for key in cythonize_option_keys}

            compiler = distutils.ccompiler.new_compiler(self.compiler)
            distutils.sysconfig.customize_compiler(compiler)

            extensions = make_extensions(_arg_options, compiler)
            extensions = cythonize(
                extensions,
                force=True,
                compiler_directives=directives,
                **cythonize_options)

            # Modify ext_modules for cython
            ext_modules.remove(dummy_extension)
            ext_modules.extend(extensions)

        build_ext.build_ext.finalize_options(self)
