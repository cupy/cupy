from __future__ import print_function
import copy
import distutils
import os
from os import path
import shutil
import subprocess
import sys
import tempfile

import pkg_resources
import setuptools
from setuptools.command import build_ext

from install import build
from install import utils


dummy_extension = setuptools.Extension('chainer', ['chainer.c'])
cython_version = '0.23.0'
MODULES = [
    {
        'name': 'cuda',
        'file': [
            'cupy.core.core',
            'cupy.core.flags',
            'cupy.core.internal',
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
        'check_method': build.check_cuda_version,
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
        'check_method': build.check_cudnn_version,
    }
]


def get_compiler_setting():
    nvcc_path = search_on_path(('nvcc', 'nvcc.exe'))
    cuda_path_default = None
    if nvcc_path is None:
        utils.print_warning('nvcc not in path.',
                            'Please set path to nvcc.')
    else:
        cuda_path_default = path.normpath(
            path.join(path.dirname(nvcc_path), '..'))

    cuda_path = os.environ.get('CUDA_PATH', '')  # Nvidia default on Windows
    if len(cuda_path) > 0 and cuda_path != cuda_path_default:
        utils.print_warning(
            'nvcc path != CUDA_PATH',
            'nvcc path: %s' % cuda_path_default,
            'CUDA_PATH: %s' % cuda_path)

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

    include_dirs = settings['include_dirs']

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
                utils.print_warning(
                    'Include files not found: %s' % module['include'],
                    'Skip installing %s support' % module['name'],
                    'Check your CPATH environment variable')
                continue

            if not check_library(compiler,
                                 libraries=module['libraries'],
                                 library_dirs=settings['library_dirs']):
                utils.print_warning(
                    'Cannot link libraries: %s' % module['libraries'],
                    'Skip installing %s support' % module['name'],
                    'Check your LIBRARY_PATH environment variable')
                continue

            if 'check_method' in module and \
               not module['check_method'](compiler, settings):
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


def get_cython_pkg():
    try:
        return pkg_resources.get_distribution('cython')
    except pkg_resources.DistributionNotFound:
        return None


def run_command(cmd):
    try:
        subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        msg = '''Command %r failed:

  command: %s
  return code: %d
  output:

%s''' % (cmd[0], ' '.join(e.cmd), e.returncode, e.output)
        raise distutils.errors.DistutilsExecError(msg)


def cythonize(extensions, force=False, annotate=False, compiler_directives={}):
    cython_location = get_cython_pkg().location
    cython_path = path.join(cython_location, 'cython.py')
    print("cython path:%s" % cython_location)
    cython_cmdbase = [sys.executable, cython_path]
    run_command(cython_cmdbase + ['--version'])

    cython_cmdbase.extend(['--fast-fail', '--verbose', '--cplus'])
    for ext in extensions:
        cmd = list(cython_cmdbase)
        for i in compiler_directives.items():
            cmd.append('--directive')
            cmd.append('%s=%s' % i)
        run_command(cmd + ext.sources)


def to_cpp_extensions(extensions):
    ret = []
    for x in extensions:
        ext = copy.copy(x)
        ext.sources = [path.splitext(f)[0] + ".cpp" for f in x.sources]
        ret.append(ext)
    return ret


def check_extensions(extensions):
    for x in extensions:
        for f in x.sources:
            if not path.isfile(f):
                msg = ('Missing file: %s\n' % f +
                       'Please install Cython.\n' +
                       'See http://docs.chainer.org/en/stable/install.html')
                raise RuntimeError(msg)


class chainer_build_ext(build_ext.build_ext):

    """`build_ext` command for cython files."""

    def finalize_options(self):
        ext_modules = self.distribution.ext_modules
        if dummy_extension in ext_modules:
            print('Executing cythonize')
            print('Options:', _arg_options)

            directive_keys = ('linetrace', 'profile')
            directives = {key: _arg_options[key] for key in directive_keys}

            cythonize_option_keys = ('annotate',)
            cythonize_options = {
                key: _arg_options[key] for key in cythonize_option_keys}

            compiler = distutils.ccompiler.new_compiler(self.compiler)
            distutils.sysconfig.customize_compiler(compiler)

            extensions = make_extensions(_arg_options, compiler)

            cython = get_cython_pkg()
            req_version = pkg_resources.parse_version(cython_version)
            if cython is not None and cython.parsed_version > req_version:
                cythonize(extensions, force=True,
                          compiler_directives=directives, **cythonize_options)

            extensions = to_cpp_extensions(extensions)
            check_extensions(extensions)

            # Modify ext_modules for cython
            ext_modules.remove(dummy_extension)
            ext_modules.extend(extensions)

        build_ext.build_ext.finalize_options(self)
