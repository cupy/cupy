from __future__ import print_function
from distutils import ccompiler
from distutils import sysconfig
import os
from os import path
import sys

import pkg_resources
import setuptools
from setuptools.command import build_ext

from install import build
from install import utils


require_cython_version = pkg_resources.parse_version('0.24.0')

MODULES = [
    {
        'name': 'cuda',
        'file': [
            # The value of the key 'file' is a list that contains extension
            # names of tuples of an extension name and a list of other source
            # files required such as .cpp files and .cu files.
            #
            #   <extension name> | (<extension name>, list of <other source>)
            #
            # The extension names are interpreted as the names of Python
            # extensions to be built as well as the names of the Cython source
            # files with appending '.pyx' extension from which the extensions
            # are built.
            'cupy.core.core',
            'cupy.core.flags',
            'cupy.core.internal',
            'cupy.cuda.cublas',
            'cupy.cuda.curand',
            'cupy.cuda.device',
            'cupy.cuda.driver',
            'cupy.cuda.memory',
            'cupy.cuda.pinned_memory',
            'cupy.cuda.profiler',
            'cupy.cuda.nvtx',
            'cupy.cuda.function',
            'cupy.cuda.runtime',
            ('cupy.cuda.thrust', ['cupy/cuda/cupy_thrust.cu']),
            'cupy.util',
        ],
        'include': [
            'cublas_v2.h',
            'cuda.h',
            'cuda_profiler_api.h',
            'cuda_runtime.h',
            'curand.h',
            'nvToolsExt.h',
            'thrust/device_ptr.h',
            'thrust/sort.h',
        ],
        'libraries': [
            'cublas',
            'cuda',
            'cudart',
            'curand',
            'nvToolsExt',
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

if sys.platform == 'win32':
    mod_cuda = MODULES[0]
    mod_cuda['file'].remove('cupy.cuda.nvtx')
    mod_cuda['include'].remove('nvToolsExt.h')
    mod_cuda['libraries'].remove('nvToolsExt')


def ensure_module_file(file):
    if isinstance(file, tuple):
        return file
    else:
        return (file, [])


def module_extension_name(file):
    return ensure_module_file(file)[0]


def module_extension_sources(file, use_cython):
    pyx, others = ensure_module_file(file)
    ext = '.pyx' if use_cython else '.cpp'
    pyx = path.join(*pyx.split('.')) + ext
    return [pyx] + others


def check_readthedocs_environment():
    return os.environ.get('READTHEDOCS', None) == 'True'


def check_library(compiler, includes=(), libraries=(),
                  include_dirs=(), library_dirs=()):

    source = ''.join(['#include <%s>\n' % header for header in includes])
    source += 'int main(int argc, char* argv[]) {return 0;}'
    try:
        build.build_and_run(compiler, source, libraries,
                            include_dirs, library_dirs)
    except Exception:
        return False
    return True


def make_extensions(options, compiler, use_cython):
    """Produce a list of Extension instances which passed to cythonize()."""

    no_cuda = options['no_cuda']
    settings = build.get_compiler_setting()

    include_dirs = settings['include_dirs']

    settings['include_dirs'] = [
        x for x in include_dirs if path.exists(x)]
    settings['library_dirs'] = [
        x for x in settings['library_dirs'] if path.exists(x)]
    if sys.platform != 'win32':
        settings['runtime_library_dirs'] = settings['library_dirs']
    if sys.platform == 'darwin':
        args = settings.setdefault('extra_link_args', [])
        args.append(
            '-Wl,' + ','.join('-rpath,' + p
                              for p in settings['library_dirs']))
        # -rpath is only supported when targetting Mac OS X 10.5 or later
        args.append('-mmacosx-version-min=10.5')

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
                    'Check your CFLAGS environment variable')
                continue

            if not check_library(compiler,
                                 libraries=module['libraries'],
                                 library_dirs=settings['library_dirs']):
                utils.print_warning(
                    'Cannot link libraries: %s' % module['libraries'],
                    'Skip installing %s support' % module['name'],
                    'Check your LDFLAGS environment variable')
                continue

            if 'check_method' in module and \
               not module['check_method'](compiler, settings):
                continue

        s = settings.copy()
        if not no_cuda:
            s['libraries'] = module['libraries']
        for f in module['file']:
            name = module_extension_name(f)
            sources = module_extension_sources(f, use_cython)
            extension = setuptools.Extension(name, sources, **s)
            ret.append(extension)
    return ret


def parse_args():
    arg_options = dict()
    arg_options['profile'] = '--cupy-profile' in sys.argv
    if arg_options['profile']:
        sys.argv.remove('--cupy-profile')

    cupy_coverage = '--cupy-coverage' in sys.argv
    if cupy_coverage:
        sys.argv.remove('--cupy-coverage')
    arg_options['linetrace'] = cupy_coverage
    arg_options['annotate'] = cupy_coverage

    arg_options['no_cuda'] = '--cupy-no-cuda' in sys.argv
    if arg_options['no_cuda']:
        sys.argv.remove('--cupy-no-cuda')
    if check_readthedocs_environment():
        arg_options['no_cuda'] = True
    return arg_options


def check_cython_version():
    try:
        import Cython
        cython_version = pkg_resources.parse_version(Cython.__version__)
        return cython_version >= require_cython_version
    except ImportError:
        return False


def cythonize(extensions, arg_options):
    import Cython.Build

    directive_keys = ('linetrace', 'profile')
    directives = {key: arg_options[key] for key in directive_keys}

    cythonize_option_keys = ('annotate',)
    cythonize_options = {key: arg_options[key]
                         for key in cythonize_option_keys}

    return Cython.Build.cythonize(
        extensions, language="c++", verbose=True,
        compiler_directives=directives, **cythonize_options)


def check_extensions(extensions):
    for x in extensions:
        for f in x.sources:
            if not path.isfile(f):
                msg = ('Missing file: %s\n' % f +
                       'Please install Cython.\n' +
                       'See http://docs.chainer.org/en/stable/install.html')
                raise RuntimeError(msg)


def get_ext_modules():
    arg_options = parse_args()
    print('Options:', arg_options)

    compiler = ccompiler.new_compiler()
    sysconfig.customize_compiler(compiler)

    use_cython = check_cython_version()
    extensions = make_extensions(arg_options, compiler, use_cython)

    if use_cython:
        extensions = cythonize(extensions, arg_options)

    check_extensions(extensions)
    return extensions


def customize_compiler_for_nvcc(compiler):
    compiler.src_extensions.append('.cu')
    default_compiler_so = compiler.compiler_so
    super = compiler._compile

    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == '.cu':
            nvcc_path = build.get_nvcc_path()
            compiler.set_executable('compiler_so', nvcc_path)
            postargs = ['-arch=sm_20',
                        '--ptxas-options=-v',
                        '-c',
                        '--compiler-options',
                        "'-fPIC'"]
        else:
            postargs = []

        super(obj, src, ext, cc_args, postargs, pp_opts)
        compiler.compiler_so = default_compiler_so

    compiler._compile = _compile


class custom_build_ext(build_ext.build_ext):

    """Custom `build_ext` command to include CUDA C source files."""

    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_ext.build_extensions(self)
