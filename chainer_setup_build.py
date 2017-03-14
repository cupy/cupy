from __future__ import print_function
from distutils import ccompiler
from distutils import errors
from distutils import sysconfig
from distutils import unixccompiler
import os
from os import path
import re
import subprocess
import sys

import pkg_resources
import setuptools
from setuptools.command import build_ext

from install import build
from install import utils


required_cython_version = pkg_resources.parse_version('0.24.0')

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
            'cupy.cuda.pinned_memory',
            'cupy.cuda.profiler',
            'cupy.cuda.nvtx',
            'cupy.cuda.function',
            'cupy.cuda.runtime',
            'cupy.util',
        ],
        'include': [
            'cublas_v2.h',
            'cuda.h',
            'cuda_profiler_api.h',
            'cuda_runtime.h',
            'curand.h',
            'nvToolsExt.h',
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
    },
    {
        'name': 'cusolver',
        'file': [
            'cupy.cuda.cusolver',
        ],
        'include': [
            'cusolverDn.h',
        ],
        'libraries': [
            'cusolver',
        ],
        'check_method': build.check_cusolver_version,
    },
    {
        # The value of the key 'file' is a list that contains extension names
        # or tuples of an extension name and a list of other souces files
        # required to build the extension such as .cpp files and .cu files.
        #
        #   <extension name> | (<extension name>, a list of <other source>)
        #
        # The extension name is also interpreted as the name of the Cython
        # source file required to build the extension with appending '.pyx'
        # file extension.
        'name': 'thrust',
        'file': [
            ('cupy.cuda.thrust', ['cupy/cuda/cupy_thrust.cu']),
        ],
        'include': [
            'thrust/device_ptr.h',
            'thrust/sort.h',
        ],
        'libraries': [
            'cudart',
        ],
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


def module_extension_sources(file, use_cython, no_cuda):
    pyx, others = ensure_module_file(file)
    ext = '.pyx' if use_cython else '.cpp'
    pyx = path.join(*pyx.split('.')) + ext

    # If CUDA SDK is not available, remove CUDA C files from extension sources
    # and use stubs defined in header files.
    if no_cuda:
        others1 = []
        for source in others:
            base, ext = os.path.splitext(source)
            if ext == '.cu':
                continue
            others1.append(source)
        others = others1

    return [pyx] + others


def check_readthedocs_environment():
    return os.environ.get('READTHEDOCS', None) == 'True'


def check_library(compiler, includes=(), libraries=(),
                  include_dirs=(), library_dirs=()):

    source = ''.join(['#include <%s>\n' % header for header in includes])
    source += 'int main(int argc, char* argv[]) {return 0;}'
    try:
        # We need to try to build a shared library because distutils
        # uses different option to build an executable and a shared library.
        # Especially when a user build an executable, distutils does not use
        # LDFLAGS environment variable.
        build.build_shlib(compiler, source, libraries,
                          include_dirs, library_dirs)
    except Exception as e:
        print(e)
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

    # This is a workaround for Anaconda.
    # Anaconda installs libstdc++ from GCC 4.8 and it is not compatible
    # with GCC 5's new ABI.
    settings['define_macros'].append(('_GLIBCXX_USE_CXX11_ABI', '0'))

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
            sources = module_extension_sources(f, use_cython, no_cuda)
            extension = setuptools.Extension(name, sources, **s)
            ret.append(extension)

    return ret


def parse_args():
    cupy_profile = '--cupy-profile' in sys.argv
    if cupy_profile:
        sys.argv.remove('--cupy-profile')
    cupy_coverage = '--cupy-coverage' in sys.argv
    if cupy_coverage:
        sys.argv.remove('--cupy-coverage')
    no_cuda = '--cupy-no-cuda' in sys.argv
    if no_cuda:
        sys.argv.remove('--cupy-no-cuda')

    arg_options = {
        'profile': cupy_profile,
        'linetrace': cupy_coverage,
        'annotate': cupy_coverage,
        'no_cuda': no_cuda,
    }
    if check_readthedocs_environment():
        arg_options['no_cuda'] = True
    return arg_options


def check_cython_version():
    try:
        import Cython
        cython_version = pkg_resources.parse_version(Cython.__version__)
        return cython_version >= required_cython_version
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
        extensions, verbose=True,
        compiler_directives=directives, **cythonize_options)


def check_extensions(extensions):
    for x in extensions:
        for f in x.sources:
            if not path.isfile(f):
                msg = ('Missing file: %s\n' % f +
                       'Please install Cython. ' +
                       'Please also check the version of Cython.\n' +
                       'See http://docs.chainer.org/en/stable/install.html')
                raise RuntimeError(msg)


def get_ext_modules():
    arg_options = parse_args()
    print('Options:', arg_options)

    # We need to call get_config_vars to initialize _config_vars in distutils
    # see #1849
    sysconfig.get_config_vars()
    compiler = ccompiler.new_compiler()
    sysconfig.customize_compiler(compiler)

    use_cython = check_cython_version()
    extensions = make_extensions(arg_options, compiler, use_cython)

    if use_cython:
        extensions = cythonize(extensions, arg_options)

    check_extensions(extensions)
    return extensions


def _nvcc_gencode_options():
    """Returns NVCC gencode options generated from NVCC command line help."""
    help_string = subprocess.check_output(
        ['nvcc', '--help']).decode('ascii').replace('\n', '')

    arch_options = re.findall("'(compute_\d{2})'", help_string)
    arch_options = sorted(list(set(arch_options)))
    arch_options = list(filter(lambda x: x >= 'compute_30', arch_options))

    code_options = re.findall("'(sm_\d{2})'", help_string)
    code_options = sorted(list(set(code_options)))
    code_options = list(filter(lambda x: x >= 'sm_30', code_options))

    pairs = []
    for code_option in code_options:
        arch_option = code_option.replace('sm_', 'compute_')
        if arch_option not in arch_options:
            msg = "No virtual architecture corresponding to '{}'.".format(
                code_option)
            raise ValueError(msg)
        pairs.append((arch_option, code_option))

    gencode_options = []
    for pair in pairs:
        gencode_options.append('-gencode=arch={},code={}'.format(*pair))

    return gencode_options


class NvidiaCCompiler(unixccompiler.UnixCCompiler):
    compiler_type = "nvidia"
    src_extensions = ['.cpp', '.cu']

    def __init__(self, verbose=0, dry_run=0, force=0):
        unixccompiler.UnixCCompiler.__init__(
            self, verbose=verbose, dry_run=dry_run, force=force)
        postargs = _nvcc_gencode_options()
        if sys.platform == 'win32':
            self.set_executables(compiler=['nvcc', '-O'] + postargs,
                                 compiler_so=['nvcc', '-O'] + postargs,
                                 compiler_cxx=['nvcc', '-O'] + postargs,
                                 linker_so=['nvcc', '-shared'],
                                 linker_exe=['nvcc'])
        else:
            postargs += ['--compiler-options=-fPIC']
            self.set_executables(compiler=['nvcc', '-O'] + postargs,
                                 compiler_so=['nvcc', '-O'] + postargs,
                                 compiler_cxx=['g++', '-O'] + postargs,
                                 linker_so=['nvcc', '-shared'],
                                 linker_exe=['nvcc'])


class custom_build_ext(build_ext.build_ext):

    """Custom `build_ext` command to include CUDA C source files."""

    def run(self):
        if build.get_nvcc_path() is not None:
            def wrap_new_compiler(func):
                def _wrap_new_compiler(*args, **kwargs):
                    try:
                        return func(*args, **kwargs)
                    except errors.DistutilsPlatformError:
                        return NvidiaCCompiler(
                            None, kwargs["dry_run"], kwargs["force"])
                return _wrap_new_compiler
            ccompiler.new_compiler = wrap_new_compiler(ccompiler.new_compiler)
            self.compiler = "nvidia"
        build_ext.build_ext.run(self)
