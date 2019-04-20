from __future__ import print_function

import argparse
import copy
from distutils import ccompiler
from distutils import errors
from distutils import msvccompiler
from distutils import sysconfig
from distutils import unixccompiler
import glob
import os
from os import path
import shutil
import sys

import pkg_resources
import setuptools
from setuptools.command import build_ext
from setuptools.command import sdist

from install import build
from install.build import PLATFORM_DARWIN
from install.build import PLATFORM_LINUX
from install.build import PLATFORM_WIN32


required_cython_version = pkg_resources.parse_version('0.28.0')
ignore_cython_versions = [
]

MODULES = [
    {
        'name': 'cuda',
        'file': [
            'cupy.core._dtype',
            'cupy.core._kernel',
            'cupy.core._routines_indexing',
            'cupy.core._routines_logic',
            'cupy.core._routines_manipulation',
            'cupy.core._routines_math',
            'cupy.core._routines_sorting',
            'cupy.core._routines_statistics',
            'cupy.core._scalar',
            'cupy.core.core',
            'cupy.core.dlpack',
            'cupy.core.flags',
            'cupy.core.internal',
            'cupy.core.fusion',
            'cupy.core.raw',
            'cupy.cuda.cublas',
            'cupy.cuda.cufft',
            'cupy.cuda.curand',
            'cupy.cuda.cusparse',
            'cupy.cuda.device',
            'cupy.cuda.driver',
            'cupy.cuda.memory',
            'cupy.cuda.memory_hook',
            'cupy.cuda.nvrtc',
            'cupy.cuda.pinned_memory',
            'cupy.cuda.profiler',
            'cupy.cuda.function',
            'cupy.cuda.stream',
            'cupy.cuda.runtime',
            'cupy.util',
        ],
        'include': [
            'cublas_v2.h',
            'cuda.h',
            'cuda_profiler_api.h',
            'cuda_runtime.h',
            'cufft.h',
            'curand.h',
            'cusparse.h',
            'nvrtc.h',
        ],
        'libraries': [
            'cublas',
            'cuda',
            'cudart',
            'cufft',
            'curand',
            'cusparse',
            'nvrtc',
        ],
        'check_method': build.check_cuda_version,
        'version_method': build.get_cuda_version,
    },
    {
        'name': 'cudnn',
        'file': [
            'cupy.cuda.cudnn',
            'cupy.cudnn',
        ],
        'include': [
            'cudnn.h',
        ],
        'libraries': [
            'cudnn',
        ],
        'check_method': build.check_cudnn_version,
        'version_method': build.get_cudnn_version,
    },
    {
        'name': 'nccl',
        'file': [
            'cupy.cuda.nccl',
        ],
        'include': [
            'nccl.h',
        ],
        'libraries': [
            'nccl',
        ],
        'check_method': build.check_nccl_version,
        'version_method': build.get_nccl_version,
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
        'check_method': build.check_cuda_version,
    },
    {
        'name': 'nvtx',
        'file': [
            'cupy.cuda.nvtx',
        ],
        'include': [
            'nvToolsExt.h',
        ],
        'libraries': [
            'nvToolsExt' if not PLATFORM_WIN32 else 'nvToolsExt64_1',
        ],
        'check_method': build.check_nvtx,
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
            'thrust/sequence.h',
            'thrust/sort.h',
        ],
        'libraries': [
            'cudart',
        ],
        'check_method': build.check_cuda_version,
    }
]


def ensure_module_file(file):
    if isinstance(file, tuple):
        return file
    else:
        return file, []


def module_extension_name(file):
    return ensure_module_file(file)[0]


def module_extension_sources(file, use_cython, no_cuda):
    pyx, others = ensure_module_file(file)
    base = path.join(*pyx.split('.'))
    if use_cython:
        pyx = base + '.pyx'
        if not os.path.exists(pyx):
            use_cython = False
            print(
                'NOTICE: Skipping cythonize as {} does not exist.'.format(pyx))
    if not use_cython:
        pyx = base + '.cpp'

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
                  include_dirs=(), library_dirs=(), define_macros=None):

    source = ''.join(['#include <%s>\n' % header for header in includes])
    source += 'int main(int argc, char* argv[]) {return 0;}'
    try:
        # We need to try to build a shared library because distutils
        # uses different option to build an executable and a shared library.
        # Especially when a user build an executable, distutils does not use
        # LDFLAGS environment variable.
        build.build_shlib(compiler, source, libraries,
                          include_dirs, library_dirs, define_macros)
    except Exception as e:
        print(e)
        sys.stdout.flush()
        return False
    return True


def preconfigure_modules(compiler, settings):
    """Returns a list of modules buildable in given environment and settings.

    For each module in MODULES list, this function checks if the module
    can be built in the current environment and reports it.
    Returns a list of module names available.
    """

    nvcc_path = build.get_nvcc_path()
    summary = [
        '',
        '************************************************************',
        '* CuPy Configuration Summary                               *',
        '************************************************************',
        '',
        'Build Environment:',
        '  Include directories: {}'.format(str(settings['include_dirs'])),
        '  Library directories: {}'.format(str(settings['library_dirs'])),
        '  nvcc command       : {}'.format(
            nvcc_path if nvcc_path else '(not found)'),
        '',
        'Environment Variables:',
    ]

    for key in ['CFLAGS', 'LDFLAGS', 'LIBRARY_PATH',
                'CUDA_PATH', 'NVTOOLSEXT_PATH', 'NVCC']:
        summary += ['  {:<16}: {}'.format(key, os.environ.get(key, '(none)'))]

    summary += [
        '',
        'Modules:',
    ]

    ret = []
    for module in MODULES:
        installed = False
        status = 'No'
        errmsg = []

        print('')
        print('-------- Configuring Module: {} --------'.format(
            module['name']))
        sys.stdout.flush()
        if not check_library(compiler,
                             includes=module['include'],
                             include_dirs=settings['include_dirs'],
                             define_macros=settings['define_macros']):
            errmsg = ['Include files not found: %s' % module['include'],
                      'Check your CFLAGS environment variable.']
        elif not check_library(compiler,
                               libraries=module['libraries'],
                               library_dirs=settings['library_dirs'],
                               define_macros=settings['define_macros']):
            errmsg = ['Cannot link libraries: %s' % module['libraries'],
                      'Check your LDFLAGS environment variable.']
        elif ('check_method' in module and
                not module['check_method'](compiler, settings)):
            # Fail on per-library condition check (version requirements etc.)
            installed = True
            errmsg = ['The library is installed but not supported.']
        elif module['name'] == 'thrust' and nvcc_path is None:
            installed = True
            errmsg = ['nvcc command could not be found in PATH.',
                      'Check your PATH environment variable.']
        else:
            installed = True
            status = 'Yes'
            ret.append(module['name'])

        if installed and 'version_method' in module:
            status += ' (version {})'.format(module['version_method'](True))

        summary += [
            '  {:<10}: {}'.format(module['name'], status)
        ]

        # If error message exists...
        if len(errmsg) != 0:
            summary += ['    -> {}'.format(m) for m in errmsg]

            # Skip checking other modules when CUDA is unavailable.
            if module['name'] == 'cuda':
                break

    if len(ret) != len(MODULES):
        if 'cuda' in ret:
            lines = [
                'WARNING: Some modules could not be configured.',
                'CuPy will be installed without these modules.',
            ]
        else:
            lines = [
                'ERROR: CUDA could not be found on your system.',
            ]
        summary += [
            '',
        ] + lines + [
            'Please refer to the Installation Guide for details:',
            'https://docs-cupy.chainer.org/en/stable/install.html',
            '',
        ]

    summary += [
        '************************************************************',
        '',
    ]

    print('\n'.join(summary))
    return ret


def _rpath_base():
    if PLATFORM_LINUX:
        return '$ORIGIN'
    elif PLATFORM_DARWIN:
        return '@loader_path'
    else:
        raise Exception('not supported on this platform')


def make_extensions(options, compiler, use_cython):
    """Produce a list of Extension instances which passed to cythonize()."""

    no_cuda = options['no_cuda']
    settings = build.get_compiler_setting()

    include_dirs = settings['include_dirs']

    settings['include_dirs'] = [
        x for x in include_dirs if path.exists(x)]
    settings['library_dirs'] = [
        x for x in settings['library_dirs'] if path.exists(x)]

    # Adjust rpath to use CUDA libraries in `cupy/_lib/*.so`) from CuPy.
    use_wheel_libs_rpath = (
        0 < len(options['wheel_libs']) and not PLATFORM_WIN32)

    # In the environment with CUDA 7.5 on Ubuntu 16.04, gcc5.3 does not
    # automatically deal with memcpy because string.h header file has
    # been changed. This is a workaround for that environment.
    # See details in the below discussions:
    # https://github.com/BVLC/caffe/issues/4046
    # https://groups.google.com/forum/#!topic/theano-users/3ihQYiTRG4E
    settings['define_macros'].append(('_FORCE_INLINES', '1'))

    if options['linetrace']:
        settings['define_macros'].append(('CYTHON_TRACE', '1'))
        settings['define_macros'].append(('CYTHON_TRACE_NOGIL', '1'))
    if no_cuda:
        settings['define_macros'].append(('CUPY_NO_CUDA', '1'))

    available_modules = []
    if no_cuda:
        available_modules = [m['name'] for m in MODULES]
    else:
        available_modules = preconfigure_modules(compiler, settings)
        if 'cuda' not in available_modules:
            raise Exception('Your CUDA environment is invalid. '
                            'Please check above error log.')

    ret = []
    for module in MODULES:
        if module['name'] not in available_modules:
            continue

        s = settings.copy()
        if not no_cuda:
            s['libraries'] = module['libraries']

        compile_args = s.setdefault('extra_compile_args', [])
        link_args = s.setdefault('extra_link_args', [])

        if module['name'] == 'cusolver':
            compile_args = s.setdefault('extra_compile_args', [])
            link_args = s.setdefault('extra_link_args', [])
            # openmp is required for cusolver
            if compiler.compiler_type == 'unix' and not PLATFORM_DARWIN:
                # In mac environment, openmp is not required.
                compile_args.append('-fopenmp')
                link_args.append('-fopenmp')
            elif compiler.compiler_type == 'msvc':
                compile_args.append('/openmp')

        original_s = s
        for f in module['file']:
            s = copy.deepcopy(original_s)
            name = module_extension_name(f)

            rpath = []
            if not options['no_rpath']:
                # Add library directories (e.g., `/usr/local/cuda/lib64`) to
                # RPATH.
                rpath += s['library_dirs']

            if use_wheel_libs_rpath:
                # Add `cupy/_lib` (where shared libraries included in wheels
                # reside) to RPATH.
                # The path is resolved relative to the module, e.g., use
                # `$ORIGIN/_lib` for `cupy/cudnn.so` and `$ORIGIN/../_lib` for
                # `cupy/cuda/cudnn.so`.
                depth = name.count('.') - 1
                rpath.append('{}{}/_lib'.format(_rpath_base(), '/..' * depth))

            if not PLATFORM_WIN32 and not PLATFORM_LINUX:
                s['runtime_library_dirs'] = rpath
            if (PLATFORM_LINUX and s['library_dirs']) or PLATFORM_DARWIN:
                ldflag = '-Wl,'
                if PLATFORM_LINUX:
                    ldflag += '--disable-new-dtags,'
                ldflag += ','.join('-rpath,' + p for p in rpath)
                args = s.setdefault('extra_link_args', [])
                args.append(ldflag)
                if PLATFORM_DARWIN:
                    # -rpath is only supported when targetting Mac OS X 10.5 or
                    # later
                    args.append('-mmacosx-version-min=10.5')

            sources = module_extension_sources(f, use_cython, no_cuda)
            extension = setuptools.Extension(name, sources, **s)
            ret.append(extension)

    return ret


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument(
        '--cupy-package-name', type=str, default='cupy',
        help='alternate package name')
    parser.add_argument(
        '--cupy-long-description', type=str, default=None,
        help='path to the long description file')
    parser.add_argument(
        '--cupy-wheel-lib', type=str, action='append', default=[],
        help='shared library to copy into the wheel '
             '(can be specified for multiple times)')
    parser.add_argument(
        '--cupy-no-rpath', action='store_true', default=False,
        help='disable adding default library directories to RPATH')
    parser.add_argument(
        '--cupy-profile', action='store_true', default=False,
        help='enable profiling for Cython code')
    parser.add_argument(
        '--cupy-coverage', action='store_true', default=False,
        help='enable coverage for Cython code')
    parser.add_argument(
        '--cupy-no-cuda', action='store_true', default=False,
        help='build CuPy with stub header file')

    opts, sys.argv = parser.parse_known_args(sys.argv)

    arg_options = {
        'package_name': opts.cupy_package_name,
        'long_description': opts.cupy_long_description,
        'wheel_libs': opts.cupy_wheel_lib,  # list
        'no_rpath': opts.cupy_no_rpath,
        'profile': opts.cupy_profile,
        'linetrace': opts.cupy_coverage,
        'annotate': opts.cupy_coverage,
        'no_cuda': opts.cupy_no_cuda,
    }
    if check_readthedocs_environment():
        arg_options['no_cuda'] = True
    return arg_options


cupy_setup_options = parse_args()
print('Options:', cupy_setup_options)


def get_package_name():
    return cupy_setup_options['package_name']


def get_long_description():
    path = cupy_setup_options['long_description']
    if path is None:
        return None
    with open(path) as f:
        return f.read()


def prepare_wheel_libs():
    """Prepare shared libraries for wheels.

    On Windows, DLLs will be placed under `cupy/cuda`.
    On other platforms, shared libraries are placed under `cupy/_libs` and
    RUNPATH will be set to this directory later.
    """
    libdirname = None
    if PLATFORM_WIN32:
        libdirname = 'cuda'
        # Clean up existing libraries.
        libfiles = glob.glob('cupy/{}/*.dll'.format(libdirname))
        for libfile in libfiles:
            print('Removing file: {}'.format(libfile))
            os.remove(libfile)
    else:
        libdirname = '_lib'
        # Clean up the library directory.
        libdir = 'cupy/{}'.format(libdirname)
        if os.path.exists(libdir):
            print('Removing directory: {}'.format(libdir))
            shutil.rmtree(libdir)
        os.mkdir(libdir)

    # Copy specified libraries to the library directory.
    libs = []
    for lib in cupy_setup_options['wheel_libs']:
        # Note: symlink is resolved by shutil.copy2.
        print('Copying library for wheel: {}'.format(lib))
        libname = path.basename(lib)
        libpath = 'cupy/{}/{}'.format(libdirname, libname)
        shutil.copy2(lib, libpath)
        libs.append('{}/{}'.format(libdirname, libname))
    return libs


try:
    import Cython
    import Cython.Build
    cython_version = pkg_resources.parse_version(Cython.__version__)
    cython_available = (
        cython_version >= required_cython_version and
        cython_version not in ignore_cython_versions)
except ImportError:
    cython_available = False


def cythonize(extensions, arg_options):
    directive_keys = ('linetrace', 'profile')
    directives = {key: arg_options[key] for key in directive_keys}

    # Embed signatures for Sphinx documentation.
    directives['embedsignature'] = True

    cythonize_option_keys = ('annotate',)
    cythonize_options = {key: arg_options[key]
                         for key in cythonize_option_keys}

    return Cython.Build.cythonize(
        extensions, verbose=True, language_level=3,
        compiler_directives=directives, **cythonize_options)


def check_extensions(extensions):
    for x in extensions:
        for f in x.sources:
            if not path.isfile(f):
                raise RuntimeError('''\
Missing file: {}
Please install Cython {} or later. Please also check the version of Cython.
See https://docs-cupy.chainer.org/en/stable/install.html for details.
'''.format(f, required_cython_version))


def get_ext_modules(use_cython=False):
    arg_options = cupy_setup_options

    # We need to call get_config_vars to initialize _config_vars in distutils
    # see #1849
    sysconfig.get_config_vars()
    compiler = ccompiler.new_compiler()
    sysconfig.customize_compiler(compiler)

    extensions = make_extensions(arg_options, compiler, use_cython)

    return extensions


def _nvcc_gencode_options(cuda_version):
    """Returns NVCC GPU code generation options."""

    # The arch_list specifies virtual architectures, such as 'compute_61', and
    # real architectures, such as 'sm_61', for which the CUDA input files are
    # to be compiled.
    #
    # The syntax of an entry of the list is
    #
    #     entry ::= virtual_arch | (virtual_arch, real_arch)
    #
    # where virtual_arch is a string which means a virtual architecture and
    # real_arch is a string which means a real architecture.
    #
    # If a virtual architecture is supplied, NVCC generates a PTX code for the
    # virtual architecture. If a pair of a virtual architecture and a real
    # architecture is supplied, NVCC generates a PTX code for the virtual
    # architecture as well as a cubin code for the real architecture.
    #
    # For example, making NVCC generate a PTX code for 'compute_60' virtual
    # architecture, the arch_list has an entry of 'compute_60'.
    #
    #     arch_list = ['compute_60']
    #
    # For another, making NVCC generate a PTX code for 'compute_61' virtual
    # architecture and a cubin code for 'sm_61' real architecture, the
    # arch_list has an entry of ('compute_61', 'sm_61').
    #
    #     arch_list = [('compute_61', 'sm_61')]

    arch_list = ['compute_30',
                 'compute_50']

    if cuda_version >= 10000:
        arch_list += [('compute_60', 'sm_60'),
                      ('compute_61', 'sm_61'),
                      ('compute_70', 'sm_70'),
                      ('compute_75', 'sm_75'),
                      'compute_70']
    elif cuda_version >= 9000:
        arch_list += [('compute_60', 'sm_60'),
                      ('compute_61', 'sm_61'),
                      ('compute_70', 'sm_70'),
                      'compute_70']
    elif cuda_version >= 8000:
        arch_list += [('compute_60', 'sm_60'),
                      ('compute_61', 'sm_61'),
                      'compute_60']

    options = []
    for arch in arch_list:
        if type(arch) is tuple:
            virtual_arch, real_arch = arch
            options.append('--generate-code=arch={},code={}'.format(
                virtual_arch, real_arch))
        else:
            options.append('--generate-code=arch={},code={}'.format(
                arch, arch))

    if sys.argv == ['setup.py', 'develop']:
        return []
    else:
        return options


class _UnixCCompiler(unixccompiler.UnixCCompiler):
    src_extensions = list(unixccompiler.UnixCCompiler.src_extensions)
    src_extensions.append('.cu')

    def _compile(self, obj, src, ext, cc_args, extra_postargs, pp_opts):
        # For sources other than CUDA C ones, just call the super class method.
        if os.path.splitext(src)[1] != '.cu':
            return unixccompiler.UnixCCompiler._compile(
                self, obj, src, ext, cc_args, extra_postargs, pp_opts)

        # For CUDA C source files, compile them with NVCC.
        _compiler_so = self.compiler_so
        try:
            nvcc_path = build.get_nvcc_path()
            base_opts = build.get_compiler_base_options()
            self.set_executable('compiler_so', nvcc_path)

            cuda_version = build.get_cuda_version()
            postargs = _nvcc_gencode_options(cuda_version) + [
                '-O2', '--compiler-options="-fPIC"']
            print('NVCC options:', postargs)

            return unixccompiler.UnixCCompiler._compile(
                self, obj, src, ext, base_opts + cc_args, postargs, pp_opts)
        finally:
            self.compiler_so = _compiler_so


class _MSVCCompiler(msvccompiler.MSVCCompiler):
    _cu_extensions = ['.cu']

    src_extensions = list(unixccompiler.UnixCCompiler.src_extensions)
    src_extensions.extend(_cu_extensions)

    def _compile_cu(self, sources, output_dir=None, macros=None,
                    include_dirs=None, debug=0, extra_preargs=None,
                    extra_postargs=None, depends=None):
        # Compile CUDA C files, mainly derived from UnixCCompiler._compile().

        macros, objects, extra_postargs, pp_opts, _build = \
            self._setup_compile(output_dir, macros, include_dirs, sources,
                                depends, extra_postargs)

        compiler_so = build.get_nvcc_path()
        cc_args = self._get_cc_args(pp_opts, debug, extra_preargs)
        cuda_version = build.get_cuda_version()
        postargs = _nvcc_gencode_options(cuda_version) + ['-O2']
        postargs += ['-Xcompiler', '/MD']
        print('NVCC options:', postargs)

        for obj in objects:
            try:
                src, ext = _build[obj]
            except KeyError:
                continue
            try:
                self.spawn(compiler_so + cc_args + [src, '-o', obj] + postargs)
            except errors.DistutilsExecError as e:
                raise errors.CompileError(str(e))

        return objects

    def compile(self, sources, **kwargs):
        # Split CUDA C sources and others.
        cu_sources = []
        other_sources = []
        for source in sources:
            if os.path.splitext(source)[1] == '.cu':
                cu_sources.append(source)
            else:
                other_sources.append(source)

        # Compile source files other than CUDA C ones.
        other_objects = msvccompiler.MSVCCompiler.compile(
            self, other_sources, **kwargs)

        # Compile CUDA C sources.
        cu_objects = self._compile_cu(cu_sources, **kwargs)

        # Return compiled object filenames.
        return other_objects + cu_objects


class sdist_with_cython(sdist.sdist):

    """Custom `sdist` command with cyhonizing."""

    def __init__(self, *args, **kwargs):
        if not cython_available:
            raise RuntimeError('Cython is required to make sdist.')
        ext_modules = get_ext_modules(True)  # get .pyx modules
        cythonize(ext_modules, cupy_setup_options)
        sdist.sdist.__init__(self, *args, **kwargs)


class custom_build_ext(build_ext.build_ext):

    """Custom `build_ext` command to include CUDA C source files."""

    def run(self):
        if build.get_nvcc_path() is not None:
            def wrap_new_compiler(func):
                def _wrap_new_compiler(*args, **kwargs):
                    try:
                        return func(*args, **kwargs)
                    except errors.DistutilsPlatformError:
                        if not PLATFORM_WIN32:
                            CCompiler = _UnixCCompiler
                        else:
                            CCompiler = _MSVCCompiler
                        return CCompiler(
                            None, kwargs['dry_run'], kwargs['force'])
                return _wrap_new_compiler
            ccompiler.new_compiler = wrap_new_compiler(ccompiler.new_compiler)
            # Intentionally causes DistutilsPlatformError in
            # ccompiler.new_compiler() function to hook.
            self.compiler = 'nvidia'
        if cython_available:
            ext_modules = get_ext_modules(True)  # get .pyx modules
            cythonize(ext_modules, cupy_setup_options)
        check_extensions(self.extensions)
        build_ext.build_ext.run(self)
