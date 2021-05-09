import argparse
import copy
from distutils import ccompiler
from distutils import errors
from distutils import msvccompiler
from distutils import sysconfig
from distutils import unixccompiler
import os
from os import path
import shutil
import sys

import pkg_resources
import setuptools
from setuptools.command import build_ext

from install import build
from install.build import PLATFORM_LINUX
from install.build import PLATFORM_WIN32


# Cython requirements (minimum version and versions known to be broken).
# Note: this must be in sync with setup_requires defined in setup.py.
required_cython_version = pkg_resources.parse_version('0.29.22')
ignore_cython_versions = [
]
use_hip = build.use_hip


# The value of the key 'file' is a list that contains extension names
# or tuples of an extension name and a list of other souces files
# required to build the extension such as .cpp files and .cu files.
#
#   <extension name> | (<extension name>, a list of <other source>)
#
# The extension name is also interpreted as the name of the Cython
# source file required to build the extension with appending '.pyx'
# file extension.
MODULES = []

cuda_files = [
    'cupy_backends.cuda.api.driver',
    'cupy_backends.cuda.api.runtime',
    'cupy_backends.cuda.libs.cublas',
    'cupy_backends.cuda.libs.curand',
    'cupy_backends.cuda.libs.cusparse',
    'cupy_backends.cuda.libs.nvrtc',
    'cupy_backends.cuda.libs.profiler',
    'cupy_backends.cuda.stream',
    'cupy._core._accelerator',
    'cupy._core._carray',
    'cupy._core._cub_reduction',
    'cupy._core._dtype',
    'cupy._core._fusion_kernel',
    'cupy._core._fusion_thread_local',
    'cupy._core._fusion_trace',
    'cupy._core._fusion_variable',
    'cupy._core._kernel',
    'cupy._core._memory_range',
    'cupy._core._optimize_config',
    'cupy._core._reduction',
    'cupy._core._routines_binary',
    'cupy._core._routines_indexing',
    'cupy._core._routines_linalg',
    'cupy._core._routines_logic',
    'cupy._core._routines_manipulation',
    'cupy._core._routines_math',
    'cupy._core._routines_sorting',
    'cupy._core._routines_statistics',
    'cupy._core._scalar',
    'cupy._core.core',
    'cupy._core.flags',
    'cupy._core.internal',
    'cupy._core.fusion',
    'cupy._core.new_fusion',
    'cupy._core.raw',
    'cupy.cuda.common',
    'cupy.cuda.cufft',
    'cupy.cuda.device',
    'cupy.cuda.memory',
    'cupy.cuda.memory_hook',
    'cupy.cuda.pinned_memory',
    'cupy.cuda.function',
    'cupy.cuda.stream',
    'cupy.cuda.texture',
    'cupy.fft._cache',
    'cupy.fft._callback',
    'cupy.lib._polynomial',
    'cupy._util'
]

if use_hip:
    # We handle nvtx (and likely any other future support) here, because
    # the HIP stubs (hip/cupy_*.h) would cause many symbols
    # to leak into all these modules even if unused. It's easier for all of
    # them to link to the same set of shared libraries.
    MODULES.append({
        # TODO(leofang): call this "rocm" or "hip" to avoid confusion?
        'name': 'cuda',
        'required': True,
        'file': cuda_files + [
            'cupy_backends.cuda.libs.nvtx',
            'cupy_backends.cuda.libs.cusolver',
            'cupy.cusolver',
        ],
        'include': [
            'hip/hip_runtime_api.h',
            'hip/hiprtc.h',
            'hipblas.h',
            'hiprand/hiprand.h',
            'hipfft.h',
            'roctx.h',
            'rocsolver.h',
        ],
        'libraries': [
            'amdhip64',  # was hiprtc and hip_hcc before ROCm 3.8.0
            'hipblas',
            'hiprand',
            'rocfft',
            'roctx64',
            'rocblas',
            'rocsolver',
        ],
        'check_method': build.check_hip_version,
        'version_method': build.get_hip_version,
    })
else:
    MODULES.append({
        'name': 'cuda',
        'required': True,
        'file': cuda_files,
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
    })

if not use_hip:
    MODULES.append({
        'name': 'cusolver',
        'required': True,
        'file': [
            'cupy_backends.cuda.libs.cusolver',
            'cupy.cusolver',
        ],
        'include': [
            'cusolverDn.h',
        ],
        'libraries': [
            'cusolver',
        ],
        'check_method': build.check_cuda_version,
    })

if not use_hip:
    MODULES.append({
        'name': 'cudnn',
        'file': [
            'cupy_backends.cuda.libs.cudnn',
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
    })

    MODULES.append({
        'name': 'nccl',
        'file': [
            'cupy_backends.cuda.libs.nccl',
        ],
        'include': [
            'nccl.h',
        ],
        'libraries': [
            'nccl',
        ],
        'check_method': build.check_nccl_version,
        'version_method': build.get_nccl_version,
    })

    MODULES.append({
        'name': 'nvtx',
        'file': [
            'cupy_backends.cuda.libs.nvtx',
        ],
        'include': [
            'nvToolsExt.h',
        ],
        'libraries': [
            'nvToolsExt' if not PLATFORM_WIN32 else 'nvToolsExt64_1',
        ],
        'check_method': build.check_nvtx,
    })

    MODULES.append({
        'name': 'cutensor',
        'file': [
            'cupy_backends.cuda.libs.cutensor',
            'cupy.cutensor',
        ],
        'include': [
            'cutensor.h',
        ],
        'libraries': [
            'cutensor',
            'cublas',
        ],
        'check_method': build.check_cutensor_version,
        'version_method': build.get_cutensor_version,
    })

    MODULES.append({
        'name': 'cub',
        'required': True,
        'file': [
            ('cupy.cuda.cub', ['cupy/cuda/cupy_cub.cu']),
        ],
        'include': [
            'cub/util_namespace.cuh',  # dummy
        ],
        'libraries': [
            'cudart',
        ],
        'check_method': build.check_cub_version,
        'version_method': build.get_cub_version,
    })

    MODULES.append({
        'name': 'jitify',
        'required': True,
        'file': [
            'cupy.cuda.jitify',
        ],
        'include': [
            'cuda.h',
            'cuda_runtime.h',
            'nvrtc.h',
        ],
        'libraries': [
            'cuda',
            'cudart',
            'nvrtc',
        ],
        'check_method': build.check_jitify_version,
        'version_method': build.get_jitify_version,
    })

    MODULES.append({
        'name': 'random',
        'required': True,
        'file': [
            'cupy.random._bit_generator',
            ('cupy.random._generator_api',
             ['cupy/random/cupy_distributions.cu']),
        ],
        'include': [
        ],
        'libraries': [
            'cudart',
            'curand',
        ],
    })

    MODULES.append({
        'name': 'cusparselt',
        'file': [
            'cupy_backends.cuda.libs.cusparselt',
        ],
        'include': [
            'cusparseLt.h',
        ],
        'libraries': [
            'cusparseLt',
        ],
        'check_method': build.check_cusparselt_version,
        'version_method': build.get_cusparselt_version,
    })

    MODULES.append({
        'name': 'cugraph',
        'file': [
            'cupy_backends.cuda.libs.cugraph',
        ],
        'include': [
            'cugraph/raft/error.hpp',  # dummy
        ],
        'libraries': [
            'cugraph',
        ],
        'check_method': build.check_cugraph_version,
        'version_method': build.get_cugraph_version,
    })

else:
    MODULES.append({
        'name': 'cub',
        'required': True,
        'file': [
            ('cupy.cuda.cub', ['cupy/cuda/cupy_cub.cu']),
        ],
        'include': [
            'hipcub/hipcub_version.hpp',  # dummy
        ],
        'libraries': [
            'amdhip64',  # was hiprtc and hip_hcc before ROCm 3.8.0
        ],
        'check_method': build.check_cub_version,
        'version_method': build.get_cub_version,
    })

    MODULES.append({
        'name': 'nccl',
        'file': [
            'cupy_backends.cuda.libs.nccl',
        ],
        'include': [
            'rccl.h',
        ],
        'libraries': [
            'rccl',
        ],
        'check_method': build.check_nccl_version,
        'version_method': build.get_nccl_version,
    })

if bool(int(os.environ.get('CUPY_SETUP_ENABLE_THRUST', 1))):
    if use_hip:
        MODULES.append({
            'name': 'thrust',
            'required': True,
            'file': [
                ('cupy.cuda.thrust', ['cupy/cuda/cupy_thrust.cu']),
            ],
            'include': [
                'thrust/version.h',
            ],
            'libraries': [
                'amdhip64',  # was hiprtc and hip_hcc before ROCm 3.8.0
            ],
        })
    else:
        MODULES.append({
            'name': 'thrust',
            'required': True,
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
            'check_method': build.check_thrust_version,
            'version_method': build.get_thrust_version,
        })

MODULES.append({
    'name': 'dlpack',
    'required': True,
    'file': [
        'cupy._core.dlpack',
    ],
    'include': [
        'cupy/dlpack/dlpack.h',
    ],
    'libraries': [],
})


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
    pyx = base + ('.pyx' if use_cython else '.cpp')

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


def get_required_modules():
    return [m['name'] for m in MODULES if m.get('required', False)]


def check_readthedocs_environment():
    return os.environ.get('READTHEDOCS', None) == 'True'


def check_library(compiler, includes=(), libraries=(),
                  include_dirs=(), library_dirs=(), define_macros=None,
                  extra_compile_args=()):

    source = ''.join(['#include <%s>\n' % header for header in includes])
    source += 'int main() {return 0;}'
    try:
        # We need to try to build a shared library because distutils
        # uses different option to build an executable and a shared library.
        # Especially when a user build an executable, distutils does not use
        # LDFLAGS environment variable.
        build.build_shlib(compiler, source, libraries,
                          include_dirs, library_dirs, define_macros,
                          extra_compile_args)
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
    hipcc_path = build.get_hipcc_path()
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
        '  hipcc command      : {}'.format(
            hipcc_path if hipcc_path else '(not found)'),
        '',
        'Environment Variables:',
    ]

    for key in ['CFLAGS', 'LDFLAGS', 'LIBRARY_PATH',
                'CUDA_PATH', 'NVTOOLSEXT_PATH', 'NVCC', 'HIPCC',
                'ROCM_HOME']:
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

        if module['name'] == 'cutensor':
            cutensor_path = os.environ.get('CUTENSOR_PATH', '')
            inc_path = os.path.join(cutensor_path, 'include')
            if os.path.exists(inc_path):
                settings['include_dirs'].append(inc_path)
            cuda_version = build.get_cuda_version()
            cuda_major = str(cuda_version // 1000)
            cuda_major_minor = cuda_major + '.' + \
                str((cuda_version // 10) % 100)
            for cuda_ver in (cuda_major_minor, cuda_major):
                lib_path = os.path.join(cutensor_path, 'lib', cuda_ver)
                if os.path.exists(lib_path):
                    settings['library_dirs'].append(lib_path)
                    break

        if module['name'] == 'cugraph':
            cugraph_path = os.environ.get('CUGRAPH_PATH', '')
            for i in 'include', 'include/cugraph':
                inc_path = os.path.join(cugraph_path, i)
                if os.path.exists(inc_path):
                    settings['include_dirs'].append(inc_path)
            lib_path = os.path.join(cugraph_path, 'lib')
            if os.path.exists(lib_path):
                settings['library_dirs'].append(lib_path)

        print('')
        print('-------- Configuring Module: {} --------'.format(
            module['name']))
        sys.stdout.flush()
        if not check_library(
                compiler,
                includes=module['include'],
                include_dirs=settings['include_dirs'],
                define_macros=settings['define_macros'],
                extra_compile_args=settings['extra_compile_args']):
            errmsg = ['Include files not found: %s' % module['include'],
                      'Check your CFLAGS environment variable.']
        elif not check_library(
                compiler,
                libraries=module['libraries'],
                library_dirs=settings['library_dirs'],
                define_macros=settings['define_macros'],
                extra_compile_args=settings['extra_compile_args']):
            errmsg = ['Cannot link libraries: %s' % module['libraries'],
                      'Check your LDFLAGS environment variable.']
        elif ('check_method' in module and
                not module['check_method'](compiler, settings)):
            # Fail on per-library condition check (version requirements etc.)
            installed = True
            errmsg = ['The library is installed but not supported.']
        elif (module['name'] in ('thrust', 'cub', 'random')
                and (nvcc_path is None and hipcc_path is None)):
            installed = True
            cmd = 'nvcc' if not use_hip else 'hipcc'
            errmsg = ['{} command could not be found in PATH.'.format(cmd),
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

    # Get a list of the CC of the devices connected to this node
    if not use_hip:
        build.check_compute_capabilities(compiler, settings)

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
            'https://docs.cupy.dev/en/stable/install.html',
            '',
        ]

    summary += [
        '************************************************************',
        '',
    ]

    print('\n'.join(summary))
    return ret, settings


def _rpath_base():
    if PLATFORM_LINUX:
        return '$ORIGIN'
    else:
        raise Exception('not supported on this platform')


def make_extensions(options, compiler, use_cython):
    """Produce a list of Extension instances which passed to cythonize()."""

    no_cuda = options['no_cuda']
    use_hip = not no_cuda and options['use_hip']
    settings = build.get_compiler_setting(use_hip)

    include_dirs = settings['include_dirs']

    settings['include_dirs'] = [
        x for x in include_dirs if path.exists(x)]
    settings['library_dirs'] = [
        x for x in settings['library_dirs'] if path.exists(x)]

    # Adjust rpath to use CUDA libraries in `cupy/.data/lib/*.so`) from CuPy.
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
    if use_hip:
        settings['define_macros'].append(('CUPY_USE_HIP', '1'))
        settings['define_macros'].append(('__HIP_PLATFORM_HCC__', '1'))

    available_modules = []
    if no_cuda:
        available_modules = [m['name'] for m in MODULES]
    else:
        available_modules, settings = preconfigure_modules(compiler, settings)
        required_modules = get_required_modules()
        if not (set(required_modules) <= set(available_modules)):
            raise Exception('Your CUDA environment is invalid. '
                            'Please check above error log.')

    ret = []
    for module in MODULES:
        if module['name'] not in available_modules:
            continue

        s = copy.deepcopy(settings)
        if not no_cuda:
            s['libraries'] = module['libraries']

        compile_args = s.setdefault('extra_compile_args', [])
        link_args = s.setdefault('extra_link_args', [])

        if module['name'] == 'cusolver':
            # cupy_backends/cupy_lapack.h has C++ template code
            compile_args.append('--std=c++11')
            # openmp is required for cusolver
            if use_hip:
                pass
            elif compiler.compiler_type == 'unix':
                compile_args.append('-fopenmp')
                link_args.append('-fopenmp')
            elif compiler.compiler_type == 'msvc':
                compile_args.append('/openmp')

        if module['name'] == 'random':
            if compiler.compiler_type == 'msvc':
                compile_args.append('-D_USE_MATH_DEFINES')

        if module['name'] == 'jitify':
            # this fixes RTD (no_cuda) builds...
            compile_args.append('--std=c++11')
            # if any change is made to the Jitify header, we force recompiling
            s['depends'] = ['./cupy/_core/include/cupy/jitify/jitify.hpp']

        if module['name'] == 'dlpack':
            # if any change is made to the DLPack header, we force recompiling
            s['depends'] = ['./cupy/_core/include/cupy/dlpack/dlpack.h']

        for f in module['file']:
            s_file = copy.deepcopy(s)
            name = module_extension_name(f)

            if name.endswith('fft._callback') and not PLATFORM_LINUX:
                continue

            rpath = []
            if not options['no_rpath']:
                # Add library directories (e.g., `/usr/local/cuda/lib64`) to
                # RPATH.
                rpath += s_file['library_dirs']

            if use_wheel_libs_rpath:
                # Add `cupy/.data/lib` (where shared libraries included in
                # wheels reside) to RPATH.
                # The path is resolved relative to the module, e.g., use
                # `$ORIGIN/../cupy/.data/lib` for `cupy/cudnn.so` and
                # `$ORIGIN/../../../cupy/.data/lib` for
                # `cupy_backends/cuda/libs/cudnn.so`.
                depth = name.count('.')
                rpath.append(
                    '{}{}/cupy/.data/lib'.format(_rpath_base(), '/..' * depth))

            if not PLATFORM_WIN32 and not PLATFORM_LINUX:
                assert False, "macOS is no longer supported"
            if (PLATFORM_LINUX and len(rpath) != 0):
                ldflag = '-Wl,'
                if PLATFORM_LINUX:
                    ldflag += '--disable-new-dtags,'
                ldflag += ','.join('-rpath,' + p for p in rpath)
                args = s_file.setdefault('extra_link_args', [])
                args.append(ldflag)

            sources = module_extension_sources(f, use_cython, no_cuda)
            extension = setuptools.Extension(name, sources, **s_file)
            ret.append(extension)

    return ret


# TODO(oktua): use enviriment variable
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
        '--cupy-wheel-include', type=str, action='append', default=[],
        help='An include file to copy into the wheel. '
             'Delimited by a colon. '
             'The former part is a full path of the source include file and '
             'the latter is the relative path within cupy wheel. '
             '(can be specified for multiple times)')
    parser.add_argument(
        '--cupy-wheel-metadata', type=str, default=None,
        help='wheel metadata (cupy/.data/_wheel.json)')
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
    # parser.add_argument(
    #     '--cupy-use-hip', action='store_true', default=False,
    #     help='build CuPy with HIP')

    opts, sys.argv = parser.parse_known_args(sys.argv)

    arg_options = {
        'package_name': opts.cupy_package_name,
        'long_description': opts.cupy_long_description,
        'wheel_libs': opts.cupy_wheel_lib,  # list
        'wheel_includes': opts.cupy_wheel_include,  # list
        'wheel_metadata': opts.cupy_wheel_metadata,
        'no_rpath': opts.cupy_no_rpath,
        'profile': opts.cupy_profile,
        'linetrace': opts.cupy_coverage,
        'annotate': opts.cupy_coverage,
        'no_cuda': opts.cupy_no_cuda,
        'use_hip': use_hip  # opts.cupy_use_hip,
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
    """Prepare shared libraries and include files for wheels.

    Shared libraries are placed under `cupy/.data/lib` and
    RUNPATH will be set to this directory later (Linux only).
    Include files are placed under `cupy/.data/include`.

    Returns the list of files (path relative to `cupy` module) to add to
    the sdist/wheel distribution.
    """
    data_dir = os.path.abspath(os.path.join('cupy', '.data'))
    if os.path.exists(data_dir):
        print('Removing directory: {}'.format(data_dir))
        shutil.rmtree(data_dir)

    # Generate list files to copy
    # tuple of (src_path, dst_path)
    files_to_copy = []

    # Library files
    for srcpath in cupy_setup_options['wheel_libs']:
        relpath = os.path.basename(srcpath)
        dstpath = os.path.join(data_dir, 'lib', relpath)
        files_to_copy.append((srcpath, dstpath))

    # Include files
    for include_path_spec in cupy_setup_options['wheel_includes']:
        srcpath, relpath = include_path_spec.rsplit(':', 1)
        dstpath = os.path.join(data_dir, 'include', relpath)
        files_to_copy.append((srcpath, dstpath))

    # Wheel meta data
    wheel_metadata = cupy_setup_options['wheel_metadata']
    if wheel_metadata:
        files_to_copy.append(
            (wheel_metadata, os.path.join(data_dir, '_wheel.json')))

    # Copy
    for srcpath, dstpath in files_to_copy:
        # Note: symlink is resolved by shutil.copy2.
        print('Copying file for wheel: {}'.format(srcpath))
        dirpath = os.path.dirname(dstpath)
        if not os.path.isdir(dirpath):
            os.makedirs(dirpath)
        shutil.copy2(srcpath, dstpath)

    return [os.path.relpath(x[1], 'cupy') for x in files_to_copy]


def cythonize(extensions, arg_options):
    # Delay importing Cython as it may be installed via setup_requires if
    # the user does not have Cython installed.
    import Cython
    import Cython.Build
    cython_version = pkg_resources.parse_version(Cython.__version__)
    if (cython_version < required_cython_version or
            cython_version in ignore_cython_versions):
        raise AssertionError(
            'Unsupported Cython version: {}'.format(cython_version))

    directive_keys = ('linetrace', 'profile')
    directives = {key: arg_options[key] for key in directive_keys}

    # Embed signatures for Sphinx documentation.
    directives['embedsignature'] = True

    cythonize_option_keys = ('annotate',)
    cythonize_options = {key: arg_options[key]
                         for key in cythonize_option_keys}

    # Compile-time constants to be used in Cython code
    compile_time_env = cythonize_options.get('compile_time_env')
    if compile_time_env is None:
        compile_time_env = {}
        cythonize_options['compile_time_env'] = compile_time_env
    compile_time_env['CUPY_CUFFT_STATIC'] = False
    compile_time_env['cython_version'] = str(cython_version)
    if arg_options['no_cuda']:  # on RTD
        compile_time_env['CUDA_VERSION'] = 0
        compile_time_env['HIP_VERSION'] = 0
    elif use_hip:  # on ROCm/HIP
        compile_time_env['CUDA_VERSION'] = 0
        compile_time_env['HIP_VERSION'] = build.get_hip_version()
    else:  # on CUDA
        compile_time_env['CUDA_VERSION'] = build.get_cuda_version()
        compile_time_env['HIP_VERSION'] = 0

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
See https://docs.cupy.dev/en/stable/install.html for details.
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

    if sys.argv == ['setup.py', 'develop']:
        return []

    envcfg = os.getenv('CUPY_NVCC_GENERATE_CODE', None)
    if envcfg is not None and envcfg != 'current':
        return ['--generate-code={}'.format(arch)
                for arch in envcfg.split(';') if len(arch) > 0]
    if envcfg == 'current' and build.get_compute_capabilities() is not None:
        ccs = build.get_compute_capabilities()
        arch_list = [
            f'compute_{cc}' if cc < 60 else (f'compute_{cc}', f'sm_{cc}')
            for cc in ccs]
    else:
        # The arch_list specifies virtual architectures, such as 'compute_61',
        # and real architectures, such as 'sm_61', for which the CUDA
        # input files are to be compiled.
        #
        # The syntax of an entry of the list is
        #
        #     entry ::= virtual_arch | (virtual_arch, real_arch)
        #
        # where virtual_arch is a string which means a virtual architecture and
        # real_arch is a string which means a real architecture.
        #
        # If a virtual architecture is supplied, NVCC generates a PTX code
        # the virtual architecture. If a pair of a virtual architecture and a
        # real architecture is supplied, NVCC generates a PTX code for the
        # virtual architecture as well as a cubin code for the real one.
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
        #
        # See the documentation of each CUDA version for the list of supported
        # architectures:
        #
        #   https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#options-for-steering-gpu-code-generation

        if cuda_version >= 11000:
            arch_list = ['compute_35',
                         'compute_50',
                         ('compute_60', 'sm_60'),
                         ('compute_61', 'sm_61'),
                         ('compute_70', 'sm_70'),
                         ('compute_75', 'sm_75'),
                         ('compute_80', 'sm_80'),
                         'compute_80']
        elif cuda_version >= 10000:
            arch_list = ['compute_30',
                         'compute_50',
                         ('compute_60', 'sm_60'),
                         ('compute_61', 'sm_61'),
                         ('compute_70', 'sm_70'),
                         ('compute_75', 'sm_75'),
                         'compute_70']
        elif cuda_version >= 9020:
            arch_list = ['compute_30',
                         'compute_50',
                         ('compute_60', 'sm_60'),
                         ('compute_61', 'sm_61'),
                         ('compute_70', 'sm_70'),
                         'compute_70']
        else:
            # This should not happen.
            assert False

    options = []
    for arch in arch_list:
        if type(arch) is tuple:
            virtual_arch, real_arch = arch
            options.append('--generate-code=arch={},code={}'.format(
                virtual_arch, real_arch))
        else:
            options.append('--generate-code=arch={},code={}'.format(
                arch, arch))

    return options


class _UnixCCompiler(unixccompiler.UnixCCompiler):
    src_extensions = list(unixccompiler.UnixCCompiler.src_extensions)
    src_extensions.append('.cu')

    def _compile(self, obj, src, ext, cc_args, extra_postargs, pp_opts):
        # For sources other than CUDA C ones, just call the super class method.
        if os.path.splitext(src)[1] != '.cu':
            return unixccompiler.UnixCCompiler._compile(
                self, obj, src, ext, cc_args, extra_postargs, pp_opts)

        if use_hip:
            return self._compile_unix_hipcc(
                obj, src, ext, cc_args, extra_postargs, pp_opts)
        else:
            return self._compile_unix_nvcc(
                obj, src, ext, cc_args, extra_postargs, pp_opts)

    def _compile_unix_nvcc(self,
                           obj, src, ext, cc_args, extra_postargs, pp_opts):
        # For CUDA C source files, compile them with NVCC.
        nvcc_path = build.get_nvcc_path()
        base_opts = build.get_compiler_base_options(nvcc_path)
        compiler_so = nvcc_path

        cuda_version = build.get_cuda_version()
        postargs = _nvcc_gencode_options(cuda_version) + [
            '-O2', '--compiler-options="-fPIC"']
        if cuda_version >= 11020:
            postargs += ['--std=c++14']
            num_threads = int(os.environ.get('CUPY_NUM_NVCC_THREADS', '2'))
            postargs += [f'-t{num_threads}']
        else:
            postargs += ['--std=c++11']
        print('NVCC options:', postargs)
        try:
            self.spawn(compiler_so + base_opts + cc_args + [src, '-o', obj] +
                       postargs)
        except errors.DistutilsExecError as e:
            raise errors.CompileError(str(e))

    def _compile_unix_hipcc(self,
                            obj, src, ext, cc_args, extra_postargs, pp_opts):
        # For CUDA C source files, compile them with HIPCC.
        rocm_path = build.get_hipcc_path()
        base_opts = build.get_compiler_base_options(rocm_path)
        compiler_so = rocm_path
        postargs = ['-O2', '-fPIC', '--include', 'hip_runtime.h']
        print('HIPCC options:', postargs)
        try:
            self.spawn(compiler_so + base_opts + cc_args + [src, '-o', obj] +
                       postargs)
        except errors.DistutilsExecError as e:
            raise errors.CompileError(str(e))

    def link(self, target_desc, objects, output_filename, *args):
        use_hipcc = False
        if use_hip:
            for i in objects:
                if any(obj in i for obj in ('cupy_thrust.o', 'cupy_cub.o')):
                    use_hipcc = True
        if use_hipcc:
            _compiler_cxx = self.compiler_cxx
            try:
                rocm_path = build.get_hipcc_path()
                self.set_executable('compiler_cxx', rocm_path)

                return unixccompiler.UnixCCompiler.link(
                    self, target_desc, objects, output_filename, *args)
            finally:
                self.compiler_cxx = _compiler_cxx
        else:
            return unixccompiler.UnixCCompiler.link(
                self, target_desc, objects, output_filename, *args)


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
        if cuda_version >= 11020:
            # MSVC 14.0 (2015) is deprecated for CUDA 11.2 but we need it
            # to build CuPy because some Python versions were built using it.
            # REF: https://wiki.python.org/moin/WindowsCompilers
            postargs += ['-allow-unsupported-compiler']
        postargs += ['-Xcompiler', '/MD', '-D_USE_MATH_DEFINES']
        # This is to compile thrust with MSVC2015
        if cuda_version >= 11020:
            postargs += ['--std=c++14']
            num_threads = int(os.environ.get('CUPY_NUM_NVCC_THREADS', '2'))
            postargs += [f'-t{num_threads}']
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
        if use_hip:
            raise RuntimeError('ROCm is not supported on Windows')

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


class custom_build_ext(build_ext.build_ext):

    """Custom `build_ext` command to include CUDA C source files."""

    def run(self):
        if (build.get_nvcc_path() is not None
                or build.get_hipcc_path() is not None):
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
        ext_modules = get_ext_modules(True)  # get .pyx modules
        cythonize(ext_modules, cupy_setup_options)
        check_extensions(self.extensions)
        build_ext.build_ext.run(self)

    def build_extensions(self):
        num_jobs = int(os.environ.get('CUPY_NUM_BUILD_JOBS', '4'))
        if num_jobs > 1:
            self.parallel = num_jobs
        super().build_extensions()
