import contextlib
import distutils.util
import os
import re
import shutil
import subprocess
import sys
import tempfile

from install import utils


PLATFORM_LINUX = sys.platform.startswith('linux')
PLATFORM_WIN32 = sys.platform.startswith('win32')

minimum_cuda_version = 9020
minimum_cudnn_version = 7600

minimum_hip_version = 305  # for ROCm 3.5.0+

_cuda_path = 'NOT_INITIALIZED'
_rocm_path = 'NOT_INITIALIZED'
_compiler_base_options = None

use_hip = bool(int(os.environ.get('CUPY_INSTALL_USE_HIP', '0')))


# Using tempfile.TemporaryDirectory would cause an error during cleanup
# due to a bug: https://bugs.python.org/issue26660
@contextlib.contextmanager
def _tempdir():
    temp_dir = tempfile.mkdtemp()
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def get_rocm_path():
    global _rocm_path

    # Use a magic word to represent the cache not filled because None is a
    # valid return value.
    if _rocm_path != 'NOT_INITIALIZED':
        return _rocm_path

    _rocm_path = os.environ.get('ROCM_HOME', '')
    return _rocm_path


def get_cuda_path():
    global _cuda_path

    # Use a magic word to represent the cache not filled because None is a
    # valid return value.
    if _cuda_path != 'NOT_INITIALIZED':
        return _cuda_path

    nvcc_path = utils.search_on_path(('nvcc', 'nvcc.exe'))
    cuda_path_default = None
    if nvcc_path is not None:
        cuda_path_default = os.path.normpath(
            os.path.join(os.path.dirname(nvcc_path), '..'))

    cuda_path = os.environ.get('CUDA_PATH', '')  # Nvidia default on Windows
    if len(cuda_path) > 0 and cuda_path != cuda_path_default:
        utils.print_warning(
            'nvcc path != CUDA_PATH',
            'nvcc path: %s' % cuda_path_default,
            'CUDA_PATH: %s' % cuda_path)

    if os.path.exists(cuda_path):
        _cuda_path = cuda_path
    elif cuda_path_default is not None:
        _cuda_path = cuda_path_default
    elif os.path.exists('/usr/local/cuda'):
        _cuda_path = '/usr/local/cuda'
    else:
        _cuda_path = None

    return _cuda_path


def get_nvcc_path():
    nvcc = os.environ.get('NVCC', None)
    if nvcc:
        return distutils.util.split_quoted(nvcc)

    cuda_path = get_cuda_path()
    if cuda_path is None:
        return None

    if PLATFORM_WIN32:
        nvcc_bin = 'bin/nvcc.exe'
    else:
        nvcc_bin = 'bin/nvcc'

    nvcc_path = os.path.join(cuda_path, nvcc_bin)
    if os.path.exists(nvcc_path):
        return [nvcc_path]
    else:
        return None


def get_hipcc_path():
    hipcc = os.environ.get('HIPCC', None)
    if hipcc:
        return distutils.util.split_quoted(hipcc)

    rocm_path = get_rocm_path()
    if rocm_path is None:
        return None

    if PLATFORM_WIN32:
        hipcc_bin = 'bin/hipcc.exe'
    else:
        hipcc_bin = 'bin/hipcc'

    hipcc_path = os.path.join(rocm_path, hipcc_bin)
    if os.path.exists(hipcc_path):
        return [hipcc_path]
    else:
        return None


def get_compiler_setting(use_hip):
    cuda_path = None
    rocm_path = None

    if use_hip:
        rocm_path = get_rocm_path()
    else:
        cuda_path = get_cuda_path()

    include_dirs = []
    library_dirs = []
    define_macros = []
    extra_compile_args = []

    if cuda_path:
        include_dirs.append(os.path.join(cuda_path, 'include'))
        if PLATFORM_WIN32:
            library_dirs.append(os.path.join(cuda_path, 'bin'))
            library_dirs.append(os.path.join(cuda_path, 'lib', 'x64'))
        else:
            library_dirs.append(os.path.join(cuda_path, 'lib64'))
            library_dirs.append(os.path.join(cuda_path, 'lib'))

    if rocm_path:
        include_dirs.append(os.path.join(rocm_path, 'include'))
        include_dirs.append(os.path.join(rocm_path, 'include', 'hip'))
        include_dirs.append(os.path.join(rocm_path, 'include', 'rocrand'))
        include_dirs.append(os.path.join(rocm_path, 'include', 'hiprand'))
        include_dirs.append(os.path.join(rocm_path, 'include', 'roctracer'))
        library_dirs.append(os.path.join(rocm_path, 'lib'))

    if use_hip:
        extra_compile_args.append('-std=c++11')

    if PLATFORM_WIN32:
        nvtoolsext_path = os.environ.get('NVTOOLSEXT_PATH', '')
        if os.path.exists(nvtoolsext_path):
            include_dirs.append(os.path.join(nvtoolsext_path, 'include'))
            library_dirs.append(os.path.join(nvtoolsext_path, 'lib', 'x64'))
        else:
            define_macros.append(('CUPY_NO_NVTX', '1'))

    # For CUB, we need the complex and CUB headers. The search precedence for
    # the latter is:
    #   1. built-in CUB (for CUDA 11+ and ROCm)
    #   2. CuPy's CUB bundle
    # Note that starting CuPy v8 we no longer use CUB_PATH

    # for <cupy/complex.cuh>
    cupy_header = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                               '../cupy/_core/include')
    global _jitify_path
    _jitify_path = os.path.join(cupy_header, 'cupy/jitify')
    if cuda_path:
        cuda_cub_path = os.path.join(cuda_path, 'include', 'cub')
        if not os.path.exists(cuda_cub_path):
            cuda_cub_path = None
    elif rocm_path:
        cuda_cub_path = os.path.join(rocm_path, 'include', 'hipcub')
        if not os.path.exists(cuda_cub_path):
            cuda_cub_path = None
    else:
        cuda_cub_path = None
    global _cub_path
    if cuda_cub_path:
        _cub_path = cuda_cub_path
    elif not use_hip:  # CuPy's bundle doesn't work for ROCm
        _cub_path = os.path.join(cupy_header, 'cupy', 'cub')
    else:
        raise Exception('Please install hipCUB and retry')
    include_dirs.insert(0, _cub_path)
    include_dirs.insert(1, cupy_header)

    return {
        'include_dirs': include_dirs,
        'library_dirs': library_dirs,
        'define_macros': define_macros,
        'language': 'c++',
        'extra_compile_args': extra_compile_args,
    }


def _match_output_lines(output_lines, regexs):
    # Matches regular expressions `regexs` against `output_lines` and finds the
    # consecutive matching lines from `output_lines`.
    # `None` is returned if no match is found.
    if len(output_lines) < len(regexs):
        return None

    matches = [None] * len(regexs)
    for i in range(len(output_lines) - len(regexs)):
        for j in range(len(regexs)):
            m = re.match(regexs[j], output_lines[i + j])
            if not m:
                break
            matches[j] = m
        else:
            # Match found
            return matches

    # No match
    return None


def get_compiler_base_options(compiler_path):
    """Returns base options for nvcc compiler.

    """
    global _compiler_base_options
    if _compiler_base_options is None:
        _compiler_base_options = _get_compiler_base_options(compiler_path)
    return _compiler_base_options


def _get_compiler_base_options(compiler_path):
    # Try compiling a dummy code.
    # If the compilation fails, try to parse the output of compilation
    # and try to compose base options according to it.
    # compiler_path is the path to nvcc (CUDA) or hipcc (ROCm/HIP)
    with _tempdir() as temp_dir:
        test_cu_path = os.path.join(temp_dir, 'test.cu')
        test_out_path = os.path.join(temp_dir, 'test.out')
        with open(test_cu_path, 'w') as f:
            f.write('int main() { return 0; }')
        proc = subprocess.Popen(
            compiler_path + ['-o', test_out_path, test_cu_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        stdoutdata, stderrdata = proc.communicate()
        stderrlines = stderrdata.split(b'\n')
        if proc.returncode != 0:

            # No supported host compiler
            matches = _match_output_lines(
                stderrlines,
                [
                    b'^ERROR: No supported gcc/g\\+\\+ host compiler found, '
                    b'but .* is available.$',
                    b'^ *Use \'nvcc (.*)\' to use that instead.$',
                ])
            if matches is not None:
                base_opts = matches[1].group(1)
                base_opts = base_opts.decode('utf8').split(' ')
                return base_opts

            # Unknown error
            raise RuntimeError(
                'Encountered unknown error while testing nvcc:\n' +
                stderrdata.decode('utf8'))

    return []


_cuda_version = None
_hip_version = None
_thrust_version = None
_cudnn_version = None
_nccl_version = None
_cutensor_version = None
_cub_path = None
_cub_version = None
_jitify_path = None
_jitify_version = None
_compute_capabilities = None
_cusparselt_version = None
_cugraph_version = None


def check_cuda_version(compiler, settings):
    global _cuda_version
    try:
        out = build_and_run(compiler, '''
        #include <cuda.h>
        #include <stdio.h>
        int main() {
          printf("%d", CUDA_VERSION);
          return 0;
        }
        ''', include_dirs=settings['include_dirs'])

    except Exception as e:
        utils.print_warning('Cannot check CUDA version', str(e))
        return False

    _cuda_version = int(out)

    if _cuda_version < minimum_cuda_version:
        utils.print_warning(
            'CUDA version is too old: %d' % _cuda_version,
            'CUDA 9.2 or newer is required')
        return False

    return True


def get_cuda_version(formatted=False):
    """Return CUDA Toolkit version cached in check_cuda_version()."""
    global _cuda_version
    if _cuda_version is None:
        msg = 'check_cuda_version() must be called first.'
        raise RuntimeError(msg)
    if formatted:
        return str(_cuda_version)
    return _cuda_version


def check_hip_version(compiler, settings):
    global _hip_version
    try:
        out = build_and_run(compiler, '''
        #include <hip/hip_version.h>
        #include <stdio.h>
        int main() {
          printf("%d", HIP_VERSION);
          return 0;
        }
        ''', include_dirs=settings['include_dirs'])

    except Exception as e:
        utils.print_warning('Cannot check HIP version', str(e))
        return False

    _hip_version = int(out)

    if _hip_version < minimum_hip_version:
        utils.print_warning(
            'ROCm/HIP version is too old: %d' % _hip_version,
            'ROCm 3.5.0 or newer is required')
        return False

    return True


def get_hip_version(formatted=False):
    """Return ROCm version cached in check_hip_version()."""
    global _hip_version
    if _hip_version is None:
        msg = 'check_hip_version() must be called first.'
        raise RuntimeError(msg)
    if formatted:
        return str(_hip_version)
    return _hip_version


def check_compute_capabilities(compiler, settings):
    """Return compute capabilities of the installed devices."""
    global _compute_capabilities
    try:
        src = '''#include <cuda_runtime_api.h>
        #include <stdio.h>
        int main() {
          cudaDeviceProp prop;
          int device_count;
          int i;
          cudaGetDeviceCount(&device_count);
          for(i=0; i < device_count; i++) {
              cudaGetDeviceProperties(&prop, i);
              printf("%d%d ", prop.major,prop.minor);
          }
          return 0;
        }
        '''
        out = build_and_run(
            compiler, src,
            include_dirs=settings['include_dirs'],
            libraries=('cudart',),
            library_dirs=settings['library_dirs'])
        _compute_capabilities = set([int(o) for o in out.split()])
    except Exception as e:
        utils.print_warning('Cannot check cuDNN version\n{0}'.format(e))
        return False

    return True


def get_compute_capabilities(formatted=False):
    return _compute_capabilities


def check_thrust_version(compiler, settings):
    global _thrust_version

    try:
        out = build_and_run(compiler, '''
        #include <thrust/version.h>
        #include <stdio.h>

        int main() {
          printf("%d", THRUST_VERSION);
          return 0;
        }
        ''', include_dirs=settings['include_dirs'])
    except Exception as e:
        utils.print_warning('Cannot check Thrust version\n{0}'.format(e))
        return False

    _thrust_version = int(out)

    return True


def get_thrust_version(formatted=False):
    """Return Thrust version cached in check_thrust_version()."""
    global _thrust_version
    if _thrust_version is None:
        msg = 'check_thrust_version() must be called first.'
        raise RuntimeError(msg)
    if formatted:
        return str(_thrust_version)
    return _thrust_version


def check_cudnn_version(compiler, settings):
    global _cudnn_version
    try:
        out = build_and_run(compiler, '''
        #include <cudnn.h>
        #include <stdio.h>
        int main() {
          printf("%d", CUDNN_VERSION);
          return 0;
        }
        ''', include_dirs=settings['include_dirs'])

    except Exception as e:
        utils.print_warning('Cannot check cuDNN version\n{0}'.format(e))
        return False

    _cudnn_version = int(out)

    if not minimum_cudnn_version <= _cudnn_version:
        min_major = str(minimum_cudnn_version)
        utils.print_warning(
            'Unsupported cuDNN version: {}'.format(str(_cudnn_version)),
            'cuDNN >=v{} is required'.format(min_major))
        return False

    return True


def get_cudnn_version(formatted=False):
    """Return cuDNN version cached in check_cudnn_version()."""
    global _cudnn_version
    if _cudnn_version is None:
        msg = 'check_cudnn_version() must be called first.'
        raise RuntimeError(msg)
    if formatted:
        return str(_cudnn_version)
    return _cudnn_version


def check_nccl_version(compiler, settings):
    global _nccl_version

    # NCCL 1.x does not provide version information.
    try:
        out = build_and_run(compiler,
                            '''
                            #ifndef CUPY_USE_HIP
                            #include <nccl.h>
                            #else
                            #include <rccl.h>
                            #endif
                            #include <stdio.h>
                            #ifdef NCCL_MAJOR
                            #ifndef NCCL_VERSION_CODE
                            #  define NCCL_VERSION_CODE \
                            (NCCL_MAJOR * 1000 + NCCL_MINOR * 100 + NCCL_PATCH)
                            #endif
                            #else
                            #  define NCCL_VERSION_CODE 0
                            #endif
                            int main() {
                              printf("%d", NCCL_VERSION_CODE);
                              return 0;
                            }
                            ''',
                            include_dirs=settings['include_dirs'],
                            define_macros=settings['define_macros'])

    except Exception as e:
        utils.print_warning('Cannot include NCCL\n{0}'.format(e))
        return False

    _nccl_version = int(out)

    return True


def get_nccl_version(formatted=False):
    """Return NCCL version cached in check_nccl_version()."""
    global _nccl_version
    if _nccl_version is None:
        msg = 'check_nccl_version() must be called first.'
        raise RuntimeError(msg)
    if formatted:
        if _nccl_version == 0:
            return '1.x'
        return str(_nccl_version)
    return _nccl_version


def check_nvtx(compiler, settings):
    if PLATFORM_WIN32:
        path = os.environ.get('NVTOOLSEXT_PATH', None)
        if path is None:
            utils.print_warning(
                'NVTX unavailable: NVTOOLSEXT_PATH is not set')
        elif not os.path.exists(path):
            utils.print_warning(
                'NVTX unavailable: NVTOOLSEXT_PATH is set but the directory '
                'does not exist')
        elif utils.search_on_path(['nvToolsExt64_1.dll']) is None:
            utils.print_warning(
                'NVTX unavailable: nvToolsExt64_1.dll not found in PATH')
        else:
            return True
        return False
    return True


def check_cub_version(compiler, settings):
    global _cub_version
    global _cub_path

    # This is guaranteed to work for any CUB source because the search
    # precedence follows that of include paths.
    # - On CUDA, CUB < 1.9.9 does not provide version.cuh and would error out
    # - On ROCm, hipCUB has the same version as rocPRIM (as of ROCm 3.5.0)
    try:
        out = build_and_run(compiler,
                            '''
                            #ifndef CUPY_USE_HIP
                            #include <cub/version.cuh>
                            #else
                            #include <hipcub/hipcub_version.hpp>
                            #endif
                            #include <stdio.h>

                            int main() {
                              #ifndef CUPY_USE_HIP
                              printf("%d", CUB_VERSION);
                              #else
                              printf("%d", HIPCUB_VERSION);
                              #endif
                              return 0;
                            }''',
                            include_dirs=settings['include_dirs'],
                            define_macros=settings['define_macros'])
    except Exception as e:
        # could be in a git submodule?
        try:
            # CuPy's bundle
            cupy_cub_include = _cub_path
            a = subprocess.run(' '.join(['git', 'describe', '--tags']),
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               shell=True, cwd=cupy_cub_include)
            if a.returncode == 0:
                tag = a.stdout.decode()[:-1]

                # CUB's tag convention changed after 1.8.0: "v1.9.0" -> "1.9.0"
                # In any case, we normalize it to be in line with CUB_VERSION
                if tag.startswith('v'):
                    tag = tag[1:]
                tag = tag.split('.')
                out = int(tag[0]) * 100000 + int(tag[1]) * 100
                try:
                    out += int(tag[2])
                except ValueError:
                    # there're local commits so tag is like 1.8.0-1-gdcbb288f,
                    # we add the number of commits to the version
                    local_patch = tag[2].split('-')
                    out += int(local_patch[0]) + int(local_patch[1])
            else:
                raise RuntimeError('Cannot determine CUB version from git tag'
                                   '\n{0}'.format(e))
        except Exception as e:
            utils.print_warning('Cannot determine CUB version\n{0}'.format(e))
            # 0: CUB is not built (makes no sense), -1: built with unknown ver
            out = -1

    _cub_version = int(out)
    settings['define_macros'].append(('CUPY_CUB_VERSION_CODE', _cub_version))
    return True  # we always build CUB


def get_cub_version(formatted=False):
    """Return CUB version cached in check_cub_version()."""
    global _cub_version
    if _cub_version is None:
        msg = 'check_cub_version() must be called first.'
        raise RuntimeError(msg)
    if formatted:
        if _cub_version == -1:
            return '<unknown>'
        return str(_cub_version)
    return _cub_version


def check_jitify_version(compiler, settings):
    global _jitify_version

    try:
        cupy_jitify_include = _jitify_path
        # Unfortunately Jitify does not have any identifiable name (branch,
        # tag, etc), so we must use the commit here
        a = subprocess.run(' '.join(['git', 'rev-parse', '--short', 'HEAD']),
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           shell=True, cwd=cupy_jitify_include)
        if a.returncode == 0:
            out = a.stdout.decode()[:-1]  # unlike elsewhere, out is a str here
        else:
            raise RuntimeError('Cannot determine Jitify version from git')
    except Exception as e:
        utils.print_warning('Cannot determine Jitify version\n{}'.format(e))
        # 0: Jitify is not built (makes no sense), -1: built with unknown ver
        out = -1

    _jitify_version = out
    settings['define_macros'].append(('CUPY_JITIFY_VERSION_CODE',
                                      _jitify_version))
    return True  # we always build Jitify


def get_jitify_version(formatted=False):
    """Return Jitify version cached in check_jitify_version()."""
    global _jitify_version
    if _jitify_version is None:
        msg = 'check_jitify_version() must be called first.'
        raise RuntimeError(msg)
    if formatted:
        if _jitify_version == -1:
            return '<unknown>'
        return _jitify_version
    raise RuntimeError('Jitify version is a commit string')


def check_cutensor_version(compiler, settings):
    global _cutensor_version
    try:
        out = build_and_run(compiler, '''
        #include <cutensor.h>
        #include <stdio.h>
        #ifdef CUTENSOR_MAJOR
        #ifndef CUTENSOR_VERSION
        #define CUTENSOR_VERSION \
                (CUTENSOR_MAJOR * 1000 + CUTENSOR_MINOR * 100 + CUTENSOR_PATCH)
        #endif
        #else
        #  define CUTENSOR_VERSION 0
        #endif
        int main(int argc, char* argv[]) {
          printf("%d", CUTENSOR_VERSION);
          return 0;
        }
        ''', include_dirs=settings['include_dirs'])

    except Exception as e:
        utils.print_warning('Cannot check cuTENSOR version\n{0}'.format(e))
        return False

    _cutensor_version = int(out)

    if _cutensor_version < 1000:
        utils.print_warning(
            'Unsupported cuTENSOR version: {}'.format(_cutensor_version)
        )
        return False

    return True


def get_cutensor_version(formatted=False):
    """Return cuTENSOR version cached in check_cutensor_version()."""
    global _cutensor_version
    if _cutensor_version is None:
        msg = 'check_cutensor_version() must be called first.'
        raise RuntimeError(msg)
    return _cutensor_version


def check_cusparselt_version(compiler, settings):
    global _cusparselt_version
    try:
        out = build_and_run(compiler, '''
        #include <cusparseLt.h>
        #include <stdio.h>
        #ifndef CUSPARSELT_VERSION
        #define CUSPARSELT_VERSION 0
        #endif
        int main(int argc, char* argv[]) {
          printf("%d", CUSPARSELT_VERSION);
          return 0;
        }
        ''', include_dirs=settings['include_dirs'])

    except Exception as e:
        utils.print_warning('Cannot check cuSPARSELt version\n{0}'.format(e))
        return False

    _cusparselt_version = int(out)
    return True


def get_cusparselt_version(formatted=False):
    """Return cuSPARSELt version cached in check_cusparselt_version()."""
    global _cusparselt_version
    if _cusparselt_version is None:
        msg = 'check_cusparselt_version() must be called first.'
        raise RuntimeError(msg)
    return _cusparselt_version


def check_cugraph_version(compiler, settings):
    global _cugraph_version
    try:
        build_and_run(compiler, '''
        #include <stdio.h>
        #include <cugraph/raft/error.hpp>
        int main(int argc, char* argv[]) {
          return 0;
        }
        ''', include_dirs=settings['include_dirs'])
    except Exception as e:
        utils.print_warning('Cannot find cuGraph header files\n{0}'.format(e))
        return False

    try:
        out = build_and_run(compiler, '''
        #include <stdio.h>
        #include <cugraph/version_config.hpp>
        int main(int argc, char* argv[]) {
          printf("%d", CUGRAPH_VERSION_MAJOR * 10000
                     + CUGRAPH_VERSION_MINOR * 100
                     + CUGRAPH_VERSION_PATCH);
          return 0;
        }
        ''', include_dirs=settings['include_dirs'])
    except Exception as e:
        utils.print_warning('Cannot find cuGRAPH version information\n{0}'.
                            format(e))
        _cugraph_version = 0
        return True

    _cugraph_version = int(out)
    return True


def get_cugraph_version(formatted=False):
    """Return cuGraph version cached in check_cugraph_version()."""
    global _cugraph_version
    if _cugraph_version is None:
        msg = 'check_cugraph_version() must be called first.'
        raise RuntimeError(msg)
    return _cugraph_version


def build_shlib(compiler, source, libraries=(),
                include_dirs=(), library_dirs=(), define_macros=None,
                extra_compile_args=()):
    with _tempdir() as temp_dir:
        fname = os.path.join(temp_dir, 'a.cpp')
        with open(fname, 'w') as f:
            f.write(source)
        objects = compiler.compile([fname], output_dir=temp_dir,
                                   include_dirs=include_dirs,
                                   macros=define_macros,
                                   extra_postargs=list(extra_compile_args))

        try:
            postargs = ['/MANIFEST'] if PLATFORM_WIN32 else []
            compiler.link_shared_lib(objects,
                                     os.path.join(temp_dir, 'a'),
                                     libraries=libraries,
                                     library_dirs=library_dirs,
                                     extra_postargs=postargs,
                                     target_lang='c++')
        except Exception as e:
            msg = 'Cannot build a stub file.\nOriginal error: {0}'.format(e)
            raise Exception(msg)


def build_and_run(compiler, source, libraries=(),
                  include_dirs=(), library_dirs=(), define_macros=None,
                  extra_compile_args=()):
    with _tempdir() as temp_dir:
        fname = os.path.join(temp_dir, 'a.cpp')
        with open(fname, 'w') as f:
            f.write(source)

        objects = compiler.compile([fname], output_dir=temp_dir,
                                   include_dirs=include_dirs,
                                   macros=define_macros,
                                   extra_postargs=list(extra_compile_args))

        try:
            postargs = ['/MANIFEST'] if PLATFORM_WIN32 else []
            compiler.link_executable(objects,
                                     os.path.join(temp_dir, 'a'),
                                     libraries=libraries,
                                     library_dirs=library_dirs,
                                     extra_postargs=postargs,
                                     target_lang='c++')
        except Exception as e:
            msg = 'Cannot build a stub file.\nOriginal error: {0}'.format(e)
            raise Exception(msg)

        try:
            out = subprocess.check_output(os.path.join(temp_dir, 'a'))
            return out

        except Exception as e:
            msg = 'Cannot execute a stub file.\nOriginal error: {0}'.format(e)
            raise Exception(msg)
