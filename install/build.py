import contextlib
import distutils.util
import os
import re
import shutil
import subprocess
import sys
import tempfile

from install import utils


PLATFORM_DARWIN = sys.platform.startswith('darwin')
PLATFORM_LINUX = sys.platform.startswith('linux')
PLATFORM_WIN32 = sys.platform.startswith('win32')

minimum_cuda_version = 8000
minimum_cudnn_version = 5000
maximum_cudnn_version = 8099

_cuda_path = 'NOT_INITIALIZED'
_rocm_path = 'NOT_INITIALIZED'
_compiler_base_options = None


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
        include_dirs.append(os.path.join(rocm_path, 'rocrand', 'include'))
        library_dirs.append(os.path.join(rocm_path, 'lib'))
        library_dirs.append(os.path.join(rocm_path, 'rocrand', 'lib'))

    if use_hip:
        extra_compile_args.append('-std=c++11')

    if PLATFORM_DARWIN:
        library_dirs.append('/usr/local/cuda/lib')

    if PLATFORM_WIN32:
        nvtoolsext_path = os.environ.get('NVTOOLSEXT_PATH', '')
        if os.path.exists(nvtoolsext_path):
            include_dirs.append(os.path.join(nvtoolsext_path, 'include'))
            library_dirs.append(os.path.join(nvtoolsext_path, 'lib', 'x64'))
        else:
            define_macros.append(('CUPY_NO_NVTX', '1'))

    # for <cupy/complex.cuh>
    cupy_header = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                               '../cupy/core/include')
    cub_path = os.environ.get('CUB_PATH', '')
    if os.path.exists(cub_path):
        include_dirs.append(cupy_header)
        include_dirs.insert(0, cub_path)
    elif cuda_path:
        cub_path = os.path.join(cuda_path, 'include', 'cub')
        if os.path.exists(cub_path):
            include_dirs.append(cupy_header)

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


def get_compiler_base_options():
    """Returns base options for nvcc compiler.

    """
    global _compiler_base_options
    if _compiler_base_options is None:
        _compiler_base_options = _get_compiler_base_options()
    return _compiler_base_options


def _get_compiler_base_options():
    # Try compiling a dummy code.
    # If the compilation fails, try to parse the output of compilation
    # and try to compose base options according to it.
    nvcc_path = get_nvcc_path()
    with _tempdir() as temp_dir:
        test_cu_path = os.path.join(temp_dir, 'test.cu')
        test_out_path = os.path.join(temp_dir, 'test.out')
        with open(test_cu_path, 'w') as f:
            f.write('int main() { return 0; }')
        proc = subprocess.Popen(
            nvcc_path + ['-o', test_out_path, test_cu_path],
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
_thrust_version = None
_cudnn_version = None
_nccl_version = None


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
            'CUDA v7.0 or newer is required')
        return False

    return True


def _format_cuda_version(version):
    return str(version)


def get_cuda_version(formatted=False):
    """Return CUDA Toolkit version cached in check_cuda_version()."""
    global _cuda_version
    if _cuda_version is None:
        msg = 'check_cuda_version() must be called first.'
        raise RuntimeError(msg)
    if formatted:
        return _format_cuda_version(_cuda_version)
    return _cuda_version


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

    if not minimum_cudnn_version <= _cudnn_version <= maximum_cudnn_version:
        min_major = _format_cuda_version(minimum_cudnn_version)
        max_major = _format_cuda_version(maximum_cudnn_version)
        utils.print_warning(
            'Unsupported cuDNN version: {}'.format(
                _format_cuda_version(_cudnn_version)),
            'cuDNN v{}= and <=v{} is required'.format(min_major, max_major))
        return False

    return True


def get_cudnn_version(formatted=False):
    """Return cuDNN version cached in check_cudnn_version()."""
    global _cudnn_version
    if _cudnn_version is None:
        msg = 'check_cudnn_version() must be called first.'
        raise RuntimeError(msg)
    if formatted:
        return _format_cuda_version(_cudnn_version)
    return _cudnn_version


def check_nccl_version(compiler, settings):
    global _nccl_version

    # NCCL 1.x does not provide version information.
    try:
        out = build_and_run(compiler, '''
        #include <nccl.h>
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
        ''', include_dirs=settings['include_dirs'])

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
        return _format_cuda_version(_nccl_version)
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
