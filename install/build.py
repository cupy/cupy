import contextlib
import distutils.util
import os
import re
import shutil
import subprocess
import sys
import tempfile

from install import utils


minimum_cuda_version = 7000
minimum_cudnn_version = 4000
maximum_cudnn_version = 7999
# Although cuda 7.0 includes cusolver,
# we tentatively support cusolver in cuda 8.0 only because
# provided functions are insufficient to implement cupy.linalg
minimum_cusolver_cuda_version = 8000

_cuda_path = 'NOT_INITIALIZED'
_compiler_base_options = None


@contextlib.contextmanager
def _tempdir():
    temp_dir = tempfile.mkdtemp()
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def get_cuda_path():
    global _cuda_path

    # Use a magic word to represent the cache not filled because None is a
    # valid return value.
    if _cuda_path is not 'NOT_INITIALIZED':
        return _cuda_path

    nvcc_path = utils.search_on_path(('nvcc', 'nvcc.exe'))
    cuda_path_default = None
    if nvcc_path is None:
        utils.print_warning('nvcc not in path.',
                            'Please set path to nvcc.')
    else:
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

    if sys.platform == 'win32':
        nvcc_bin = 'bin/nvcc.exe'
    else:
        nvcc_bin = 'bin/nvcc'

    nvcc_path = os.path.join(cuda_path, nvcc_bin)
    if os.path.exists(nvcc_path):
        return [nvcc_path]
    else:
        return None


def get_compiler_setting():
    cuda_path = get_cuda_path()

    include_dirs = []
    library_dirs = []
    define_macros = []

    if cuda_path:
        include_dirs.append(os.path.join(cuda_path, 'include'))
        if sys.platform == 'win32':
            library_dirs.append(os.path.join(cuda_path, 'bin'))
            library_dirs.append(os.path.join(cuda_path, 'lib', 'x64'))
        else:
            library_dirs.append(os.path.join(cuda_path, 'lib64'))
            library_dirs.append(os.path.join(cuda_path, 'lib'))
    if sys.platform == 'darwin':
        library_dirs.append('/usr/local/cuda/lib')

    if sys.platform == 'win32':
        nvtoolsext_path = os.environ.get('NVTOOLSEXT_PATH', '')
        if os.path.exists(nvtoolsext_path):
            include_dirs.append(os.path.join(nvtoolsext_path, 'include'))
            library_dirs.append(os.path.join(nvtoolsext_path, 'lib', 'x64'))
        else:
            define_macros.append(('CUPY_NO_NVTX', '1'))

    return {
        'include_dirs': include_dirs,
        'library_dirs': library_dirs,
        'define_macros': define_macros,
        'language': 'c++',
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


def check_cuda_version(compiler, settings):
    global _cuda_version
    try:
        out = build_and_run(compiler, '''
        #include <cuda.h>
        #include <stdio.h>
        int main(int argc, char* argv[]) {
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


def get_cuda_version():
    """Return CUDA Toolkit version cached in check_cuda_version()."""
    global _cuda_version
    if _cuda_version is None:
        msg = 'check_cuda_version() must be called first.'
        raise Exception(msg)
    return _cuda_version


def check_cudnn_version(compiler, settings):
    try:
        out = build_and_run(compiler, '''
        #include <cudnn.h>
        #include <stdio.h>
        int main(int argc, char* argv[]) {
          printf("%d", CUDNN_VERSION);
          return 0;
        }
        ''', include_dirs=settings['include_dirs'])

    except Exception as e:
        utils.print_warning('Cannot check cuDNN version\n{0}'.format(e))
        return False

    cudnn_version = int(out)

    if not minimum_cudnn_version <= cudnn_version <= maximum_cudnn_version:
        min_major = minimum_cudnn_version // 1000
        max_major = maximum_cudnn_version // 1000
        utils.print_warning(
            'Unsupported cuDNN version: %d' % cudnn_version,
            'cuDNN v%d<= and <=v%d is required' % (min_major, max_major))
        return False

    return True


def check_nccl_version(compiler, settings):
    # NCCL does not provide version information.
    # It only check whether there is nccl.h.
    try:
        build_and_run(compiler, '''
        #include <nccl.h>
        int main(int argc, char* argv[]) {
          return 0;
        }
        ''', include_dirs=settings['include_dirs'])

    except Exception as e:
        utils.print_warning('Cannot include NCCL\n{0}'.format(e))
        return False

    return True


def check_cusolver_version(compiler, settings):
    # As an initial cusolver does not have cusolverGetProperty,
    # we use CUDA_VERSION instead.
    try:
        out = build_and_run(compiler, '''
        #include <cuda.h>
        #include <stdio.h>
        int main(int argc, char* argv[]) {
          printf("%d", CUDA_VERSION);
          return 0;
        }
        ''', include_dirs=settings['include_dirs'])

    except Exception as e:
        utils.print_warning('Cannot check CUDA version', str(e))
        return False

    cuda_version = int(out)

    if cuda_version < minimum_cusolver_cuda_version:
        return False

    return True


def build_shlib(compiler, source, libraries=(),
                include_dirs=(), library_dirs=()):
    with _tempdir() as temp_dir:
        fname = os.path.join(temp_dir, 'a.cpp')
        with open(fname, 'w') as f:
            f.write(source)

        objects = compiler.compile([fname], output_dir=temp_dir,
                                   include_dirs=include_dirs)

        try:
            postargs = ['/MANIFEST'] if sys.platform == 'win32' else []
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
                  include_dirs=(), library_dirs=()):
    with _tempdir() as temp_dir:
        fname = os.path.join(temp_dir, 'a.cpp')
        with open(fname, 'w') as f:
            f.write(source)

        objects = compiler.compile([fname], output_dir=temp_dir,
                                   include_dirs=include_dirs)

        try:
            postargs = ['/MANIFEST'] if sys.platform == 'win32' else []
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
