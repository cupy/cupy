from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ast
import contextlib
import os
import re
import shutil
import subprocess
import sys
import tempfile

from os.path import join as pjoin
from pkg_resources import resource_filename

import distutils
from distutils import ccompiler
from distutils import errors
from distutils import msvccompiler
from distutils import sysconfig
from distutils import unixccompiler

import cupy


def print_warning(*lines):
    print('**************************************************')
    for line in lines:
        print('*** WARNING: %s' % line)
    print('**************************************************')


def get_path(key):
    return os.environ.get(key, '').split(os.pathsep)


def search_on_path(filenames):
    for p in get_path('PATH'):
        for filename in filenames:
            full = os.path.join(p, filename)
            if os.path.exists(full):
                return os.path.abspath(full)
    return None


PLATFORM_DARWIN = sys.platform.startswith('darwin')
PLATFORM_LINUX = sys.platform.startswith('linux')
PLATFORM_WIN32 = sys.platform.startswith('win32')

minimum_cuda_version = 8000
minimum_cudnn_version = 5000
maximum_cudnn_version = 7999

_cuda_path = 'NOT_INITIALIZED'
_compiler_base_options = None
_cuda_info = None


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

    nvcc_path = search_on_path(('nvcc', 'nvcc.exe'))
    cuda_path_default = None
    if nvcc_path is None:
        print_warning('nvcc not in path.',
                      'Please set path to nvcc.')
    else:
        cuda_path_default = os.path.normpath(
            os.path.join(os.path.dirname(nvcc_path), '..'))

    cuda_path = os.environ.get('CUDA_PATH', '')  # Nvidia default on Windows
    if len(cuda_path) > 0 and cuda_path != cuda_path_default:
        print_warning(
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
        return distutil.split_quoted(nvcc)

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


def get_compiler_setting():
    cuda_path = get_cuda_path()

    include_dirs = []
    library_dirs = []
    define_macros = []

    if cuda_path:
        include_dirs.append(os.path.join(cuda_path, 'include'))
        if PLATFORM_WIN32:
            library_dirs.append(os.path.join(cuda_path, 'bin'))
            library_dirs.append(os.path.join(cuda_path, 'lib', 'x64'))
        else:
            library_dirs.append(os.path.join(cuda_path, 'lib64'))
            library_dirs.append(os.path.join(cuda_path, 'lib'))
    if PLATFORM_DARWIN:
        library_dirs.append('/usr/local/cuda/lib')

    if PLATFORM_WIN32:
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


def _get_cuda_info():
    nvcc_path = get_nvcc_path()

    code = '''
    #include <cuda.h>
    #include <stdio.h>
    int main(int argc, char* argv[]) {
        int nDevices;
        cudaGetDeviceCount(&nDevices);

        printf("{\\n");
        printf("'cuda_version': %d,\\n", CUDA_VERSION);
        printf("'devices': [\\n");

        for(int d=0; d < nDevices; ++d) {
            cudaDeviceProp props;
            cudaGetDeviceProperties(&props, d);

            printf("{\\n");
            printf("'name'                  :'%s',\\n", props.name);
            printf("'major'                 :  %d,\\n", props.major);
            printf("'minor'                 :  %d,\\n", props.minor);
            printf("'total_global_mem'      :  %u,\\n", props.totalGlobalMem);
            printf("'warp_size'             :  %u,\\n", props.warpSize);
            printf("'max_threads_per_block' :  %u,\\n", props.maxThreadsPerBlock);
            printf("'max_thread_size'       : (%u,%u,%u),\\n",
                                                        props.maxThreadsDim[0],
                                                        props.maxThreadsDim[1],
                                                        props.maxThreadsDim[2]);
            printf("'max_grid_size'         : (%u,%u,%u),\\n",
                                                        props.maxGridSize[0],
                                                        props.maxGridSize[1],
                                                        props.maxGridSize[2]);
            printf("'device_overlap'        :  %s,\\n", props.deviceOverlap ? "True" : "False");
            printf("'async_engines'         :  %u,\\n", props.asyncEngineCount);
            printf("'multiprocessors'       :  %u,\\n", props.multiProcessorCount);
            printf("},\\n");
        }

        printf("]\\n");
        printf("}\\n");

        return 0;
    }
    '''  # noqa

    with _tempdir() as temp_dir:
        test_cu_path = os.path.join(temp_dir, 'test.cu')
        test_out_path = os.path.join(temp_dir, 'test.out')

        with open(test_cu_path, 'w') as f:
            f.write(code)

        proc = subprocess.Popen(
            nvcc_path + ['-o', test_out_path, test_cu_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)

        stdoutdata, stderrdata = proc.communicate()
        stderrlines = stderrdata.split(b'\n')

        if proc.returncode != 0:
            raise RuntimeError("Cannot determine "
                               "compute architecture {0}"
                               .format(stderrdata))

        try:
            out = subprocess.check_output(test_out_path)
        except Exception as e:
            msg = 'Cannot execute a stub file.\nOriginal error: {0}'.format(e)
            raise Exception(msg)

        return ast.literal_eval(out)


def get_cuda_info():
    global _cuda_info

    if _cuda_info is None:
        _cuda_info = _get_cuda_info()

    return _cuda_info


def _format_cuda_version(version):
    return str(version)


def get_cuda_version(formatted=False):
    """Return CUDA Toolkit version cached in check_cuda_version()."""
    _cuda_version = get_cuda_info()['cuda_version']

    if _cuda_version < minimum_cuda_version:
        raise ValueError('CUDA version is too old: %d'
                         'CUDA v7.0 or newer is required' % _cuda_version)

    return str(_cuda_version) if formatted else _cuda_version


def get_gencode_options():
    return ["--generate-code=arch=compute_{a},code=sm_{a}".format(
            a=dev['major']*10 + dev['minor'])
            for dev in get_cuda_info()['devices']]


def build_and_run(compiler, source, libraries=(),
                  include_dirs=(), library_dirs=(), define_macros=None):
    with _tempdir() as temp_dir:
        fname = os.path.join(temp_dir, 'a.cpp')
        with open(fname, 'w') as f:
            f.write(source)

        objects = compiler.compile([fname], output_dir=temp_dir,
                                   include_dirs=include_dirs,
                                   macros=define_macros)

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
            nvcc_path = get_nvcc_path()
            base_opts = get_compiler_base_options()
            self.set_executable('compiler_so', nvcc_path)

            cuda_version = get_cuda_version()
            postargs = get_gencode_options() + [
                '-O2', '--compiler-options="-fPIC"']
            postargs += extra_postargs
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

        compiler_so = get_nvcc_path()
        cc_args = self._get_cc_args(pp_opts, debug, extra_preargs)
        cuda_version = get_cuda_version()
        postargs = get_gencode_options() + ['-O2']
        postargs += ['-Xcompiler', '/MD']
        postargs += extra_postargs
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


_compiler = None


def get_compiler():
    global _compiler

    if _compiler is None:
        if not PLATFORM_WIN32:
            _compiler = _UnixCCompiler()
        else:
            _compiler = _MSVCCompiler()

    return _compiler


@contextlib.contextmanager
def stdchannel_redirected(stdchannel, dest_filename):
    """
    A context manager to temporarily redirect stdout or stderr

    e.g.:

    with stdchannel_redirected(sys.stderr, os.devnull):
        if compiler.has_function('clock_gettime', libraries=['rt']):
            libraries.append('rt')
    """

    try:
        oldstdchannel = os.dup(stdchannel.fileno())
        dest_file = open(dest_filename, 'w')
        os.dup2(dest_file.fileno(), stdchannel.fileno())

        yield
    finally:
        if oldstdchannel is not None:
            os.dup2(oldstdchannel, stdchannel.fileno())
        if dest_file is not None:
            dest_file.close()


def compile_using_nvcc(source, options=None, arch=None, filename='kern.cu'):
    options = options or []

    if arch is None:
        cuda_info = get_cuda_info()
        arch = min([dev['major']*10 + dev['minor']
                   for dev in cuda_info['devices']])

    cc = get_compiler()
    settings = get_compiler_setting()
    arch = "--generate-code=arch=compute_{a},code=sm_{a}".format(a=arch)

    options += ('-cubin',)

    cupy_path = resource_filename("cupy", pjoin("core", "include"))
    settings['include_dirs'].append(cupy_path)

    with _tempdir() as tmpdir:
        tmpfile = pjoin(tmpdir, filename)

        with open(tmpfile, "w") as f:
            f.write(source)

        try:
            stderr_file = pjoin(tmpdir, "stderr.txt")

            with stdchannel_redirected(sys.stderr, stderr_file):
                objects = cc.compile([tmpfile],
                                     include_dirs=settings['include_dirs'],
                                     macros=settings['define_macros'],
                                     extra_postargs=options)
        except errors.CompileError as e:
            # Obtain nvcc error output
            with open(stderr_file, "r") as f:
                errs = f.read()

            # Format source with line numbers
            formatted_source = ["%-5d %s" % (i, l) for i, l
                                in enumerate(code.split('\n'), 1)]

            lines = ["The following source code",
                     formatted_source,
                     "",
                     "created the following compilation errors",
                     "",
                     errs.strip(),
                     str(e).strip()]

            ex = errors.CompileError("\n".join(lines))
            raise (ex, None, sys.exc_info()[2])

        # Should only be one file
        assert len(objects) == 1

        # Return the cubin binary
        with open(objects[0], "rb") as f:
            return f.read()
